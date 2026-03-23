# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_hf_file_to_dict
from vllm.transformers_utils.repo_utils import file_or_path_exists

logger = init_logger(__name__)

_WAN_NATIVE_T2V_HINT = "wan2.2-t2v-a14b"
_WAN_NATIVE_ADAPTER_DIR = ".vllm_omni_wan22_t2v_native"
_WAN_NATIVE_ADAPTER_READY = ".ready"

_HIGH_NOISE_SUBDIR = "high_noise_model"
_LOW_NOISE_SUBDIR = "low_noise_model"
_TOKENIZER_SUBDIR = "google/umt5-xxl"
_T5_PTH = "models_t5_umt5-xxl-enc-bf16.pth"
_VAE_PTH_CANDIDATES = ("Wan2.1_VAE.pth", "Wan2.2_VAE.pth")


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    state = payload
    if isinstance(state, dict):
        for key in ("state_dict", "model", "module"):
            nested = state.get(key)
            if isinstance(nested, dict):
                state = nested
                break

    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(state)!r}")

    return {str(k): v for k, v in state.items() if isinstance(v, torch.Tensor)}


def _torch_load_cpu(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _strip_prefixes(state: dict[str, torch.Tensor], prefixes: tuple[str, ...]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        new_k = k
        for p in prefixes:
            if p and new_k.startswith(p):
                new_k = new_k[len(p) :]
                break
        out[new_k] = v
    return out


def _best_effort_load(module: torch.nn.Module, raw_state: dict[str, torch.Tensor], name: str) -> None:
    target_keys = set(module.state_dict().keys())
    candidates = [
        raw_state,
        _strip_prefixes(raw_state, ("module.", "model.", "vae.", "first_stage_model.")),
    ]

    best_state = None
    best_overlap = -1
    for candidate in candidates:
        overlap = len(set(candidate.keys()) & target_keys)
        if overlap > best_overlap:
            best_overlap = overlap
            best_state = candidate

    if best_state is None or best_overlap <= 0:
        raise RuntimeError(f"Cannot map {name} checkpoint keys to target module parameters.")

    missing, unexpected = module.load_state_dict(best_state, strict=False)
    logger.info(
        "Loaded native WAN %s checkpoint (matched=%d, missing=%d, unexpected=%d).",
        name,
        best_overlap,
        len(missing),
        len(unexpected),
    )


def _safe_link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.symlink(src, dst, target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _find_vae_checkpoint(model_dir: Path) -> Path:
    for name in _VAE_PTH_CANDIDATES:
        path = model_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing VAE checkpoint. Expected one of: {', '.join(_VAE_PTH_CANDIDATES)}")


def looks_like_wan22_native_t2v(model_name_or_path: str) -> bool:
    lower = model_name_or_path.lower()
    if _WAN_NATIVE_T2V_HINT in lower and "diffusers" not in lower:
        return True
    return "wan2.2" in lower and "t2v" in lower and "diffusers" not in lower


def is_local_wan22_native_t2v_layout(model_dir: str) -> bool:
    if not os.path.isdir(model_dir):
        return False
    base = Path(model_dir)
    has_high = (base / _HIGH_NOISE_SUBDIR / "config.json").exists()
    has_low = (base / _LOW_NOISE_SUBDIR / "config.json").exists()
    has_t5 = (base / _T5_PTH).exists()
    has_tokenizer = (base / _TOKENIZER_SUBDIR / "config.json").exists()
    has_vae = any((base / name).exists() for name in _VAE_PTH_CANDIDATES)
    return has_high and has_low and has_t5 and has_tokenizer and has_vae


def has_wan22_native_t2v_remote_layout(model_name_or_path: str) -> bool:
    if is_local_wan22_native_t2v_layout(model_name_or_path):
        return True

    if not looks_like_wan22_native_t2v(model_name_or_path):
        return False

    try:
        return (
            file_or_path_exists(model_name_or_path, f"{_HIGH_NOISE_SUBDIR}/config.json", revision=None)
            and file_or_path_exists(model_name_or_path, f"{_LOW_NOISE_SUBDIR}/config.json", revision=None)
            and file_or_path_exists(model_name_or_path, _T5_PTH, revision=None)
        )
    except Exception:
        return False


def load_wan22_native_t2v_transformer_config(model_name_or_path: str) -> dict[str, Any] | None:
    if os.path.isdir(model_name_or_path):
        cfg_path = Path(model_name_or_path) / _HIGH_NOISE_SUBDIR / "config.json"
        if cfg_path.exists():
            with cfg_path.open(encoding="utf-8") as f:
                return json.load(f)
        return None

    try:
        cfg = get_hf_file_to_dict(f"{_HIGH_NOISE_SUBDIR}/config.json", model_name_or_path)
    except Exception:
        cfg = None
    return cfg


def prepare_local_wan22_native_t2v_for_vllm(model_dir: str) -> str:
    if not is_local_wan22_native_t2v_layout(model_dir):
        raise ValueError(f"Not a WAN native T2V local checkpoint layout: {model_dir}")

    model_path = Path(model_dir).resolve()
    adapter_dir = model_path / _WAN_NATIVE_ADAPTER_DIR
    ready_path = adapter_dir / _WAN_NATIVE_ADAPTER_READY
    if ready_path.exists():
        return str(adapter_dir)

    adapter_dir.mkdir(parents=True, exist_ok=True)
    lock_path = adapter_dir / ".lock"

    lock_fp = None
    try:
        lock_fp = open(lock_path, "w", encoding="utf-8")
        try:
            import fcntl

            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
        except Exception:
            # Best-effort lock; continue without fcntl on unsupported platforms.
            pass

        if ready_path.exists():
            return str(adapter_dir)

        # 1) Reuse transformer/tokenizer assets by linking/copying.
        _safe_link_or_copy(model_path / _HIGH_NOISE_SUBDIR, adapter_dir / "transformer")
        _safe_link_or_copy(model_path / _LOW_NOISE_SUBDIR, adapter_dir / "transformer_2")
        _safe_link_or_copy(model_path / _TOKENIZER_SUBDIR, adapter_dir / "tokenizer")

        # 2) Convert T5 encoder checkpoint to HF-style text_encoder folder.
        from transformers import AutoConfig, UMT5EncoderModel

        text_encoder_dir = adapter_dir / "text_encoder"
        if not text_encoder_dir.exists():
            text_encoder_dir.mkdir(parents=True, exist_ok=True)
            t5_config = AutoConfig.from_pretrained(str(model_path), subfolder=_TOKENIZER_SUBDIR, local_files_only=True)
            text_encoder = UMT5EncoderModel(t5_config)
            t5_ckpt = _torch_load_cpu(model_path / _T5_PTH)
            t5_state = _extract_state_dict(t5_ckpt)
            _best_effort_load(text_encoder, t5_state, "text_encoder")
            text_encoder.save_pretrained(text_encoder_dir, safe_serialization=True)

        # 3) Convert Wan VAE checkpoint to diffusers AutoencoderKLWan folder.
        from diffusers.models.autoencoders import AutoencoderKLWan

        vae_dir = adapter_dir / "vae"
        if not vae_dir.exists():
            vae_dir.mkdir(parents=True, exist_ok=True)
            vae = AutoencoderKLWan()
            vae_ckpt_path = _find_vae_checkpoint(model_path)
            vae_ckpt = _torch_load_cpu(vae_ckpt_path)
            vae_state = _extract_state_dict(vae_ckpt)
            _best_effort_load(vae, vae_state, "vae")
            vae.save_pretrained(vae_dir, safe_serialization=True)

        # 4) Write minimal model_index.json for vllm-omni diffusion initialization.
        try:
            import diffusers

            diffusers_version = getattr(diffusers, "__version__", "unknown")
        except Exception:
            diffusers_version = "unknown"

        model_index = {
            "_class_name": "WanPipeline",
            "_diffusers_version": diffusers_version,
            "expand_timesteps": False,
            "transformer": ["vllm_omni", "WanTransformer3DModel"],
            "transformer_2": ["vllm_omni", "WanTransformer3DModel"],
            "text_encoder": ["transformers", "UMT5EncoderModel"],
            "tokenizer": ["transformers", "AutoTokenizer"],
            "vae": ["diffusers", "AutoencoderKLWan"],
        }
        with (adapter_dir / "model_index.json").open("w", encoding="utf-8") as f:
            json.dump(model_index, f, indent=2, ensure_ascii=False)

        ready_path.write_text("ok\n", encoding="utf-8")
        logger.info("Prepared WAN native T2V checkpoint for vLLM-Omni at: %s", adapter_dir)
        return str(adapter_dir)
    finally:
        if lock_fp is not None:
            try:
                lock_fp.close()
            except Exception:
                pass
