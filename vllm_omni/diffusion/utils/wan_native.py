# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_hf_file_to_dict
from vllm.transformers_utils.repo_utils import file_or_path_exists

logger = init_logger(__name__)

_WAN_NATIVE_ADAPTER_DIR = ".vllm_omni_wan22_native"
_WAN_NATIVE_ADAPTER_READY = ".ready"

_HIGH_NOISE_SUBDIR = "high_noise_model"
_LOW_NOISE_SUBDIR = "low_noise_model"
_TOKENIZER_SUBDIR_CANDIDATES = ("google/umt5-xxl", "tokenizer")
_T5_PTH_CANDIDATES = (
    "models_t5_umt5-xxl-enc-bf16.pth",
    "models_t5_umt5-xxl-enc-fp16.pth",
    "models_t5_umt5-xxl-enc.pth",
)
_VAE_PTH_CANDIDATES = ("Wan2.1_VAE.pth", "Wan2.2_VAE.pth", "vae.pth")
_LOCAL_LAYOUT_SCAN_MAX_DEPTH = 4


def _find_best_tensor_dict(payload: Any) -> dict[str, torch.Tensor]:
    best: dict[str, torch.Tensor] = {}
    best_count = 0

    def _visit(node: Any) -> None:
        nonlocal best, best_count
        if isinstance(node, dict):
            tensor_items = {str(k): v for k, v in node.items() if isinstance(v, torch.Tensor)}
            if len(tensor_items) > best_count:
                best = tensor_items
                best_count = len(tensor_items)
            for v in node.values():
                if isinstance(v, dict):
                    _visit(v)

    _visit(payload)
    return best


def _flatten_tensor_dict(payload: Any, prefix: str = "", out: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    if out is None:
        out = {}
    if isinstance(payload, dict):
        for k, v in payload.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, torch.Tensor):
                out[key] = v
            elif isinstance(v, dict):
                _flatten_tensor_dict(v, key, out)
    return out


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    state = payload
    if isinstance(state, dict):
        for key in ("state_dict", "model", "module"):
            nested = state.get(key)
            if isinstance(nested, dict):
                state = nested
                break

    flattened = _flatten_tensor_dict(state)
    if flattened:
        return flattened

    tensor_state = _find_best_tensor_dict(state)
    if tensor_state:
        return tensor_state

    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(state)!r}")
    raise RuntimeError("No tensor state dict found in checkpoint payload.")


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


def _strip_prefixes_repeated(state: dict[str, torch.Tensor], prefixes: tuple[str, ...]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        new_k = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if p and new_k.startswith(p):
                    new_k = new_k[len(p) :]
                    changed = True
                    break
        out[new_k] = v
    return out


def _remap_wan_blocks_text_encoder_to_umt5(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map WAN text encoder keys (blocks.*) to HF UMT5 encoder keys."""
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = key

        if key == "token_embedding.weight":
            # WAN native uses token_embedding, while HF UMT5 checkpoints keep
            # the shared embedding table as "shared.weight".
            new_key = "shared.weight"
        elif key == "norm.weight":
            new_key = "encoder.final_layer_norm.weight"
        elif key == "norm.bias":
            new_key = "encoder.final_layer_norm.bias"
        else:
            m = re.match(r"^blocks\.(\d+)\.(.+)$", key)
            if m:
                layer_idx = m.group(1)
                suffix = m.group(2)
                prefix = f"encoder.block.{layer_idx}."
                suffix_map = {
                    "norm1.weight": "layer.0.layer_norm.weight",
                    "norm1.bias": "layer.0.layer_norm.bias",
                    "attn.q.weight": "layer.0.SelfAttention.q.weight",
                    "attn.k.weight": "layer.0.SelfAttention.k.weight",
                    "attn.v.weight": "layer.0.SelfAttention.v.weight",
                    "attn.o.weight": "layer.0.SelfAttention.o.weight",
                    "attn.q.bias": "layer.0.SelfAttention.q.bias",
                    "attn.k.bias": "layer.0.SelfAttention.k.bias",
                    "attn.v.bias": "layer.0.SelfAttention.v.bias",
                    "attn.o.bias": "layer.0.SelfAttention.o.bias",
                    "norm2.weight": "layer.1.layer_norm.weight",
                    "norm2.bias": "layer.1.layer_norm.bias",
                    "ffn.gate.0.weight": "layer.1.DenseReluDense.wi_0.weight",
                    "ffn.gate.0.bias": "layer.1.DenseReluDense.wi_0.bias",
                    "ffn.fc1.weight": "layer.1.DenseReluDense.wi_1.weight",
                    "ffn.fc1.bias": "layer.1.DenseReluDense.wi_1.bias",
                    "ffn.fc2.weight": "layer.1.DenseReluDense.wo.weight",
                    "ffn.fc2.bias": "layer.1.DenseReluDense.wo.bias",
                    # WAN native position bias table name.
                    "pos_embedding.embedding.weight": "layer.0.SelfAttention.relative_attention_bias.weight",
                }
                if suffix in suffix_map:
                    new_key = prefix + suffix_map[suffix]
                else:
                    # Skip unknown keys by default.
                    continue

        out[new_key] = value
    return out


def _best_effort_load(module: torch.nn.Module, raw_state: dict[str, torch.Tensor], name: str) -> None:
    target_state = module.state_dict()
    target_keys = set(target_state.keys())
    common_prefixes = ("module.", "model.", "vae.", "first_stage_model.")
    candidates = [raw_state, _strip_prefixes(raw_state, common_prefixes)]

    if name == "text_encoder":
        text_prefixes = (
            "module.",
            "model.",
            "text_encoder.",
            "t5.",
            "umt5.",
            "transformer.",
            "backbone.",
            "network.",
            "encoder_decoder.",
        )
        candidates.append(_strip_prefixes_repeated(raw_state, text_prefixes))
        candidates.append(_remap_wan_blocks_text_encoder_to_umt5(raw_state))
        candidates.append(_strip_prefixes_repeated(_remap_wan_blocks_text_encoder_to_umt5(raw_state), text_prefixes))

    best_state = None
    best_overlap = -1
    for candidate in candidates:
        compatible = {}
        for k, v in candidate.items():
            t = target_state.get(k)
            if t is None:
                continue
            if tuple(getattr(v, "shape", ())) != tuple(getattr(t, "shape", ())):
                continue
            compatible[k] = v

        overlap = len(compatible)
        if overlap > best_overlap:
            best_overlap = overlap
            best_state = compatible

    if best_state is None or best_overlap <= 0:
        sample_source_keys = list(raw_state.keys())[:12]
        sample_target_keys = list(target_keys)[:12]
        raise RuntimeError(
            f"Cannot map {name} checkpoint keys to target module parameters. "
            f"sample_source_keys={sample_source_keys}; sample_target_keys={sample_target_keys}"
        )

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


def _first_existing_file(model_dir: Path, candidates: tuple[str, ...]) -> Path | None:
    for name in candidates:
        path = model_dir / name
        if path.exists():
            return path
    return None


def _first_existing_tokenizer_subdir(model_dir: Path) -> str | None:
    for subdir in _TOKENIZER_SUBDIR_CANDIDATES:
        subdir_path = model_dir / subdir
        if not subdir_path.is_dir():
            continue
        if any(
            (subdir_path / name).exists()
            for name in (
                "config.json",
                "tokenizer_config.json",
                "tokenizer.json",
                "spiece.model",
                "special_tokens_map.json",
            )
        ):
            return subdir
    return None


def _find_tensor_by_suffix(state: dict[str, torch.Tensor], suffixes: tuple[str, ...]) -> torch.Tensor | None:
    for key, value in state.items():
        for suffix in suffixes:
            if key.endswith(suffix):
                return value
    return None


def _infer_umt5_encoder_config_from_state(
    state: dict[str, torch.Tensor],
    default_relative_attention_num_buckets: int = 32,
    fallback_vocab_size: int | None = None,
):
    """Infer a UMT5 encoder config from checkpoint tensor shapes."""
    from transformers import UMT5Config

    state = _strip_prefixes(state, ("module.", "model.", "text_encoder."))

    shared = state.get("shared.weight")
    if shared is None:
        shared = state.get("encoder.embed_tokens.weight")
    if shared is None:
        shared = _find_tensor_by_suffix(
            state,
            (
                "shared.weight",
                "embed_tokens.weight",
                "word_embeddings.weight",
                "token_embedding.weight",
            ),
        )
    if shared is not None and shared.ndim == 2:
        vocab_size, d_model = int(shared.shape[0]), int(shared.shape[1])
    else:
        if fallback_vocab_size is None:
            raise RuntimeError("Cannot infer UMT5 config: missing shared/embed token weights and no fallback vocab.")
        vocab_size, d_model = int(fallback_vocab_size), 4096

    q_weight = state.get("encoder.block.0.layer.0.SelfAttention.q.weight")
    if q_weight is None:
        q_weight = _find_tensor_by_suffix(
            state,
            (
                "SelfAttention.q.weight",
                "self_attn.q_proj.weight",
                "self_attention.q_proj.weight",
                "attn.q.weight",
            ),
        )
    if q_weight is None or q_weight.ndim != 2:
        inner_dim = 4096
    elif int(q_weight.shape[1]) == d_model:
        inner_dim = int(q_weight.shape[0])
    elif int(q_weight.shape[0]) == d_model:
        inner_dim = int(q_weight.shape[1])
    else:
        inner_dim = int(max(q_weight.shape))

    relative_bias = state.get("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")
    if relative_bias is None:
        relative_bias = _find_tensor_by_suffix(
            state,
            (
                "relative_attention_bias.weight",
                "self_attn.relative_attention_bias.weight",
                "pos_embedding.embedding.weight",
            ),
        )
    if relative_bias is not None and relative_bias.ndim == 2:
        relative_attention_num_buckets = int(relative_bias.shape[0])
        num_heads = int(relative_bias.shape[1])
    else:
        relative_attention_num_buckets = default_relative_attention_num_buckets
        common_head_candidates = (64, 48, 40, 32, 24, 16, 8, 4, 2, 1)
        num_heads = next((h for h in common_head_candidates if inner_dim % h == 0 and d_model % h == 0), 8)

    if num_heads <= 0 or inner_dim % num_heads != 0:
        raise RuntimeError(
            f"Cannot infer UMT5 config: invalid heads (inner_dim={inner_dim}, num_heads={num_heads})."
        )
    d_kv = inner_dim // num_heads

    wi_0 = state.get("encoder.block.0.layer.1.DenseReluDense.wi_0.weight")
    wi = state.get("encoder.block.0.layer.1.DenseReluDense.wi.weight")
    if wi_0 is None:
        wi_0 = _find_tensor_by_suffix(
            state,
            (
                "DenseReluDense.wi_0.weight",
                "ffn.gate.weight",
                "ffn.gate.0.weight",
                "mlp.gate_proj.weight",
            ),
        )
    if wi is None:
        wi = _find_tensor_by_suffix(
            state,
            (
                "DenseReluDense.wi.weight",
                "DenseReluDense.wi_1.weight",
                "ffn.wi.weight",
                "ffn.fc1.weight",
                "mlp.up_proj.weight",
            ),
        )
    if wi_0 is not None and wi_0.ndim == 2:
        d_ff = int(wi_0.shape[0] if int(wi_0.shape[1]) == d_model else wi_0.shape[1])
        feed_forward_proj = "gated-gelu"
    elif wi is not None and wi.ndim == 2:
        d_ff = int(wi.shape[0] if int(wi.shape[1]) == d_model else wi.shape[1])
        feed_forward_proj = "relu"
    else:
        d_ff = 10240
        feed_forward_proj = "gated-gelu"

    layer_pattern = re.compile(r"(?:encoder\.(?:block|layers)|blocks)\.(\d+)\.")
    layer_ids = []
    for key in state:
        m = layer_pattern.search(key)
        if m:
            layer_ids.append(int(m.group(1)))
    num_layers = max(layer_ids) + 1 if layer_ids else 24

    return UMT5Config(
        vocab_size=vocab_size,
        d_model=d_model,
        d_kv=d_kv,
        d_ff=d_ff,
        num_layers=num_layers,
        num_decoder_layers=num_layers,
        num_heads=num_heads,
        relative_attention_num_buckets=relative_attention_num_buckets,
        feed_forward_proj=feed_forward_proj,
    )


def _has_model_weights(dir_path: Path) -> bool:
    if not dir_path.is_dir():
        return False
    return any(
        (dir_path / name).exists()
        for name in (
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )
    )


def _has_tokenizer_files(dir_path: Path) -> bool:
    if not dir_path.is_dir():
        return False
    return any(
        (dir_path / name).exists()
        for name in (
            "config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "spiece.model",
            "special_tokens_map.json",
        )
    )


def _is_adapter_ready(adapter_dir: Path) -> bool:
    if not adapter_dir.is_dir():
        return False
    if not (adapter_dir / "model_index.json").exists():
        return False
    if not (adapter_dir / "transformer" / "config.json").exists():
        return False
    if not (adapter_dir / "transformer_2" / "config.json").exists():
        return False
    if not _has_tokenizer_files(adapter_dir / "tokenizer"):
        return False
    if not ((adapter_dir / "text_encoder" / "config.json").exists() and _has_model_weights(adapter_dir / "text_encoder")):
        return False
    if not ((adapter_dir / "vae" / "config.json").exists() and _has_model_weights(adapter_dir / "vae")):
        return False
    return True


def _find_local_diffusers_i2v_ref_dir(model_path: Path) -> Path | None:
    """Best-effort locate a local Wan2.2-I2V-A14B-Diffusers directory."""
    env_path = os.environ.get("VLLM_OMNI_WAN_I2V_DIFFUSERS_REF")
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path).expanduser().resolve())

    name = model_path.name
    if name.endswith("-Diffusers"):
        candidates.append(model_path)
    else:
        candidates.append(model_path.parent / f"{name}-Diffusers")
        candidates.append(model_path.parent / "Wan2.2-I2V-A14B-Diffusers")
        if "I2V-A14B" in name:
            candidates.append(model_path.parent / name.replace("I2V-A14B", "I2V-A14B-Diffusers"))

    for cand in candidates:
        if not cand.is_dir():
            continue
        text_encoder_ok = (cand / "text_encoder" / "config.json").exists() and _has_model_weights(cand / "text_encoder")
        tokenizer_ok = _has_tokenizer_files(cand / "tokenizer")
        if text_encoder_ok and tokenizer_ok:
            return cand
    return None


def _has_high_low_configs(model_dir: Path) -> bool:
    return (model_dir / _HIGH_NOISE_SUBDIR / "config.json").exists() and (
        model_dir / _LOW_NOISE_SUBDIR / "config.json"
    ).exists()


def resolve_local_wan22_native_root(model_dir: str, max_depth: int = _LOCAL_LAYOUT_SCAN_MAX_DEPTH) -> str | None:
    """Resolve actual WAN native checkpoint root from a local path.

    The user may pass a parent directory (e.g. ModelScope cache root). This
    function searches a few levels down for a directory containing both
    `high_noise_model/config.json` and `low_noise_model/config.json`.
    """
    if not os.path.isdir(model_dir):
        return None

    start = Path(model_dir).resolve()
    if _has_high_low_configs(start):
        return str(start)

    queue: list[tuple[Path, int]] = [(start, 0)]
    candidates: list[Path] = []

    while queue:
        current, depth = queue.pop(0)
        if depth >= max_depth:
            continue

        try:
            subdirs = [p for p in current.iterdir() if p.is_dir()]
        except Exception:
            continue

        for subdir in subdirs:
            if _has_high_low_configs(subdir):
                candidates.append(subdir)
            queue.append((subdir, depth + 1))

    if not candidates:
        return None

    # Prefer the shallowest candidate path.
    candidates.sort(key=lambda p: (len(p.parts), len(str(p))))
    return str(candidates[0])


def _infer_wan22_native_model_class_from_name(model_name_or_path: str) -> str:
    lower = model_name_or_path.lower()
    if "i2v" in lower or "ti2v" in lower:
        return "WanImageToVideoPipeline"
    # This adapter branch targets Wan2.2 I2V native checkpoints.
    return "WanImageToVideoPipeline"


def infer_wan22_native_model_class(model_name_or_path: str, transformer_cfg: dict[str, Any] | None = None) -> str:
    if os.path.isdir(model_name_or_path):
        model_dir = Path(model_name_or_path)
        has_image_encoder = (model_dir / "image_encoder" / "config.json").exists()
        has_image_processor = (model_dir / "image_processor").exists()
        if has_image_encoder and has_image_processor:
            return "WanImageToVideoPipeline"

    if transformer_cfg is not None:
        image_dim = transformer_cfg.get("image_dim")
        added_kv_proj_dim = transformer_cfg.get("added_kv_proj_dim")
        if image_dim is not None and added_kv_proj_dim is not None:
            return "WanImageToVideoPipeline"
    return _infer_wan22_native_model_class_from_name(model_name_or_path)


def looks_like_wan22_native_checkpoint(model_name_or_path: str) -> bool:
    lower = model_name_or_path.lower()
    if "diffusers" in lower:
        return False
    return "wan2.2" in lower and ("i2v" in lower or "ti2v" in lower)


def is_local_wan22_native_layout(model_dir: str) -> bool:
    resolved_root = resolve_local_wan22_native_root(model_dir)
    if resolved_root is None:
        return False
    base = Path(resolved_root)
    has_high = (base / _HIGH_NOISE_SUBDIR / "config.json").exists()
    has_low = (base / _LOW_NOISE_SUBDIR / "config.json").exists()
    has_t5 = _first_existing_file(base, _T5_PTH_CANDIDATES) is not None
    has_tokenizer = _first_existing_tokenizer_subdir(base) is not None
    has_vae = _first_existing_file(base, _VAE_PTH_CANDIDATES) is not None
    return has_high and has_low and has_t5 and has_tokenizer and has_vae


def is_local_wan22_native_candidate_layout(model_dir: str) -> bool:
    """Lightweight detection for WAN native layout before full asset checks.

    This is used by routing logic (model-type/stage config resolution) so we
    don't fail early. Full asset validation still happens in
    `prepare_local_wan22_native_for_vllm`.
    """
    return resolve_local_wan22_native_root(model_dir) is not None


def has_wan22_native_remote_layout(model_name_or_path: str) -> bool:
    if is_local_wan22_native_layout(model_name_or_path):
        return True

    if not looks_like_wan22_native_checkpoint(model_name_or_path):
        return False

    try:
        has_high = file_or_path_exists(model_name_or_path, f"{_HIGH_NOISE_SUBDIR}/config.json", revision=None)
        has_low = file_or_path_exists(model_name_or_path, f"{_LOW_NOISE_SUBDIR}/config.json", revision=None)
        has_t5 = any(file_or_path_exists(model_name_or_path, name, revision=None) for name in _T5_PTH_CANDIDATES)
        return has_high and has_low and has_t5
    except Exception:
        return False


def has_wan22_native_remote_candidate_layout(model_name_or_path: str) -> bool:
    if is_local_wan22_native_candidate_layout(model_name_or_path):
        return True

    if not looks_like_wan22_native_checkpoint(model_name_or_path):
        return False

    try:
        has_high = file_or_path_exists(model_name_or_path, f"{_HIGH_NOISE_SUBDIR}/config.json", revision=None)
        has_low = file_or_path_exists(model_name_or_path, f"{_LOW_NOISE_SUBDIR}/config.json", revision=None)
        return has_high and has_low
    except Exception:
        return False


def load_wan22_native_transformer_config(model_name_or_path: str) -> dict[str, Any] | None:
    if os.path.isdir(model_name_or_path):
        resolved_root = resolve_local_wan22_native_root(model_name_or_path)
        if resolved_root is None:
            return None
        cfg_path = Path(resolved_root) / _HIGH_NOISE_SUBDIR / "config.json"
        if cfg_path.exists():
            with cfg_path.open(encoding="utf-8") as f:
                return json.load(f)
        return None

    try:
        cfg = get_hf_file_to_dict(f"{_HIGH_NOISE_SUBDIR}/config.json", model_name_or_path)
    except Exception:
        cfg = None
    return cfg


def prepare_local_wan22_native_for_vllm(model_dir: str, prefer_model_class_name: str | None = None) -> str:
    resolved_root = resolve_local_wan22_native_root(model_dir)
    if resolved_root is None:
        raise ValueError(
            "Not a WAN native local checkpoint candidate layout. "
            "Expected to find high_noise_model/config.json and low_noise_model/config.json "
            f"under: {model_dir}"
        )
    model_path = Path(resolved_root).resolve()

    adapter_dir = model_path / _WAN_NATIVE_ADAPTER_DIR
    ready_path = adapter_dir / _WAN_NATIVE_ADAPTER_READY
    if ready_path.exists() and _is_adapter_ready(adapter_dir):
        return str(adapter_dir)
    if ready_path.exists():
        logger.warning("Found stale WAN adapter marker, rebuilding adapter dir: %s", adapter_dir)
        try:
            ready_path.unlink()
        except FileNotFoundError:
            pass

    tokenizer_subdir = _first_existing_tokenizer_subdir(model_path)
    t5_ckpt_path = _first_existing_file(model_path, _T5_PTH_CANDIDATES)
    vae_ckpt_path = _first_existing_file(model_path, _VAE_PTH_CANDIDATES)

    missing_assets: list[str] = []
    if tokenizer_subdir is None:
        missing_assets.append(f"tokenizer subdir in one of {list(_TOKENIZER_SUBDIR_CANDIDATES)}")
    if t5_ckpt_path is None:
        missing_assets.append(f"T5 checkpoint in one of {list(_T5_PTH_CANDIDATES)}")
    if vae_ckpt_path is None:
        missing_assets.append(f"VAE checkpoint in one of {list(_VAE_PTH_CANDIDATES)}")
    if missing_assets:
        raise FileNotFoundError(
            "Missing required WAN native assets: "
            + ", ".join(missing_assets)
            + f". model_dir={model_path}"
        )

    adapter_dir.mkdir(parents=True, exist_ok=True)
    lock_path = adapter_dir / ".lock"

    lock_fp = None
    try:
        lock_fp = open(lock_path, "w", encoding="utf-8")
        try:
            import fcntl

            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
        except Exception:
            pass

        if ready_path.exists() and _is_adapter_ready(adapter_dir):
            return str(adapter_dir)

        # 1) Reuse transformer assets.
        _safe_link_or_copy(model_path / _HIGH_NOISE_SUBDIR, adapter_dir / "transformer")
        _safe_link_or_copy(model_path / _LOW_NOISE_SUBDIR, adapter_dir / "transformer_2")

        # 2) Prefer text assets from a local Diffusers reference checkpoint when available.
        diffusers_ref_dir = _find_local_diffusers_i2v_ref_dir(model_path)
        if diffusers_ref_dir is not None:
            logger.info("Using local Diffusers reference for text assets: %s", diffusers_ref_dir)

        # Tokenizer directory
        tokenizer_dst = adapter_dir / "tokenizer"
        if tokenizer_dst.exists() and not _has_tokenizer_files(tokenizer_dst):
            shutil.rmtree(tokenizer_dst, ignore_errors=True)
        if diffusers_ref_dir is not None:
            _safe_link_or_copy(diffusers_ref_dir / "tokenizer", tokenizer_dst)
        else:
            _safe_link_or_copy(model_path / tokenizer_subdir, tokenizer_dst)

        # 3) Prepare text_encoder in HF format.
        from transformers import AutoConfig, UMT5EncoderModel

        text_encoder_dir = adapter_dir / "text_encoder"
        if text_encoder_dir.exists() and not (
            (text_encoder_dir / "config.json").exists() and _has_model_weights(text_encoder_dir)
        ):
            shutil.rmtree(text_encoder_dir, ignore_errors=True)
        if not text_encoder_dir.exists():
            if diffusers_ref_dir is not None and (diffusers_ref_dir / "text_encoder").is_dir():
                _safe_link_or_copy(diffusers_ref_dir / "text_encoder", text_encoder_dir)
            else:
                text_encoder_dir.mkdir(parents=True, exist_ok=True)
                t5_ckpt = _torch_load_cpu(t5_ckpt_path)
                t5_state = _extract_state_dict(t5_ckpt)
                try:
                    t5_config = AutoConfig.from_pretrained(
                        str(model_path),
                        subfolder=tokenizer_subdir,
                        local_files_only=True,
                    )
                    logger.info(
                        "Loaded UMT5 config from local tokenizer subdir: %s",
                        model_path / tokenizer_subdir,
                    )
                except Exception:
                    fallback_vocab_size = None
                    try:
                        from transformers import AutoTokenizer

                        tokenizer_obj = AutoTokenizer.from_pretrained(
                            str(model_path / tokenizer_subdir),
                            local_files_only=True,
                        )
                        fallback_vocab_size = len(tokenizer_obj)
                    except Exception:
                        fallback_vocab_size = None
                    t5_config = _infer_umt5_encoder_config_from_state(
                        t5_state,
                        fallback_vocab_size=fallback_vocab_size,
                    )
                    logger.warning(
                        "Tokenizer subdir %s has no model config.json; inferred UMT5 config from checkpoint shapes.",
                        model_path / tokenizer_subdir,
                    )
                text_encoder = UMT5EncoderModel(t5_config)
                try:
                    _best_effort_load(text_encoder, t5_state, "text_encoder")
                except RuntimeError as e:
                    raise RuntimeError(
                        f"{e}. If you have Wan2.2-I2V-A14B-Diffusers locally, "
                        "set VLLM_OMNI_WAN_I2V_DIFFUSERS_REF to that path and retry."
                    ) from e
                text_encoder.save_pretrained(text_encoder_dir, safe_serialization=True)

        # 4) Convert Wan VAE checkpoint to diffusers AutoencoderKLWan folder.
        from diffusers.models.autoencoders import AutoencoderKLWan

        vae_dir = adapter_dir / "vae"
        if vae_dir.exists() and not ((vae_dir / "config.json").exists() and _has_model_weights(vae_dir)):
            shutil.rmtree(vae_dir, ignore_errors=True)
        if not vae_dir.exists():
            vae_dir.mkdir(parents=True, exist_ok=True)
            vae = AutoencoderKLWan()
            vae_ckpt = _torch_load_cpu(vae_ckpt_path)
            vae_state = _extract_state_dict(vae_ckpt)
            _best_effort_load(vae, vae_state, "vae")
            vae.save_pretrained(vae_dir, safe_serialization=True)

        # Optional CLIP image encoder assets for WAN I2V checkpoints.
        has_image_encoder = (model_path / "image_encoder" / "config.json").exists()
        has_image_processor = (model_path / "image_processor").exists()
        if has_image_encoder and has_image_processor:
            _safe_link_or_copy(model_path / "image_encoder", adapter_dir / "image_encoder")
            _safe_link_or_copy(model_path / "image_processor", adapter_dir / "image_processor")
        elif has_image_encoder or has_image_processor:
            logger.warning(
                "Detected partial I2V image encoder assets in %s, image encoder conditioning may be disabled.",
                model_path,
            )

        # 5) Write minimal model_index.json for vllm-omni diffusion initialization.
        try:
            import diffusers

            diffusers_version = getattr(diffusers, "__version__", "unknown")
        except Exception:
            diffusers_version = "unknown"

        high_noise_cfg = load_wan22_native_transformer_config(str(model_path))
        model_class_name = prefer_model_class_name or infer_wan22_native_model_class(str(model_path), high_noise_cfg)

        model_index: dict[str, Any] = {
            "_class_name": model_class_name,
            "_diffusers_version": diffusers_version,
            "expand_timesteps": False,
            "transformer": ["vllm_omni", "WanTransformer3DModel"],
            "transformer_2": ["vllm_omni", "WanTransformer3DModel"],
            "text_encoder": ["transformers", "UMT5EncoderModel"],
            "tokenizer": ["transformers", "AutoTokenizer"],
            "vae": ["diffusers", "AutoencoderKLWan"],
        }
        if has_image_encoder and has_image_processor:
            model_index["image_encoder"] = ["transformers", "CLIPVisionModel"]
            model_index["image_processor"] = ["transformers", "CLIPImageProcessor"]

        with (adapter_dir / "model_index.json").open("w", encoding="utf-8") as f:
            json.dump(model_index, f, indent=2)

        ready_path.write_text("ok\n", encoding="utf-8")
        logger.info("Prepared WAN native checkpoint for vLLM-Omni at: %s", adapter_dir)
        return str(adapter_dir)
    finally:
        if lock_fp is not None:
            try:
                lock_fp.close()
            except Exception:
                pass
