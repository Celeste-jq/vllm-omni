from __future__ import annotations

import json
import shutil
import tempfile
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


def _device_to_string(device: torch.device) -> str:
    if device.index is None:
        return device.type
    return f"{device.type}:{device.index}"


def _normalize_dtype_name(dtype: Any) -> str | None:
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        mapping = {
            torch.bfloat16: "bfloat16",
            torch.float16: "float16",
            torch.float32: "float32",
        }
        return mapping.get(dtype, str(dtype).removeprefix("torch."))
    dtype_str = str(dtype)
    return dtype_str.removeprefix("torch.")


def _resolve_runtime_device(vllm_config: VllmConfig) -> torch.device:
    try:
        from vllm_omni.platforms import current_omni_platform

        return current_omni_platform.get_torch_device()
    except Exception:
        pass

    device = getattr(getattr(vllm_config, "device_config", None), "device", None)
    if isinstance(device, torch.device):
        return device
    if device:
        return torch.device(device)
    return torch.device("cpu")


def _prepare_runtime_model_dir(
    model_path: str | Path,
    *,
    target_device: torch.device,
    target_dtype: str | None,
) -> str:
    source_dir = Path(model_path)
    config_path = source_dir / "config.json"
    if not config_path.exists():
        return str(source_dir)

    config_text = config_path.read_text()
    config_dict = json.loads(config_text)
    desired_device = target_device.type
    desired_dtype = target_dtype or config_dict.get("dtype")

    if config_dict.get("device") == desired_device and config_dict.get("dtype") == desired_dtype:
        return str(source_dir)

    digest = sha256(f"{source_dir.resolve()}:{config_text}:{desired_device}:{desired_dtype}".encode()).hexdigest()[:16]
    runtime_dir = Path(tempfile.gettempdir()) / "vllm_omni_voxcpm_runtime" / digest
    runtime_dir.mkdir(parents=True, exist_ok=True)

    for entry in source_dir.iterdir():
        target = runtime_dir / entry.name
        if entry.name == "config.json" or target.exists():
            continue
        try:
            target.symlink_to(entry, target_is_directory=entry.is_dir())
        except OSError as exc:
            logger.warning(
                "Falling back to copying VoxCPM runtime artifact %s into %s because symlink creation failed: %s",
                entry,
                runtime_dir,
                exc,
            )
            if entry.is_dir():
                shutil.copytree(entry, target, dirs_exist_ok=True)
            else:
                shutil.copy2(entry, target)

    patched_config = dict(config_dict)
    patched_config["device"] = desired_device
    if desired_dtype is not None:
        patched_config["dtype"] = desired_dtype
    (runtime_dir / "config.json").write_text(json.dumps(patched_config, indent=2, sort_keys=True))
    return str(runtime_dir)


@contextmanager
def _force_cuda_available_for_npu(device: torch.device):
    if device.type != "npu":
        yield
        return

    with patch("torch.cuda.is_available", return_value=True):
        yield


def _is_torchcodec_load_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "torchcodec" in message or "load_with_torchcodec" in message


def _load_audio_with_soundfile(
    prompt_wav_path: str,
    *,
    sample_rate: int,
) -> torch.Tensor:
    try:
        import soundfile as sf
    except ImportError:
        raise

    audio_np, source_sr = sf.read(prompt_wav_path, dtype="float32", always_2d=True)
    audio = torch.from_numpy(np.ascontiguousarray(audio_np.T))

    if audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)

    if int(source_sr) != int(sample_rate):
        try:
            import torchaudio
        except ImportError as exc:
            raise ImportError("torchaudio is required for resampling prompt audio.") from exc
        audio = torchaudio.functional.resample(audio, int(source_sr), int(sample_rate))

    return audio


def _build_prompt_cache_with_soundfile(model: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    if args:
        prompt_text = args[0]
        prompt_wav_path = args[1] if len(args) > 1 else kwargs.get("prompt_wav_path")
    else:
        prompt_text = kwargs.get("prompt_text")
        prompt_wav_path = kwargs.get("prompt_wav_path")

    if not prompt_text or not prompt_wav_path:
        raise ValueError("prompt_text and prompt_wav_path are required")

    audio = _load_audio_with_soundfile(prompt_wav_path, sample_rate=int(model.sample_rate))

    patch_len = model.patch_size * model.chunk_size
    if audio.size(1) % patch_len != 0:
        padding_size = patch_len - audio.size(1) % patch_len
        audio = torch.nn.functional.pad(audio, (padding_size, 0))

    audio_feat = model.audio_vae.encode(audio.to(model.device), model.sample_rate).cpu()
    audio_feat = audio_feat.view(
        model.audio_vae.latent_dim,
        -1,
        model.patch_size,
    ).permute(1, 2, 0)

    return {
        "prompt_text": prompt_text,
        "audio_feat": audio_feat,
    }
