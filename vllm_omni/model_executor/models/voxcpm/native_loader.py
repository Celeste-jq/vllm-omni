from __future__ import annotations

import json
from pathlib import Path

import torch

from .import_utils import _import_voxcpm_audio_vae_classes, _import_voxcpm_model_class
from .runtime_utils import _force_cuda_available_for_npu, _prepare_runtime_model_dir
from .stage_wrappers import _DirectVoxCPMAudioVAE, _DirectVoxCPMLatentGenerator


def _load_native_voxcpm_model(
    model_path: str,
    *,
    device: torch.device,
    dtype: str | None,
):
    VoxCPMModel = _import_voxcpm_model_class()
    runtime_model_path = _prepare_runtime_model_dir(model_path, target_device=device, target_dtype=dtype)

    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.set_device(device)

    with _force_cuda_available_for_npu(device):
        tts_model = VoxCPMModel.from_local(
            runtime_model_path,
            optimize=device.type == "cuda",
        )

    return tts_model


def _load_native_voxcpm_latent_generator(
    model_path: str,
    *,
    device: torch.device,
    dtype: str | None,
) -> _DirectVoxCPMLatentGenerator:
    return _DirectVoxCPMLatentGenerator(_load_native_voxcpm_model(model_path, device=device, dtype=dtype))


def _load_native_voxcpm_audio_vae(
    model_path: str,
    *,
    device: torch.device,
) -> _DirectVoxCPMAudioVAE:
    AudioVAE, AudioVAEConfig = _import_voxcpm_audio_vae_classes()
    runtime_model_path = _prepare_runtime_model_dir(model_path, target_device=device, target_dtype="float32")
    config_dict = json.loads((Path(runtime_model_path) / "config.json").read_text())
    audio_vae_config = config_dict.get("audio_vae_config")
    if audio_vae_config is not None:
        audio_vae = AudioVAE(config=AudioVAEConfig(**audio_vae_config))
    else:
        audio_vae = AudioVAE()

    state_dict = torch.load(
        Path(runtime_model_path) / "audiovae.pth",
        map_location="cpu",
        weights_only=True,
    )["state_dict"]
    audio_vae.load_state_dict(state_dict, strict=True)
    audio_vae = audio_vae.to(device=device, dtype=torch.float32).eval()
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.set_device(device)
    patch_size = int(config_dict.get("patch_size", 2))
    return _DirectVoxCPMAudioVAE(audio_vae, patch_size=patch_size)
