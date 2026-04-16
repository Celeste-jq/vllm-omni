from __future__ import annotations

from typing import Any

import torch

VOXCPM_LATENT_MAGIC = 131071


def _coerce_scalar_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar tensor, got shape={tuple(value.shape)}")
        return int(value.detach().cpu().item())
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        if len(value) != 1:
            raise ValueError(f"Expected single-element container, got len={len(value)}")
        return _coerce_scalar_int(value[0], default=default)
    return int(value)


def extract_left_context_size(info: dict[str, Any] | None, default: int = 0) -> int:
    if not isinstance(info, dict):
        return default
    return max(0, _coerce_scalar_int(info.get("left_context_size"), default=default))


def serialize_latent_to_codes(latent: Any) -> list[int]:
    latent_tensor = latent if isinstance(latent, torch.Tensor) else torch.as_tensor(latent)
    latent_tensor = latent_tensor.detach().cpu().contiguous()
    if latent_tensor.ndim == 3:
        if latent_tensor.shape[0] != 1:
            raise ValueError(f"Expected batch=1 latent tensor, got shape={tuple(latent_tensor.shape)}")
        latent_tensor = latent_tensor.squeeze(0)
    if latent_tensor.ndim != 2:
        raise ValueError(f"Unsupported latent_audio_feat shape for async chunk: {tuple(latent_tensor.shape)}")
    latent_dim, time_dim = int(latent_tensor.shape[0]), int(latent_tensor.shape[1])
    packed = latent_tensor.to(torch.bfloat16).contiguous().view(torch.uint16).reshape(-1).to(torch.int32)
    return [VOXCPM_LATENT_MAGIC, latent_dim, time_dim, *packed.tolist()]


def recover_latent_from_input_ids(input_ids: torch.Tensor | None) -> torch.Tensor | None:
    if input_ids is None or input_ids.numel() == 0:
        return None
    flat_ids = input_ids.detach().reshape(-1).to("cpu")
    if flat_ids.numel() < 4 or int(flat_ids[0].item()) != VOXCPM_LATENT_MAGIC:
        return None
    latent_dim = int(flat_ids[1].item())
    time_dim = int(flat_ids[2].item())
    payload = flat_ids[3:]
    expected = latent_dim * time_dim
    if latent_dim <= 0 or time_dim <= 0:
        raise ValueError(f"Invalid VoxCPM latent header: latent_dim={latent_dim}, time_dim={time_dim}")
    if int(payload.numel()) != expected:
        raise ValueError(
            "Invalid VoxCPM latent payload size: "
            f"expected={expected}, actual={int(payload.numel())}, "
            f"latent_dim={latent_dim}, time_dim={time_dim}"
        )
    packed = payload.to(dtype=torch.int32).to(torch.uint16)
    return packed.view(torch.bfloat16).to(torch.float32).reshape(1, latent_dim, time_dim)
