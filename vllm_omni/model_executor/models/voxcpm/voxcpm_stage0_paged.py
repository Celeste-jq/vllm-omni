from __future__ import annotations

from typing import Any

import torch

from .voxcpm_request_state import Stage0RequestState


def _canonicalize_patch_tensor(patch: Any) -> torch.Tensor:
    patch_tensor = patch if isinstance(patch, torch.Tensor) else torch.as_tensor(patch)
    patch_tensor = patch_tensor.detach().to(torch.float32)
    if patch_tensor.ndim not in (2, 3):
        raise ValueError(f"Unsupported latent patch shape: {tuple(patch_tensor.shape)}")
    return patch_tensor.contiguous()


def append_latent_patch(state: Stage0RequestState, patch: Any) -> int:
    patch_tensor = _canonicalize_patch_tensor(patch)
    if patch_tensor.ndim == 2:
        patches = [patch_tensor]
    else:
        patches = [chunk.contiguous() for chunk in patch_tensor]

    if not patches:
        return 0

    state.generated_latent_history.extend(patches)
    state.generated_latent_frames += len(patches)
    state.pending_latent_patch = patches[-1]
    return len(patches)


def resolve_left_context_frames(
    codec_left_context_frames: int | None = None,
    streaming_prefix_len: int | None = None,
) -> int:
    if codec_left_context_frames is not None:
        return max(0, int(codec_left_context_frames))
    if streaming_prefix_len is None:
        return 0
    return max(0, int(streaming_prefix_len) - 1)


def maybe_emit_window(state: Stage0RequestState, codec_left_context_frames: int) -> dict[str, Any] | None:
    new_frames = state.generated_latent_frames - state.emitted_latent_frames
    if new_frames <= 0:
        return None

    left_context_size = min(max(0, int(codec_left_context_frames)), state.emitted_latent_frames)
    start_index = state.generated_latent_frames - (left_context_size + new_frames)
    window_patches = state.generated_latent_history[start_index : state.generated_latent_frames]
    if not window_patches:
        return None

    state.emitted_latent_frames = state.generated_latent_frames
    return {
        "latent_audio_feat": torch.stack(window_patches, dim=0),
        "left_context_size": left_context_size,
        "finished": False,
    }


def maybe_emit_terminal_payload(state: Stage0RequestState) -> dict[str, Any] | None:
    if not state.terminal_payload_pending or state.terminal_payload_sent:
        return None
    state.terminal_payload_pending = False
    state.terminal_payload_sent = True
    return {
        "latent_audio_feat": None,
        "left_context_size": 0,
        "finished": True,
    }
