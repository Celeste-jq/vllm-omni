from __future__ import annotations

import dataclasses
from typing import Any

import torch


@dataclasses.dataclass
class Stage0RequestState:
    request_id: str
    request_start_time: float = 0.0
    prefill_completed: bool = False
    is_stopping: bool = False
    terminal_payload_pending: bool = False
    terminal_payload_sent: bool = False
    decode_step_count: int = 0
    prompt_cache: dict[str, Any] | None = None
    prefill_artifacts: dict[str, Any] | None = None
    curr_embed_for_next: torch.Tensor | None = None
    prev_feat_embed: torch.Tensor | None = None
    curr_prefix_feat_cond: torch.Tensor | None = None
    precomputed_stop_logits: torch.Tensor | None = None
    pending_latent_patch: torch.Tensor | None = None
    generated_latent_history: list[torch.Tensor] = dataclasses.field(default_factory=list)
    generated_latent_frames: int = 0
    emitted_latent_frames: int = 0
    streaming_prefix_len: int = 3
    codec_left_context_frames: int | None = None
    min_len: int = 2
    max_len: int = 4096
    cfg_value: float = 2.0
    inference_timesteps: int = 10


@dataclasses.dataclass
class Stage1RequestState:
    request_id: str
    decoded_chunk_count: int = 0
    emitted_audio_samples: int = 0
    last_left_context_size: int = 0
    terminal_payload_seen: bool = False
    terminal_output_sent: bool = False
