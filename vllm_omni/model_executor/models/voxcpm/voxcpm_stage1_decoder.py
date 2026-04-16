from __future__ import annotations

import torch


class VoxCPMStage1Decoder:
    def __init__(self, stream_audio_patch_samples: int) -> None:
        self.stream_audio_patch_samples = max(1, int(stream_audio_patch_samples))

    def trim_left_context(self, audio: torch.Tensor, left_context_size: int) -> torch.Tensor:
        stream = audio.detach().reshape(-1).to(torch.float32)
        prefix_samples = max(0, int(left_context_size)) * self.stream_audio_patch_samples
        if prefix_samples <= 0:
            return stream
        return stream[prefix_samples:]
