# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.model_executor.models.voxcpm.voxcpm_stage1_decoder import VoxCPMStage1Decoder

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_trim_left_context_audio_by_patch_count():
    decoder = VoxCPMStage1Decoder(stream_audio_patch_samples=4)
    audio = torch.arange(12, dtype=torch.float32)

    trimmed = decoder.trim_left_context(audio, left_context_size=2)

    assert trimmed.tolist() == [8.0, 9.0, 10.0, 11.0]


def test_trim_left_context_audio_no_context_returns_original():
    decoder = VoxCPMStage1Decoder(stream_audio_patch_samples=4)
    audio = torch.arange(6, dtype=torch.float32)

    trimmed = decoder.trim_left_context(audio, left_context_size=0)

    assert trimmed.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
