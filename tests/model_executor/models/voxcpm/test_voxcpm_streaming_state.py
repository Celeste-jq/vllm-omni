# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.model_executor.models.voxcpm.voxcpm_request_state import Stage0RequestState
from vllm_omni.model_executor.models.voxcpm.voxcpm_stage0_paged import (
    append_latent_patch,
    maybe_emit_terminal_payload,
    maybe_emit_window,
    resolve_left_context_frames,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_stage0_request_state_defaults():
    state = Stage0RequestState(request_id="req-1")

    assert state.request_id == "req-1"
    assert state.prefill_completed is False
    assert state.is_stopping is False
    assert state.terminal_payload_pending is False
    assert state.terminal_payload_sent is False
    assert state.generated_latent_history == []
    assert state.generated_latent_frames == 0
    assert state.emitted_latent_frames == 0


def test_resolve_left_context_frames_prefers_codec_setting():
    assert resolve_left_context_frames(codec_left_context_frames=2, streaming_prefix_len=5) == 2


def test_resolve_left_context_frames_falls_back_to_streaming_prefix_len():
    assert resolve_left_context_frames(codec_left_context_frames=None, streaming_prefix_len=3) == 2


def test_maybe_emit_window_emits_only_new_patch_then_overlapping_window():
    state = Stage0RequestState(request_id="req-2")

    append_latent_patch(state, torch.tensor([[1.0], [2.0]], dtype=torch.float32))
    first = maybe_emit_window(state, codec_left_context_frames=1)
    assert first is not None
    assert first["left_context_size"] == 0
    assert first["finished"] is False
    torch.testing.assert_close(
        first["latent_audio_feat"],
        torch.tensor([[[1.0], [2.0]]], dtype=torch.float32),
    )

    append_latent_patch(state, torch.tensor([[3.0], [4.0]], dtype=torch.float32))
    second = maybe_emit_window(state, codec_left_context_frames=1)
    assert second is not None
    assert second["left_context_size"] == 1
    assert second["finished"] is False
    torch.testing.assert_close(
        second["latent_audio_feat"],
        torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=torch.float32),
    )


def test_maybe_emit_terminal_payload_only_emits_once():
    state = Stage0RequestState(request_id="req-3", terminal_payload_pending=True)

    first = maybe_emit_terminal_payload(state)
    second = maybe_emit_terminal_payload(state)

    assert first == {
        "latent_audio_feat": None,
        "left_context_size": 0,
        "finished": True,
    }
    assert second is None
