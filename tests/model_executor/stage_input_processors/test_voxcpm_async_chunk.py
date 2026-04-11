# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.voxcpm import (
    latent2vae_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _req(*, finished: bool):
    return SimpleNamespace(
        is_finished=lambda: finished,
    )


def test_accepts_transfer_manager_keyword():
    payload = latent2vae_async_chunk(
        transfer_manager=object(),
        pooling_output={"latent_audio_feat": torch.ones((1, 64, 4))},
        request=_req(finished=False),
        is_finished=False,
    )

    assert payload is not None
    assert "latent_audio_feat" in payload
    assert payload["finished"] is False


def test_finished_empty_payload_emits_terminal_marker():
    payload = latent2vae_async_chunk(
        transfer_manager=object(),
        pooling_output=None,
        request=_req(finished=True),
        is_finished=True,
    )

    assert payload == {
        "code_predictor_codes": [0],
        "finished": True,
    }


def test_normalizes_tensor_finished_flag_to_python_bool():
    payload = latent2vae_async_chunk(
        transfer_manager=object(),
        pooling_output={"latent_audio_feat": torch.ones((1, 64, 4))},
        request=_req(finished=torch.tensor(True)),
        is_finished=torch.tensor(False),
    )

    assert payload is not None
    assert payload["finished"] is True
    assert isinstance(payload["finished"], bool)


def test_normalizes_singleton_container_finished_flag():
    payload = latent2vae_async_chunk(
        transfer_manager=object(),
        pooling_output=None,
        request=_req(finished=[True]),
        is_finished=[False],
    )

    assert payload == {
        "code_predictor_codes": [0],
        "finished": True,
    }
