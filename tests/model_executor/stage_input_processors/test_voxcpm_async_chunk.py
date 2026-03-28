# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.voxcpm import (
    latent2vae_async_chunk,
    voxcpm_pooler_streaming_has_more,
)


def _req(*, finished: bool = False):
    return SimpleNamespace(is_finished=lambda: finished)


def _tm(chunk_patches: int = 8):
    return SimpleNamespace(
        connector=SimpleNamespace(config={"extra": {"latent_chunk_patches": chunk_patches}})
    )


def test_voxcpm_pooler_streaming_has_more_handles_tensor_and_list():
    assert voxcpm_pooler_streaming_has_more(
        {"voxcpm_streaming_continue": torch.tensor(1.0, dtype=torch.float32)}
    ) is True
    assert voxcpm_pooler_streaming_has_more(
        {"voxcpm_streaming_continue": [torch.tensor(0.0, dtype=torch.float32)]}
    ) is False
    assert voxcpm_pooler_streaming_has_more({}) is None


def test_async_chunk_streaming_payload_keeps_request_open_when_more_chunks_exist():
    latent = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    payload = latent2vae_async_chunk(
        transfer_manager=_tm(),
        pooling_output={
            "latent_audio_feat": latent,
            "sr": torch.tensor(22050, dtype=torch.int32),
            "voxcpm_streaming_continue": torch.tensor(1.0, dtype=torch.float32),
        },
        request=_req(finished=False),
        is_finished=False,
    )

    assert payload is not None
    assert payload["finished"] is False
    assert payload["sr"] == 22050
    assert torch.equal(payload["latent_audio_feat"], latent.float())


def test_async_chunk_streaming_payload_finishes_on_last_chunk():
    latent = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    payload = latent2vae_async_chunk(
        transfer_manager=_tm(),
        pooling_output={
            "latent_audio_feat": latent,
            "voxcpm_streaming_continue": torch.tensor(0.0, dtype=torch.float32),
        },
        request=_req(finished=False),
        is_finished=False,
    )

    assert payload is not None
    assert payload["finished"] is True
    assert payload["sr"] == 24000


def test_async_chunk_fallback_slices_3d_latents_by_configured_patch_count():
    latent = torch.arange(5 * 2 * 4, dtype=torch.float32).reshape(5, 2, 4)
    payloads = latent2vae_async_chunk(
        transfer_manager=_tm(chunk_patches=2),
        pooling_output={
            "latent_audio_feat": latent,
            "sr": torch.tensor(24000, dtype=torch.int32),
        },
        request=_req(finished=True),
        is_finished=True,
    )

    assert isinstance(payloads, list)
    assert len(payloads) == 3
    assert [tuple(p["latent_audio_feat"].shape) for p in payloads] == [(2, 2, 4), (2, 2, 4), (1, 2, 4)]
    assert [p["finished"] for p in payloads] == [False, False, True]


def test_async_chunk_fallback_slices_2d_latents_by_configured_patch_count():
    latent = torch.arange(4 * 5, dtype=torch.float32).reshape(4, 5)
    payloads = latent2vae_async_chunk(
        transfer_manager=_tm(chunk_patches=2),
        pooling_output={"latent_audio_feat": latent},
        request=_req(finished=True),
        is_finished=True,
    )

    assert isinstance(payloads, list)
    assert len(payloads) == 3
    assert [tuple(p["latent_audio_feat"].shape) for p in payloads] == [(4, 2), (4, 2), (4, 1)]
    assert [p["finished"] for p in payloads] == [False, False, True]


@pytest.mark.parametrize(
    "pooling_output,finished,expected",
    [
        (None, False, None),
        (None, True, {"latent_audio_feat": None, "sr": None, "finished": True}),
    ],
)
def test_async_chunk_handles_missing_pooling_output(pooling_output, finished, expected):
    payload = latent2vae_async_chunk(
        transfer_manager=_tm(),
        pooling_output=pooling_output,
        request=_req(finished=finished),
        is_finished=finished,
    )
    assert payload == expected
