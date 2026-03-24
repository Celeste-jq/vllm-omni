import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.voxcpm.voxcpm import (
    _DirectVoxCPMAudioVAE,
    _DirectVoxCPMLatentGenerator,
    _normalize_dtype_name,
    _prepare_runtime_model_dir,
    VoxCPMForConditionalGeneration,
)
from vllm_omni.model_executor.stage_input_processors.voxcpm import latent2vae, latent2vae_async_chunk

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_prepare_runtime_model_dir_rewrites_device_and_dtype(tmp_path: Path):
    model_dir = tmp_path / "voxcpm-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"device": "cuda", "dtype": "float16", "foo": "bar"}))
    (model_dir / "model.safetensors").write_text("weights")

    runtime_dir = Path(
        _prepare_runtime_model_dir(
            model_dir,
            target_device=torch.device("npu"),
            target_dtype="bfloat16",
        )
    )

    assert runtime_dir != model_dir
    rendered = json.loads((runtime_dir / "config.json").read_text())
    assert rendered["device"] == "npu"
    assert rendered["dtype"] == "bfloat16"
    assert rendered["foo"] == "bar"
    assert (runtime_dir / "model.safetensors").exists()


def test_prepare_runtime_model_dir_reuses_source_when_already_compatible(tmp_path: Path):
    model_dir = tmp_path / "voxcpm-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"device": "npu", "dtype": "bfloat16"}))

    runtime_dir = _prepare_runtime_model_dir(
        model_dir,
        target_device=torch.device("npu"),
        target_dtype="bfloat16",
    )

    assert runtime_dir == str(model_dir)


def test_normalize_dtype_name_handles_torch_dtype():
    assert _normalize_dtype_name(torch.bfloat16) == "bfloat16"
    assert _normalize_dtype_name(torch.float16) == "float16"
    assert _normalize_dtype_name("torch.float32") == "float32"


def test_direct_latent_generator_forwards_expected_kwargs(tmp_path: Path):
    class _FakeTTSModel:
        sample_rate = 24000

        def __init__(self):
            self.build_prompt_cache_calls = []
            self.generate_calls = []

        def build_prompt_cache(self, **kwargs):
            self.build_prompt_cache_calls.append(kwargs)
            return {"cache": True}

        def generate_with_prompt_cache(self, **kwargs):
            self.generate_calls.append(kwargs)
            return torch.zeros(1), torch.zeros(1), torch.ones((3, 2, 4), dtype=torch.float32)

    prompt_wav = tmp_path / "prompt.wav"
    prompt_wav.write_bytes(b"RIFF")

    tts_model = _FakeTTSModel()
    generator = _DirectVoxCPMLatentGenerator(tts_model)
    latents = generator.generate_latents(
        text="hello\nworld",
        prompt_wav_path=str(prompt_wav),
        prompt_text="ref",
        cfg_value=1.5,
        inference_timesteps=8,
        min_len=3,
        max_len=64,
    )

    assert tuple(latents.shape) == (3, 2, 4)
    assert tts_model.build_prompt_cache_calls == [{"prompt_text": "ref", "prompt_wav_path": str(prompt_wav)}]
    assert tts_model.generate_calls[0]["target_text"] == "hello world"
    assert tts_model.generate_calls[0]["prompt_cache"] == {"cache": True}
    assert tts_model.generate_calls[0]["cfg_value"] == 1.5


def test_audio_vae_prepare_latents_for_decode():
    class _FakeAudioVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.sample_rate = 24000
            self.latent_dim = 4
            self.param = torch.nn.Parameter(torch.zeros(1))

        def decode(self, z):
            return z.sum(dim=1, keepdim=True)

    decoder = _DirectVoxCPMAudioVAE(_FakeAudioVAE())
    latents = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    prepared = decoder._prepare_latents_for_decode(latents)

    assert tuple(prepared.shape) == (1, 4, 6)


def test_latent2vae_async_chunk_uses_omni_stream_flags():
    class _Req:
        external_req_id = "r1"

        def is_finished(self):
            return False

    tm = SimpleNamespace()
    latent = torch.ones(2, 3, dtype=torch.float32)
    out = latent2vae_async_chunk(
        transfer_manager=tm,
        pooling_output={
            "latent_audio_feat": latent,
            "omni_stream_continue": torch.tensor(1, dtype=torch.int32),
            "omni_stream_gen_exhausted": torch.tensor(0, dtype=torch.int32),
        },
        request=_Req(),
        is_finished=False,
    )
    assert out is not None
    assert out["finished"] is False
    assert "latent_audio_feat" in out

    out_last = latent2vae_async_chunk(
        transfer_manager=tm,
        pooling_output={
            "latent_audio_feat": latent,
            "omni_stream_continue": torch.tensor(0, dtype=torch.int32),
            "omni_stream_gen_exhausted": torch.tensor(1, dtype=torch.int32),
        },
        request=_Req(),
        is_finished=False,
    )
    assert out_last is not None
    assert out_last["finished"] is True


def test_latent2vae_wraps_stage_outputs():
    latent = torch.ones((3, 2, 4), dtype=torch.float32)
    stage_output = SimpleNamespace(
        request_id="req-1",
        outputs=[SimpleNamespace(multimodal_output={"latent_audio_feat": latent, "sr": torch.tensor(24000)})],
    )
    stage = SimpleNamespace(engine_outputs=[stage_output])

    prompts = latent2vae([stage], [0])

    assert len(prompts) == 1
    assert prompts[0]["prompt_token_ids"] == [0]
    assert torch.equal(prompts[0]["additional_information"]["latent_audio_feat"], latent)
    assert prompts[0]["additional_information"]["sample_rate"] == [24000]


def test_voxcpm_load_weights_uses_native_loader_without_consuming_iterator():
    model = VoxCPMForConditionalGeneration.__new__(VoxCPMForConditionalGeneration)
    torch.nn.Module.__init__(model)
    load_calls: list[str] = []

    def _fake_ensure_model_loaded():
        load_calls.append("loaded")

    model._ensure_model_loaded = _fake_ensure_model_loaded

    def _weights():
        raise AssertionError("vLLM weight iterator should not be consumed for native VoxCPM loading")
        yield ("unused", torch.zeros(1))

    loaded = model.load_weights(_weights())

    assert loaded == set()
    assert load_calls == ["loaded"]
