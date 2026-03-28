import json
import sys
from pathlib import Path
from types import SimpleNamespace
import types

import pytest
import torch

from vllm_omni.model_executor.models.voxcpm.voxcpm import (
    _DirectVoxCPMAudioVAE,
    _DirectVoxCPMLatentGenerator,
    _VoxCPMLatentStreamState,
    _build_prompt_cache_with_audio_load_fallback,
    _init_voxcpm_latent_stream_state,
    _normalize_dtype_name,
    _prepare_runtime_model_dir,
    _step_voxcpm_latent_stream_state,
    VoxCPMForConditionalGeneration,
)
from vllm_omni.model_executor.stage_input_processors.voxcpm import latent2vae

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


def test_build_prompt_cache_falls_back_to_soundfile(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    fake_soundfile = types.ModuleType("soundfile")

    def _fake_sf_read(path, dtype="float32", always_2d=True):
        assert dtype == "float32"
        assert always_2d is True
        assert path.endswith("prompt.wav")
        return [[0.1], [0.2], [0.3]], 16000

    fake_soundfile.read = _fake_sf_read
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)

    native_voxcpm_module = types.ModuleType("voxcpm.model.voxcpm")
    native_voxcpm_module.torchaudio = types.SimpleNamespace()

    def _fail_load(*args, **kwargs):
        raise RuntimeError("TorchCodec is required for load_with_torchcodec")

    native_voxcpm_module.torchaudio.load = _fail_load

    voxcpm_pkg = types.ModuleType("voxcpm")
    voxcpm_model_pkg = types.ModuleType("voxcpm.model")
    voxcpm_pkg.model = voxcpm_model_pkg
    voxcpm_model_pkg.voxcpm = native_voxcpm_module
    monkeypatch.setitem(sys.modules, "voxcpm", voxcpm_pkg)
    monkeypatch.setitem(sys.modules, "voxcpm.model", voxcpm_model_pkg)
    monkeypatch.setitem(sys.modules, "voxcpm.model.voxcpm", native_voxcpm_module)

    class _FakeTTSModel:
        def __init__(self):
            self.calls = 0

        def build_prompt_cache(self, *, prompt_text, prompt_wav_path):
            self.calls += 1
            audio, sr = native_voxcpm_module.torchaudio.load(prompt_wav_path)
            return {
                "prompt_text": prompt_text,
                "shape": tuple(audio.shape),
                "sample_rate": sr,
            }

    prompt_wav = tmp_path / "prompt.wav"
    prompt_wav.write_bytes(b"RIFF")

    tts_model = _FakeTTSModel()
    prompt_cache = _build_prompt_cache_with_audio_load_fallback(
        tts_model,
        prompt_text="ref",
        prompt_wav_path=str(prompt_wav),
    )

    assert tts_model.calls == 2
    assert prompt_cache == {
        "prompt_text": "ref",
        "shape": (1, 3),
        "sample_rate": 16000,
    }


def test_init_voxcpm_latent_stream_state_preserves_max_inf():
    class _FakeKVCache:
        def __init__(self):
            self._step = 0

        def fill_caches(self, *_args, **_kwargs):
            return None

        def step(self):
            cur = self._step
            self._step += 1
            return cur

    class _FakeLM:
        def __init__(self):
            self.kv_cache = _FakeKVCache()

        def embed_tokens(self, tokens):
            b, t = tokens.shape
            return torch.ones((b, t, 4), dtype=torch.float32)

        def __call__(self, *, inputs_embeds, is_causal=True):
            del is_causal
            return inputs_embeds + 1.0, object()

        def forward_step(self, hidden, positions):
            del positions
            return hidden + 2.0

    class _FakeFeatEncoder:
        def __call__(self, feat):
            return feat.mean(dim=2)

    class _FakeProj:
        def __call__(self, x):
            return x

    class _FakeTTSModel:
        patch_size = 2
        device = torch.device("cpu")

        def __init__(self):
            self.audio_vae = SimpleNamespace(latent_dim=4)
            self.base_lm = _FakeLM()
            self.residual_lm = _FakeLM()
            self.feat_encoder = _FakeFeatEncoder()
            self.enc_to_lm_proj = _FakeProj()
            self.fsq_layer = _FakeProj()
            self.lm_to_dit_proj = _FakeProj()
            self.res_to_dit_proj = _FakeProj()
            self.config = SimpleNamespace(
                dtype="bfloat16",
                lm_config=SimpleNamespace(scale_emb=1.0, use_mup=False),
            )
            self.audio_start_token = 3

        def text_tokenizer(self, text):
            return [1] * len(text.split())

    state = _init_voxcpm_latent_stream_state(
        _FakeTTSModel(),
        target_text="a b c d",
        prompt_cache=None,
        min_len=2,
        max_len=100,
        inference_timesteps=10,
        cfg_value=2.0,
        retry_badcase_ratio_threshold=6.0,
    )

    assert isinstance(state, _VoxCPMLatentStreamState)
    assert state.max_len == 34


def test_step_voxcpm_latent_stream_state_advances_one_native_step():
    class _FakeKVCache:
        def __init__(self):
            self._step = 0

        def fill_caches(self, *_args, **_kwargs):
            return None

        def step(self):
            cur = self._step
            self._step += 1
            return cur

    class _FakeLM:
        def __init__(self):
            self.kv_cache = _FakeKVCache()
            self.forward_step_calls = 0

        def embed_tokens(self, tokens):
            b, t = tokens.shape
            return torch.ones((b, t, 4), dtype=torch.float32)

        def __call__(self, *, inputs_embeds, is_causal=True):
            del is_causal
            return inputs_embeds + 1.0, object()

        def forward_step(self, hidden, positions):
            del positions
            self.forward_step_calls += 1
            return hidden + 2.0

    class _FakeFeatEncoder:
        def __call__(self, feat):
            return feat.mean(dim=2)

    class _FakeProj:
        def __call__(self, x):
            return x

    class _FakeStopHead:
        def __init__(self):
            self.calls = 0

        def __call__(self, hidden):
            del hidden
            self.calls += 1
            if self.calls == 1:
                return torch.tensor([[1.0, 0.0]], dtype=torch.float32)
            return torch.tensor([[0.0, 1.0]], dtype=torch.float32)

    class _FakeFeatDecoder:
        def __init__(self):
            self.calls = 0

        def __call__(self, *, mu, patch_size, cond, n_timesteps, cfg_value):
            del mu, cond, n_timesteps, cfg_value
            self.calls += 1
            value = float(self.calls)
            return torch.full((1, 4, patch_size), value, dtype=torch.float32)

    class _FakeTTSModel:
        patch_size = 2
        device = torch.device("cpu")

        def __init__(self):
            self.audio_vae = SimpleNamespace(latent_dim=4)
            self.base_lm = _FakeLM()
            self.residual_lm = _FakeLM()
            self.feat_encoder = _FakeFeatEncoder()
            self.enc_to_lm_proj = _FakeProj()
            self.fsq_layer = _FakeProj()
            self.lm_to_dit_proj = _FakeProj()
            self.res_to_dit_proj = _FakeProj()
            self.stop_proj = _FakeProj()
            self.stop_actn = _FakeProj()
            self.stop_head = _FakeStopHead()
            self.feat_decoder = _FakeFeatDecoder()
            self.config = SimpleNamespace(
                dtype="bfloat16",
                lm_config=SimpleNamespace(scale_emb=1.0, use_mup=False),
            )
            self.audio_start_token = 3

        def text_tokenizer(self, text):
            return [1] * len(text.split())

    tts = _FakeTTSModel()
    state = _init_voxcpm_latent_stream_state(
        tts,
        target_text="hello world",
        prompt_cache=None,
        min_len=0,
        max_len=8,
        inference_timesteps=10,
        cfg_value=2.0,
        retry_badcase_ratio_threshold=6.0,
    )

    patch_1, has_more_1 = _step_voxcpm_latent_stream_state(tts, state)
    patch_2, has_more_2 = _step_voxcpm_latent_stream_state(tts, state)

    assert tuple(patch_1.shape) == (1, 2, 4)
    assert tuple(patch_2.shape) == (1, 2, 4)
    assert has_more_1 is True
    assert has_more_2 is False
    assert state.done is True
    assert state.step_idx == 2
    assert tts.base_lm.forward_step_calls == 1
    assert tts.residual_lm.forward_step_calls == 1


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
