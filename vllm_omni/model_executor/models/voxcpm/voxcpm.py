from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import wave
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Iterator
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.stage_input_processors.voxcpm import DEFAULT_LATENT_CHUNK_PATCHES

logger = init_logger(__name__)


def _import_voxcpm_model_class():
    try:
        from voxcpm.model.voxcpm import VoxCPMModel

        return VoxCPMModel
    except ImportError:
        pass

    candidates: list[Path] = []
    env_path = os.environ.get("VLLM_OMNI_VOXCPM_CODE_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    repo_root = Path(__file__).resolve().parents[4]
    candidates.append(repo_root.parent / "VoxCPM" / "src")

    for candidate in candidates:
        if not candidate.exists():
            continue
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
        try:
            from voxcpm.model.voxcpm import VoxCPMModel

            return VoxCPMModel
        except ImportError:
            continue

    raise ImportError(
        "Failed to import VoxCPMModel. Install the `voxcpm` package or set "
        "`VLLM_OMNI_VOXCPM_CODE_PATH` to the VoxCPM `src` directory."
    )


def _import_voxcpm_audio_vae_classes():
    try:
        from voxcpm.modules.audiovae import AudioVAE, AudioVAEConfig

        return AudioVAE, AudioVAEConfig
    except ImportError:
        pass

    candidates: list[Path] = []
    env_path = os.environ.get("VLLM_OMNI_VOXCPM_CODE_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    repo_root = Path(__file__).resolve().parents[4]
    candidates.append(repo_root.parent / "VoxCPM" / "src")

    for candidate in candidates:
        if not candidate.exists():
            continue
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
        try:
            from voxcpm.modules.audiovae import AudioVAE, AudioVAEConfig

            return AudioVAE, AudioVAEConfig
        except ImportError:
            continue

    raise ImportError(
        "Failed to import VoxCPM AudioVAE. Install the `voxcpm` package or set "
        "`VLLM_OMNI_VOXCPM_CODE_PATH` to the VoxCPM `src` directory."
    )


def _device_to_string(device: torch.device) -> str:
    if device.index is None:
        return device.type
    return f"{device.type}:{device.index}"


def _normalize_dtype_name(dtype: Any) -> str | None:
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        mapping = {
            torch.bfloat16: "bfloat16",
            torch.float16: "float16",
            torch.float32: "float32",
        }
        return mapping.get(dtype, str(dtype).removeprefix("torch."))
    dtype_str = str(dtype)
    return dtype_str.removeprefix("torch.")


def _resolve_runtime_device(vllm_config: VllmConfig) -> torch.device:
    try:
        from vllm_omni.platforms import current_omni_platform

        return current_omni_platform.get_torch_device()
    except Exception:
        pass

    device = getattr(getattr(vllm_config, "device_config", None), "device", None)
    if isinstance(device, torch.device):
        return device
    if device:
        return torch.device(device)
    return torch.device("cpu")


def _prepare_runtime_model_dir(
    model_path: str | Path,
    *,
    target_device: torch.device,
    target_dtype: str | None,
) -> str:
    source_dir = Path(model_path)
    config_path = source_dir / "config.json"
    if not config_path.exists():
        return str(source_dir)

    config_dict = json.loads(config_path.read_text())
    desired_device = target_device.type
    desired_dtype = target_dtype or config_dict.get("dtype")

    if config_dict.get("device") == desired_device and config_dict.get("dtype") == desired_dtype:
        return str(source_dir)

    digest = sha256(
        f"{source_dir.resolve()}:{config_path.read_text()}:{desired_device}:{desired_dtype}".encode("utf-8")
    ).hexdigest()[:16]
    runtime_dir = Path(tempfile.gettempdir()) / "vllm_omni_voxcpm_runtime" / digest
    runtime_dir.mkdir(parents=True, exist_ok=True)

    for entry in source_dir.iterdir():
        target = runtime_dir / entry.name
        if entry.name == "config.json" or target.exists():
            continue
        try:
            target.symlink_to(entry, target_is_directory=entry.is_dir())
        except OSError:
            if entry.is_dir():
                shutil.copytree(entry, target, dirs_exist_ok=True)
            else:
                shutil.copy2(entry, target)

    patched_config = dict(config_dict)
    patched_config["device"] = desired_device
    if desired_dtype is not None:
        patched_config["dtype"] = desired_dtype
    (runtime_dir / "config.json").write_text(json.dumps(patched_config, indent=2, sort_keys=True))
    return str(runtime_dir)


@contextmanager
def _force_cuda_available_for_npu(device: torch.device):
    if device.type != "npu":
        yield
        return

    with patch("torch.cuda.is_available", return_value=True):
        yield


def _voxcpm_native_model_dtype(tts_model: Any) -> torch.dtype:
    try:
        from voxcpm.model.utils import get_dtype

        return get_dtype(tts_model.config.dtype)
    except Exception:
        return torch.bfloat16


def _voxcpm_iter_latent_patches_native(
    tts_model: Any,
    *,
    target_text: str,
    prompt_cache: dict[str, Any] | None,
    min_len: int,
    max_len: int,
    inference_timesteps: int,
    cfg_value: float,
    retry_badcase_ratio_threshold: float,
    streaming_prefix_len: int = 3,
) -> Iterator[tuple[torch.Tensor, bool]]:
    """Walk native ``VoxCPMModel._inference(..., streaming=True)`` one AR step at a time.

    Each yield is a *new* latent patch ``(1, P, D)`` on CPU float32, plus ``has_more`` until
    the final patch (``has_more=False``). The Omni forward batches up to
    ``latent_chunk_patches`` yields per scheduler step so each SHM payload matches the legacy
    time-chunk granularity instead of one patch per step.
    """
    target_text = " ".join(target_text.split())
    if prompt_cache is None:
        prompt_audio_feat = torch.empty(
            (0, tts_model.patch_size, tts_model.audio_vae.latent_dim),
            dtype=torch.float32,
        )
        text = target_text
    else:
        prompt_audio_feat = prompt_cache["audio_feat"]
        prompt_text = prompt_cache["prompt_text"]
        text = prompt_text + target_text

    text_token = torch.LongTensor(tts_model.text_tokenizer(text))
    text_token = torch.cat(
        [
            text_token,
            torch.tensor(
                [tts_model.audio_start_token],
                dtype=torch.int32,
                device=text_token.device,
            ),
        ],
        dim=-1,
    )

    audio_length = prompt_audio_feat.size(0)
    text_length = text_token.shape[0]
    text_pad_token = torch.zeros(audio_length, dtype=torch.int32, device=text_token.device)
    audio_pad_feat = torch.zeros(
        (text_token.shape[0], tts_model.patch_size, tts_model.audio_vae.latent_dim),
        dtype=torch.float32,
        device=text_token.device,
    )
    text_token = torch.cat([text_token, text_pad_token])
    audio_feat = torch.cat([audio_pad_feat, prompt_audio_feat], dim=0)
    text_mask = torch.cat([torch.ones(text_length), torch.zeros(audio_length)]).type(torch.int32).to(text_token.device)
    audio_mask = torch.cat([torch.zeros(text_length), torch.ones(audio_length)]).type(torch.int32).to(text_token.device)

    text_token = text_token.unsqueeze(0).to(tts_model.device)
    text_mask = text_mask.unsqueeze(0).to(tts_model.device)
    audio_feat = audio_feat.unsqueeze(0).to(tts_model.device).to(_voxcpm_native_model_dtype(tts_model))
    audio_mask = audio_mask.unsqueeze(0).to(tts_model.device)

    target_text_length = len(tts_model.text_tokenizer(target_text))
    max_inf = min(int(target_text_length * retry_badcase_ratio_threshold + 10), max_len)

    inference_gen = tts_model._inference(
        text_token,
        text_mask,
        audio_feat,
        audio_mask,
        min_len=min_len,
        max_len=max_inf,
        inference_timesteps=inference_timesteps,
        cfg_value=cfg_value,
        streaming=True,
        streaming_prefix_len=streaming_prefix_len,
    )

    prev: torch.Tensor | None = None
    for _feat_pred, pred_feat_seq in inference_gen:
        last = pred_feat_seq[-1]
        patch = last.squeeze(0).detach().float().cpu().contiguous()
        if prev is not None:
            yield prev, True
        prev = patch
    if prev is not None:
        yield prev, False


class _DirectVoxCPMLatentGenerator:
    def __init__(self, tts_model: Any):
        self.tts_model = tts_model
        self.sample_rate = int(getattr(tts_model, "sample_rate", 24000))

    def generate_latents(
        self,
        *,
        text: str,
        prompt_wav_path: str | None = None,
        prompt_text: str | None = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        min_len: int = 2,
        max_len: int = 4096,
        retry_badcase: bool = True,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,
    ) -> torch.Tensor:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("target text must be a non-empty string")
        if (prompt_wav_path is None) != (prompt_text is None):
            raise ValueError("prompt_wav_path and prompt_text must both be provided or both be None")
        if prompt_wav_path is not None and not os.path.exists(prompt_wav_path):
            raise FileNotFoundError(f"prompt_wav_path does not exist: {prompt_wav_path}")

        prompt_cache = None
        if prompt_wav_path is not None and prompt_text is not None:
            prompt_cache = self.tts_model.build_prompt_cache(
                prompt_text=prompt_text,
                prompt_wav_path=prompt_wav_path,
            )

        _, _, pred_audio_feat = self.tts_model.generate_with_prompt_cache(
            target_text=" ".join(text.split()),
            prompt_cache=prompt_cache,
            min_len=min_len,
            max_len=max_len,
            inference_timesteps=inference_timesteps,
            cfg_value=cfg_value,
            retry_badcase=retry_badcase,
            retry_badcase_max_times=retry_badcase_max_times,
            retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
        )
        return pred_audio_feat.detach().cpu().to(torch.float32)


class _DirectVoxCPMAudioVAE:
    def __init__(self, audio_vae: nn.Module):
        self.audio_vae = audio_vae
        self.sample_rate = int(getattr(audio_vae, "sample_rate", 24000))
        self.latent_dim = int(getattr(audio_vae, "latent_dim", 64))

    def _prepare_latents_for_decode(self, latent_audio_feat: Any) -> torch.Tensor:
        latents = latent_audio_feat
        if not isinstance(latents, torch.Tensor):
            latents = torch.tensor(latents, dtype=torch.float32)
        latents = latents.detach().to(torch.float32)

        if latents.ndim == 3:
            if latents.shape[-1] == self.latent_dim:
                latents = rearrange(latents, "t p d -> 1 d (t p)")
            elif latents.shape[1] == self.latent_dim:
                latents = latents.contiguous()
            else:
                raise ValueError(f"Unsupported latent_audio_feat shape: {tuple(latents.shape)}")
        elif latents.ndim == 2:
            if latents.shape[0] == self.latent_dim:
                latents = latents.unsqueeze(0)
            elif latents.shape[1] == self.latent_dim:
                latents = rearrange(latents, "t d -> 1 d t")
            else:
                raise ValueError(f"Unsupported latent_audio_feat shape: {tuple(latents.shape)}")
        else:
            raise ValueError(f"Unsupported latent_audio_feat ndim: {latents.ndim}")

        return latents

    @torch.no_grad()
    def decode(self, latent_audio_feat: Any) -> torch.Tensor:
        latents = self._prepare_latents_for_decode(latent_audio_feat)
        device = next(self.audio_vae.parameters()).device
        audio = self.audio_vae.decode(latents.to(device=device, dtype=torch.float32))
        return audio.squeeze(1).reshape(-1).detach().cpu().to(torch.float32)


def _load_native_voxcpm_model(
    model_path: str,
    *,
    device: torch.device,
    dtype: str | None,
):
    VoxCPMModel = _import_voxcpm_model_class()
    runtime_model_path = _prepare_runtime_model_dir(model_path, target_device=device, target_dtype=dtype)

    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.set_device(device)

    with _force_cuda_available_for_npu(device):
        tts_model = VoxCPMModel.from_local(
            runtime_model_path,
            optimize=device.type == "cuda",
        )

    return tts_model


def _load_native_voxcpm_latent_generator(
    model_path: str,
    *,
    device: torch.device,
    dtype: str | None,
) -> _DirectVoxCPMLatentGenerator:
    return _DirectVoxCPMLatentGenerator(_load_native_voxcpm_model(model_path, device=device, dtype=dtype))


def _load_native_voxcpm_audio_vae(
    model_path: str,
    *,
    device: torch.device,
) -> _DirectVoxCPMAudioVAE:
    AudioVAE, AudioVAEConfig = _import_voxcpm_audio_vae_classes()
    runtime_model_path = _prepare_runtime_model_dir(model_path, target_device=device, target_dtype="float32")
    config_dict = json.loads((Path(runtime_model_path) / "config.json").read_text())
    audio_vae_config = config_dict.get("audio_vae_config")
    if audio_vae_config is not None:
        audio_vae = AudioVAE(config=AudioVAEConfig(**audio_vae_config))
    else:
        audio_vae = AudioVAE()

    state_dict = torch.load(
        Path(runtime_model_path) / "audiovae.pth",
        map_location="cpu",
        weights_only=True,
    )["state_dict"]
    audio_vae.load_state_dict(state_dict, strict=True)
    audio_vae = audio_vae.to(device=device, dtype=torch.float32).eval()
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.set_device(device)
    return _DirectVoxCPMAudioVAE(audio_vae)


class VoxCPMForConditionalGeneration(nn.Module):
    input_modalities = "audio"
    _LATENT_STAGES = {"latent_generator", "latent", "ar_dit"}
    _VAE_STAGES = {"vae", "audio_vae"}

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        del prefix
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model
        self.model_stage = getattr(vllm_config.model_config, "model_stage", "latent_generator")
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True
        self._pipeline = None
        # Latent chunk streaming: iterator keyed by ``omni_req_id``; each forward drains up to
        # ``latent_chunk_patches`` native steps into one SHM chunk.
        self._voxcpm_latent_patch_iters: dict[str, Iterator[tuple[torch.Tensor, bool]]] = {}

    def _ensure_model_loaded(self):
        if self._pipeline is not None:
            return

        target_device = _resolve_runtime_device(self.vllm_config)
        model_dtype = getattr(self.vllm_config.model_config, "dtype", None)
        normalized_dtype = _normalize_dtype_name(model_dtype)
        if self.model_stage in self._LATENT_STAGES:
            self._pipeline = _load_native_voxcpm_latent_generator(
                self.model_path,
                device=target_device,
                dtype=normalized_dtype,
            )
        elif self.model_stage in self._VAE_STAGES:
            self._pipeline = _load_native_voxcpm_audio_vae(
                self.model_path,
                device=target_device,
            )
        else:
            raise ValueError(
                f"Unsupported VoxCPM model_stage: {self.model_stage}. "
                "pure_voxcpm only supports split-stage latent_generator/vae inference."
            )

        logger.info(
            "Loaded VoxCPM stage '%s' on %s",
            self.model_stage,
            _device_to_string(target_device),
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load VoxCPM via its native runtime instead of vLLM's HF weight loader.

        VoxCPM stages are constructed from the original local model directory using
        ``VoxCPMModel.from_local`` / ``AudioVAE`` inside ``_ensure_model_loaded``.
        The standard vLLM weight iterator is therefore not applicable here.
        """
        del weights
        self._ensure_model_loaded()
        return set()

    @staticmethod
    def _extract_val(info: dict[str, Any], key: str, default: Any) -> Any:
        value = info.get(key, default)
        if isinstance(value, list):
            return value[0] if value else default
        return value

    @staticmethod
    def _normalize_audio_samples(samples: Any) -> np.ndarray:
        if isinstance(samples, torch.Tensor):
            return samples.detach().cpu().float().reshape(-1).numpy()
        return np.asarray(samples, dtype=np.float32).reshape(-1)

    @classmethod
    def _normalize_ref_audio(cls, ref_audio: Any) -> tuple[np.ndarray, int]:
        if isinstance(ref_audio, str):
            raise TypeError("String ref_audio should be handled as a path before waveform normalization.")

        if isinstance(ref_audio, dict):
            sr = ref_audio.get("sample_rate") or ref_audio.get("sampling_rate") or ref_audio.get("sr")
            samples = None
            for key in ("audio", "wav", "samples", "array", "waveform"):
                if key in ref_audio and ref_audio[key] is not None:
                    samples = ref_audio[key]
                    break
            if sr is None or samples is None:
                raise ValueError("ref_audio dict must contain waveform data and sample rate.")
            return cls._normalize_audio_samples(samples), int(sr)

        if isinstance(ref_audio, (list, tuple)):
            if len(ref_audio) == 1:
                return cls._normalize_ref_audio(ref_audio[0])
            if len(ref_audio) == 2 and np.isscalar(ref_audio[1]):
                return cls._normalize_audio_samples(ref_audio[0]), int(ref_audio[1])

        raise TypeError(f"Unsupported ref_audio format: {type(ref_audio)!r}")

    def _latent_chunk_patches_cfg(self) -> int:
        """Patches per SHM chunk (connector ``extra.latent_chunk_patches``), default 8."""
        mc = getattr(self.vllm_config, "model_config", None)
        cc = getattr(mc, "stage_connector_config", None) if mc is not None else None
        extra: Any
        if isinstance(cc, dict):
            extra = cc.get("extra", cc)
        else:
            extra = getattr(cc, "extra", None) if cc is not None else None
        if isinstance(extra, dict) and extra.get("latent_chunk_patches") is not None:
            return max(1, int(extra["latent_chunk_patches"]))
        return DEFAULT_LATENT_CHUNK_PATCHES

    def _write_temp_prompt_wav(waveform: np.ndarray, sample_rate: int) -> str:
        prompt_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        prompt_file.close()

        wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
        wav = np.clip(wav, -1.0, 1.0)
        pcm16 = (wav * 32767.0).astype(np.int16)
        with wave.open(prompt_file.name, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(int(sample_rate))
            wav_file.writeframes(pcm16.tobytes())

        return prompt_file.name

    @classmethod
    def _resolve_prompt_inputs(cls, info: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
        prompt_text = cls._extract_val(info, "prompt_text", None)
        prompt_wav_path = cls._extract_val(info, "prompt_wav_path", None)
        if prompt_wav_path:
            if prompt_text is None:
                prompt_text = cls._extract_val(info, "ref_text", None)
            return prompt_wav_path, prompt_text, None

        ref_audio = cls._extract_val(info, "ref_audio", None)
        ref_text = cls._extract_val(info, "ref_text", None)
        if ref_audio is None or ref_text is None:
            return None, None, None

        if isinstance(ref_audio, str):
            return ref_audio, ref_text, None

        waveform, sample_rate = cls._normalize_ref_audio(ref_audio)
        temp_prompt_wav = cls._write_temp_prompt_wav(waveform, sample_rate)
        return temp_prompt_wav, ref_text, temp_prompt_wav

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> None:
        return None

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        del input_ids, positions, intermediate_tensors, inputs_embeds, kwargs
        self._ensure_model_loaded()

        infos = runtime_additional_information or [{}]
        sample_rate = int(getattr(self._pipeline, "sample_rate", 24000))
        if self.model_stage in self._VAE_STAGES:
            if all(self._extract_val(info, "latent_audio_feat", None) is None for info in infos):
                return OmniOutput(
                    text_hidden_states=None,
                    multimodal_outputs={
                        "model_outputs": [torch.zeros((0,), dtype=torch.float32) for _ in infos],
                        "sr": [torch.tensor(sample_rate, dtype=torch.int32) for _ in infos],
                    },
                )
        else:
            texts = [self._extract_val(info, "text", "") for info in infos]
            if all(not text for text in texts):
                return OmniOutput(
                    text_hidden_states=None,
                    multimodal_outputs={
                        "latent_audio_feat": [torch.zeros((0,), dtype=torch.float32) for _ in infos],
                        "sr": [torch.tensor(sample_rate, dtype=torch.int32) for _ in infos],
                    },
                )

        use_native_latent_stream = (
            self.model_stage in self._LATENT_STAGES
            and getattr(self.vllm_config.model_config, "async_chunk", False)
        )

        outputs: list[torch.Tensor] = []
        sample_rates: list[torch.Tensor] = []
        stream_continue: list[bool] = []
        for info in infos:
            if self.model_stage in self._VAE_STAGES:
                latent_audio_feat = self._extract_val(info, "latent_audio_feat", None)
                audio_tensor = self._pipeline.decode(latent_audio_feat)
                outputs.append(audio_tensor.float().cpu())
                sample_rates.append(torch.tensor(sample_rate, dtype=torch.int32))
                continue

            text = self._extract_val(info, "text", "")
            cfg_value = float(self._extract_val(info, "cfg_value", 2.0))
            inference_timesteps = int(self._extract_val(info, "inference_timesteps", 10))
            min_len = int(self._extract_val(info, "min_len", 2))
            max_len = int(self._extract_val(info, "max_len", self._extract_val(info, "max_new_tokens", 4096)))
            retry_badcase = bool(self._extract_val(info, "retry_badcase", True))
            retry_badcase_max_times = int(self._extract_val(info, "retry_badcase_max_times", 3))
            retry_badcase_ratio_threshold = float(self._extract_val(info, "retry_badcase_ratio_threshold", 6.0))
            streaming_prefix_len = int(self._extract_val(info, "streaming_prefix_len", 3))

            prompt_wav_path, prompt_text, temp_prompt_wav = self._resolve_prompt_inputs(info)
            try:
                if self.model_stage in self._LATENT_STAGES:
                    if use_native_latent_stream:
                        req_id = str(self._extract_val(info, "omni_req_id", "0"))
                        tts = self._pipeline.tts_model
                        if req_id not in self._voxcpm_latent_patch_iters:
                            prompt_cache = None
                            if prompt_wav_path is not None and prompt_text is not None:
                                prompt_cache = tts.build_prompt_cache(
                                    prompt_text=prompt_text,
                                    prompt_wav_path=prompt_wav_path,
                                )
                            self._voxcpm_latent_patch_iters[req_id] = _voxcpm_iter_latent_patches_native(
                                tts,
                                target_text=" ".join(text.split()),
                                prompt_cache=prompt_cache,
                                min_len=min_len,
                                max_len=max_len,
                                inference_timesteps=inference_timesteps,
                                cfg_value=cfg_value,
                                retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
                                streaming_prefix_len=streaming_prefix_len,
                            )
                        it = self._voxcpm_latent_patch_iters[req_id]
                        chunk_cap = self._latent_chunk_patches_cfg()
                        patches: list[torch.Tensor] = []
                        last_hm = False
                        for _ in range(chunk_cap):
                            try:
                                patch_cpu, last_hm = next(it)
                            except StopIteration:
                                last_hm = False
                                self._voxcpm_latent_patch_iters.pop(req_id, None)
                                break
                            patches.append(patch_cpu)
                            if not last_hm:
                                self._voxcpm_latent_patch_iters.pop(req_id, None)
                                break
                        if not patches:
                            outputs.append(torch.zeros((0, 1, 1), dtype=torch.float32))
                            stream_continue.append(False)
                        else:
                            stacked = torch.cat(patches, dim=0)
                            outputs.append(stacked)
                            stream_continue.append(last_hm)
                    else:
                        latent_audio_feat = self._pipeline.generate_latents(
                            text=text,
                            prompt_wav_path=prompt_wav_path,
                            prompt_text=prompt_text,
                            cfg_value=cfg_value,
                            inference_timesteps=inference_timesteps,
                            min_len=min_len,
                            max_len=max_len,
                            retry_badcase=retry_badcase,
                            retry_badcase_max_times=retry_badcase_max_times,
                            retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
                        )
                        outputs.append(latent_audio_feat.float().cpu())
            finally:
                if temp_prompt_wav is not None and os.path.exists(temp_prompt_wav):
                    os.unlink(temp_prompt_wav)

            sample_rates.append(torch.tensor(sample_rate, dtype=torch.int32))

        output_key = "latent_audio_feat" if self.model_stage in self._LATENT_STAGES else "model_outputs"
        mm: dict[str, Any] = {output_key: outputs, "sr": sample_rates}
        # 0/1 float tensor per request: msgspec IPC rejects raw bool in pooling_output.
        if use_native_latent_stream and stream_continue:
            mm["voxcpm_streaming_continue"] = [
                torch.tensor(1.0 if c else 0.0, dtype=torch.float32) for c in stream_continue
            ]
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs=mm,
        )

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        del batch_size, dtype, device
        return {}
