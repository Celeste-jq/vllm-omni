from __future__ import annotations

import os
import tempfile
import wave
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .voxcpm_native_loader import _load_native_voxcpm_audio_vae, _load_native_voxcpm_latent_generator
from .voxcpm_runtime_utils import _device_to_string, _normalize_dtype_name, _resolve_runtime_device

logger = init_logger(__name__)


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
        self.inject_omni_request_id_into_runtime_info = True
        self._pipeline = None
        self._latent_stream_gens: dict[str, Any] = {}
        self._ar_emit_stop_token = True

    def _runner_hidden_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        """Device and dtype for tensors consumed by NPU or GPU AR runners."""
        device = _resolve_runtime_device(self.vllm_config)
        model_config = getattr(self.vllm_config, "model_config", None)
        dtype = getattr(model_config, "dtype", torch.float32) if model_config is not None else torch.float32
        return device, dtype

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
        """Load VoxCPM via its native runtime instead of vLLM's HF weight loader."""
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
            sample_rate = ref_audio.get("sample_rate") or ref_audio.get("sampling_rate") or ref_audio.get("sr")
            samples = None
            for key in ("audio", "wav", "samples", "array", "waveform"):
                if key in ref_audio and ref_audio[key] is not None:
                    samples = ref_audio[key]
                    break
            if sample_rate is None or samples is None:
                raise ValueError("ref_audio dict must contain waveform data and sample rate.")
            return cls._normalize_audio_samples(samples), int(sample_rate)

        if isinstance(ref_audio, (list, tuple)):
            if len(ref_audio) == 1:
                return cls._normalize_ref_audio(ref_audio[0])
            if len(ref_audio) == 2 and np.isscalar(ref_audio[1]):
                return cls._normalize_audio_samples(ref_audio[0]), int(ref_audio[1])

        raise TypeError(f"Unsupported ref_audio format: {type(ref_audio)!r}")

    @staticmethod
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

    def _get_vocab_size(self) -> int:
        model_config = getattr(self.vllm_config, "model_config", None)
        if model_config is not None:
            getter = getattr(model_config, "get_vocab_size", None)
            if callable(getter):
                try:
                    return int(getter())
                except Exception:
                    pass
            hf_config = getattr(model_config, "hf_text_config", None)
            if hf_config is not None and hasattr(hf_config, "vocab_size"):
                return int(hf_config.vocab_size)
        return 32000

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> torch.Tensor:
        del sampling_metadata
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states

        if hidden_states is None:
            device, dtype = self._runner_hidden_device_dtype()
            hidden_states = torch.zeros((0, 1), device=device, dtype=dtype)
        if hidden_states.ndim == 1:
            hidden_states = hidden_states.unsqueeze(-1)
        elif hidden_states.ndim > 2:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

        vocab_size = self._get_vocab_size()
        num_rows = int(hidden_states.shape[0])
        logits = torch.zeros(
            (num_rows, vocab_size),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        eos_id = 2 if vocab_size > 2 else 0
        safe_id = 1 if vocab_size > 1 and 1 != eos_id else 0
        emit_stop = getattr(self, "_ar_emit_stop_token", True)
        if num_rows > 0:
            if emit_stop:
                logits[:, eos_id] = 1.0e6
            else:
                logits[:, eos_id] = -1.0e9
                logits[:, safe_id] = 1.0e6
        return logits

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
        del positions, intermediate_tensors, inputs_embeds, kwargs
        self._ensure_model_loaded()
        out_device, out_dtype = self._runner_hidden_device_dtype()
        if input_ids is not None and input_ids.device.type == out_device.type:
            out_device = input_ids.device

        infos = runtime_additional_information or [{}]
        sample_rate = int(getattr(self._pipeline, "sample_rate", 24000))
        async_chunk = bool(getattr(self.vllm_config.model_config, "async_chunk", False))
        if self.model_stage in self._VAE_STAGES:
            if all(self._extract_val(info, "latent_audio_feat", None) is None for info in infos):
                self._ar_emit_stop_token = True
                return OmniOutput(
                    text_hidden_states=torch.zeros((0, 1), device=out_device, dtype=out_dtype),
                    multimodal_outputs={
                        "model_outputs": [torch.zeros((0,), dtype=torch.float32) for _ in infos],
                        "sr": [torch.tensor(sample_rate, dtype=torch.int32) for _ in infos],
                    },
                )
        else:
            texts = [self._extract_val(info, "text", "") for info in infos]
            if all(not text for text in texts):
                self._ar_emit_stop_token = True
                return OmniOutput(
                    text_hidden_states=torch.zeros((0, 1), device=out_device, dtype=out_dtype),
                    multimodal_outputs={
                        "latent_audio_feat": [torch.zeros((0,), dtype=torch.float32) for _ in infos],
                        "sr": [torch.tensor(sample_rate, dtype=torch.int32) for _ in infos],
                    },
                )

        outputs: list[torch.Tensor] = []
        sample_rates: list[torch.Tensor] = []
        last_chunk_flags: list[bool] | None = [] if (self.model_stage in self._LATENT_STAGES and async_chunk) else None
        for info in infos:
            if self.model_stage in self._VAE_STAGES:
                latent_audio_feat = self._extract_val(info, "latent_audio_feat", None)
                audio_tensor = self._pipeline.decode(
                    latent_audio_feat,
                    trim_streaming_patch=async_chunk,
                )
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

            request_key = str(info.get("_omni_req_id", "0"))
            created_temp: str | None = None

            if self.model_stage in self._LATENT_STAGES and async_chunk:
                if request_key not in self._latent_stream_gens:
                    prompt_wav_path, prompt_text, temp_prompt_wav = self._resolve_prompt_inputs(info)
                    created_temp = temp_prompt_wav
                    self._latent_stream_gens[request_key] = self._pipeline.iter_latent_chunks_streaming(
                        text=text,
                        prompt_wav_path=prompt_wav_path,
                        prompt_text=prompt_text,
                        cfg_value=cfg_value,
                        inference_timesteps=inference_timesteps,
                        min_len=min_len,
                        max_len=max_len,
                        streaming_prefix_len=streaming_prefix_len,
                        retry_badcase=False,
                        retry_badcase_max_times=retry_badcase_max_times,
                        retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
                    )
                generator = self._latent_stream_gens[request_key]
                try:
                    chunk_latent, is_last = next(generator)
                except StopIteration:
                    self._latent_stream_gens.pop(request_key, None)
                    outputs.append(torch.zeros((0,), dtype=torch.float32))
                    assert last_chunk_flags is not None
                    last_chunk_flags.append(True)
                else:
                    if is_last:
                        self._latent_stream_gens.pop(request_key, None)
                    outputs.append(chunk_latent.detach().float().cpu())
                    assert last_chunk_flags is not None
                    last_chunk_flags.append(bool(is_last))
                finally:
                    if created_temp is not None and os.path.exists(created_temp):
                        os.unlink(created_temp)
                sample_rates.append(torch.tensor(sample_rate, dtype=torch.int32))
                continue

            prompt_wav_path, prompt_text, temp_prompt_wav = self._resolve_prompt_inputs(info)
            try:
                if self.model_stage in self._LATENT_STAGES:
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
        multimodal_outputs: dict[str, Any] = {output_key: outputs, "sr": sample_rates}
        if outputs:
            outputs_tensor = torch.stack(outputs)
            if outputs_tensor.ndim == 1:
                text_hidden_states = outputs_tensor.unsqueeze(-1)
            else:
                text_hidden_states = outputs_tensor.reshape(-1, outputs_tensor.shape[-1])
        else:
            text_hidden_states = torch.zeros((0, 1), device=out_device, dtype=out_dtype)
        text_hidden_states = text_hidden_states.to(device=out_device, dtype=out_dtype)

        if self.model_stage in self._LATENT_STAGES and async_chunk and last_chunk_flags:
            self._ar_emit_stop_token = all(last_chunk_flags)
        else:
            self._ar_emit_stop_token = True

        return OmniOutput(
            text_hidden_states=text_hidden_states,
            multimodal_outputs=multimodal_outputs,
        )

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        del batch_size, dtype, device
        return {}
