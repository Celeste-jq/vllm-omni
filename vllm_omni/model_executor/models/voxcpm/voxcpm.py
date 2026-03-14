from __future__ import annotations

import os
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


def _import_voxcpm_class():
    try:
        from voxcpm.core import VoxCPM

        return VoxCPM
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
            from voxcpm.core import VoxCPM

            return VoxCPM
        except ImportError:
            continue

    raise ImportError(
        "Failed to import VoxCPM. Install the `voxcpm` package or set "
        "`VLLM_OMNI_VOXCPM_CODE_PATH` to the VoxCPM `src` directory."
    )


class VoxCPMForConditionalGeneration(nn.Module):
    input_modalities = "audio"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        del prefix
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True
        self._pipeline = None

    def _ensure_model_loaded(self):
        if self._pipeline is not None:
            return

        VoxCPM = _import_voxcpm_class()
        self._pipeline = VoxCPM(
            voxcpm_model_path=self.model_path,
            zipenhancer_model_path=None,
            enable_denoiser=False,
            optimize=False,
        )

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
        texts = [self._extract_val(info, "text", "") for info in infos]
        if all(not text for text in texts):
            sample_rate = int(getattr(self._pipeline.tts_model, "sample_rate", 24000))
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "model_outputs": [torch.zeros((0,), dtype=torch.float32) for _ in infos],
                    "sr": [torch.tensor(sample_rate, dtype=torch.int32) for _ in infos],
                },
            )

        outputs: list[torch.Tensor] = []
        sample_rates: list[torch.Tensor] = []
        for info in infos:
            text = self._extract_val(info, "text", "")
            cfg_value = float(self._extract_val(info, "cfg_value", 2.0))
            inference_timesteps = int(self._extract_val(info, "inference_timesteps", 10))
            min_len = int(self._extract_val(info, "min_len", 2))
            max_len = int(self._extract_val(info, "max_len", self._extract_val(info, "max_new_tokens", 4096)))
            normalize = bool(self._extract_val(info, "normalize", False))
            denoise = bool(self._extract_val(info, "denoise", False))
            retry_badcase = bool(self._extract_val(info, "retry_badcase", True))
            retry_badcase_max_times = int(self._extract_val(info, "retry_badcase_max_times", 3))
            retry_badcase_ratio_threshold = float(self._extract_val(info, "retry_badcase_ratio_threshold", 6.0))

            prompt_wav_path, prompt_text, temp_prompt_wav = self._resolve_prompt_inputs(info)
            try:
                audio_np = self._pipeline.generate(
                    text=text,
                    prompt_wav_path=prompt_wav_path,
                    prompt_text=prompt_text,
                    cfg_value=cfg_value,
                    inference_timesteps=inference_timesteps,
                    min_len=min_len,
                    max_len=max_len,
                    normalize=normalize,
                    denoise=denoise,
                    retry_badcase=retry_badcase,
                    retry_badcase_max_times=retry_badcase_max_times,
                    retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
                )
            finally:
                if temp_prompt_wav is not None and os.path.exists(temp_prompt_wav):
                    os.unlink(temp_prompt_wav)

            if isinstance(audio_np, np.ndarray):
                outputs.append(torch.from_numpy(audio_np).float())
            elif isinstance(audio_np, torch.Tensor):
                outputs.append(audio_np.float().cpu())
            else:
                outputs.append(torch.tensor(audio_np, dtype=torch.float32))

            sample_rates.append(torch.tensor(int(self._pipeline.tts_model.sample_rate), dtype=torch.int32))

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": outputs, "sr": sample_rates},
        )

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        del batch_size, dtype, device
        return {}
