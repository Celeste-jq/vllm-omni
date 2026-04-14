from __future__ import annotations

import logging
import os
import time
from collections.abc import Generator
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange

logger = logging.getLogger(__name__)


class _DirectVoxCPMLatentGenerator:
    def __init__(self, tts_model: Any):
        self.tts_model = tts_model
        self.sample_rate = int(getattr(tts_model, "sample_rate", 24000))
        self._profile_stream_inner = os.getenv("VOXCPM_PROFILE_STREAM_INNER", "0") == "1"

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

        gen_kw = dict(
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
        latent_entry = getattr(self.tts_model, "generate_latents_with_prompt_cache", None)
        if latent_entry is not None:
            _, _, pred_audio_feat = latent_entry(**gen_kw)
        else:
            try:
                _, _, pred_audio_feat = self.tts_model.generate_with_prompt_cache(
                    **gen_kw,
                    latents_only=True,
                )
            except TypeError:
                _, _, pred_audio_feat = self.tts_model.generate_with_prompt_cache(**gen_kw)
        return pred_audio_feat.detach().cpu().to(torch.float32)

    def iter_latent_chunks_streaming(
        self,
        *,
        text: str,
        prompt_wav_path: str | None = None,
        prompt_text: str | None = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        min_len: int = 2,
        max_len: int = 4096,
        streaming_prefix_len: int = 3,
        retry_badcase: bool = False,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,
        profile_tag: str | None = None,
    ) -> Generator[tuple[torch.Tensor, bool], None, None]:
        """Yield ``(latent_window, is_last_chunk)`` for Omni async_chunk latent to VAE."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("target text must be a non-empty string")
        if (prompt_wav_path is None) != (prompt_text is None):
            raise ValueError("prompt_wav_path and prompt_text must both be provided or both be None")
        if prompt_wav_path is not None and not os.path.exists(prompt_wav_path):
            raise FileNotFoundError(f"prompt_wav_path does not exist: {prompt_wav_path}")

        prompt_cache = None
        prompt_cache_ms = 0.0
        if prompt_wav_path is not None and prompt_text is not None:
            prompt_cache_start = time.perf_counter()
            prompt_cache = self.tts_model.build_prompt_cache(
                prompt_text=prompt_text,
                prompt_wav_path=prompt_wav_path,
            )
            prompt_cache_ms = (time.perf_counter() - prompt_cache_start) * 1000.0

        gen_kw = dict(
            target_text=" ".join(text.split()),
            prompt_cache=prompt_cache,
            min_len=min_len,
            max_len=max_len,
            inference_timesteps=inference_timesteps,
            cfg_value=cfg_value,
            retry_badcase=retry_badcase,
            retry_badcase_max_times=retry_badcase_max_times,
            retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
            streaming_prefix_len=streaming_prefix_len,
            profile_tag=profile_tag,
        )
        stream_source = "generate_latents_with_prompt_cache_streaming"
        create_start = time.perf_counter()
        stream_entry = getattr(self.tts_model, "generate_latents_with_prompt_cache_streaming", None)
        if stream_entry is not None:
            gen = stream_entry(**gen_kw)
        else:
            fallback_stream_entry = getattr(self.tts_model, "generate_with_prompt_cache_streaming", None)
            if fallback_stream_entry is not None:
                stream_source = "generate_with_prompt_cache_streaming"
                gen = fallback_stream_entry(**gen_kw, latents_only=True)
            else:
                stream_source = "_generate_with_prompt_cache"
                gen = self.tts_model._generate_with_prompt_cache(streaming=True, latents_only=True, **gen_kw)
        create_ms = (time.perf_counter() - create_start) * 1000.0

        iterator = iter(gen)
        bootstrap_start = time.perf_counter()
        previous = next(iterator, None)
        bootstrap_ms = (time.perf_counter() - bootstrap_start) * 1000.0
        if self._profile_stream_inner:
            logger.warning(
                "[VoxCPM][stream-inner] tag=%s source=%s text_len=%d prompt_cache=%s prompt_cache_ms=%.3f create_ms=%.3f bootstrap_ms=%.3f first_none=%s",
                profile_tag or "-",
                stream_source,
                len(text),
                prompt_cache is not None,
                prompt_cache_ms,
                create_ms,
                bootstrap_ms,
                previous is None,
            )
        chunk_idx = 0
        while previous is not None:
            step_start = time.perf_counter()
            current = next(iterator, None)
            step_ms = (time.perf_counter() - step_start) * 1000.0
            _, _target_tok, chunk_latent = previous
            if not isinstance(chunk_latent, torch.Tensor):
                chunk_latent = torch.as_tensor(chunk_latent)
            if self._profile_stream_inner:
                logger.warning(
                    "[VoxCPM][stream-inner] tag=%s idx=%d yielded_shape=%s step_ms=%.3f is_last=%s",
                    profile_tag or "-",
                    chunk_idx,
                    tuple(chunk_latent.shape),
                    step_ms,
                    current is None,
                )
            yield chunk_latent, current is None
            previous = current
            chunk_idx += 1


class _DirectVoxCPMAudioVAE:
    def __init__(self, audio_vae: nn.Module, *, patch_size: int = 2):
        self.audio_vae = audio_vae
        self.sample_rate = int(getattr(audio_vae, "sample_rate", 24000))
        self.latent_dim = int(getattr(audio_vae, "latent_dim", 64))
        self.patch_size = int(patch_size)
        self._chunk_size = int(getattr(audio_vae, "chunk_size", 1))
        self._stream_audio_patch_samples = max(1, self.patch_size * self._chunk_size)

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
    def decode(self, latent_audio_feat: Any, *, trim_streaming_patch: bool = False) -> torch.Tensor:
        latents = self._prepare_latents_for_decode(latent_audio_feat)
        device = next(self.audio_vae.parameters()).device
        raw = self.audio_vae.decode(latents.to(device=device, dtype=torch.float32))
        if isinstance(raw, dict):
            audio = raw.get("audio")
            if audio is None:
                audio = next(v for v in raw.values() if isinstance(v, torch.Tensor))
        else:
            audio = raw
        if audio.dim() == 3:
            stream = audio.squeeze(1)
        elif audio.dim() == 2:
            stream = audio
        else:
            stream = audio.reshape(audio.shape[0], -1)
        if trim_streaming_patch:
            stream = stream[..., -self._stream_audio_patch_samples :]
        return stream.reshape(-1).detach().cpu().to(torch.float32)
