from __future__ import annotations

import os
import tempfile
import time
import wave
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

from ..voxcpm2.minicpm4_paged import MiniCPM4PagedForVoxCPM2, MiniCPM4PagedResidualLM
from .voxcpm_loader import _resolve_runtime_device, load_native_voxcpm_model
from .voxcpm_request_state import Stage0RequestState
from .voxcpm_stage0_paged import append_latent_patch, maybe_emit_window, resolve_left_context_frames

logger = init_logger(__name__)


def _resolve_lm_cfg(config: Any) -> Any:
    lm_cfg = getattr(config, "lm_config", config)
    if isinstance(lm_cfg, dict):

        class _Cfg:
            pass

        cfg = _Cfg()
        for key, value in lm_cfg.items():
            setattr(cfg, key, value)
        return cfg
    return lm_cfg


class VoxCPMStage0PagedForConditionalGeneration(nn.Module):
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"base_lm.": "model."})

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self._lm_cfg = _resolve_lm_cfg(self.config)

        self.have_multimodal_outputs = True
        self.has_preprocess = True
        self.has_postprocess = True

        self.model = MiniCPM4PagedForVoxCPM2(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.residual_model = MiniCPM4PagedResidualLM(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "residual_model"),
        )
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

        self._tts: nn.Module | None = None
        self._device = str(_resolve_runtime_device(vllm_config))
        self._side_dtype = torch.bfloat16

        self._hidden_size = int(getattr(self._lm_cfg, "hidden_size"))
        self._vocab_size = int(getattr(self._lm_cfg, "vocab_size"))
        self._patch_size = getattr(self.config, "patch_size", 2)
        self._feat_dim = getattr(self.config, "feat_dim", 64)
        self._sample_rate = getattr(self.config, "sample_rate", 24000)
        self._max_decode_steps = 2000
        self._max_batch_size = getattr(vllm_config.scheduler_config, "max_num_seqs", 1)

        self._active_states: dict[str, Stage0RequestState] = {}
        self._current_request_id: str | None = None
        self._pending_requests: list[tuple[str, bool, torch.Tensor | None, int]] = []
        self._results_queue: list[tuple[str, torch.Tensor | None]] = []
        self._latent_queue: list[tuple[str, dict[str, Any] | None]] = []
        self._deferred_cleanup_ids: set[str] = set()

    @property
    def tts(self) -> nn.Module:
        assert self._tts is not None, "Model not loaded yet"
        return self._tts

    def _get_or_create_state(self, request_id: str) -> Stage0RequestState:
        if request_id not in self._active_states:
            self._active_states[request_id] = Stage0RequestState(request_id=request_id)
        return self._active_states[request_id]

    def _switch_to_request(self, request_id: str) -> Stage0RequestState:
        if request_id != self._current_request_id:
            self._current_request_id = request_id
        return self._get_or_create_state(request_id)

    def _cleanup_request(self, request_id: str) -> None:
        self._active_states.pop(request_id, None)
        if self._current_request_id == request_id:
            self._current_request_id = None

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        self._deferred_cleanup_ids.update(finished_req_ids)

    def _flush_deferred_cleanup(self) -> None:
        for req_id in self._deferred_cleanup_ids:
            state = self._active_states.get(req_id)
            if state is not None and not state.terminal_payload_sent:
                continue
            self._cleanup_request(req_id)
        self._deferred_cleanup_ids = {
            req_id
            for req_id in self._deferred_cleanup_ids
            if req_id in self._active_states
        }

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

    @staticmethod
    def _extract_val(info: dict[str, Any], key: str, default: Any) -> Any:
        value = info.get(key, default)
        if isinstance(value, list):
            return value[0] if value else default
        return value

    def _build_prompt_cache_from_info(self, info: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        prompt_text = self._extract_val(info, "prompt_text", None)
        prompt_wav_path = self._extract_val(info, "prompt_wav_path", None)
        ref_audio = self._extract_val(info, "ref_audio", None)
        ref_text = self._extract_val(info, "ref_text", None)

        temp_prompt_wav: str | None = None
        if prompt_wav_path:
            effective_prompt_text = prompt_text or ref_text
            if effective_prompt_text:
                return self.tts.build_prompt_cache(
                    prompt_text=effective_prompt_text,
                    prompt_wav_path=prompt_wav_path,
                ), None

        if ref_audio is None or ref_text is None:
            return None, None

        if isinstance(ref_audio, str):
            return self.tts.build_prompt_cache(prompt_text=ref_text, prompt_wav_path=ref_audio), None

        waveform, sample_rate = self._normalize_ref_audio(ref_audio)
        temp_prompt_wav = self._write_temp_prompt_wav(waveform, sample_rate)
        return self.tts.build_prompt_cache(prompt_text=ref_text, prompt_wav_path=temp_prompt_wav), temp_prompt_wav

    def _build_prefill_inputs(self, token_ids: list[int], dev: Any, req_id: str = "default") -> dict[str, torch.Tensor]:
        tts = self.tts
        dtype = self._side_dtype
        state = self._active_states.get(req_id)
        cache = state.prompt_cache if state else None
        prompt_audio_feat = cache["audio_feat"] if cache is not None else torch.empty(
            (0, self._patch_size, self._feat_dim),
            dtype=torch.float32,
        )

        prompt_text = cache["prompt_text"] if cache is not None else ""
        prompt_ids = list(tts.text_tokenizer(prompt_text)) if prompt_text else []
        all_ids = prompt_ids + token_ids

        text_token = torch.tensor(all_ids, dtype=torch.int32)
        text_token = torch.cat([text_token, torch.tensor([tts.audio_start_token], dtype=torch.int32)], dim=-1)

        audio_length = prompt_audio_feat.size(0)
        text_length = text_token.shape[0]
        text_pad_token = torch.zeros(audio_length, dtype=torch.int32)
        audio_pad_feat = torch.zeros((text_length, self._patch_size, self._feat_dim), dtype=torch.float32)

        text_token = torch.cat([text_token, text_pad_token], dim=0)
        audio_feat = torch.cat([audio_pad_feat, prompt_audio_feat], dim=0)
        text_mask = torch.cat([torch.ones(text_length), torch.zeros(audio_length)]).to(torch.int32)
        audio_mask = torch.cat([torch.zeros(text_length), torch.ones(audio_length)]).to(torch.int32)

        return {
            "text_token": text_token.unsqueeze(0).to(dev),
            "audio_feat": audio_feat.unsqueeze(0).to(dev).to(dtype),
            "text_mask": text_mask.unsqueeze(0).to(dev),
            "audio_mask": audio_mask.unsqueeze(0).to(dev),
        }

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        del input_embeds
        additional = info_dict.get("additional_information")
        if isinstance(additional, dict):
            merged = {k: v for k, v in info_dict.items() if k != "additional_information"}
            for k, v in additional.items():
                merged.setdefault(k, v)
            info_dict = merged

        span_len = int(input_ids.shape[0])
        dev = input_ids.device
        req_id = str(
            self._extract_val(
                info_dict,
                "request_id",
                self._extract_val(info_dict, "_omni_req_id", "default"),
            )
        )
        is_prefill = span_len > 1
        state = self._get_or_create_state(req_id)

        if is_prefill:
            pending_ids = {rid for rid, *_ in self._pending_requests}
            pending_ids.add(req_id)
            if self._current_request_id:
                pending_ids.add(self._current_request_id)
            for rid in [
                active_req_id
                for active_req_id, active_state in self._active_states.items()
                if active_req_id not in pending_ids
                and (active_state.prefill_completed or active_state.terminal_payload_sent)
            ]:
                self._cleanup_request(rid)

            state.request_start_time = 0.0
            state.prefill_completed = False
            state.is_stopping = False
            state.terminal_payload_pending = False
            state.terminal_payload_sent = False
            state.decode_step_count = 0
            state.curr_embed_for_next = None
            state.prev_feat_embed = None
            state.curr_prefix_feat_cond = None
            state.precomputed_stop_logits = None
            state.pending_latent_patch = None
            state.generated_latent_history = []
            state.generated_latent_frames = 0
            state.emitted_latent_frames = 0
            state.streaming_prefix_len = int(self._extract_val(info_dict, "streaming_prefix_len", 3))
            state.codec_left_context_frames = self._extract_val(info_dict, "codec_left_context_frames", None)
            state.min_len = int(self._extract_val(info_dict, "min_len", 2))
            state.max_len = int(self._extract_val(info_dict, "max_len", self._extract_val(info_dict, "max_new_tokens", 4096)))
            state.cfg_value = float(self._extract_val(info_dict, "cfg_value", 2.0))
            state.inference_timesteps = int(self._extract_val(info_dict, "inference_timesteps", 10))

            token_ids = input_ids.tolist()
            bos_token_id = getattr(self.config, "bos_token_id", None)
            if token_ids and bos_token_id is not None and token_ids[0] == bos_token_id:
                token_ids = token_ids[1:]

            temp_prompt_wav: str | None = None
            try:
                state.prompt_cache, temp_prompt_wav = self._build_prompt_cache_from_info(info_dict)
                inputs = self._build_prefill_inputs(token_ids, dev, req_id)
            finally:
                if temp_prompt_wav is not None and os.path.exists(temp_prompt_wav):
                    os.unlink(temp_prompt_wav)

            tts = self.tts
            feat_embed = tts.enc_to_lm_proj(tts.feat_encoder(inputs["audio_feat"]))
            text_embed = self.model.embed_input_ids(inputs["text_token"].to(dev))
            text_mask, feat_mask = inputs["text_mask"], inputs["audio_mask"]
            embeds = (text_mask.unsqueeze(-1) * text_embed + feat_mask.unsqueeze(-1) * feat_embed).squeeze(0)
            state.prefill_artifacts = {
                "text_mask": text_mask,
                "feat_mask": feat_mask,
                "feat": inputs["audio_feat"],
                "feat_embed": feat_embed,
            }
        else:
            curr = state.curr_embed_for_next
            if curr is not None:
                embeds = curr.to(dev, dtype=self._side_dtype).reshape(1, -1)
            else:
                embeds = torch.zeros(1, self._hidden_size, device=dev, dtype=self._side_dtype)

        self._pending_requests.append((req_id, is_prefill, embeds, span_len))
        return input_ids, embeds, {}

    def _prepare_residual_prefill(self, state: Stage0RequestState, base_lm_out: torch.Tensor, dev: Any):
        tts = self.tts
        artifacts = state.prefill_artifacts or {}
        text_mask = artifacts["text_mask"]
        feat_mask = artifacts["feat_mask"]
        feat = artifacts["feat"]
        feat_embed = artifacts["feat_embed"]
        state.prefill_artifacts = None

        tts_len = text_mask.shape[1]
        scaffold_len = base_lm_out.shape[0]
        if scaffold_len < tts_len:
            pad = torch.zeros(
                tts_len - scaffold_len,
                base_lm_out.shape[-1],
                device=base_lm_out.device,
                dtype=base_lm_out.dtype,
            )
            enc_out = torch.cat([base_lm_out, pad], dim=0).unsqueeze(0)
        else:
            enc_out = base_lm_out.unsqueeze(0)

        prefix_feat_cond = (
            feat[:, -1, ...]
            if feat.shape[1] > 0
            else torch.zeros(1, self._patch_size, self._feat_dim, device=dev, dtype=self._side_dtype)
        )
        enc_outputs = tts.fsq_layer(enc_out) * feat_mask.unsqueeze(-1) + enc_out * text_mask.unsqueeze(-1)
        lm_hidden = enc_outputs[:, -1, :]
        residual_input = enc_outputs + feat_mask.unsqueeze(-1) * feat_embed
        meta = {"lm_hidden": lm_hidden, "prefix_feat_cond": prefix_feat_cond}
        return residual_input.squeeze(0), meta

    def _prepare_residual_decode(self, state: Stage0RequestState, base_lm_out: torch.Tensor):
        tts = self.tts
        state.decode_step_count += 1
        if state.decode_step_count >= min(self._max_decode_steps, state.max_len):
            logger.warning("MAX_DECODE_STEPS for %s (%d), forcing stop", state.request_id, state.decode_step_count)
            state.is_stopping = True

        h = base_lm_out.unsqueeze(0) if base_lm_out.ndim == 1 else base_lm_out
        lm_h = tts.fsq_layer(h)
        if lm_h.ndim == 1:
            lm_h = lm_h.unsqueeze(0)

        prev = state.prev_feat_embed.to(self._side_dtype)
        if prev.ndim == 1:
            prev = prev.unsqueeze(0)
        res_input = lm_h + prev
        return res_input, {"new_lm_hidden": lm_h}

    def _run_dit(self, state: Stage0RequestState, lm_hidden: torch.Tensor, residual_hidden: torch.Tensor) -> torch.Tensor:
        tts = self.tts
        dit_hidden = tts.lm_to_dit_proj(lm_hidden) + tts.res_to_dit_proj(residual_hidden)
        cond = state.curr_prefix_feat_cond
        if cond is None:
            cond = torch.zeros(self._patch_size, self._feat_dim, device=dit_hidden.device, dtype=self._side_dtype)
        if cond.ndim == 2:
            cond = cond.unsqueeze(0)
        pred_feat = tts.feat_decoder(
            mu=dit_hidden,
            patch_size=self._patch_size,
            cond=cond.transpose(1, 2).contiguous(),
            n_timesteps=state.inference_timesteps,
            cfg_value=state.cfg_value,
        ).transpose(1, 2)
        return pred_feat

    def _finish_prefill(self, state: Stage0RequestState, meta: dict[str, Any], res_out: torch.Tensor):
        tts = self.tts
        lm_hidden = meta["lm_hidden"]
        prefix_feat_cond = meta["prefix_feat_cond"]
        residual_hidden = res_out[-1:, :]

        state.precomputed_stop_logits = tts.stop_head(tts.stop_actn(tts.stop_proj(lm_hidden))).detach()
        state.curr_prefix_feat_cond = prefix_feat_cond[0].detach()
        pred_feat = self._run_dit(state, lm_hidden, residual_hidden)

        with torch.no_grad():
            curr_embed = tts.enc_to_lm_proj(tts.feat_encoder(pred_feat.unsqueeze(1))).squeeze(1)

        state.curr_embed_for_next = curr_embed.detach()
        state.prev_feat_embed = curr_embed.detach()
        state.curr_prefix_feat_cond = pred_feat[0].detach()
        state.decode_step_count = 0
        state.request_start_time = time.perf_counter()
        state.prefill_completed = True
        append_latent_patch(state, pred_feat[0].detach().to(torch.float32).cpu())

    def _finish_decode(self, state: Stage0RequestState, meta: dict[str, Any], res_out: torch.Tensor):
        tts = self.tts
        lm_hidden = meta["new_lm_hidden"]
        residual_hidden = res_out.unsqueeze(0) if res_out.ndim == 1 else res_out

        pred_feat = self._run_dit(state, lm_hidden, residual_hidden)
        next_embed = tts.enc_to_lm_proj(tts.feat_encoder(pred_feat.unsqueeze(1))).squeeze(1)

        state.precomputed_stop_logits = tts.stop_head(tts.stop_actn(tts.stop_proj(lm_hidden))).detach()
        state.curr_embed_for_next = next_embed.detach()
        state.prev_feat_embed = next_embed.detach()
        state.curr_prefix_feat_cond = pred_feat[0].detach()
        append_latent_patch(state, pred_feat[0].detach().to(torch.float32).cpu())

        if state.precomputed_stop_logits is not None and state.decode_step_count >= state.min_len:
            state.is_stopping = bool(state.precomputed_stop_logits[0, 1] > state.precomputed_stop_logits[0, 0])

    def _queue_payload(self, state: Stage0RequestState) -> dict[str, Any] | None:
        payload = maybe_emit_window(
            state,
            resolve_left_context_frames(
                codec_left_context_frames=state.codec_left_context_frames,
                streaming_prefix_len=state.streaming_prefix_len,
            ),
        )
        if payload is None:
            return None
        payload["finished"] = False
        if state.is_stopping:
            state.terminal_payload_pending = True
        return payload

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        del kwargs

        model_output = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        if isinstance(model_output, IntermediateTensors):
            return model_output
        scaffold_hidden = model_output[0] if isinstance(model_output, tuple) else model_output

        token_offset = 0
        residual_inputs: list[torch.Tensor] = []
        residual_positions: list[torch.Tensor] = []
        req_metas: list[tuple[Stage0RequestState, bool, dict[str, Any]]] = []

        for req_id, is_prefill, _req_embeds, n in self._pending_requests:
            state = self._switch_to_request(req_id)
            req_hidden = scaffold_hidden[token_offset : token_offset + n]
            req_pos = positions[token_offset : token_offset + n]

            if not is_prefill and state.terminal_payload_pending and not state.terminal_payload_sent:
                token_offset += n
                state.terminal_payload_pending = False
                state.terminal_payload_sent = True
                self._results_queue.append((req_id, None))
                self._latent_queue.append(
                    (
                        req_id,
                        {
                            "latent_audio_feat": None,
                            "left_context_size": 0,
                            "finished": True,
                        },
                    )
                )
                continue
            elif is_prefill:
                res_input, meta = self._prepare_residual_prefill(state, req_hidden, input_ids.device)
            elif state.prefill_completed:
                res_input, meta = self._prepare_residual_decode(state, req_hidden)
            else:
                token_offset += n
                self._results_queue.append((req_id, None))
                self._latent_queue.append((req_id, None))
                continue

            residual_inputs.append(res_input)
            residual_positions.append(req_pos)
            req_metas.append((state, is_prefill, meta))
            token_offset += n

        if residual_inputs:
            batch_in = torch.cat(residual_inputs, dim=0)
            batch_pos = torch.cat(residual_positions, dim=0)
            batch_out = self.residual_model(batch_pos, batch_in)

            offset = 0
            for idx, (state, is_prefill, meta) in enumerate(req_metas):
                n = residual_inputs[idx].shape[0]
                res_out = batch_out[offset : offset + n]
                offset += n

                if is_prefill:
                    self._finish_prefill(state, meta, res_out)
                else:
                    self._finish_decode(state, meta, res_out)

                self._results_queue.append((state.request_id, state.precomputed_stop_logits))
                self._latent_queue.append((state.request_id, self._queue_payload(state)))

        self._pending_requests.clear()
        self._flush_deferred_cleanup()
        return scaffold_hidden

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> torch.Tensor | None:
        del sampling_metadata
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None

        bsz = hidden_states.shape[0]
        logits = torch.full(
            (bsz, self._vocab_size),
            float("-inf"),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        eos_id = 2 if logits.shape[1] > 2 else 0
        safe_id = 1 if logits.shape[1] > 1 and 1 != eos_id else 0

        if self._results_queue:
            for i, (req_id, stop_logits) in enumerate(self._results_queue):
                if i >= bsz:
                    break
                state = self._active_states.get(req_id)
                if state is not None and state.is_stopping and state.terminal_payload_sent:
                    logits[i, eos_id] = 1.0
                    continue
                should_stop = False
                if stop_logits is not None and state is not None and state.decode_step_count >= state.min_len:
                    should_stop = bool(stop_logits[0, 1] > stop_logits[0, 0])
                    state.is_stopping = should_stop
                if should_stop:
                    logits[i, safe_id] = 1.0
                else:
                    logits[i, safe_id] = 1.0
            self._results_queue.clear()
        else:
            logits[:, safe_id] = 1.0
        return logits

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        del kwargs
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        mm: dict[str, Any] = {}
        if self._latent_queue:
            payload_by_req = {rid: payload for rid, payload in self._latent_queue}
            order = [r for r, _ in self._latent_queue]
            mm["latent_audio_feat"] = [
                (payload_by_req.get(r) or {}).get("latent_audio_feat")
                for r in order
            ]
            mm["left_context_size"] = [
                torch.tensor(int((payload_by_req.get(r) or {}).get("left_context_size", 0)), dtype=torch.int32)
                for r in order
            ]
            mm["finished"] = [
                torch.tensor(bool((payload_by_req.get(r) or {}).get("finished", False)), dtype=torch.bool)
                for r in order
            ]
            mm["sr"] = [torch.tensor(self._sample_rate, dtype=torch.int32) for _ in order]
            self._latent_queue.clear()

        return OmniOutput(text_hidden_states=model_outputs, multimodal_outputs=mm)

    def postprocess(self, hidden_states: torch.Tensor, **info: Any) -> dict[str, Any]:
        del hidden_states, info
        return {}

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def _base_lm_only(ws):
            for name, tensor in ws:
                if name.startswith("base_lm."):
                    yield name, tensor

        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(_base_lm_only(weights), mapper=self.hf_to_vllm_mapper)

        model_path = self.vllm_config.model_config.model
        target_device = _resolve_runtime_device(self.vllm_config)
        native = load_native_voxcpm_model(
            model_path,
            device=target_device,
            dtype=str(getattr(self.vllm_config.model_config, "dtype", "bfloat16")).removeprefix("torch."),
        )
        self._tts = native.to(target_device)
        self._side_dtype = self._tts.fusion_concat_proj.weight.dtype if hasattr(self._tts, "fusion_concat_proj") else (
            self._tts.enc_to_lm_proj.weight.dtype
        )
        self._device = str(target_device)
        self._patch_size = self._tts.patch_size
        self._feat_dim = self._tts.feat_dim
        self._sample_rate = self._tts.sample_rate

        n = self.residual_model.load_weights_from_native(self._tts.residual_lm)
        for name, _ in self.residual_model.named_parameters():
            loaded.add(f"residual_model.{name}")
        logger.info("VoxCPM1.5 paged stage0: loaded %d params into paged residual_model", n)

        del self._tts.base_lm
        self._tts.base_lm = None
        del self._tts.residual_lm
        self._tts.residual_lm = None
        if target_device.type == "cuda":
            torch.cuda.empty_cache()

        logger.info(
            "Loaded VoxCPM1.5 paged stage0 (patch=%d, feat_dim=%d, dtype=%s)",
            self._patch_size,
            self._feat_dim,
            self._side_dtype,
        )
        return loaded
