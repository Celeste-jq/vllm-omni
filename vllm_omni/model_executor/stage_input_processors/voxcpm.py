from __future__ import annotations

from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt

logger = init_logger(__name__)

# Default latent time-slices per SHM chunk (override via connector extra.latent_chunk_patches).
DEFAULT_LATENT_CHUNK_PATCHES = 8
DEFAULT_LATENT_LEFT_CONTEXT_PATCHES = 0


def voxcpm_pooler_streaming_has_more(pooling_output: dict[str, Any]) -> bool | None:
    """Parse native streaming flag from Stage0 ``pooling_output``.

    Value must be a **tensor** (0/1) for vLLM IPC (msgspec); raw bool breaks decode.
    Returns ``None`` if this is not a native-streaming pooler payload.
    """
    if "voxcpm_streaming_continue" not in pooling_output:
        return None
    val = pooling_output["voxcpm_streaming_continue"]
    if isinstance(val, (list, tuple)):
        if not val:
            return False
        val = val[0]
    if isinstance(val, torch.Tensor):
        return bool(val.detach().cpu().reshape(-1)[0].item() > 0.5)
    return bool(val)


def latent2vae(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    del prompt, requires_multimodal_data

    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")

    source_outputs = stage_list[source_stage_id].engine_outputs
    if source_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    vae_inputs: list[OmniTokensPrompt] = []
    for source_output in source_outputs:
        output = source_output.outputs[0]
        multimodal_output = getattr(output, "multimodal_output", None)
        if not isinstance(multimodal_output, dict) or "latent_audio_feat" not in multimodal_output:
            raise ValueError(
                "VoxCPM latent stage output missing 'latent_audio_feat'. "
                f"request_id={getattr(source_output, 'request_id', None)}"
            )

        additional_information = {
            "latent_audio_feat": multimodal_output["latent_audio_feat"],
        }
        if "sr" in multimodal_output:
            additional_information["sample_rate"] = [int(multimodal_output["sr"])]

        vae_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0],
                additional_information=additional_information,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return vae_inputs


def latent2vae_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> dict[str, Any] | list[dict[str, Any]] | None:
    """Stage0 -> Stage1 streaming.

    - **Chunk streaming** (``voxcpm_streaming_continue`` in ``pooling_output``): one SHM payload
      per ``save_async``. Each payload aggregates up to ``latent_chunk_patches`` native patches
      along time (same boundaries as the sliced fallback), so Stage-1 can recv/decode after
      each chunk instead of waiting for the full latent.
    - **Sliced fallback** (no such key): split a full latent tensor into multiple payloads
      (returns a list) when Stage0 produced the entire latent in one forward (non-streaming).
    """
    finished = bool(is_finished or (request.is_finished() if hasattr(request, "is_finished") else False))

    if not isinstance(pooling_output, dict):
        if finished:
            return {"latent_audio_feat": None, "sr": None, "finished": True}
        return None

    # Chunk streaming: one pooling_output per scheduler step → one SHM put per aggregated chunk.
    # Optional latent left-context window (similar to codec left-context in Qwen3-TTS).
    # When enabled, each payload carries [left_context + current_chunk], and
    # left_context_size indicates how many leading patches belong to context.
    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size_cfg = int(cfg.get("latent_chunk_patches", DEFAULT_LATENT_CHUNK_PATCHES))
    left_ctx_cfg = int(cfg.get("latent_left_context_patches", DEFAULT_LATENT_LEFT_CONTEXT_PATCHES))
    if chunk_size_cfg <= 0:
        chunk_size_cfg = DEFAULT_LATENT_CHUNK_PATCHES
    if left_ctx_cfg < 0:
        left_ctx_cfg = DEFAULT_LATENT_LEFT_CONTEXT_PATCHES

    if "voxcpm_streaming_continue" in pooling_output:
        latent = pooling_output.get("latent_audio_feat")
        sr = pooling_output.get("sr")
        if sr is not None and isinstance(sr, torch.Tensor):
            t = sr.detach().cpu()
            sr = int(t.item()) if t.numel() == 1 else int(t[0].item())
        elif sr is not None and hasattr(sr, "__len__") and len(sr) > 0:
            sr = int(sr[0]) if isinstance(sr[0], (int, float)) else int(sr[0].item())
        else:
            sr = 24000
        hm = voxcpm_pooler_streaming_has_more(pooling_output)
        has_more = bool(hm) if hm is not None else False
        payload_finished = bool(finished) or (not has_more)
        if latent is None:
            return {"latent_audio_feat": None, "sr": sr, "finished": True} if payload_finished else None
        if isinstance(latent, torch.Tensor):
            latent = latent.detach().cpu().float().contiguous()
        else:
            latent = torch.tensor(latent, dtype=torch.float32)
        current_chunk_patches = int(latent.shape[0]) if latent.ndim >= 1 else -1
        if left_ctx_cfg > 0:
            req_id = getattr(request, "external_req_id", None) or getattr(request, "request_id", "0")
            payload_state = getattr(transfer_manager, "request_payload", None)
            if payload_state is None:
                payload_state = {}
                transfer_manager.request_payload = payload_state
            req_state = payload_state.get(req_id) or {}
            prev_tail = req_state.get("_latent_tail")
            if isinstance(prev_tail, torch.Tensor) and prev_tail.numel() > 0:
                latent_window = torch.cat([prev_tail, latent], dim=0).contiguous()
                left_context_size = int(prev_tail.shape[0])
            else:
                latent_window = latent
                left_context_size = 0
            keep = min(left_ctx_cfg, int(latent_window.shape[0]))
            req_state["_latent_tail"] = latent_window[-keep:].contiguous() if keep > 0 else None
            payload_state[req_id] = req_state
            latent = latent_window
        else:
            left_context_size = 0
        n_patches = int(latent.shape[0]) if isinstance(latent, torch.Tensor) and latent.ndim >= 1 else -1
        shape_str = tuple(latent.shape) if isinstance(latent, torch.Tensor) else ()
        logger.info(
            "[VoxCPM stream] Stage-0 latent chunk send "
            "(req=%s, current_patches=%s, window_patches=%s, shape=%s, "
            "left_ctx_cfg=%s, left_context_size=%s, has_more=%s, shm_finished=%s)",
            getattr(request, "external_req_id", None) or getattr(request, "request_id", "?"),
            current_chunk_patches,
            n_patches,
            shape_str,
            left_ctx_cfg,
            left_context_size,
            has_more,
            payload_finished,
        )
        return {
            "latent_audio_feat": latent,
            "sr": sr,
            "left_context_size": left_context_size,
            "finished": payload_finished,
        }

    latent = pooling_output.get("latent_audio_feat")
    sr = pooling_output.get("sr")
    if sr is not None and isinstance(sr, torch.Tensor):
        t = sr.detach().cpu()
        sr = int(t.item()) if t.numel() == 1 else int(t[0].item())
    elif sr is not None and hasattr(sr, "__len__") and len(sr) > 0:
        sr = int(sr[0]) if isinstance(sr[0], (int, float)) else int(sr[0].item())
    else:
        sr = 24000

    if latent is None:
        return [{"latent_audio_feat": None, "sr": sr, "finished": True}] if finished else None

    if isinstance(latent, torch.Tensor):
        latent = latent.detach().cpu().float()
    else:
        latent = torch.tensor(latent, dtype=torch.float32)

    chunk_size = chunk_size_cfg

    if latent.ndim == 3:
        t_len, _p, _d = latent.shape
        if chunk_size <= 0:
            chunk_size = DEFAULT_LATENT_CHUNK_PATCHES
        payloads: list[dict[str, Any]] = []
        for start in range(0, t_len, chunk_size):
            end = min(start + chunk_size, t_len)
            chunk = latent[start:end]
            payloads.append(
                {
                    "latent_audio_feat": chunk.contiguous(),
                    "sr": sr,
                    "left_context_size": 0,
                    "finished": finished and end >= t_len,
                }
            )
        if not payloads:
            payloads = [{"latent_audio_feat": latent[0:0], "sr": sr, "finished": True}]
        logger.info(
            "[VoxCPM stream] Stage-0 emitting %d latent chunks (T=%d, chunk_patches=%d)",
            len(payloads),
            t_len,
            chunk_size,
        )
        return payloads

    if latent.ndim == 2:
        _d, t_len = latent.shape
        if chunk_size <= 0:
            chunk_size = DEFAULT_LATENT_CHUNK_PATCHES
        payloads = []
        for start in range(0, t_len, chunk_size):
            end = min(start + chunk_size, t_len)
            chunk = latent[:, start:end].contiguous()
            payloads.append(
                {
                    "latent_audio_feat": chunk,
                    "sr": sr,
                    "left_context_size": 0,
                    "finished": finished and end >= t_len,
                }
            )
        if not payloads:
            payloads = [{"latent_audio_feat": latent[:, 0:0], "sr": sr, "finished": True}]
        logger.info(
            "[VoxCPM stream] Stage-0 emitting %d latent chunks (T=%d, chunk_patches=%d)",
            len(payloads),
            t_len,
            chunk_size,
        )
        return payloads

    return [{"latent_audio_feat": latent, "sr": sr, "left_context_size": 0, "finished": True}]
