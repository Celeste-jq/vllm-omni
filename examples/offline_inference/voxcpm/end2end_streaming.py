"""Offline VoxCPM with vLLM-Omni async_chunk: stream final-stage audio via AsyncOmni.

Requires stage config with ``async_chunk: true`` (default: ``voxcpm.yaml``).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

import soundfile as sf
import torch

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni import AsyncOmni, Omni

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STAGE_ASYNC = REPO_ROOT / "vllm_omni" / "model_executor" / "stage_configs" / "voxcpm.yaml"
DEFAULT_STAGE_SYNC = REPO_ROOT / "vllm_omni" / "model_executor" / "stage_configs" / "voxcpm_no_async_chunk.yaml"

logger = logging.getLogger(__name__)


def _build_prompt(args) -> dict[str, Any]:
    additional_information: dict[str, list[Any]] = {
        "text": [args.text],
        "cfg_value": [args.cfg_value],
        "inference_timesteps": [args.inference_timesteps],
        "min_len": [args.min_len],
        "max_new_tokens": [args.max_new_tokens],
    }
    if args.streaming_prefix_len is not None:
        additional_information["streaming_prefix_len"] = [args.streaming_prefix_len]
    if args.ref_audio:
        additional_information["ref_audio"] = [args.ref_audio]
    if args.ref_text:
        additional_information["ref_text"] = [args.ref_text]
    return {
        "prompt_token_ids": [1],
        "additional_information": additional_information,
    }


def _extract_audio_tensor(mm: dict[str, Any]) -> torch.Tensor:
    audio = mm.get("audio", mm.get("model_outputs"))
    if audio is None:
        raise ValueError("No audio output found in multimodal output.")
    if isinstance(audio, list):
        parts = [torch.as_tensor(a).float().cpu().reshape(-1) for a in audio]
        audio = torch.cat(parts, dim=-1) if parts else torch.zeros(0)
    if not isinstance(audio, torch.Tensor):
        audio = torch.as_tensor(audio)
    return audio.float().cpu().reshape(-1)


def _extract_sample_rate(mm: dict[str, Any]) -> int:
    sr_raw = mm.get("sr", 24000)
    if isinstance(sr_raw, list) and sr_raw:
        sr_raw = sr_raw[-1]
    if hasattr(sr_raw, "item"):
        return int(sr_raw.item())
    return int(sr_raw)


def parse_args():
    p = FlexibleArgumentParser(description="VoxCPM Omni async_chunk streaming (AsyncOmni) or one-shot fallback")
    p.add_argument("--model", type=str, default=os.environ.get("VOXCPM_MODEL"))
    p.add_argument("--text", type=str, default="Streaming VoxCPM test via vLLM Omni async_chunk.")
    p.add_argument("--ref-audio", type=str, default=None)
    p.add_argument("--ref-text", type=str, default=None)
    p.add_argument(
        "--stage-configs-path",
        type=str,
        default=str(DEFAULT_STAGE_ASYNC),
        help="Use voxcpm.yaml (async_chunk) for streaming; voxcpm_no_async_chunk.yaml for --sync.",
    )
    p.add_argument(
        "--sync",
        action="store_true",
        help="Use synchronous Omni (no async_chunk); overrides default config to voxcpm_no_async_chunk.yaml if path still default.",
    )
    p.add_argument("--cfg-value", type=float, default=2.0)
    p.add_argument("--inference-timesteps", type=int, default=10)
    p.add_argument("--min-len", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--streaming-prefix-len", type=int, default=None, help="VoxCPM streaming window (optional).")
    p.add_argument("--output-dir", type=str, default="output_audio_streaming")
    p.add_argument("--stage-init-timeout", type=int, default=600)
    p.add_argument("--log-stats", action="store_true")
    args = p.parse_args()
    if not args.model:
        p.error("--model is required unless $VOXCPM_MODEL is set")
    if (args.ref_audio is None) != (args.ref_text is None):
        p.error("--ref-audio and --ref-text must be provided together")
    if args.sync and args.stage_configs_path == str(DEFAULT_STAGE_ASYNC):
        args.stage_configs_path = str(DEFAULT_STAGE_SYNC)
    return args


async def _run_async_streaming(args) -> Path:
    prompt = _build_prompt(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    omni = AsyncOmni(
        model=args.model,
        stage_configs_path=args.stage_configs_path,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
    )

    request_id = f"stream_{uuid.uuid4().hex[:8]}"
    # VAE may return either growing cumulative waveforms or one non-overlapping segment per chunk
    # (after ``trim_streaming_patch`` in VoxCPM VAE decode). Support both.
    delta_chunks: list[torch.Tensor] = []
    sample_rate = 24000
    t0 = time.perf_counter()
    chunk_i = 0
    prev_total_samples = 0
    

    t_start = time.perf_counter()
    async for stage_output in omni.generate(prompt, request_id=request_id):
        ro = stage_output.request_output
        seq = ro.outputs[0] if hasattr(ro, "outputs") and ro.outputs else None
        if seq is None:
            continue
        mm = seq.multimodal_output
        if not isinstance(mm, dict):
            continue
        sample_rate = _extract_sample_rate(mm)
        try:
            w = _extract_audio_tensor(mm)
            n = int(w.numel())
            if n == 0:
                continue
            if n > prev_total_samples:
                delta = w.reshape(-1)[prev_total_samples:]
            else:
                delta = w.reshape(-1)
            delta_chunks.append(delta)
            if n > prev_total_samples:
                prev_total_samples = n
            else:
                prev_total_samples += int(delta.numel())
            logger.info(
                "chunk=%d delta_samples=%d buf_len=%d finished=%s elapsed_ms=%.1f",
                chunk_i,
                int(delta.numel()),
                n,
                stage_output.finished,
                (time.perf_counter() - t0) * 1000.0,
            )
            chunk_i += 1
        except ValueError:
            if not stage_output.finished:
                logger.debug("skip non-audio partial output chunk=%d", chunk_i)

    if not delta_chunks:
        raise RuntimeError("No audio chunks received; check stage config (async_chunk + connectors) and logs.")

    audio_cat = torch.cat([c.reshape(-1) for c in delta_chunks], dim=0)
    out_path = out_dir / f"output_{request_id}.wav"
    sf.write(str(out_path), audio_cat.numpy(), sample_rate, format="WAV")
    logger.info("Wrote %s (samples=%d sr=%d)", out_path, int(audio_cat.numel()), sample_rate)
    
    elapsed = time.perf_counter() - t_start
    print(f"Generation finished in {elapsed:.2f}s")


    return out_path


def _run_sync_oneshot(args) -> Path:
    prompt = _build_prompt(args)
    out_dir = Path(args.output_dir)
    omni = Omni(
        model=args.model,
        stage_configs_path=args.stage_configs_path,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
    )

    t_start = time.perf_counter()
    saved: list[Path] = []
    
    for stage_outputs in omni.generate([prompt]):
        for output in stage_outputs.request_output:
            mm = output.outputs[0].multimodal_output
            out_path = out_dir / f"output_{output.request_id}.wav"
            out_dir.mkdir(parents=True, exist_ok=True)
            audio = _extract_audio_tensor(mm).numpy()
            sf.write(str(out_path), audio, _extract_sample_rate(mm), format="WAV")
            saved.append(out_path)

    elapsed = time.perf_counter() - t_start
    print(f"Generation finished in {elapsed:.2f}s")

    if not saved:
        raise RuntimeError("No output from Omni.generate")
    return saved[0]


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    print(f"Model: {args.model}")
    print(f"Stage config: {args.stage_configs_path}")
    if args.sync:
        path = _run_sync_oneshot(args)
        print(f"Saved (sync): {path}")
    else:
        path = asyncio.run(_run_async_streaming(args))
        print(f"Saved (streaming): {path}")


if __name__ == "__main__":
    main()
