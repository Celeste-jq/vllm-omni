# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Evaluate WAN fast_layernorm effect on image-to-video.

This script runs image_to_video.py twice with the same inputs:
1) fast_on  : VLLM_OMNI_WAN_FAST_LAYERNORM_ENABLE=1
2) fast_off : VLLM_OMNI_WAN_FAST_LAYERNORM_ENABLE=0

Then it reports:
- Runtime metrics (wall time and parsed "Total generation time")
- Log evidence for fast layernorm hit/disable
- Output video difference metrics (frame-level MAE / RMSE / PSNR)

Example:
  python examples/offline_inference/image_to_video/evaluate_wan_fast_layernorm_effect_i2v.py \
    --model /path/to/Wan2.2-I2V-A14B-Diffusers \
    --image /path/to/input.jpg \
    --prompt "A cat playing with yarn." \
    --num-inference-steps 8 \
    --height 480 --width 832 --num-frames 17 \
    --tensor-parallel-size 4
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
ENV_ENABLE = "VLLM_OMNI_WAN_FAST_LAYERNORM_ENABLE"
ENV_IMPL_MODE = "VLLM_OMNI_WAN_FAST_LAYERNORM_IMPL_MODE"
GEN_TIME_PATTERN = re.compile(r"Total generation time:\s*([0-9.]+)\s*seconds")
FAST_HIT_PATTERN = "WAN fast layernorm enabled via mindiesd.fast_layernorm"
FAST_OFF_PATTERN = "WAN fast layernorm disabled by"


def parse_mode(value: str) -> int:
    try:
        mode = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid impl_mode value: {value!r}") from exc
    if mode not in (0, 1, 2):
        raise argparse.ArgumentTypeError(f"impl_mode must be 0/1/2, got {mode}")
    return mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate WAN fast_layernorm effect for image-to-video.")
    parser.add_argument("--python-executable", default=sys.executable, help="Python executable to run image_to_video.py.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="WAN I2V model ID or local path.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--prompt", default="", help="Prompt.")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--guidance-scale", type=float, default=5.0, help="CFG guidance scale.")
    parser.add_argument("--num-inference-steps", type=int, default=8, help="Sampling steps.")
    parser.add_argument("--height", type=int, default=480, help="Output height.")
    parser.add_argument("--width", type=int, default=832, help="Output width.")
    parser.add_argument("--num-frames", type=int, default=17, help="Number of frames.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="TP size passed to image_to_video.py.")
    parser.add_argument("--impl-mode-on", type=parse_mode, default=0, help="impl_mode when fast_on.")
    parser.add_argument("--impl-mode-off", type=parse_mode, default=0, help="impl_mode when fast_off.")
    parser.add_argument(
        "--output-dir",
        default="outputs_wan_fast_layernorm_effect_i2v",
        help="Output directory for videos and logs.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra args for image_to_video.py. Use '--' separator before extra args.",
    )
    return parser.parse_args()


def build_cmd(args: argparse.Namespace, output_path: Path) -> list[str]:
    script_path = Path(__file__).with_name("image_to_video.py")
    cmd = [
        args.python_executable,
        str(script_path),
        "--model",
        args.model,
        "--image",
        args.image,
        "--prompt",
        args.prompt,
        "--negative-prompt",
        args.negative_prompt,
        "--seed",
        str(args.seed),
        "--guidance-scale",
        str(args.guidance_scale),
        "--num-inference-steps",
        str(args.num_inference_steps),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--num-frames",
        str(args.num_frames),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--output",
        str(output_path),
    ]
    extra_args = args.extra_args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    cmd.extend(extra_args)
    return cmd


def extract_generation_time(text: str) -> float | None:
    match = GEN_TIME_PATTERN.search(text)
    if not match:
        return None
    return float(match.group(1))


def run_case(case_name: str, enable: int, impl_mode: int, args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    output_path = output_dir / f"{case_name}.mp4"
    cmd = build_cmd(args, output_path)
    env = os.environ.copy()
    env[ENV_ENABLE] = str(enable)
    env[ENV_IMPL_MODE] = str(impl_mode)

    cmd_str = " ".join(shlex.quote(x) for x in cmd)
    print(f"\n[{case_name}] {ENV_ENABLE}={enable} {ENV_IMPL_MODE}={impl_mode} {cmd_str}")
    start = time.perf_counter()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    wall_seconds = time.perf_counter() - start

    merged = (proc.stdout or "") + "\n" + (proc.stderr or "")
    generation_seconds = extract_generation_time(merged)
    fast_hit = FAST_HIT_PATTERN in merged
    fast_off_hit = FAST_OFF_PATTERN in merged

    log_path = output_dir / f"{case_name}.log"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"{ENV_ENABLE}={enable}\n")
        f.write(f"{ENV_IMPL_MODE}={impl_mode}\n")
        f.write(f"cmd={cmd_str}\n")
        f.write(f"returncode={proc.returncode}\n")
        f.write(f"fast_hit={fast_hit}\n")
        f.write(f"fast_off_hit={fast_off_hit}\n")
        f.write("\n===== STDOUT =====\n")
        f.write(proc.stdout or "")
        f.write("\n===== STDERR =====\n")
        f.write(proc.stderr or "")

    print(
        f"[{case_name}] returncode={proc.returncode}, wall={wall_seconds:.3f}s, "
        f"gen_time={'N/A' if generation_seconds is None else f'{generation_seconds:.3f}'}, "
        f"fast_hit={fast_hit}, fast_off_hit={fast_off_hit}"
    )
    return {
        "case_name": case_name,
        "returncode": proc.returncode,
        "output_path": output_path,
        "log_path": log_path,
        "wall_seconds": wall_seconds,
        "generation_seconds": generation_seconds,
        "fast_hit": fast_hit,
        "fast_off_hit": fast_off_hit,
    }


def _load_video_frames(video_path: Path) -> np.ndarray:
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise RuntimeError("imageio is required for video diff metrics. Please install imageio[ffmpeg].") from exc
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required for video diff metrics. Please install numpy.") from exc

    frames = iio.imread(str(video_path), index=None)
    if frames.ndim != 4:
        raise RuntimeError(f"Unexpected video array shape {frames.shape} from {video_path}")
    return frames


def compare_videos(video_on: Path, video_off: Path) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required for video diff metrics. Please install numpy.") from exc

    frames_on = _load_video_frames(video_on).astype(np.float32)
    frames_off = _load_video_frames(video_off).astype(np.float32)

    min_frames = min(frames_on.shape[0], frames_off.shape[0])
    if min_frames == 0:
        raise RuntimeError("At least one video has zero frames.")

    if frames_on.shape[0] != frames_off.shape[0]:
        print(
            f"[WARN] frame count differs: fast_on={frames_on.shape[0]}, fast_off={frames_off.shape[0]}; "
            f"comparing first {min_frames} frames."
        )

    frames_on = frames_on[:min_frames]
    frames_off = frames_off[:min_frames]

    diff = frames_on - frames_off
    abs_diff = np.abs(diff)
    sq_diff = diff * diff

    mae = float(abs_diff.mean() / 255.0)
    rmse = float(np.sqrt(sq_diff.mean()) / 255.0)
    mse_255 = float(sq_diff.mean())
    psnr = float("inf") if mse_255 == 0.0 else float(10.0 * math.log10((255.0 * 255.0) / mse_255))

    per_frame_mae = abs_diff.mean(axis=(1, 2, 3)) / 255.0
    per_frame_psnr = []
    for i in range(min_frames):
        mse_i = float((diff[i] * diff[i]).mean())
        psnr_i = float("inf") if mse_i == 0.0 else float(10.0 * math.log10((255.0 * 255.0) / mse_i))
        per_frame_psnr.append(psnr_i)

    return {
        "frame_count_compared": min_frames,
        "resolution": tuple(int(x) for x in frames_on.shape[1:3]),
        "mae_norm_0_1": mae,
        "rmse_norm_0_1": rmse,
        "psnr_db": psnr,
        "max_frame_mae_norm_0_1": float(np.max(per_frame_mae)),
        "min_frame_psnr_db": float(np.min(per_frame_psnr)),
    }


def _fmt(v: float | None) -> str:
    return "N/A" if v is None else f"{v:.3f}"


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    on_output = output_dir / "fast_on.mp4"
    off_output = output_dir / "fast_off.mp4"
    on_cmd = build_cmd(args, on_output)
    off_cmd = build_cmd(args, off_output)

    if args.dry_run:
        print(
            f"{ENV_ENABLE}=1 {ENV_IMPL_MODE}={args.impl_mode_on} "
            f"{' '.join(shlex.quote(x) for x in on_cmd)}"
        )
        print(
            f"{ENV_ENABLE}=0 {ENV_IMPL_MODE}={args.impl_mode_off} "
            f"{' '.join(shlex.quote(x) for x in off_cmd)}"
        )
        return 0

    on_result = run_case("fast_on", enable=1, impl_mode=args.impl_mode_on, args=args, output_dir=output_dir)
    off_result = run_case("fast_off", enable=0, impl_mode=args.impl_mode_off, args=args, output_dir=output_dir)

    print("\n=== Runtime Summary ===")
    print("case     | returncode | wall_s | gen_s | fast_hit | fast_off_hit")
    for item in [on_result, off_result]:
        print(
            f"{item['case_name']:<8} | {item['returncode']:<10} | {item['wall_seconds']:.3f} | "
            f"{_fmt(item['generation_seconds'])} | {item['fast_hit']} | {item['fast_off_hit']}"
        )

    if on_result["generation_seconds"] is not None and off_result["generation_seconds"] is not None:
        denom = off_result["generation_seconds"]
        if denom > 0:
            speedup = (denom - on_result["generation_seconds"]) / denom * 100.0
            print(f"speedup_gen_time = {speedup:.2f}% (positive => fast_on faster)")

    if on_result["returncode"] != 0 or off_result["returncode"] != 0:
        print("\nAt least one case failed. Skip video quality diff. Check logs:")
        print(f"- fast_on : {on_result['log_path']}")
        print(f"- fast_off: {off_result['log_path']}")
        return 1

    try:
        diff_metrics = compare_videos(on_result["output_path"], off_result["output_path"])
    except Exception as e:
        print(f"\nVideo diff metrics unavailable: {e}")
        print("Install imageio + ffmpeg to enable frame-level metrics.")
        return 0

    print("\n=== Video Difference Metrics (fast_on vs fast_off) ===")
    print(f"frames_compared: {diff_metrics['frame_count_compared']}")
    print(f"resolution: {diff_metrics['resolution']}")
    print(f"MAE (0-1): {diff_metrics['mae_norm_0_1']:.6f}")
    print(f"RMSE (0-1): {diff_metrics['rmse_norm_0_1']:.6f}")
    print(f"PSNR (dB): {diff_metrics['psnr_db']:.3f}")
    print(f"Max frame MAE (0-1): {diff_metrics['max_frame_mae_norm_0_1']:.6f}")
    print(f"Min frame PSNR (dB): {diff_metrics['min_frame_psnr_db']:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
