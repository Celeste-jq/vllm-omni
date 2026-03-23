# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
A/B benchmark for WAN fast layernorm on image-to-video.

This script runs image_to_video.py in two cases:
1) fast_on  : VLLM_OMNI_WAN_FAST_LAYERNORM_ENABLE=1
2) fast_off : VLLM_OMNI_WAN_FAST_LAYERNORM_ENABLE=0

It records wall time, parsed generation time, and key fast-layernorm logs.

Example:
  python examples/offline_inference/image_to_video/benchmark_wan_fast_layernorm_ab_i2v.py \
    --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
    --image /path/to/input.jpg \
    --prompt "A cinematic camera move around the subject." \
    --num-inference-steps 8 \
    --height 480 --width 832 --num-frames 17 \
    --repeats 3 \
    -- --tensor-parallel-size 1 --enforce-eager
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="A/B benchmark for WAN fast layernorm on image-to-video.")
    parser.add_argument("--python-executable", default=sys.executable, help="Python executable used to launch runs.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="WAN I2V model ID or local path.")
    parser.add_argument("--image", required=True, help="Input image path for image_to_video.py.")
    parser.add_argument("--prompt", default="", help="Generation prompt.")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--guidance-scale", type=float, default=5.0, help="CFG guidance scale.")
    parser.add_argument("--num-inference-steps", type=int, default=8, help="Sampling steps.")
    parser.add_argument("--height", type=int, default=480, help="Output height.")
    parser.add_argument("--width", type=int, default=832, help="Output width.")
    parser.add_argument("--num-frames", type=int, default=17, help="Number of frames.")
    parser.add_argument("--repeats", type=int, default=1, help="Runs per case.")
    parser.add_argument("--impl-mode-on", type=parse_mode, default=0, help="impl_mode for fast_on case.")
    parser.add_argument(
        "--impl-mode-off",
        type=parse_mode,
        default=0,
        help="impl_mode for fast_off case (kept for traceability; fast path is disabled).",
    )
    parser.add_argument("--output-dir", default="outputs_wan_fast_layernorm_ab_i2v", help="Output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue benchmarking remaining runs when one run fails.",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra args passed to image_to_video.py. Use '--' separator before extra args.",
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
        "--output",
        str(output_path),
    ]

    extra_args = args.extra_args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    cmd.extend(extra_args)
    return cmd


def extract_generation_time(output_text: str) -> float | None:
    match = GEN_TIME_PATTERN.search(output_text)
    if not match:
        return None
    return float(match.group(1))


def run_once(
    case_name: str,
    enable: int,
    impl_mode: int,
    run_idx: int,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict:
    output_path = output_dir / f"{case_name}_run{run_idx:02d}.mp4"
    cmd = build_cmd(args, output_path)
    env = os.environ.copy()
    env[ENV_ENABLE] = str(enable)
    env[ENV_IMPL_MODE] = str(impl_mode)

    cmd_str = " ".join(shlex.quote(x) for x in cmd)
    print(f"\n[{case_name} run={run_idx}] {ENV_ENABLE}={enable} {ENV_IMPL_MODE}={impl_mode} {cmd_str}")

    start = time.perf_counter()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    wall_seconds = time.perf_counter() - start

    merged_output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    generation_time = extract_generation_time(merged_output)
    fast_hit = FAST_HIT_PATTERN in merged_output
    fast_off_hit = FAST_OFF_PATTERN in merged_output

    log_path = output_dir / f"{case_name}_run{run_idx:02d}.log"
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

    status = "ok" if proc.returncode == 0 else "fail"
    gen_str = "N/A" if generation_time is None else f"{generation_time:.3f}"
    print(
        f"[{case_name} run={run_idx}] {status}, wall={wall_seconds:.3f}s, "
        f"gen_time={gen_str}, fast_hit={fast_hit}, fast_off_hit={fast_off_hit}"
    )

    return {
        "case_name": case_name,
        "run_idx": run_idx,
        "returncode": proc.returncode,
        "wall_seconds": wall_seconds,
        "generation_seconds": generation_time,
        "fast_hit": fast_hit,
        "fast_off_hit": fast_off_hit,
        "output_path": str(output_path),
        "log_path": str(log_path),
    }


def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def summarize_case(case_name: str, rows: list[dict]) -> dict:
    success_rows = [r for r in rows if r["returncode"] == 0]
    wall_values = [r["wall_seconds"] for r in success_rows]
    gen_values = [r["generation_seconds"] for r in success_rows if r["generation_seconds"] is not None]
    return {
        "case_name": case_name,
        "runs": len(rows),
        "success": len(success_rows),
        "avg_wall_seconds": _avg(wall_values),
        "avg_generation_seconds": _avg(gen_values),
        "fast_hit_runs": sum(1 for r in rows if r["fast_hit"]),
        "fast_off_hit_runs": sum(1 for r in rows if r["fast_off_hit"]),
    }


def _fmt_float(v: float | None) -> str:
    return "N/A" if v is None else f"{v:.3f}"


def print_summary(rows: list[dict]) -> None:
    on_rows = [r for r in rows if r["case_name"] == "fast_on"]
    off_rows = [r for r in rows if r["case_name"] == "fast_off"]
    on = summarize_case("fast_on", on_rows)
    off = summarize_case("fast_off", off_rows)

    print("\n=== A/B Summary (I2V) ===")
    print("case     | success/runs | avg_wall_s | avg_gen_s | fast_hit_runs | fast_off_hit_runs")
    print(
        "fast_on  | "
        f"{on['success']}/{on['runs']} | {_fmt_float(on['avg_wall_seconds'])} | "
        f"{_fmt_float(on['avg_generation_seconds'])} | {on['fast_hit_runs']} | {on['fast_off_hit_runs']}"
    )
    print(
        "fast_off | "
        f"{off['success']}/{off['runs']} | {_fmt_float(off['avg_wall_seconds'])} | "
        f"{_fmt_float(off['avg_generation_seconds'])} | {off['fast_hit_runs']} | {off['fast_off_hit_runs']}"
    )

    if on["avg_generation_seconds"] is not None and off["avg_generation_seconds"] is not None:
        speedup = (off["avg_generation_seconds"] - on["avg_generation_seconds"]) / off["avg_generation_seconds"] * 100.0
        print(f"\nEstimated speedup (gen_time): {speedup:.2f}% (positive means fast_on is faster)")
    else:
        print("\nEstimated speedup (gen_time): N/A (missing generation time in logs)")

    print("\nLog evidence check:")
    print(f"- fast_on should hit '{FAST_HIT_PATTERN}'")
    print(f"- fast_off should hit '{FAST_OFF_PATTERN}'")


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        ("fast_on", 1, args.impl_mode_on),
        ("fast_off", 0, args.impl_mode_off),
    ]

    if args.dry_run:
        for case_name, enable, impl_mode in cases:
            for run_idx in range(1, args.repeats + 1):
                output_path = output_dir / f"{case_name}_run{run_idx:02d}.mp4"
                cmd = build_cmd(args, output_path)
                cmd_str = " ".join(shlex.quote(x) for x in cmd)
                print(f"{ENV_ENABLE}={enable} {ENV_IMPL_MODE}={impl_mode} {cmd_str}")
        return 0

    rows: list[dict] = []
    for case_name, enable, impl_mode in cases:
        for run_idx in range(1, args.repeats + 1):
            row = run_once(case_name, enable, impl_mode, run_idx, args, output_dir)
            rows.append(row)
            if row["returncode"] != 0 and not args.continue_on_error:
                print_summary(rows)
                return row["returncode"]

    print_summary(rows)
    return 0 if all(item["returncode"] == 0 for item in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
