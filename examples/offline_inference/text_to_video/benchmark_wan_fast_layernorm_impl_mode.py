# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark WAN fast-layernorm impl_mode (0/1/2) with the existing text_to_video example.

This script runs:
  examples/offline_inference/text_to_video/text_to_video.py
multiple times with different environment variable settings:
  VLLM_OMNI_WAN_FAST_LAYERNORM_IMPL_MODE=<mode>

Usage example:
  python examples/offline_inference/text_to_video/benchmark_wan_fast_layernorm_impl_mode.py \
    --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --num-inference-steps 8 \
    --height 480 --width 832 --num-frames 17 \
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

DEFAULT_MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
ENV_IMPL_MODE = "VLLM_OMNI_WAN_FAST_LAYERNORM_IMPL_MODE"
GEN_TIME_PATTERN = re.compile(r"Total generation time:\s*([0-9.]+)\s*seconds")


def parse_modes(modes: str) -> list[int]:
    parsed: list[int] = []
    for item in modes.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            mode = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid mode value: {item!r}") from exc
        if mode not in (0, 1, 2):
            raise argparse.ArgumentTypeError(f"impl_mode must be 0/1/2, got {mode}")
        parsed.append(mode)
    if not parsed:
        raise argparse.ArgumentTypeError("No valid impl_mode found in --modes")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark WAN fast-layernorm impl_mode.")
    parser.add_argument("--python-executable", default=sys.executable, help="Python executable used to launch runs.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="WAN model ID or local path.")
    parser.add_argument("--prompt", default="A panda surfing a giant wave at sunset.", help="Generation prompt.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--guidance-scale", type=float, default=4.0, help="CFG guidance scale.")
    parser.add_argument("--num-inference-steps", type=int, default=8, help="Sampling steps.")
    parser.add_argument("--height", type=int, default=480, help="Output height.")
    parser.add_argument("--width", type=int, default=832, help="Output width.")
    parser.add_argument("--num-frames", type=int, default=17, help="Number of frames.")
    parser.add_argument("--modes", type=parse_modes, default=[0, 1, 2], help="Comma-separated impl_mode list.")
    parser.add_argument("--output-dir", default="outputs_wan_fast_layernorm_bench", help="Output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue benchmarking remaining modes when one mode fails.",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra args passed to text_to_video.py. Use '--' separator before extra args.",
    )
    return parser.parse_args()


def build_cmd(args: argparse.Namespace, output_path: Path) -> list[str]:
    script_path = Path(__file__).with_name("text_to_video.py")
    cmd = [
        args.python_executable,
        str(script_path),
        "--model",
        args.model,
        "--prompt",
        args.prompt,
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


def extract_generation_time(stdout: str) -> float | None:
    match = GEN_TIME_PATTERN.search(stdout)
    if not match:
        return None
    return float(match.group(1))


def run_single_mode(mode: int, args: argparse.Namespace, output_dir: Path) -> dict:
    output_path = output_dir / f"wan_impl_mode_{mode}.mp4"
    cmd = build_cmd(args, output_path)
    env = os.environ.copy()
    env[ENV_IMPL_MODE] = str(mode)

    print(f"\n[mode={mode}] {' '.join(shlex.quote(c) for c in cmd)}")
    start = time.perf_counter()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    wall_seconds = time.perf_counter() - start

    log_path = output_dir / f"run_mode_{mode}.log"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"{ENV_IMPL_MODE}={mode}\n")
        f.write(f"cmd={' '.join(shlex.quote(c) for c in cmd)}\n")
        f.write(f"returncode={proc.returncode}\n")
        f.write("\n===== STDOUT =====\n")
        f.write(proc.stdout or "")
        f.write("\n===== STDERR =====\n")
        f.write(proc.stderr or "")

    generation_time = extract_generation_time(proc.stdout or "")
    if proc.returncode == 0:
        print(
            f"[mode={mode}] success, wall={wall_seconds:.3f}s, "
            f"gen_time={generation_time if generation_time is not None else 'N/A'}"
        )
    else:
        print(f"[mode={mode}] failed, returncode={proc.returncode}, see {log_path}")
        if proc.stderr:
            print(proc.stderr.strip())

    return {
        "mode": mode,
        "returncode": proc.returncode,
        "wall_seconds": wall_seconds,
        "generation_seconds": generation_time,
        "output_path": str(output_path),
        "log_path": str(log_path),
    }


def print_summary(results: list[dict]) -> None:
    print("\n=== Summary ===")
    print("mode | status | wall_seconds | generation_seconds | output")
    for item in results:
        status = "ok" if item["returncode"] == 0 else "fail"
        wall = f"{item['wall_seconds']:.3f}"
        gen = "N/A" if item["generation_seconds"] is None else f"{item['generation_seconds']:.3f}"
        print(f"{item['mode']} | {status} | {wall} | {gen} | {item['output_path']}")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).with_name("text_to_video.py")
    if not script_path.exists():
        print(f"Cannot find script: {script_path}")
        return 2

    if args.dry_run:
        for mode in args.modes:
            output_path = output_dir / f"wan_impl_mode_{mode}.mp4"
            cmd = build_cmd(args, output_path)
            print(f"{ENV_IMPL_MODE}={mode} {' '.join(shlex.quote(c) for c in cmd)}")
        return 0

    results: list[dict] = []
    for mode in args.modes:
        result = run_single_mode(mode, args, output_dir)
        results.append(result)
        if result["returncode"] != 0 and not args.continue_on_error:
            print_summary(results)
            return result["returncode"]

    print_summary(results)
    return 0 if all(item["returncode"] == 0 for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
