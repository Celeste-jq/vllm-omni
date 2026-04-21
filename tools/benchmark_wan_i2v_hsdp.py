#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline vs HSDP Wan2.2-I2V profiling experiments and collect comparable metrics."
    )
    parser.add_argument("--model", required=True, help="Wan2.2 I2V model path or ID.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--prompt", default="", help="Text prompt.")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--guidance-scale", type=float, default=5.0, help="CFG scale.")
    parser.add_argument("--guidance-scale-high", type=float, default=None, help="Optional separate CFG scale.")
    parser.add_argument("--height", type=int, required=True, help="Output height.")
    parser.add_argument("--width", type=int, required=True, help="Output width.")
    parser.add_argument("--num-frames", type=int, default=81, help="Number of frames.")
    parser.add_argument("--num-inference-steps", type=int, default=40, help="Number of denoising steps.")
    parser.add_argument("--boundary-ratio", type=float, default=0.875, help="MoE boundary ratio.")
    parser.add_argument("--frame-rate", type=float, default=None, help="Optional model frame rate.")
    parser.add_argument("--flow-shift", type=float, default=5.0, help="Scheduler flow_shift.")
    parser.add_argument(
        "--sample-solver",
        type=str,
        default="unipc",
        choices=["unipc", "euler"],
        help="Sampling solver.",
    )
    parser.add_argument("--fps", type=int, default=16, help="Output fps.")
    parser.add_argument("--nproc-per-node", type=int, default=4, help="torchrun worker count.")
    parser.add_argument("--ulysses-degree", type=int, default=1, help="Ulysses sequence parallel degree.")
    parser.add_argument("--ring-degree", type=int, default=1, help="Ring sequence parallel degree.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--cfg-parallel-size", type=int, default=1, choices=[1, 2], help="CFG parallel size.")
    parser.add_argument("--vae-patch-parallel-size", type=int, default=1, help="VAE patch parallel size.")
    parser.add_argument("--hsdp-shard-size", type=int, default=-1, help="HSDP shard size for the HSDP scenario.")
    parser.add_argument("--hsdp-replicate-size", type=int, default=1, help="HSDP replicate size for the HSDP scenario.")
    parser.add_argument("--enable-cpu-offload", action="store_true", help="Enable CPU offload.")
    parser.add_argument("--enable-layerwise-offload", action="store_true", help="Enable layerwise offload.")
    parser.add_argument("--vae-use-slicing", action="store_true", help="Enable VAE slicing.")
    parser.add_argument("--vae-use-tiling", action="store_true", help="Enable VAE tiling.")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile.")
    parser.add_argument(
        "--cache-backend",
        type=str,
        default="none",
        choices=["none", "cache_dit", "tea_cache"],
        help="Optional cache acceleration backend.",
    )
    parser.add_argument(
        "--enable-diffusion-pipeline-profiler",
        action="store_true",
        help="Enable the coarse diffusion pipeline profiler inside profile_wan_i2v.py.",
    )
    parser.add_argument("--warmup-runs", type=int, default=1, help="Warmup runs before the profiled run.")
    parser.add_argument("--profile-memory", action="store_true", help="Enable profile_memory in torch_npu profiler.")
    parser.add_argument("--with-stack", action="store_true", help="Enable with_modules-like metadata in profiler.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_runs/wan_i2v_hsdp",
        help="Directory for scenario artifacts and comparison outputs.",
    )
    return parser.parse_args()


def _append_option(command: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    command.extend([flag, str(value)])


def _append_flag(command: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        command.append(flag)


def build_profile_command(
    scenario_name: str,
    nproc_per_node: int,
    common_args: dict[str, Any],
    scenario_args: dict[str, Any],
    output_dir: str,
) -> list[str]:
    command = [
        "torchrun",
        f"--nproc_per_node={nproc_per_node}",
        "tools/profile_wan_i2v.py",
        "--model",
        str(common_args["model"]),
        "--image",
        str(common_args["image"]),
        "--prompt",
        str(common_args["prompt"]),
        "--height",
        str(common_args["height"]),
        "--width",
        str(common_args["width"]),
        "--num-frames",
        str(common_args["num_frames"]),
        "--num-inference-steps",
        str(common_args["num_inference_steps"]),
        "--ulysses-degree",
        str(common_args["ulysses_degree"]),
        "--ring-degree",
        str(common_args["ring_degree"]),
        "--tensor-parallel-size",
        str(common_args["tensor_parallel_size"]),
        "--cfg-parallel-size",
        str(common_args["cfg_parallel_size"]),
        "--vae-patch-parallel-size",
        str(common_args["vae_patch_parallel_size"]),
        "--guidance-scale",
        str(common_args.get("guidance_scale", 5.0)),
        "--boundary-ratio",
        str(common_args.get("boundary_ratio", 0.875)),
        "--flow-shift",
        str(common_args.get("flow_shift", 5.0)),
        "--sample-solver",
        str(common_args.get("sample_solver", "unipc")),
        "--fps",
        str(common_args.get("fps", 16)),
        "--warmup-runs",
        str(common_args.get("warmup_runs", 1)),
        "--profiling-output-dir",
        output_dir,
        "--output",
        str(Path(output_dir).parent / f"{scenario_name}.mp4"),
    ]

    _append_option(command, "--negative-prompt", common_args.get("negative_prompt"))
    _append_option(command, "--seed", common_args.get("seed"))
    _append_option(command, "--guidance-scale-high", common_args.get("guidance_scale_high"))
    _append_option(command, "--frame-rate", common_args.get("frame_rate"))
    _append_option(command, "--cache-backend", common_args.get("cache_backend"))
    _append_flag(command, "--enable-cpu-offload", bool(common_args.get("enable_cpu_offload")))
    _append_flag(command, "--enable-layerwise-offload", bool(common_args.get("enable_layerwise_offload")))
    _append_flag(command, "--vae-use-slicing", bool(common_args.get("vae_use_slicing")))
    _append_flag(command, "--vae-use-tiling", bool(common_args.get("vae_use_tiling")))
    _append_flag(command, "--enforce-eager", bool(common_args.get("enforce_eager")))
    _append_flag(
        command,
        "--enable-diffusion-pipeline-profiler",
        bool(common_args.get("enable_diffusion_pipeline_profiler")),
    )
    _append_flag(command, "--profile-memory", bool(common_args.get("profile_memory")))
    _append_flag(command, "--with-stack", bool(common_args.get("with_stack")))

    if scenario_args.get("use_hsdp"):
        command.append("--use-hsdp")
        if scenario_args.get("hsdp_shard_size") is not None:
            _append_option(command, "--hsdp-shard-size", scenario_args.get("hsdp_shard_size"))
        if scenario_args.get("hsdp_replicate_size") is not None:
            _append_option(command, "--hsdp-replicate-size", scenario_args.get("hsdp_replicate_size"))

    return command


def classify_failure(returncode: int, stdout: str, stderr: str) -> dict[str, Any]:
    joined = f"{stdout}\n{stderr}".lower()
    oom_patterns = [
        "out of memory",
        "oom",
        "npu out of memory",
        "cuda out of memory",
    ]
    is_oom = returncode != 0 and any(pattern in joined for pattern in oom_patterns)
    return {
        "status": "oom" if is_oom else ("ok" if returncode == 0 else "failed"),
        "oom": is_oom,
    }


def aggregate_run_metadata_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    wall_times = [payload.get("wall_time_sec") for payload in payloads if payload.get("wall_time_sec") is not None]
    peak_allocated = [
        payload.get("memory", {}).get("peak_allocated_bytes")
        for payload in payloads
        if payload.get("memory", {}).get("peak_allocated_bytes") is not None
    ]
    peak_reserved = [
        payload.get("memory", {}).get("peak_reserved_bytes")
        for payload in payloads
        if payload.get("memory", {}).get("peak_reserved_bytes") is not None
    ]

    return {
        "world_size": len(payloads),
        "max_wall_time_sec": max(wall_times) if wall_times else None,
        "avg_wall_time_sec": sum(wall_times) / len(wall_times) if wall_times else None,
        "memory": {
            "max_peak_allocated_bytes": max(peak_allocated) if peak_allocated else None,
            "max_peak_reserved_bytes": max(peak_reserved) if peak_reserved else None,
        },
        "ranks": payloads,
    }


def _aggregate_summary_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    phase_totals: dict[str, dict[str, float]] = {}
    total_profiled_sec = 0.0
    for payload in payloads:
        total_profiled_sec += float(payload.get("total_profiled_sec", 0.0))
        for phase_name, phase_data in payload.get("phases", {}).items():
            aggregate = phase_totals.setdefault(phase_name, {"count": 0.0, "total_sec": 0.0})
            aggregate["count"] += float(phase_data.get("count", 0))
            aggregate["total_sec"] += float(phase_data.get("total_sec", 0.0))

    phases: dict[str, dict[str, float]] = {}
    for phase_name, aggregate in phase_totals.items():
        total_sec = aggregate["total_sec"]
        count = int(aggregate["count"])
        phases[phase_name] = {
            "count": count,
            "total_sec": total_sec,
            "avg_sec": (total_sec / count) if count else 0.0,
            "pct": (total_sec / total_profiled_sec * 100.0) if total_profiled_sec else 0.0,
        }
    return {
        "total_profiled_sec": total_profiled_sec,
        "phases": phases,
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _load_or_aggregate_run_metadata(scenario_dir: Path) -> dict[str, Any] | None:
    aggregate_path = scenario_dir / "aggregate_run_metadata.json"
    payload = _load_json(aggregate_path)
    if payload is not None:
        return payload

    rank_payloads: list[dict[str, Any]] = []
    for metadata_path in sorted(scenario_dir.glob("rank*/run_metadata.json")):
        rank_payload = _load_json(metadata_path)
        if rank_payload is not None:
            rank_payloads.append(rank_payload)
    if not rank_payloads:
        return None
    return aggregate_run_metadata_payloads(rank_payloads)


def _load_or_aggregate_summary(scenario_dir: Path) -> dict[str, Any] | None:
    aggregate_path = scenario_dir / "aggregate_summary.json"
    payload = _load_json(aggregate_path)
    if payload is not None:
        return payload

    rank_payloads: list[dict[str, Any]] = []
    for summary_path in sorted(scenario_dir.glob("rank*/summary.json")):
        summary_payload = _load_json(summary_path)
        if summary_payload is not None:
            rank_payloads.append(summary_payload)
    if not rank_payloads:
        return None
    return _aggregate_summary_payloads(rank_payloads)


def load_scenario_metrics(scenario_dir: Path) -> dict[str, Any]:
    summary_payload = _load_or_aggregate_summary(scenario_dir)
    metadata_payload = _load_or_aggregate_run_metadata(scenario_dir)

    metrics: dict[str, Any] = {
        "wall_time_sec": metadata_payload.get("max_wall_time_sec") if metadata_payload else None,
        "total_profiled_sec": summary_payload.get("total_profiled_sec") if summary_payload else None,
        "max_peak_allocated_bytes": (
            metadata_payload.get("memory", {}).get("max_peak_allocated_bytes") if metadata_payload else None
        ),
        "max_peak_reserved_bytes": (
            metadata_payload.get("memory", {}).get("max_peak_reserved_bytes") if metadata_payload else None
        ),
    }

    if summary_payload is not None:
        for phase_name, phase_data in summary_payload.get("phases", {}).items():
            metrics[f"{phase_name}_pct"] = phase_data.get("pct")

    return metrics


def pct_delta(baseline: float | int | None, candidate: float | int | None, inverse: bool = False) -> float | None:
    if baseline in (None, 0) or candidate is None:
        return None
    baseline_value = float(baseline)
    candidate_value = float(candidate)
    if inverse:
        return round((baseline_value - candidate_value) / baseline_value * 100.0, 2)
    return round((candidate_value - baseline_value) / baseline_value * 100.0, 2)


def render_markdown_summary(comparison: dict[str, Any]) -> str:
    baseline = comparison.get("baseline", {})
    hsdp = comparison.get("hsdp", {})
    deltas = comparison.get("deltas", {})

    return "\n".join(
        [
            "# Wan2.2 I2V HSDP Comparison",
            "",
            "## Scenario Status",
            "",
            f"- baseline: `{baseline.get('status')}`",
            f"- hsdp: `{hsdp.get('status')}`",
            "",
            "## Metrics",
            "",
            "| field | baseline | hsdp | delta |",
            "| --- | ---: | ---: | ---: |",
            f"| `wall_time_sec` | {baseline.get('metrics', {}).get('wall_time_sec')} | {hsdp.get('metrics', {}).get('wall_time_sec')} | {deltas.get('wall_time_pct')} |",
            f"| `max_peak_allocated_bytes` | {baseline.get('metrics', {}).get('max_peak_allocated_bytes')} | {hsdp.get('metrics', {}).get('max_peak_allocated_bytes')} | {deltas.get('peak_allocated_pct')} |",
            f"| `max_peak_reserved_bytes` | {baseline.get('metrics', {}).get('max_peak_reserved_bytes')} | {hsdp.get('metrics', {}).get('max_peak_reserved_bytes')} | {deltas.get('peak_reserved_pct')} |",
            "",
            "## Deltas",
            "",
            f"- `wall_time_pct`: {deltas.get('wall_time_pct')}",
            f"- `peak_allocated_pct`: {deltas.get('peak_allocated_pct')}",
            f"- `peak_reserved_pct`: {deltas.get('peak_reserved_pct')}",
            "",
        ]
    )


def write_rows_to_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _scenario_metrics_row(name: str, result: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "scenario": name,
        "status": result.get("status"),
        "oom": result.get("oom"),
        "returncode": result.get("returncode"),
    }
    for key, value in result.get("metrics", {}).items():
        row[key] = value
    return row


def _phase_rows(name: str, scenario_dir: Path) -> list[dict[str, Any]]:
    summary_payload = _load_or_aggregate_summary(scenario_dir)
    if summary_payload is None:
        return []
    rows: list[dict[str, Any]] = []
    for phase_name, phase_data in sorted(summary_payload.get("phases", {}).items()):
        rows.append(
            {
                "scenario": name,
                "phase": phase_name,
                "pct": phase_data.get("pct"),
                "total_sec": phase_data.get("total_sec"),
                "count": phase_data.get("count"),
            }
        )
    return rows


def run_scenario(
    scenario_name: str,
    nproc_per_node: int,
    common_args: dict[str, Any],
    scenario_args: dict[str, Any],
    root_dir: Path,
) -> dict[str, Any]:
    scenario_dir = root_dir / scenario_name
    profile_dir = scenario_dir / "profiling"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    command = build_profile_command(
        scenario_name=scenario_name,
        nproc_per_node=nproc_per_node,
        common_args=common_args,
        scenario_args=scenario_args,
        output_dir=str(profile_dir),
    )
    _write_text(scenario_dir / "command.txt", " ".join(command) + "\n")

    start_time = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    command_wall_time_sec = time.perf_counter() - start_time

    _write_text(scenario_dir / "stdout.log", completed.stdout)
    _write_text(scenario_dir / "stderr.log", completed.stderr)

    failure_info = classify_failure(completed.returncode, completed.stdout, completed.stderr)
    metrics = load_scenario_metrics(profile_dir)
    metrics["command_wall_time_sec"] = round(command_wall_time_sec, 6)

    return {
        "scenario": scenario_name,
        "status": failure_info["status"],
        "oom": failure_info["oom"],
        "returncode": completed.returncode,
        "command": command,
        "paths": {
            "scenario_dir": str(scenario_dir),
            "profiling_dir": str(profile_dir),
            "stdout_log": str(scenario_dir / "stdout.log"),
            "stderr_log": str(scenario_dir / "stderr.log"),
        },
        "metrics": metrics,
    }


def build_comparison(baseline: dict[str, Any], hsdp: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline": baseline,
        "hsdp": hsdp,
        "deltas": {
            "wall_time_pct": pct_delta(
                baseline.get("metrics", {}).get("wall_time_sec"),
                hsdp.get("metrics", {}).get("wall_time_sec"),
                inverse=False,
            ),
            "command_wall_time_pct": pct_delta(
                baseline.get("metrics", {}).get("command_wall_time_sec"),
                hsdp.get("metrics", {}).get("command_wall_time_sec"),
                inverse=False,
            ),
            "peak_allocated_pct": pct_delta(
                baseline.get("metrics", {}).get("max_peak_allocated_bytes"),
                hsdp.get("metrics", {}).get("max_peak_allocated_bytes"),
                inverse=True,
            ),
            "peak_reserved_pct": pct_delta(
                baseline.get("metrics", {}).get("max_peak_reserved_bytes"),
                hsdp.get("metrics", {}).get("max_peak_reserved_bytes"),
                inverse=True,
            ),
            "total_profiled_sec_pct": pct_delta(
                baseline.get("metrics", {}).get("total_profiled_sec"),
                hsdp.get("metrics", {}).get("total_profiled_sec"),
                inverse=False,
            ),
        },
    }


def namespace_to_common_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.model,
        "image": args.image,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "guidance_scale": args.guidance_scale,
        "guidance_scale_high": args.guidance_scale_high,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "boundary_ratio": args.boundary_ratio,
        "frame_rate": args.frame_rate,
        "flow_shift": args.flow_shift,
        "sample_solver": args.sample_solver,
        "fps": args.fps,
        "ulysses_degree": args.ulysses_degree,
        "ring_degree": args.ring_degree,
        "tensor_parallel_size": args.tensor_parallel_size,
        "cfg_parallel_size": args.cfg_parallel_size,
        "vae_patch_parallel_size": args.vae_patch_parallel_size,
        "enable_cpu_offload": args.enable_cpu_offload,
        "enable_layerwise_offload": args.enable_layerwise_offload,
        "vae_use_slicing": args.vae_use_slicing,
        "vae_use_tiling": args.vae_use_tiling,
        "enforce_eager": args.enforce_eager,
        "cache_backend": args.cache_backend,
        "enable_diffusion_pipeline_profiler": args.enable_diffusion_pipeline_profiler,
        "warmup_runs": args.warmup_runs,
        "profile_memory": args.profile_memory,
        "with_stack": args.with_stack,
    }


def main() -> None:
    args = parse_args()
    root_dir = Path(args.output_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    common_args = namespace_to_common_args(args)
    baseline_result = run_scenario(
        scenario_name="baseline",
        nproc_per_node=args.nproc_per_node,
        common_args=common_args,
        scenario_args={"use_hsdp": False},
        root_dir=root_dir,
    )
    hsdp_result = run_scenario(
        scenario_name="hsdp",
        nproc_per_node=args.nproc_per_node,
        common_args=common_args,
        scenario_args={
            "use_hsdp": True,
            "hsdp_shard_size": args.hsdp_shard_size,
            "hsdp_replicate_size": args.hsdp_replicate_size,
        },
        root_dir=root_dir,
    )

    comparison = build_comparison(baseline_result, hsdp_result)
    _write_json(root_dir / "comparison.json", comparison)
    write_rows_to_csv(
        root_dir / "scenario_summary.csv",
        [
            _scenario_metrics_row("baseline", baseline_result),
            _scenario_metrics_row("hsdp", hsdp_result),
        ],
    )
    write_rows_to_csv(
        root_dir / "phase_summary.csv",
        _phase_rows("baseline", root_dir / "baseline" / "profiling")
        + _phase_rows("hsdp", root_dir / "hsdp" / "profiling"),
    )
    _write_text(root_dir / "comparison.md", render_markdown_summary(comparison))

    print(f"[HSDP Benchmark] comparison json: {root_dir / 'comparison.json'}")
    print(f"[HSDP Benchmark] scenario summary: {root_dir / 'scenario_summary.csv'}")
    print(f"[HSDP Benchmark] phase summary: {root_dir / 'phase_summary.csv'}")
    print(f"[HSDP Benchmark] markdown summary: {root_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
