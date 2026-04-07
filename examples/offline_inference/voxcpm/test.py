"""Run a fixed VoxCPM offline-example test matrix.

This script reuses ``end2end.py`` and covers both stage-config routes:
- streaming: ``voxcpm.yaml``
- sync: ``voxcpm_no_async_chunk.yaml``

Scenarios:
- warmup + single TTS
- warmup + single voice cloning
- warmup + batch TTS
- warmup + batch voice cloning
- no warmup + single TTS
- no warmup + single voice cloning
"""

from __future__ import annotations

import argparse
import ast
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
END2END_SCRIPT = Path(__file__).with_name("end2end.py")
DEFAULT_STAGE_ASYNC = REPO_ROOT / "vllm_omni" / "model_executor" / "stage_configs" / "voxcpm.yaml"
DEFAULT_STAGE_SYNC = REPO_ROOT / "vllm_omni" / "model_executor" / "stage_configs" / "voxcpm_no_async_chunk.yaml"

SINGLE_TTS_TEXT = "This is a single text-to-speech smoke test for VoxCPM on vLLM Omni."
SINGLE_CLONE_TEXT = "This sentence is synthesized with the cloned voice for validation."
BATCH_TTS_TEXTS = [
    "The first batch text-to-speech sample validates sequential batch execution.",
    "The second batch text-to-speech sample checks another prompt in the same file.",
    "The third batch text-to-speech sample completes the sequential batch path.",
]
BATCH_CLONE_TEXTS = [
    "The first cloned sample validates sequential batch voice cloning.",
    "The second cloned sample checks the same reference voice on another prompt.",
    "The third cloned sample finishes the shared-reference clone batch path.",
]


@dataclass(frozen=True, slots=True)
class ModeSpec:
    name: str
    stage_config: Path


@dataclass(frozen=True, slots=True)
class CaseSpec:
    name: str
    warmup_runs: int
    prompt_kind: str
    voice_clone: bool


@dataclass(frozen=True, slots=True)
class CaseResult:
    mode: str
    case: str
    returncode: int
    elapsed_s: float
    output_dir: Path
    log_path: Path
    request_summaries: list[dict[str, Any]]

    @property
    def ok(self) -> bool:
        return self.returncode == 0


MODE_SPECS = [
    ModeSpec(name="streaming", stage_config=DEFAULT_STAGE_ASYNC),
    ModeSpec(name="sync", stage_config=DEFAULT_STAGE_SYNC),
]

CASE_SPECS = [
    CaseSpec(name="warmup_single_tts", warmup_runs=1, prompt_kind="single", voice_clone=False),
    CaseSpec(name="warmup_single_clone", warmup_runs=1, prompt_kind="single", voice_clone=True),
    CaseSpec(name="warmup_batch_tts", warmup_runs=1, prompt_kind="batch", voice_clone=False),
    CaseSpec(name="warmup_batch_clone", warmup_runs=1, prompt_kind="batch", voice_clone=True),
    CaseSpec(name="cold_single_tts", warmup_runs=0, prompt_kind="single", voice_clone=False),
    CaseSpec(name="cold_single_clone", warmup_runs=0, prompt_kind="single", voice_clone=True),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VoxCPM offline example smoke tests.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Local VoxCPM model directory.",
    )
    parser.add_argument(
        "--ref-audio",
        type=str,
        required=True,
        help="Reference audio path for voice cloning scenarios.",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        required=True,
        help="Real transcript of the reference audio for voice cloning scenarios.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to launch end2end.py.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(Path(__file__).with_name("test_outputs")),
        help="Root directory for generated inputs and per-case outputs.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Forwarded to end2end.py for each case. Default 1.",
    )
    parser.add_argument(
        "--cfg-value",
        type=float,
        default=None,
        help="Optional cfg override forwarded to end2end.py.",
    )
    parser.add_argument(
        "--inference-timesteps",
        type=int,
        default=None,
        help="Optional inference-timesteps override forwarded to end2end.py.",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=None,
        help="Optional min-len override forwarded to end2end.py.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Optional max-new-tokens override forwarded to end2end.py.",
    )
    parser.add_argument(
        "--streaming-prefix-len",
        type=int,
        default=None,
        help="Optional streaming-prefix-len override forwarded to end2end.py.",
    )
    parser.add_argument(
        "--stage-init-timeout",
        type=int,
        default=600,
        help="Stage initialization timeout forwarded to end2end.py.",
    )
    return parser.parse_args()


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prepare_batch_inputs(output_root: Path) -> tuple[Path, Path]:
    input_dir = output_root / "inputs"
    batch_tts_path = input_dir / "batch_tts_prompts.txt"
    batch_clone_path = input_dir / "batch_clone_prompts.txt"
    _write_lines(batch_tts_path, BATCH_TTS_TEXTS)
    _write_lines(batch_clone_path, BATCH_CLONE_TEXTS)
    return batch_tts_path, batch_clone_path


def _base_command(args: argparse.Namespace, mode: ModeSpec, output_dir: Path) -> list[str]:
    cmd = [
        args.python,
        str(END2END_SCRIPT),
        "--model",
        args.model,
        "--stage-configs-path",
        str(mode.stage_config),
        "--output-dir",
        str(output_dir),
        "--num-runs",
        str(args.num_runs),
        "--stage-init-timeout",
        str(args.stage_init_timeout),
        "--log-stats",
    ]
    if args.cfg_value is not None:
        cmd.extend(["--cfg-value", str(args.cfg_value)])
    if args.inference_timesteps is not None:
        cmd.extend(["--inference-timesteps", str(args.inference_timesteps)])
    if args.min_len is not None:
        cmd.extend(["--min-len", str(args.min_len)])
    if args.max_new_tokens is not None:
        cmd.extend(["--max-new-tokens", str(args.max_new_tokens)])
    if args.streaming_prefix_len is not None:
        cmd.extend(["--streaming-prefix-len", str(args.streaming_prefix_len)])
    return cmd


def _extract_summary_blocks(log_text: str) -> list[dict[str, Any]]:
    marker = "[Summary]"
    results: list[dict[str, Any]] = []
    cursor = 0
    while True:
        marker_idx = log_text.find(marker, cursor)
        if marker_idx < 0:
            break
        brace_idx = log_text.find("{", marker_idx)
        if brace_idx < 0:
            break

        depth = 0
        in_single = False
        in_double = False
        escaped = False
        end_idx: int | None = None
        for pos in range(brace_idx, len(log_text)):
            ch = log_text[pos]
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if in_single:
                if ch == "'":
                    in_single = False
                continue
            if in_double:
                if ch == '"':
                    in_double = False
                continue
            if ch == "'":
                in_single = True
                continue
            if ch == '"':
                in_double = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end_idx = pos + 1
                    break

        if end_idx is None:
            break

        block = log_text[brace_idx:end_idx]
        try:
            parsed = ast.literal_eval(block)
        except Exception:
            cursor = end_idx
            continue
        if isinstance(parsed, dict):
            results.append(parsed)
        cursor = end_idx
    return results


def _normalize_request_summaries(summary_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for summary in summary_blocks:
        overall = summary.get("overall_summary", {})
        request_id = None
        stage_table = summary.get("stage_table", [])
        e2e_table = summary.get("e2e_table", [])
        if stage_table and isinstance(stage_table[0], dict):
            request_id = stage_table[0].get("request_id")
        if request_id is None and e2e_table and isinstance(e2e_table[0], dict):
            request_id = e2e_table[0].get("request_id")
        if request_id is None:
            request_id = f"request_{len(normalized) + 1:03d}"

        stage_wall_times: dict[str, float] = {}
        for key, value in overall.items():
            if key.startswith("e2e_stage_") and key.endswith("_wall_time_ms"):
                stage_name = key[len("e2e_") : -len("_wall_time_ms")]
                stage_wall_times[stage_name] = float(value)

        e2e_stats = e2e_table[0] if e2e_table and isinstance(e2e_table[0], dict) else {}
        normalized.append(
            {
                "request_id": request_id,
                "stage_wall_time_ms": stage_wall_times,
                "e2e_total_ms": float(e2e_stats.get("e2e_total_ms", 0.0)),
                "e2e_total_tokens": int(e2e_stats.get("e2e_total_tokens", 0)),
                "transfers_total_time_ms": float(e2e_stats.get("transfers_total_time_ms", 0.0)),
                "transfers_total_kbytes": float(e2e_stats.get("transfers_total_kbytes", 0.0)),
            }
        )
    return normalized


def _print_request_summaries(request_summaries: list[dict[str, Any]]) -> None:
    if not request_summaries:
        print("未解析到 stage 耗时摘要。")
        return
    print("每个 request 的 stage 耗时:")
    for item in request_summaries:
        stage_parts = [
            f"{stage_name}={stage_ms:.2f}ms"
            for stage_name, stage_ms in sorted(item["stage_wall_time_ms"].items())
        ]
        stage_text = ", ".join(stage_parts) if stage_parts else "无 stage 数据"
        print(
            f"- {item['request_id']}: {stage_text}, "
            f"e2e={item['e2e_total_ms']:.2f}ms, tokens={item['e2e_total_tokens']}"
        )


def _build_case_command(
    args: argparse.Namespace,
    mode: ModeSpec,
    case: CaseSpec,
    *,
    batch_tts_path: Path,
    batch_clone_path: Path,
    output_dir: Path,
) -> list[str]:
    cmd = _base_command(args, mode, output_dir)
    cmd.extend(["--warmup-runs", str(case.warmup_runs)])

    if case.prompt_kind == "single":
        text = SINGLE_CLONE_TEXT if case.voice_clone else SINGLE_TTS_TEXT
        cmd.extend(["--text", text])
    else:
        prompt_path = batch_clone_path if case.voice_clone else batch_tts_path
        cmd.extend(["--txt-prompts", str(prompt_path)])

    if case.voice_clone:
        cmd.extend(
            [
                "--ref-audio",
                args.ref_audio,
                "--ref-text",
                args.ref_text,
            ]
        )
    return cmd


def _run_case(
    args: argparse.Namespace,
    mode: ModeSpec,
    case: CaseSpec,
    *,
    batch_tts_path: Path,
    batch_clone_path: Path,
    output_root: Path,
) -> CaseResult:
    case_output_dir = output_root / mode.name / case.name
    case_output_dir.mkdir(parents=True, exist_ok=True)
    case_log_path = case_output_dir / "run.log"
    cmd = _build_case_command(
        args,
        mode,
        case,
        batch_tts_path=batch_tts_path,
        batch_clone_path=batch_clone_path,
        output_dir=case_output_dir,
    )

    print()
    print("=" * 80)
    print(f"[{mode.name}] {case.name}")
    print(f"输出目录: {case_output_dir}")
    print(shlex.join(cmd))

    start = time.perf_counter()
    captured_lines: list[str] = []
    with case_log_path.open("w", encoding="utf-8") as log_fp:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_fp.write(line)
            captured_lines.append(line)
        process.wait()
    elapsed_s = time.perf_counter() - start
    completed_returncode = int(process.returncode or 0)
    summary_blocks = _extract_summary_blocks("".join(captured_lines))
    request_summaries = _normalize_request_summaries(summary_blocks)
    _print_request_summaries(request_summaries)
    summary_json_path = case_output_dir / "summary.json"
    summary_json_path.write_text(
        json.dumps(request_summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    status = "PASS" if completed_returncode == 0 else "FAIL"
    print(f"[{mode.name}] {case.name} -> {status} ({elapsed_s:.2f}s)")

    return CaseResult(
        mode=mode.name,
        case=case.name,
        returncode=completed_returncode,
        elapsed_s=elapsed_s,
        output_dir=case_output_dir,
        log_path=case_log_path,
        request_summaries=request_summaries,
    )


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    batch_tts_path, batch_clone_path = _prepare_batch_inputs(output_root)

    print(f"Model: {args.model}")
    print(f"Reference audio: {args.ref_audio}")
    print(f"Reference text: {args.ref_text}")
    print(f"Python: {args.python}")
    print(f"Output root: {output_root}")
    print(f"Cases: {len(MODE_SPECS) * len(CASE_SPECS)}")

    results: list[CaseResult] = []
    for mode in MODE_SPECS:
        for case in CASE_SPECS:
            results.append(
                _run_case(
                    args,
                    mode,
                    case,
                    batch_tts_path=batch_tts_path,
                    batch_clone_path=batch_clone_path,
                    output_root=output_root,
                )
            )

    passed = sum(1 for result in results if result.ok)
    failed = [result for result in results if not result.ok]

    print()
    print("=" * 80)
    print("汇总:")
    for result in results:
        status = "PASS" if result.ok else f"FAIL({result.returncode})"
        print(f"- [{result.mode}] {result.case}: {status} ({result.elapsed_s:.2f}s)")
        for item in result.request_summaries:
            stage_parts = [
                f"{stage_name}={stage_ms:.2f}ms"
                for stage_name, stage_ms in sorted(item["stage_wall_time_ms"].items())
            ]
            stage_text = ", ".join(stage_parts) if stage_parts else "无 stage 数据"
            print(
                f"  request={item['request_id']}, {stage_text}, "
                f"e2e={item['e2e_total_ms']:.2f}ms"
            )

    print(f"通过: {passed}/{len(results)}")
    results_json_path = output_root / "results.json"
    results_json_path.write_text(
        json.dumps(
            [
                {
                    "mode": result.mode,
                    "case": result.case,
                    "returncode": result.returncode,
                    "elapsed_s": result.elapsed_s,
                    "output_dir": str(result.output_dir),
                    "log_path": str(result.log_path),
                    "request_summaries": result.request_summaries,
                }
                for result in results
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"结果汇总已写入: {results_json_path}")
    if failed:
        print("失败用例:")
        for result in failed:
            print(f"- [{result.mode}] {result.case}: 输出目录 {result.output_dir}, 日志 {result.log_path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
