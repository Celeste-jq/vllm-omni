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
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

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
    parser.add_argument(
        "--log-stats",
        action="store_true",
        help="Forward --log-stats to end2end.py.",
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
    if args.log_stats:
        cmd.append("--log-stats")
    return cmd


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
    completed = subprocess.run(cmd, check=False)
    elapsed_s = time.perf_counter() - start
    status = "PASS" if completed.returncode == 0 else "FAIL"
    print(f"[{mode.name}] {case.name} -> {status} ({elapsed_s:.2f}s)")

    return CaseResult(
        mode=mode.name,
        case=case.name,
        returncode=completed.returncode,
        elapsed_s=elapsed_s,
        output_dir=case_output_dir,
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

    print(f"通过: {passed}/{len(results)}")
    if failed:
        print("失败用例:")
        for result in failed:
            print(f"- [{result.mode}] {result.case}: 输出目录 {result.output_dir}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
