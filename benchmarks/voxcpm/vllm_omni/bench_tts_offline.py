"""Offline VoxCPM benchmark for vLLM Omni.

Supports both:
- sync one-shot (Omni.generate)
- streaming (AsyncOmni.generate with async_chunk config)
- text-only synthesis
- voice cloning
- text/clone batch inputs from txt or jsonl
- fixed smoke matrix equivalent to the old examples/offline_inference/voxcpm/test.py
"""

from __future__ import annotations

import asyncio
import ast
import json
import logging
import os
import shlex
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni import AsyncOmni, Omni

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCH_SCRIPT = Path(__file__).resolve()
DEFAULT_STAGE_ASYNC = REPO_ROOT / "vllm_omni" / "model_executor" / "stage_configs" / "voxcpm_async_chunk.yaml"
DEFAULT_STAGE_SYNC = REPO_ROOT / "vllm_omni" / "model_executor" / "stage_configs" / "voxcpm.yaml"
DEFAULT_MATRIX_OUTPUT_ROOT = BENCH_SCRIPT.parents[1] / "results" / "offline_matrix"

logger = logging.getLogger(__name__)

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
class PromptSpec:
    text: str
    label: str
    ref_audio: str | None = None
    ref_text: str | None = None


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


def _require_soundfile():
    try:
        import soundfile as sf  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("soundfile is required to write VoxCPM benchmark WAV outputs. Install it with: pip install soundfile") from exc
    return sf


def _build_prompt(
    args,
    *,
    text: str,
    ref_audio: str | None = None,
    ref_text: str | None = None,
    global_request_id: str | None = None,
) -> dict[str, Any]:
    additional_information: dict[str, list[Any]] = {
        "text": [text],
        "cfg_value": [args.cfg_value],
        "inference_timesteps": [args.inference_timesteps],
        "min_len": [args.min_len],
        "max_new_tokens": [args.max_new_tokens],
    }
    if args.streaming_prefix_len is not None:
        additional_information["streaming_prefix_len"] = [args.streaming_prefix_len]

    if ref_audio:
        additional_information["ref_audio"] = [ref_audio]
    if ref_text:
        additional_information["ref_text"] = [ref_text]
    if global_request_id is not None:
        additional_information["global_request_id"] = [global_request_id]

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


def _emit_offline_metrics(
    *,
    request_id: str,
    elapsed_s: float,
    first_audio_elapsed: float | None,
    audio_duration_s: float,
) -> None:
    metrics = {
        "request_id": request_id,
        "ttfp_ms": round(first_audio_elapsed * 1000.0, 3) if first_audio_elapsed is not None else None,
        "audio_duration_s": round(audio_duration_s, 6),
        "rtf": round(elapsed_s / audio_duration_s, 6) if audio_duration_s > 0 else None,
    }
    print(f"[OfflineMetrics] {metrics}")


def _save_wav(mm: dict[str, Any], output_dir: Path, request_id: str) -> Path:
    sf = _require_soundfile()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"output_{request_id}.wav"
    sf.write(
        output_path,
        _extract_audio_tensor(mm).float().cpu().clamp(-1.0, 1.0).numpy(),
        _extract_sample_rate(mm),
        format="WAV",
        subtype="PCM_16",
    )
    return output_path


def _iter_request_multimodal_outputs(request_output: Any):
    outputs = getattr(request_output, "outputs", None)
    if outputs:
        for output in outputs:
            mm = getattr(output, "multimodal_output", None)
            if isinstance(mm, dict):
                yield mm

    mm = getattr(request_output, "multimodal_output", None)
    if isinstance(mm, dict):
        yield mm


def _read_non_empty_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _load_prompt_specs(args) -> list[PromptSpec]:
    specs: list[PromptSpec] = []

    if args.txt_prompts is not None:
        texts = _read_non_empty_lines(args.txt_prompts)
        if not texts:
            raise ValueError(f"No prompts found in {args.txt_prompts}")
        for idx, text in enumerate(texts, start=1):
            specs.append(
                PromptSpec(
                    text=text,
                    label=f"item{idx:03d}",
                    ref_audio=args.ref_audio,
                    ref_text=args.ref_text,
                )
            )
        return specs

    if args.jsonl_prompts is not None:
        with open(args.jsonl_prompts, encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{args.jsonl_prompts}:{line_no} is not valid JSON: {exc}") from exc
                if not isinstance(item, dict):
                    raise ValueError(f"{args.jsonl_prompts}:{line_no} must be a JSON object")

                text = item.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"{args.jsonl_prompts}:{line_no} requires non-empty string field 'text'")

                ref_audio = item.get("ref_audio", args.ref_audio)
                ref_text = item.get("ref_text", args.ref_text)
                if (ref_audio is None) != (ref_text is None):
                    raise ValueError(
                        f"{args.jsonl_prompts}:{line_no} must provide both 'ref_audio' and 'ref_text' together"
                    )

                specs.append(
                    PromptSpec(
                        text=text.strip(),
                        label=f"item{len(specs) + 1:03d}",
                        ref_audio=ref_audio,
                        ref_text=ref_text,
                    )
                )

        if not specs:
            raise ValueError(f"No prompts found in {args.jsonl_prompts}")
        return specs

    specs.append(
        PromptSpec(
            text=args.text,
            label="item001",
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
        )
    )
    return specs


def _build_prompt_for_spec(args, spec: PromptSpec, *, global_request_id: str | None = None) -> dict[str, Any]:
    return _build_prompt(
        args,
        text=spec.text,
        ref_audio=spec.ref_audio,
        ref_text=spec.ref_text,
        global_request_id=global_request_id,
    )


def _count_voice_clone_prompts(prompt_specs: list[PromptSpec]) -> int:
    return sum(1 for spec in prompt_specs if spec.ref_audio is not None)


def _get_warmup_specs(prompt_specs: list[PromptSpec]) -> list[PromptSpec]:
    return prompt_specs[:1]


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


def _extract_summary_blocks(log_text: str) -> list[dict[str, Any]]:
    return _extract_literal_blocks(log_text, "[Summary]")


def _extract_offline_metrics_blocks(log_text: str) -> list[dict[str, Any]]:
    return _extract_literal_blocks(log_text, "[OfflineMetrics]")


def _extract_literal_blocks(log_text: str, marker: str) -> list[dict[str, Any]]:
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


def _normalize_request_summaries(
    summary_blocks: list[dict[str, Any]],
    offline_metrics: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    offline_metrics = offline_metrics or {}
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
        metrics = offline_metrics.get(str(request_id), {})
        normalized.append(
            {
                "request_id": request_id,
                "stage_wall_time_ms": stage_wall_times,
                "e2e_total_ms": float(e2e_stats.get("e2e_total_ms", 0.0)),
                "e2e_total_tokens": int(e2e_stats.get("e2e_total_tokens", 0)),
                "transfers_total_time_ms": float(e2e_stats.get("transfers_total_time_ms", 0.0)),
                "transfers_total_kbytes": float(e2e_stats.get("transfers_total_kbytes", 0.0)),
                "ttfp_ms": float(metrics["ttfp_ms"]) if metrics.get("ttfp_ms") is not None else None,
                "audio_duration_s": float(metrics.get("audio_duration_s", 0.0)),
                "rtf": float(metrics["rtf"]) if metrics.get("rtf") is not None else None,
            }
        )
    return normalized


def _collect_request_summaries_from_log(log_text: str) -> list[dict[str, Any]]:
    summary_blocks = _extract_summary_blocks(log_text)
    metrics_blocks = _extract_offline_metrics_blocks(log_text)
    metrics_by_request_id = {
        str(item["request_id"]): item
        for item in metrics_blocks
        if isinstance(item, dict) and item.get("request_id") is not None
    }
    return _normalize_request_summaries(summary_blocks, metrics_by_request_id)


def _print_request_summaries(request_summaries: list[dict[str, Any]]) -> None:
    if not request_summaries:
        print("No stage timing summary was parsed.")
        return
    print("Per-request stage timings:")
    for item in request_summaries:
        stage_parts = [
            f"{stage_name}={stage_ms:.2f}ms" for stage_name, stage_ms in sorted(item["stage_wall_time_ms"].items())
        ]
        stage_text = ", ".join(stage_parts) if stage_parts else "no stage data"
        ttfp_text = f", ttfp={item['ttfp_ms']:.2f}ms" if item.get("ttfp_ms") is not None else ""
        rtf_text = f", rtf={item['rtf']:.3f}" if item.get("rtf") is not None else ""
        print(
            f"- {item['request_id']}: {stage_text}, e2e={item['e2e_total_ms']:.2f}ms, "
            f"tokens={item['e2e_total_tokens']}{ttfp_text}{rtf_text}"
        )


def _base_matrix_command(args: Any, mode: ModeSpec, output_dir: Path) -> list[str]:
    cmd = [
        args.python,
        str(BENCH_SCRIPT),
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
    if args.log_stats:
        cmd.append("--log-stats")
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
    if args.enable_profiler:
        profiler_dir = Path(args.profiler_dir) if args.profiler_dir is not None else (output_dir / "profiler")
        cmd.append("--enable-profiler")
        cmd.extend(["--profiler-dir", str(profiler_dir)])
        cmd.extend(["--profiler-wait-seconds", str(args.profiler_wait_seconds)])
        if args.profiler_stages is not None:
            cmd.append("--profiler-stages")
            cmd.extend(str(stage_id) for stage_id in args.profiler_stages)
    return cmd


def _build_matrix_case_command(
    args: Any,
    mode: ModeSpec,
    case: CaseSpec,
    *,
    batch_tts_path: Path,
    batch_clone_path: Path,
    output_dir: Path,
) -> list[str]:
    cmd = _base_matrix_command(args, mode, output_dir)
    cmd.extend(["--warmup-runs", str(case.warmup_runs)])

    if case.prompt_kind == "single":
        text = SINGLE_CLONE_TEXT if case.voice_clone else SINGLE_TTS_TEXT
        cmd.extend(["--text", text])
    else:
        prompt_path = batch_clone_path if case.voice_clone else batch_tts_path
        cmd.extend(["--txt-prompts", str(prompt_path)])

    if case.voice_clone:
        cmd.extend(["--ref-audio", args.ref_audio, "--ref-text", args.ref_text])
    return cmd


def _run_matrix_case(
    args: Any,
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
    cmd = _build_matrix_case_command(
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
    print(f"Output directory: {case_output_dir}")
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
    returncode = int(process.returncode or 0)
    request_summaries = _collect_request_summaries_from_log("".join(captured_lines))
    _print_request_summaries(request_summaries)
    summary_json_path = case_output_dir / "summary.json"
    summary_json_path.write_text(json.dumps(request_summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    status = "PASS" if returncode == 0 else "FAIL"
    print(f"[{mode.name}] {case.name} -> {status} ({elapsed_s:.2f}s)")
    return CaseResult(
        mode=mode.name,
        case=case.name,
        returncode=returncode,
        elapsed_s=elapsed_s,
        output_dir=case_output_dir,
        log_path=case_log_path,
        request_summaries=request_summaries,
    )


def _run_full_matrix(args: Any) -> int:
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
                _run_matrix_case(
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
    print("Summary:")
    for result in results:
        status = "PASS" if result.ok else f"FAIL({result.returncode})"
        print(f"- [{result.mode}] {result.case}: {status} ({result.elapsed_s:.2f}s)")
        for item in result.request_summaries:
            stage_parts = [
                f"{stage_name}={stage_ms:.2f}ms" for stage_name, stage_ms in sorted(item["stage_wall_time_ms"].items())
            ]
            stage_text = ", ".join(stage_parts) if stage_parts else "no stage data"
            ttfp_text = f", ttfp={item['ttfp_ms']:.2f}ms" if item.get("ttfp_ms") is not None else ""
            rtf_text = f", rtf={item['rtf']:.3f}" if item.get("rtf") is not None else ""
            print(f"  request={item['request_id']}, {stage_text}, e2e={item['e2e_total_ms']:.2f}ms{ttfp_text}{rtf_text}")

    print(f"Passed: {passed}/{len(results)}")
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
    print(f"Wrote results summary to: {results_json_path}")

    if failed:
        print("Failed cases:")
        for result in failed:
            print(f"- [{result.mode}] {result.case}: output dir {result.output_dir}, log {result.log_path}")
        return 1
    return 0


def _build_profiled_stage_config(
    stage_configs_path: str,
    profiler_dir: str,
) -> str:
    stage_config_path = Path(stage_configs_path)
    yaml_text = stage_config_path.read_text(encoding="utf-8")
    injected_lines: list[str] = []
    injected_count = 0

    for line in yaml_text.splitlines():
        injected_lines.append(line)
        if line.strip() != "engine_args:":
            continue
        indent = line[: len(line) - len(line.lstrip())]
        child_indent = indent + "  "
        grandchild_indent = child_indent + "  "
        injected_lines.extend(
            [
                f"{child_indent}profiler_config:",
                f'{grandchild_indent}profiler: "torch"',
                f'{grandchild_indent}torch_profiler_dir: "{profiler_dir}"',
                f"{grandchild_indent}torch_profiler_with_stack: true",
            ]
        )
        injected_count += 1

    if injected_count == 0:
        raise ValueError(f"No engine_args block found in stage config: {stage_configs_path}")

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        suffix=".yaml",
        prefix=f"{stage_config_path.stem}_profile_",
    )
    tmp.write("\n".join(injected_lines) + "\n")
    tmp.close()
    return tmp.name


def parse_args():
    parser = FlexibleArgumentParser(
        description="Offline split-stage VoxCPM inference with vLLM Omni (auto sync/streaming by stage config)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("VOXCPM_MODEL"),
        help="Local VoxCPM model directory. Defaults to $VOXCPM_MODEL.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="This is a split-stage VoxCPM synthesis example running on vLLM Omni.",
        help="Text to synthesize. Ignored when --txt-prompts or --jsonl-prompts is used.",
    )
    parser.add_argument(
        "--txt-prompts",
        type=str,
        default=None,
        help="Path to a .txt file with one synthesis text per line.",
    )
    parser.add_argument(
        "--jsonl-prompts",
        type=str,
        default=None,
        help="Path to a .jsonl file. Each line must contain at least {'text': ...}; clone rows can also set ref_audio/ref_text, and ref_text must be the real transcript of ref_audio.",
    )
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help="Optional reference audio path for voice cloning. With --txt-prompts, the same reference is applied to every line.",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default=None,
        help="Real transcript of the reference audio. Placeholder text or mismatched text will usually produce noisy/electronic clone audio.",
    )
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default=str(DEFAULT_STAGE_SYNC),
        help="Stage config YAML path. Routing is selected only from this path.",
    )
    parser.add_argument(
        "--cfg-value",
        type=float,
        default=2.0,
        help="Classifier-free guidance value for VoxCPM.",
    )
    parser.add_argument(
        "--inference-timesteps",
        type=int,
        default=10,
        help="Number of inference timesteps.",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=2,
        help="Minimum generated token length.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Maximum generated token length.",
    )
    parser.add_argument(
        "--streaming-prefix-len",
        type=int,
        default=None,
        help="VoxCPM streaming window (optional, streaming mode only).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output WAV files.",
    )
    parser.add_argument(
        "--matrix",
        choices=["none", "full"],
        default="none",
        help="Run a fixed offline smoke matrix. 'full' matches the old examples/offline_inference/voxcpm/test.py coverage.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Root directory for matrix outputs. Used only with --matrix full.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used for recursive matrix runs. Used only with --matrix full.",
    )
    parser.add_argument(
        "--stage-init-timeout",
        type=int,
        default=600,
        help="Stage initialization timeout in seconds.",
    )
    parser.add_argument(
        "--log-stats",
        action="store_true",
        help="Enable vLLM Omni stats logging.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of full inference runs (same prompt each time). Default 1.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Optional number of warmup passes before measured runs. Warmup uses only the first prompt and does not save outputs.",
    )
    parser.add_argument(
        "--enable-profiler",
        action="store_true",
        help="Enable torch profiler for the configured stages. A temporary profiled stage config is generated automatically.",
    )
    parser.add_argument(
        "--profiler-dir",
        type=str,
        default=None,
        help="Directory for profiler traces. Defaults to <output-dir>/profiler when profiling is enabled.",
    )
    parser.add_argument(
        "--profiler-stages",
        type=int,
        nargs="*",
        default=None,
        help="Optional stage ids to profile. Defaults to all stages that have profiler_config.",
    )
    parser.add_argument(
        "--profiler-wait-seconds",
        type=float,
        default=30.0,
        help="Seconds to wait after stop_profile for trace files to flush.",
    )
    args = parser.parse_args()

    if not args.model:
        parser.error("--model is required unless $VOXCPM_MODEL is set")
    if args.txt_prompts is not None and args.jsonl_prompts is not None:
        parser.error("--txt-prompts and --jsonl-prompts are mutually exclusive")
    if (args.ref_audio is None) != (args.ref_text is None):
        parser.error("--ref-audio and --ref-text must be provided together")
    if args.num_runs < 1:
        parser.error("--num-runs must be >= 1")
    if args.warmup_runs < 0:
        parser.error("--warmup-runs must be >= 0")
    if args.matrix == "full":
        if args.ref_audio is None or args.ref_text is None:
            parser.error("--matrix full requires --ref-audio and --ref-text because clone cases are included")
        if args.output_root is None:
            args.output_root = str(DEFAULT_MATRIX_OUTPUT_ROOT)
        return args
    if args.output_dir is None:
        args.output_dir = (
            "output_audio_streaming" if _is_streaming_stage_config(args.stage_configs_path) else "output_audio"
        )
    if args.enable_profiler and args.profiler_dir is None:
        args.profiler_dir = str(Path(args.output_dir) / "profiler")
    try:
        args.prompt_specs = _load_prompt_specs(args)
    except ValueError as exc:
        parser.error(str(exc))

    return args


def _is_streaming_stage_config(stage_configs_path: str) -> bool:
    cfg_name = Path(stage_configs_path).name.lower()
    # Keep routing purely config-path based:
    # - voxcpm.yaml => sync
    # - voxcpm_async_chunk.yaml => streaming
    return "async_chunk" in cfg_name


async def _collect_streaming_audio(
    omni: AsyncOmni,
    args: Any,
    spec: PromptSpec,
    request_id: str,
    *,
    phase_label: str,
    prompt_index: int,
    prompt_count: int,
    print_prompt: bool = False,
) -> tuple[torch.Tensor, int, float, float | None]:
    prompt = _build_prompt_for_spec(args, spec, global_request_id=request_id)
    delta_chunks: list[torch.Tensor] = []
    sample_rate = 24000
    chunk_i = 0
    prev_total_samples = 0
    t_start = time.perf_counter()
    first_audio_elapsed: float | None = None

    if print_prompt:
        print(f"---prompt---:{prompt}")

    async for stage_output in omni.generate(prompt, request_id=request_id):
        mm = getattr(stage_output, "multimodal_output", None)
        if not isinstance(mm, dict):
            ro = getattr(stage_output, "request_output", None)
            if ro is None:
                continue
            mm = getattr(ro, "multimodal_output", None)
            if not isinstance(mm, dict) and getattr(ro, "outputs", None):
                seq = ro.outputs[0]
                mm = getattr(seq, "multimodal_output", None)
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
                prev_total_samples = n
            else:
                delta = w.reshape(-1)
                prev_total_samples += int(delta.numel())
            delta_chunks.append(delta)
            if first_audio_elapsed is None and int(delta.numel()) > 0:
                first_audio_elapsed = time.perf_counter() - t_start
            logger.info(
                "%s prompt=%d/%d chunk=%d delta_samples=%d buf_len=%d finished=%s",
                phase_label,
                prompt_index + 1,
                prompt_count,
                chunk_i,
                int(delta.numel()),
                n,
                stage_output.finished,
            )
            chunk_i += 1
        except ValueError:
            if not stage_output.finished:
                logger.debug("skip non-audio partial output chunk=%d", chunk_i)

    if not delta_chunks:
        raise RuntimeError("No audio chunks received; check stage config and logs.")

    audio_cat = torch.cat([c.reshape(-1) for c in delta_chunks], dim=0)
    elapsed = time.perf_counter() - t_start
    return audio_cat, sample_rate, elapsed, first_audio_elapsed


async def _run_streaming_single(
    omni: AsyncOmni,
    args: Any,
    spec: PromptSpec,
    output_dir: Path,
    request_id: str,
    *,
    run_index: int,
    num_runs: int,
    prompt_index: int,
    prompt_count: int,
) -> Path:
    audio_cat, sample_rate, elapsed, first_audio_elapsed = await _collect_streaming_audio(
        omni,
        args,
        spec,
        request_id,
        phase_label=f"run={run_index + 1}/{num_runs}",
        prompt_index=prompt_index,
        prompt_count=prompt_count,
        print_prompt=(run_index == 0 and prompt_index == 0),
    )
    output_path = output_dir / f"output_run{run_index + 1}_{spec.label}.wav"
    sf.write(
        output_path,
        audio_cat.float().cpu().clamp(-1.0, 1.0).numpy(),
        sample_rate,
        format="WAV",
        subtype="PCM_16",
    )
    audio_duration_s = float(audio_cat.numel()) / float(sample_rate) if sample_rate > 0 else 0.0
    ttfp_text = f", ttfp={first_audio_elapsed:.2f}s" if first_audio_elapsed is not None else ""
    rtf_text = f", rtf={elapsed / audio_duration_s:.3f}" if audio_duration_s > 0 else ""
    print(
        f"Saved (streaming) run {run_index + 1}/{num_runs}, "
        f"prompt {prompt_index + 1}/{prompt_count}: {output_path} ({elapsed:.2f}s{ttfp_text}{rtf_text})"
    )
    _emit_offline_metrics(
        request_id=request_id,
        elapsed_s=elapsed,
        first_audio_elapsed=first_audio_elapsed,
        audio_duration_s=audio_duration_s,
    )
    return output_path


async def _run_streaming_warmup(args, omni: AsyncOmni) -> None:
    if args.warmup_runs == 0:
        return

    warmup_specs = _get_warmup_specs(args.prompt_specs)
    print(
        f"Warmup: {args.warmup_runs} run(s) using the first prompt "
        f"({len(warmup_specs)} prompt(s)); outputs will be discarded."
    )
    for warmup_index in range(args.warmup_runs):
        t_warmup = time.perf_counter()
        tasks = []
        for prompt_index, spec in enumerate(warmup_specs):
            request_id = f"warmup_stream_{warmup_index + 1}_{spec.label}_{uuid.uuid4().hex[:8]}"
            tasks.append(
                _collect_streaming_audio(
                    omni,
                    args,
                    spec,
                    request_id,
                    phase_label=f"warmup={warmup_index + 1}/{args.warmup_runs}",
                    prompt_index=prompt_index,
                    prompt_count=len(warmup_specs),
                )
            )
        results = await asyncio.gather(*tasks)
        total_samples = sum(int(audio.numel()) for audio, _, _, _ in results)
        warmup_ttfps = [ttfp for _, _, _, ttfp in results if ttfp is not None]
        ttfp_text = f", ttfp={min(warmup_ttfps):.2f}s" if warmup_ttfps else ""
        print(
            f"Warmup (streaming) {warmup_index + 1}/{args.warmup_runs} finished: "
            f"{len(results)} prompt(s), {total_samples} sample(s) "
            f"({time.perf_counter() - t_warmup:.2f}s{ttfp_text})"
        )


async def _run_streaming(args) -> list[Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    omni = AsyncOmni(
        model=args.model,
        stage_configs_path=args.stage_configs_path,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
    )

    await _run_streaming_warmup(args, omni)
    profiler_started = False
    if args.enable_profiler:
        profile_prefix = f"voxcpm_streaming_{int(time.time())}"
        stages_text = args.profiler_stages if args.profiler_stages is not None else "all-configured"
        print(f"Starting profiler (streaming): stages={stages_text}, dir={args.profiler_dir}")
        await omni.start_profile(profile_prefix=profile_prefix, stages=args.profiler_stages)
        profiler_started = True
    t_total = time.perf_counter()
    total_elapsed = 0.0
    paths: list[Path] = []
    prompt_specs: list[PromptSpec] = args.prompt_specs
    try:
        for run in range(args.num_runs):
            for prompt_index, spec in enumerate(prompt_specs):
                request_id = f"stream_{run + 1}_{spec.label}_{uuid.uuid4().hex[:8]}"
                paths.append(
                    await _run_streaming_single(
                        omni,
                        args,
                        spec,
                        output_dir,
                        request_id,
                        run_index=run,
                        num_runs=args.num_runs,
                        prompt_index=prompt_index,
                        prompt_count=len(prompt_specs),
                    )
                )
        total_elapsed = time.perf_counter() - t_total
    finally:
        if profiler_started:
            print("Stopping profiler (streaming)...")
            await omni.stop_profile(stages=args.profiler_stages)
            if args.profiler_wait_seconds > 0:
                print(f"Waiting {args.profiler_wait_seconds:.1f}s for profiler traces to flush...")
                await asyncio.sleep(args.profiler_wait_seconds)

    print(
        f"All streaming runs finished: {args.num_runs} run(s), "
        f"{len(prompt_specs)} prompt(s), {len(paths)} file(s) in {total_elapsed:.2f}s total"
    )
    return paths


def _run_sync(args) -> list[Path]:
    output_dir = Path(args.output_dir)

    omni = Omni(
        model=args.model,
        stage_configs_path=args.stage_configs_path,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
    )

    def _run_sync_single(
        spec: PromptSpec,
        *,
        request_prefix: str,
        save_outputs: bool,
        run_index: int | None = None,
    ) -> tuple[list[Path], int, float | None, float, float, str]:
        global_request_id = f"{request_prefix}_{spec.label}"
        prompt = _build_prompt_for_spec(args, spec, global_request_id=global_request_id)
        if save_outputs and run_index == 0 and spec.label == "item001":
            print(f"---prompt---:{prompt}")

        saved_paths: list[Path] = []
        output_count = 0
        first_audio_elapsed: float | None = None
        total_audio_duration_s = 0.0
        metrics_request_id = global_request_id
        t_start = time.perf_counter()
        for stage_outputs in omni.generate(prompt):
            request_output = stage_outputs.request_output
            if request_output is None:
                continue
            request_output_id = getattr(request_output, "request_id", None)
            if isinstance(request_output_id, str) and request_output_id:
                metrics_request_id = request_output_id
            for j, mm in enumerate(_iter_request_multimodal_outputs(request_output)):
                output_count += 1
                if first_audio_elapsed is None:
                    try:
                        audio_tensor = _extract_audio_tensor(mm)
                        if int(audio_tensor.numel()) > 0:
                            first_audio_elapsed = time.perf_counter() - t_start
                        total_audio_duration_s += float(audio_tensor.numel()) / float(_extract_sample_rate(mm))
                    except ValueError:
                        pass
                else:
                    try:
                        audio_tensor = _extract_audio_tensor(mm)
                        total_audio_duration_s += float(audio_tensor.numel()) / float(_extract_sample_rate(mm))
                    except ValueError:
                        pass
                if not save_outputs:
                    continue
                save_stem = f"run{run_index + 1}_{spec.label}" if j == 0 else f"run{run_index + 1}_{spec.label}_{j}"
                saved_paths.append(_save_wav(mm, output_dir, save_stem))

        if output_count == 0:
            raise RuntimeError("No output from Omni.generate")
        elapsed_s = time.perf_counter() - t_start
        return saved_paths, output_count, first_audio_elapsed, elapsed_s, total_audio_duration_s, metrics_request_id

    if args.warmup_runs:
        warmup_specs = _get_warmup_specs(args.prompt_specs)
        print(
            f"Warmup: {args.warmup_runs} run(s) using the first prompt "
            f"({len(warmup_specs)} prompt(s)); outputs will be discarded."
        )
        for warmup_index in range(args.warmup_runs):
            t_warmup = time.perf_counter()
            _, output_count, first_audio_elapsed, elapsed_s, audio_duration_s, _ = _run_sync_single(
                warmup_specs[0],
                request_prefix=f"warmup_sync{warmup_index + 1}",
                save_outputs=False,
            )
            ttfp_text = f", ttfp={first_audio_elapsed:.2f}s" if first_audio_elapsed is not None else ""
            rtf_text = f", rtf={elapsed_s / audio_duration_s:.3f}" if audio_duration_s > 0 else ""
            print(
                f"Warmup (sync) {warmup_index + 1}/{args.warmup_runs} finished: "
                f"{output_count} output(s) ({time.perf_counter() - t_warmup:.2f}s{ttfp_text}{rtf_text})"
            )

    profiler_started = False
    if args.enable_profiler:
        profile_prefix = f"voxcpm_sync_{int(time.time())}"
        stages_text = args.profiler_stages if args.profiler_stages is not None else "all-configured"
        print(f"Starting profiler (sync): stages={stages_text}, dir={args.profiler_dir}")
        omni.start_profile(profile_prefix=profile_prefix, stages=args.profiler_stages)
        profiler_started = True

    t_total = time.perf_counter()
    total_elapsed = 0.0
    saved_paths: list[Path] = []
    prompt_specs: list[PromptSpec] = args.prompt_specs
    try:
        for run in range(args.num_runs):
            t_run = time.perf_counter()
            run_paths: list[Path] = []
            for prompt_index, spec in enumerate(prompt_specs):
                prompt_paths, _, first_audio_elapsed, elapsed_s, audio_duration_s, metrics_request_id = _run_sync_single(
                    spec,
                    request_prefix=f"sync_run{run + 1}_{prompt_index + 1:03d}",
                    save_outputs=True,
                    run_index=run,
                )
                run_paths.extend(prompt_paths)
                ttfp_text = f", ttfp={first_audio_elapsed:.2f}s" if first_audio_elapsed is not None else ""
                rtf_text = f", rtf={elapsed_s / audio_duration_s:.3f}" if audio_duration_s > 0 else ""
                print(
                    f"Saved (sync) run {run + 1}/{args.num_runs}, "
                    f"prompt {prompt_index + 1}/{len(prompt_specs)}: {len(prompt_paths)} file(s){ttfp_text}{rtf_text}"
                )
                _emit_offline_metrics(
                    request_id=metrics_request_id,
                    elapsed_s=elapsed_s,
                    first_audio_elapsed=first_audio_elapsed,
                    audio_duration_s=audio_duration_s,
                )

            saved_paths.extend(run_paths)
            print(
                f"Run {run + 1}/{args.num_runs} finished: {len(run_paths)} file(s) ({time.perf_counter() - t_run:.2f}s)"
            )
            for path in run_paths:
                print(f"  {path}")

        total_elapsed = time.perf_counter() - t_total
    finally:
        if profiler_started:
            print("Stopping profiler (sync)...")
            omni.stop_profile(stages=args.profiler_stages)
            if args.profiler_wait_seconds > 0:
                print(f"Waiting {args.profiler_wait_seconds:.1f}s for profiler traces to flush...")
                time.sleep(args.profiler_wait_seconds)

    print(
        f"All sync runs finished: {args.num_runs} run(s), "
        f"{len(prompt_specs)} prompt(s), {len(saved_paths)} file(s) in {total_elapsed:.2f}s total"
    )
    return saved_paths


def main(args) -> int:
    logging.basicConfig(level=logging.INFO)
    if args.matrix == "full":
        return _run_full_matrix(args)

    profiled_stage_config_path: str | None = None
    original_stage_config_path = args.stage_configs_path
    if args.enable_profiler:
        Path(args.profiler_dir).mkdir(parents=True, exist_ok=True)
        profiled_stage_config_path = _build_profiled_stage_config(
            args.stage_configs_path,
            str(Path(args.profiler_dir).resolve()),
        )
        args.stage_configs_path = profiled_stage_config_path

    is_streaming = _is_streaming_stage_config(args.stage_configs_path)
    voice_clone_count = _count_voice_clone_prompts(args.prompt_specs)
    print(f"Model: {args.model}")
    print(f"Stage config: {original_stage_config_path}")
    print(f"Route: {'streaming' if is_streaming else 'sync'} (from stage-configs-path)")
    print(f"Prompt count: {len(args.prompt_specs)}")
    print("Batch mode: sequential (aligned with native VoxCPM)")
    print(f"Warmup runs: {args.warmup_runs}")
    print(f"Voice cloning prompts: {voice_clone_count}/{len(args.prompt_specs)}")
    if args.enable_profiler:
        print(f"Profiler: enabled (dir={args.profiler_dir}, stages={args.profiler_stages or 'all-configured'})")
        print(f"Profiled stage config: {args.stage_configs_path}")
    if voice_clone_count:
        print("Voice cloning note: --ref-text/ref_text must match the spoken content of the reference audio.")
    print(f"Num runs: {args.num_runs}")
    try:
        if is_streaming:
            asyncio.run(_run_streaming(args))
        else:
            _run_sync(args)
    finally:
        if profiled_stage_config_path is not None and os.path.exists(profiled_stage_config_path):
            os.unlink(profiled_stage_config_path)
    return 0


if __name__ == "__main__":
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    raise SystemExit(main(parse_args()))
