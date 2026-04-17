from __future__ import annotations

import json
import time
from collections import OrderedDict
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Callable


class PhaseRecorder:
    def __init__(self) -> None:
        self.records: list[tuple[str, float]] = []
        self._active_phases: list[str] = []

    def add(self, name: str, duration_sec: float) -> None:
        self.records.append((name, duration_sec))

    def is_active(self, name: str) -> bool:
        return name in self._active_phases

    def push_active(self, name: str) -> None:
        self._active_phases.append(name)

    def pop_active(self, name: str) -> None:
        for index in range(len(self._active_phases) - 1, -1, -1):
            if self._active_phases[index] == name:
                del self._active_phases[index]
                return


def classify_dit_phase(model_name: str) -> str:
    return "DIT_LOW" if model_name == "transformer_2" else "DIT_HIGH"


def build_summary_from_records(records: list[tuple[str, float]]) -> dict[str, dict[str, float | int]]:
    summary: OrderedDict[str, dict[str, float | int]] = OrderedDict()
    for name, duration in records:
        item = summary.setdefault(name, {"count": 0, "total_sec": 0.0, "avg_sec": 0.0, "pct": 0.0})
        item["count"] = int(item["count"]) + 1
        item["total_sec"] = float(item["total_sec"]) + float(duration)

    total_profiled_sec = sum(float(item["total_sec"]) for item in summary.values())
    for item in summary.values():
        item["avg_sec"] = float(item["total_sec"]) / int(item["count"])
        item["pct"] = (float(item["total_sec"]) / total_profiled_sec * 100.0) if total_profiled_sec > 0 else 0.0
    return summary


def build_summary_payload(records: list[tuple[str, float]]) -> dict[str, Any]:
    phases = build_summary_from_records(records)
    total_profiled_sec = sum(float(item["total_sec"]) for item in phases.values())
    return {
        "total_profiled_sec": total_profiled_sec,
        "phases": phases,
    }


def _write_summary_payload(output_dir: Path, payload: dict[str, Any], *, stem: str = "summary") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    phases = payload["phases"]
    total_profiled_sec = float(payload["total_profiled_sec"])

    (output_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    lines = [f"total_profiled_sec: {total_profiled_sec:.6f}"]
    for name, item in phases.items():
        lines.append(
            f"{name}: count={item['count']} total_sec={float(item['total_sec']):.6f} "
            f"avg_sec={float(item['avg_sec']):.6f} pct={float(item['pct']):.2f}"
        )
    (output_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")


def write_summary_outputs(output_dir: Path, records: list[tuple[str, float]]) -> None:
    _write_summary_payload(output_dir, build_summary_payload(records), stem="summary")


def aggregate_summary_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    phases: OrderedDict[str, dict[str, float | int]] = OrderedDict()
    for payload in payloads:
        for name, item in payload.get("phases", {}).items():
            merged = phases.setdefault(name, {"count": 0, "total_sec": 0.0, "avg_sec": 0.0, "pct": 0.0})
            merged["count"] = int(merged["count"]) + int(item.get("count", 0))
            merged["total_sec"] = float(merged["total_sec"]) + float(item.get("total_sec", 0.0))

    total_profiled_sec = sum(float(item["total_sec"]) for item in phases.values())
    for item in phases.values():
        count = int(item["count"])
        total_sec = float(item["total_sec"])
        item["avg_sec"] = total_sec / count if count > 0 else 0.0
        item["pct"] = (total_sec / total_profiled_sec * 100.0) if total_profiled_sec > 0 else 0.0

    return {
        "total_profiled_sec": total_profiled_sec,
        "phases": phases,
    }


def write_aggregate_summary(output_dir: Path, payloads: list[dict[str, Any]]) -> None:
    _write_summary_payload(output_dir, aggregate_summary_payloads(payloads), stem="aggregate_summary")


@contextmanager
def npu_range(name: str):
    range_id = None
    mstx = None
    try:
        import torch_npu

        mstx = getattr(torch_npu.npu, "mstx", None)
        if mstx is not None:
            range_id = mstx.range_start(name)
    except Exception:
        mstx = None
        range_id = None

    try:
        yield
    finally:
        if mstx is not None and range_id is not None:
            mstx.range_end(range_id)


def _synchronize() -> None:
    try:
        from vllm_omni.platforms import current_omni_platform

        if current_omni_platform.is_available():
            current_omni_platform.synchronize()
    except Exception:
        return


@contextmanager
def phase(recorder: PhaseRecorder, name: str):
    if recorder.is_active(name):
        yield
        return

    recorder.push_active(name)
    _synchronize()
    start = time.perf_counter()
    try:
        with npu_range(name):
            yield
    finally:
        _synchronize()
        recorder.add(name, time.perf_counter() - start)
        recorder.pop_active(name)


def _wrap_bound_method(
    owner: Any,
    method_name: str,
    wrapper_factory: Callable[[Callable[..., Any]], Callable[..., Any]],
) -> None:
    original = getattr(owner, method_name)
    wrapped = wrapper_factory(original)
    setattr(owner, method_name, wrapped)


def _resolve_model_name(pipeline: Any, current_model: Any = None) -> str:
    model_name = getattr(current_model, "_profile_name", None)
    if model_name is not None:
        return model_name
    if hasattr(pipeline, "transformer_2") and current_model is getattr(pipeline, "transformer_2", None):
        return "transformer_2"
    return "transformer"


def _resolve_dit_phase_name(
    pipeline: Any,
    *,
    current_model: Any = None,
    positive_kwargs: dict[str, Any] | None = None,
    negative_kwargs: dict[str, Any] | None = None,
) -> str:
    if current_model is None and positive_kwargs is not None:
        current_model = positive_kwargs.get("current_model")
    if current_model is None and negative_kwargs is not None:
        current_model = negative_kwargs.get("current_model")
    return classify_dit_phase(_resolve_model_name(pipeline, current_model))


def apply_pipeline_profiling_patch(pipeline: Any, recorder: PhaseRecorder) -> None:
    if getattr(pipeline, "_wan_i2v_profile_patch_applied", False):
        return

    if hasattr(pipeline, "encode_prompt"):
        def wrap_encode_prompt(original: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(original)
            def inner(*args: Any, **kwargs: Any) -> Any:
                with phase(recorder, "TEXT_ENCODER"):
                    return original(*args, **kwargs)

            return inner

        _wrap_bound_method(pipeline, "encode_prompt", wrap_encode_prompt)

    if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "encode"):
        def wrap_vae_encode(original: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(original)
            def inner(*args: Any, **kwargs: Any) -> Any:
                with phase(recorder, "VAE_ENCODE"):
                    return original(*args, **kwargs)

            return inner

        _wrap_bound_method(pipeline.vae, "encode", wrap_vae_encode)

    if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "decode"):
        def wrap_vae_decode(original: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(original)
            def inner(*args: Any, **kwargs: Any) -> Any:
                with phase(recorder, "VAE_DECODE"):
                    return original(*args, **kwargs)

            return inner

        _wrap_bound_method(pipeline.vae, "decode", wrap_vae_decode)

    if hasattr(pipeline, "predict_noise"):
        def wrap_predict_noise(original: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(original)
            def inner(*args: Any, **kwargs: Any) -> Any:
                current_model = kwargs.get("current_model")
                if current_model is None and args:
                    current_model = args[0]
                phase_name = _resolve_dit_phase_name(pipeline, current_model=current_model)
                with phase(recorder, phase_name):
                    return original(*args, **kwargs)

            return inner

        _wrap_bound_method(pipeline, "predict_noise", wrap_predict_noise)

    if hasattr(pipeline, "predict_noise_maybe_with_cfg"):
        def wrap_predict_noise_maybe_with_cfg(original: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(original)
            def inner(*args: Any, **kwargs: Any) -> Any:
                positive_kwargs = kwargs.get("positive_kwargs")
                negative_kwargs = kwargs.get("negative_kwargs")
                if positive_kwargs is None and len(args) >= 3:
                    positive_kwargs = args[2]
                if negative_kwargs is None and len(args) >= 4:
                    negative_kwargs = args[3]
                phase_name = _resolve_dit_phase_name(
                    pipeline,
                    positive_kwargs=positive_kwargs,
                    negative_kwargs=negative_kwargs,
                )
                pipeline._wan_i2v_profile_last_dit_phase = phase_name
                with phase(recorder, phase_name):
                    return original(*args, **kwargs)

            return inner

        _wrap_bound_method(pipeline, "predict_noise_maybe_with_cfg", wrap_predict_noise_maybe_with_cfg)

    if hasattr(pipeline, "scheduler_step_maybe_with_cfg"):
        def wrap_scheduler_step_maybe_with_cfg(original: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(original)
            def inner(*args: Any, **kwargs: Any) -> Any:
                phase_name = getattr(pipeline, "_wan_i2v_profile_last_dit_phase", None)
                if phase_name is None:
                    return original(*args, **kwargs)
                with phase(recorder, phase_name):
                    return original(*args, **kwargs)

            return inner

        _wrap_bound_method(pipeline, "scheduler_step_maybe_with_cfg", wrap_scheduler_step_maybe_with_cfg)

    pipeline._wan_i2v_profile_patch_applied = True
