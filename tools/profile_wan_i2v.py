#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.wan_i2v_profile_patch import (
    PhaseRecorder,
    apply_pipeline_profiling_patch,
    write_aggregate_summary,
    write_summary_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile Wan2.2-I2V in vLLM-Omni with MindIE-style module tags.")
    parser.add_argument("--model", default="Wan-AI/Wan2.2-I2V-A14B-Diffusers", help="Wan2.2 I2V model ID or path.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--prompt", default="", help="Text prompt.")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--guidance-scale", type=float, default=5.0, help="CFG scale.")
    parser.add_argument(
        "--guidance-scale-high",
        type=float,
        default=None,
        help="Optional separate CFG scale for the high-noise branch.",
    )
    parser.add_argument("--height", type=int, default=None, help="Video height. Auto-inferred when omitted.")
    parser.add_argument("--width", type=int, default=None, help="Video width. Auto-inferred when omitted.")
    parser.add_argument("--num-frames", type=int, default=81, help="Number of frames.")
    parser.add_argument("--num-inference-steps", type=int, default=40, help="Number of denoising steps.")
    parser.add_argument("--boundary-ratio", type=float, default=0.875, help="MoE boundary split ratio.")
    parser.add_argument("--frame-rate", type=float, default=None, help="Model frame rate. Defaults to fps.")
    parser.add_argument("--flow-shift", type=float, default=5.0, help="Scheduler flow_shift.")
    parser.add_argument(
        "--sample-solver",
        type=str,
        default="unipc",
        choices=["unipc", "euler"],
        help="Sampling solver.",
    )
    parser.add_argument("--fps", type=int, default=16, help="Output video fps.")
    parser.add_argument("--output", type=str, default="i2v_output.mp4", help="Output video path.")
    parser.add_argument("--ulysses-degree", type=int, default=1, help="Ulysses sequence parallel degree.")
    parser.add_argument("--ring-degree", type=int, default=1, help="Ring sequence parallel degree.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--cfg-parallel-size", type=int, default=1, choices=[1, 2], help="CFG parallel size.")
    parser.add_argument("--vae-patch-parallel-size", type=int, default=1, help="VAE patch parallel size.")
    parser.add_argument("--use-hsdp", action="store_true", help="Enable HSDP.")
    parser.add_argument("--hsdp-shard-size", type=int, default=-1, help="HSDP shard size.")
    parser.add_argument("--hsdp-replicate-size", type=int, default=1, help="HSDP replicate size.")
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
        help="Also enable the built-in coarse diffusion pipeline profiler.",
    )
    parser.add_argument(
        "--profiling-output-dir",
        type=str,
        default="profiling_runs/wan_i2v_profile",
        help="Root directory for traces and summaries.",
    )
    parser.add_argument("--warmup-runs", type=int, default=1, help="Number of warmup runs before profiling.")
    parser.add_argument("--profile-memory", action="store_true", help="Enable profile_memory in torch_npu profiler.")
    parser.add_argument("--with-stack", action="store_true", help="Enable with_modules/stack-like profiler metadata.")
    return parser.parse_args()


def build_cache_config(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.cache_backend == "cache_dit":
        return {
            "Fn_compute_blocks": 1,
            "Bn_compute_blocks": 0,
            "max_warmup_steps": 4,
            "residual_diff_threshold": 0.24,
            "max_continuous_cached_steps": 3,
            "enable_taylorseer": False,
            "taylorseer_order": 1,
            "scm_steps_mask_policy": None,
            "scm_steps_policy": "dynamic",
        }
    if args.cache_backend == "tea_cache":
        return {
            "rel_l1_thresh": 0.2,
        }
    return None


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _device_memory_backend():
    import torch

    for backend_name in ("npu", "cuda", "xpu", "musa"):
        backend = getattr(torch, backend_name, None)
        if backend is None:
            continue
        is_available = getattr(backend, "is_available", None)
        if callable(is_available) and is_available():
            return backend_name, backend
    return None, None


def reset_peak_memory_stats_if_available() -> None:
    _, backend = _device_memory_backend()
    if backend is None:
        return
    reset_peak_memory_stats = getattr(backend, "reset_peak_memory_stats", None)
    if callable(reset_peak_memory_stats):
        reset_peak_memory_stats()


def collect_memory_stats() -> dict[str, Any]:
    backend_name, backend = _device_memory_backend()
    if backend is None:
        return {
            "device_type": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }

    peak_allocated = getattr(backend, "max_memory_allocated", None)
    peak_reserved = getattr(backend, "max_memory_reserved", None)
    return {
        "device_type": backend_name,
        "peak_allocated_bytes": int(peak_allocated()) if callable(peak_allocated) else None,
        "peak_reserved_bytes": int(peak_reserved()) if callable(peak_reserved) else None,
    }


def write_run_metadata(rank_dir: Path, payload: dict[str, Any]) -> None:
    rank_dir.mkdir(parents=True, exist_ok=True)
    (rank_dir / "run_metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def aggregate_run_metadata_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    peak_allocated_values = [
        payload.get("memory", {}).get("peak_allocated_bytes")
        for payload in payloads
        if payload.get("memory", {}).get("peak_allocated_bytes") is not None
    ]
    peak_reserved_values = [
        payload.get("memory", {}).get("peak_reserved_bytes")
        for payload in payloads
        if payload.get("memory", {}).get("peak_reserved_bytes") is not None
    ]
    wall_times = [payload.get("wall_time_sec") for payload in payloads if payload.get("wall_time_sec") is not None]

    return {
        "world_size": len(payloads),
        "max_wall_time_sec": max(wall_times) if wall_times else None,
        "avg_wall_time_sec": sum(wall_times) / len(wall_times) if wall_times else None,
        "memory": {
            "max_peak_allocated_bytes": max(peak_allocated_values) if peak_allocated_values else None,
            "max_peak_reserved_bytes": max(peak_reserved_values) if peak_reserved_values else None,
        },
        "ranks": payloads,
    }


def write_aggregate_run_metadata(root_dir: Path, payloads: list[dict[str, Any]]) -> None:
    aggregate_payload = aggregate_run_metadata_payloads(payloads)
    (root_dir / "aggregate_run_metadata.json").write_text(
        json.dumps(aggregate_payload, indent=2, sort_keys=True) + "\n"
    )


def _is_distributed_ready() -> bool:
    import torch

    return torch.distributed.is_available() and torch.distributed.is_initialized()


def barrier_if_needed() -> None:
    import torch

    if _is_distributed_ready():
        torch.distributed.barrier()


def rank_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def calculate_dimensions(image: PIL.Image.Image, max_area: int = 480 * 832, mod_value: int = 16) -> tuple[int, int]:
    import numpy as np

    aspect_ratio = image.height / image.width
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    return height, width


def build_runtime(args: argparse.Namespace):
    import torch

    from vllm.config import CompilationConfig, DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.v1.worker.workspace import init_workspace_manager

    from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
    from vllm_omni.diffusion.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.platforms import current_omni_platform

    rank, local_rank, world_size = rank_info()
    parallel_config = DiffusionParallelConfig(
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        cfg_parallel_size=args.cfg_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        vae_patch_parallel_size=args.vae_patch_parallel_size,
        use_hsdp=args.use_hsdp,
        hsdp_shard_size=args.hsdp_shard_size,
        hsdp_replicate_size=args.hsdp_replicate_size,
    )
    if world_size != parallel_config.world_size:
        raise ValueError(
            f"WORLD_SIZE={world_size} 与并行配置 world_size={parallel_config.world_size} 不一致。"
            "请检查 torchrun 参数与并行配置。"
        )

    od_config = OmniDiffusionConfig.from_kwargs(
        model=args.model,
        model_class_name="WanImageToVideoPipeline",
        dtype=torch.bfloat16,
        parallel_config=parallel_config,
        enable_cpu_offload=args.enable_cpu_offload,
        enable_layerwise_offload=args.enable_layerwise_offload,
        vae_use_slicing=args.vae_use_slicing,
        vae_use_tiling=args.vae_use_tiling,
        boundary_ratio=args.boundary_ratio,
        flow_shift=args.flow_shift,
        enforce_eager=args.enforce_eager,
        cache_backend=args.cache_backend,
        cache_config=build_cache_config(args) or {},
        enable_diffusion_pipeline_profiler=args.enable_diffusion_pipeline_profiler,
    )

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(od_config.master_port))
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    device = current_omni_platform.get_torch_device(local_rank)
    current_omni_platform.set_device(device)

    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(),
        device_config=DeviceConfig(device=device),
    )
    vllm_config.parallel_config.tensor_parallel_size = parallel_config.tensor_parallel_size
    vllm_config.parallel_config.data_parallel_size = parallel_config.data_parallel_size
    vllm_config.parallel_config.enable_expert_parallel = parallel_config.enable_expert_parallel

    with (
        set_forward_context(vllm_config=vllm_config, omni_diffusion_config=od_config),
        set_current_vllm_config(vllm_config),
    ):
        init_distributed_environment(world_size=world_size, rank=rank, local_rank=local_rank)
        initialize_model_parallel(
            data_parallel_size=parallel_config.data_parallel_size,
            cfg_parallel_size=parallel_config.cfg_parallel_size,
            sequence_parallel_size=parallel_config.sequence_parallel_size,
            ulysses_degree=parallel_config.ulysses_degree,
            ring_degree=parallel_config.ring_degree,
            tensor_parallel_size=parallel_config.tensor_parallel_size,
            pipeline_parallel_size=parallel_config.pipeline_parallel_size,
            fully_shard_degree=parallel_config.hsdp_shard_size if parallel_config.use_hsdp else 1,
            hsdp_replicate_size=parallel_config.hsdp_replicate_size if parallel_config.use_hsdp else 1,
            enable_expert_parallel=parallel_config.enable_expert_parallel,
        )
        init_workspace_manager(device)

    return {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": device,
        "od_config": od_config,
        "vllm_config": vllm_config,
    }


def load_runner(runtime: dict[str, Any]):
    from vllm.config import set_current_vllm_config

    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.registry import get_diffusion_post_process_func, get_diffusion_pre_process_func
    from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner

    runner = DiffusionModelRunner(
        vllm_config=runtime["vllm_config"],
        od_config=runtime["od_config"],
        device=runtime["device"],
    )
    with (
        set_forward_context(vllm_config=runtime["vllm_config"], omni_diffusion_config=runtime["od_config"]),
        set_current_vllm_config(runtime["vllm_config"]),
    ):
        runner.load_model(load_format=runtime["od_config"].diffusion_load_format)

    return runner, get_diffusion_pre_process_func(runtime["od_config"]), get_diffusion_post_process_func(
        runtime["od_config"]
    )


def build_request(args: argparse.Namespace, runtime: dict[str, Any]):
    import torch

    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    frame_rate = args.frame_rate if args.frame_rate is not None else float(args.fps)
    generator = torch.Generator(device=runtime["device"]).manual_seed(args.seed)
    prompt = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "multi_modal_data": {"image": args.image},
    }
    sampling_params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        generator=generator,
        guidance_scale=args.guidance_scale,
        guidance_scale_2=args.guidance_scale_high,
        boundary_ratio=args.boundary_ratio,
        num_inference_steps=args.num_inference_steps,
        num_frames=args.num_frames,
        frame_rate=frame_rate,
        extra_args={
            "sample_solver": args.sample_solver,
            "flow_shift": args.flow_shift,
        },
    )
    return OmniDiffusionRequest(
        prompts=[prompt],
        sampling_params=sampling_params,
        request_ids=[f"wan_i2v_rank{runtime['rank']}"],
        request_id=f"wan_i2v_rank{runtime['rank']}",
    )


def maybe_preprocess_request(request, pre_process_func):
    if pre_process_func is None:
        return request
    return pre_process_func(request)


def execute_request(runner, runtime: dict[str, Any], request):
    from vllm.config import set_current_vllm_config

    from vllm_omni.diffusion.forward_context import set_forward_context

    with (
        set_forward_context(vllm_config=runtime["vllm_config"], omni_diffusion_config=runtime["od_config"]),
        set_current_vllm_config(runtime["vllm_config"]),
    ):
        return runner.execute_model(request)


def maybe_postprocess_output(output, request, post_process_func, enable_cpu_offload: bool, rank: int = 0):
    import torch

    output_data = output.output
    if enable_cpu_offload and isinstance(output_data, torch.Tensor) and output_data.device.type != "cpu":
        output_data = output_data.cpu()

    if post_process_func is None or rank != 0:
        return output_data

    if "sampling_params" in inspect.signature(post_process_func).parameters:
        return post_process_func(output_data, sampling_params=request.sampling_params)
    return post_process_func(output_data)


def extract_video_payload(processed: Any) -> Any:
    if isinstance(processed, dict) and "video" in processed:
        return processed["video"]
    return processed


def normalize_frame(frame: Any) -> np.ndarray:
    import numpy as np
    import PIL.Image
    import torch

    if isinstance(frame, torch.Tensor):
        frame_tensor = frame.detach().cpu()
        if frame_tensor.dim() == 4 and frame_tensor.shape[0] == 1:
            frame_tensor = frame_tensor[0]
        if frame_tensor.dim() == 3 and frame_tensor.shape[0] in (3, 4):
            frame_tensor = frame_tensor.permute(1, 2, 0)
        if frame_tensor.is_floating_point():
            frame_tensor = frame_tensor.clamp(-1, 1) * 0.5 + 0.5
        return frame_tensor.float().numpy()
    if isinstance(frame, np.ndarray):
        frame_array = frame
        if frame_array.ndim == 4 and frame_array.shape[0] == 1:
            frame_array = frame_array[0]
        if np.issubdtype(frame_array.dtype, np.integer):
            frame_array = frame_array.astype(np.float32) / 255.0
        return frame_array
    if isinstance(frame, PIL.Image.Image):
        return np.asarray(frame).astype(np.float32) / 255.0
    return np.asarray(frame)


def ensure_frame_list(video_array: Any) -> Any:
    import numpy as np

    if isinstance(video_array, list):
        if len(video_array) == 0:
            return video_array
        first_item = video_array[0]
        if isinstance(first_item, np.ndarray):
            if first_item.ndim == 5:
                return list(first_item[0])
            if first_item.ndim == 4:
                return list(first_item)
            if first_item.ndim == 3:
                return video_array
        return video_array
    if isinstance(video_array, np.ndarray):
        if video_array.ndim == 5:
            return list(video_array[0])
        if video_array.ndim == 4:
            return list(video_array)
        if video_array.ndim == 3:
            return [video_array]
    return video_array


def save_video(video: Any, output_path: Path, fps: int) -> None:
    import numpy as np
    import torch

    from diffusers.utils import export_to_video

    if isinstance(video, torch.Tensor):
        video_tensor = video.detach().cpu()
        if video_tensor.dim() == 5:
            if video_tensor.shape[1] in (3, 4):
                video_tensor = video_tensor[0].permute(1, 2, 3, 0)
            else:
                video_tensor = video_tensor[0]
        elif video_tensor.dim() == 4 and video_tensor.shape[0] in (3, 4):
            video_tensor = video_tensor.permute(1, 2, 3, 0)
        if video_tensor.is_floating_point():
            video_tensor = video_tensor.clamp(-1, 1) * 0.5 + 0.5
        video_array = video_tensor.float().numpy()
    elif isinstance(video, np.ndarray):
        video_array = video[0] if video.ndim == 5 else video
        if np.issubdtype(video_array.dtype, np.integer):
            video_array = video_array.astype(np.float32) / 255.0
    elif isinstance(video, list):
        if len(video) == 0:
            raise ValueError("No video frames found in output.")
        video_array = [normalize_frame(frame) for frame in video]
    else:
        video_array = video

    video_array = ensure_frame_list(video_array)
    ensure_parent(output_path)
    export_to_video(video_array, str(output_path), fps=fps)


@contextmanager
def torch_npu_profile_context(trace_dir: Path, args: argparse.Namespace):
    try:
        import torch_npu
    except ImportError as exc:
        raise RuntimeError("当前环境缺少 torch_npu，无法采集 NPU profiling。") from exc

    trace_dir.mkdir(parents=True, exist_ok=True)
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        msprof_tx=True,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        l2_cache=False,
        op_attr=False,
        data_simplification=True,
        record_op_args=False,
        gc_detect_threshold=None,
    )
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        profile_memory=args.profile_memory,
        with_modules=args.with_stack,
        experimental_config=experimental_config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(trace_dir)),
    ) as prof:
        yield prof


def run_once(args: argparse.Namespace, runtime: dict[str, Any], runner, pre_process_func, post_process_func):
    request = build_request(args, runtime)
    request = maybe_preprocess_request(request, pre_process_func)
    output = execute_request(runner, runtime, request)
    return maybe_postprocess_output(
        output,
        request,
        post_process_func,
        runtime["od_config"].enable_cpu_offload,
        rank=runtime["rank"],
    )


def aggregate_rank_summaries(root_dir: Path, world_size: int) -> None:
    payloads: list[dict[str, Any]] = []
    for rank in range(world_size):
        summary_path = root_dir / f"rank{rank}" / "summary.json"
        if summary_path.exists():
            payloads.append(json.loads(summary_path.read_text()))
    if payloads:
        write_aggregate_summary(root_dir, payloads)


def aggregate_rank_run_metadata(root_dir: Path, world_size: int) -> None:
    payloads: list[dict[str, Any]] = []
    for rank in range(world_size):
        metadata_path = root_dir / f"rank{rank}" / "run_metadata.json"
        if metadata_path.exists():
            payloads.append(json.loads(metadata_path.read_text()))
    if payloads:
        write_aggregate_run_metadata(root_dir, payloads)


def main() -> None:
    args = parse_args()
    import PIL.Image

    from vllm_omni.diffusion.distributed.parallel_state import destroy_distributed_env
    from vllm_omni.platforms import current_omni_platform

    if args.height is None or args.width is None:
        image = PIL.Image.open(args.image).convert("RGB")
        calc_height, calc_width = calculate_dimensions(image)
        args.height = args.height or calc_height
        args.width = args.width or calc_width

    root_dir = Path(args.profiling_output_dir)
    runtime = build_runtime(args)
    rank_dir = root_dir / f"rank{runtime['rank']}"
    trace_dir = rank_dir / "trace"
    recorder = PhaseRecorder()

    try:
        runner, pre_process_func, post_process_func = load_runner(runtime)
        apply_pipeline_profiling_patch(runner.pipeline, recorder)

        for _ in range(max(args.warmup_runs, 0)):
            _ = run_once(args, runtime, runner, pre_process_func, post_process_func)
            recorder.records.clear()
            barrier_if_needed()
            if current_omni_platform.is_available():
                current_omni_platform.empty_cache()

        barrier_if_needed()
        reset_peak_memory_stats_if_available()
        start_time = time.perf_counter()
        with torch_npu_profile_context(trace_dir, args):
            processed = run_once(args, runtime, runner, pre_process_func, post_process_func)
        wall_time_sec = time.perf_counter() - start_time
        write_summary_outputs(rank_dir, recorder.records)
        write_run_metadata(
            rank_dir,
            {
                "rank": runtime["rank"],
                "local_rank": runtime["local_rank"],
                "world_size": runtime["world_size"],
                "wall_time_sec": wall_time_sec,
                "memory": collect_memory_stats(),
                "parallel": {
                    "ulysses_degree": args.ulysses_degree,
                    "ring_degree": args.ring_degree,
                    "tensor_parallel_size": args.tensor_parallel_size,
                    "cfg_parallel_size": args.cfg_parallel_size,
                    "vae_patch_parallel_size": args.vae_patch_parallel_size,
                    "use_hsdp": args.use_hsdp,
                    "hsdp_shard_size": args.hsdp_shard_size,
                    "hsdp_replicate_size": args.hsdp_replicate_size,
                },
                "request": {
                    "height": args.height,
                    "width": args.width,
                    "num_frames": args.num_frames,
                    "num_inference_steps": args.num_inference_steps,
                    "prompt": args.prompt,
                    "image": args.image,
                },
            },
        )

        barrier_if_needed()
        if runtime["rank"] == 0:
            processed_video = extract_video_payload(processed)
            save_video(processed_video, Path(args.output), fps=args.fps)
            aggregate_rank_summaries(root_dir, runtime["world_size"])
            aggregate_rank_run_metadata(root_dir, runtime["world_size"])
            print(f"[Profiling] trace dir: {trace_dir}")
            print(f"[Profiling] rank0 summary: {rank_dir / 'summary.json'}")
            if (root_dir / "aggregate_summary.json").exists():
                print(f"[Profiling] aggregate summary: {root_dir / 'aggregate_summary.json'}")
            if (root_dir / "aggregate_run_metadata.json").exists():
                print(f"[Profiling] aggregate run metadata: {root_dir / 'aggregate_run_metadata.json'}")
            print(f"[Profiling] saved video: {args.output}")
    finally:
        destroy_distributed_env()


if __name__ == "__main__":
    main()
