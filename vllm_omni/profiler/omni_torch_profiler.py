# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import subprocess
from typing import Literal

import torch
from typing_extensions import override
from vllm.config import ProfilerConfig
from vllm.config.profiler import _is_uri_path
from vllm.logger import init_logger
from vllm.profiler.wrapper import WorkerProfiler

from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

TorchProfilerActivity = Literal["CPU", "CUDA", "XPU", "NPU"]
TorchProfilerActivityMap = {
    "CPU": torch.profiler.ProfilerActivity.CPU,
    "CUDA": torch.profiler.ProfilerActivity.CUDA,
    "XPU": torch.profiler.ProfilerActivity.XPU,
}


class OmniTorchProfilerWrapper(WorkerProfiler):
    """Omni-specific torch profiler that inherits vLLM's WorkerProfiler lifecycle.

    Adds on top of WorkerProfiler:
    - Custom trace file naming with stage/rank info
    - Background gzip compression via subprocess
    - Returns trace file paths from get_results() for orchestrator collection
    - NPU profiler support via torch_npu when running on Ascend NPU
    """

    def __init__(
        self,
        profiler_config: ProfilerConfig,
        worker_name: str,
        local_rank: int,
        activities: list[TorchProfilerActivity] | None = None,
    ) -> None:
        super().__init__(profiler_config)

        self._is_npu = current_omni_platform.is_npu()

        if activities is None:
            if self._is_npu:
                activities = ["CPU", "NPU"]
            else:
                activities = ["CPU", "CUDA"]

        self.local_rank = local_rank
        self.profiler_config = profiler_config
        self._trace_dir = profiler_config.torch_profiler_dir
        self._use_gzip = profiler_config.torch_profiler_use_gzip
        self._trace_filename: str | None = None
        self._trace_path: str | None = None
        self._table_path: str | None = None

        if local_rank in (None, 0):
            logger.info_once(
                "Omni torch profiling enabled. Traces will be saved to: %s",
                self._trace_dir,
                scope="local",
            )

        self.dump_cpu_time_total = "CPU" in activities and len(activities) == 1

        if self._is_npu:
            self.profiler = self._create_npu_profiler(profiler_config, activities)
        else:
            self.profiler = torch.profiler.profile(
                activities=[TorchProfilerActivityMap[a] for a in activities],
                record_shapes=profiler_config.torch_profiler_record_shapes,
                profile_memory=profiler_config.torch_profiler_with_memory,
                with_stack=profiler_config.torch_profiler_with_stack,
                with_flops=profiler_config.torch_profiler_with_flops,
                on_trace_ready=self._on_trace_ready,
            )

    def _create_npu_profiler(
        self,
        profiler_config: ProfilerConfig,
        activities: list[TorchProfilerActivity],
    ):
        """Create NPU-specific profiler using torch_npu.profiler."""
        import torch_npu

        # Map activity names to torch_npu profiler activities
        npu_activities = []
        for activity in activities:
            if activity == "CPU":
                npu_activities.append(torch_npu.profiler.ProfilerActivity.CPU)
            elif activity == "NPU":
                npu_activities.append(torch_npu.profiler.ProfilerActivity.NPU)

        # NPU-specific experimental config for detailed profiling
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=torch_npu.profiler.ExportType.Text,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            msprof_tx=False,
            aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
            l2_cache=False,
            op_attr=False,
            data_simplification=True,
            record_op_args=False,
            gc_detect_threshold=None,
        )

        # Set up trace directory for NPU - tensorboard_trace_handler creates
        # its own subdirectory structure
        rank = self.local_rank
        npu_trace_dir = os.path.join(self._trace_dir, f"npu_rank{rank}")
        os.makedirs(npu_trace_dir, exist_ok=True)
        self._trace_path = npu_trace_dir

        return torch_npu.profiler.profile(
            activities=npu_activities,
            with_stack=False,
            profile_memory=profiler_config.torch_profiler_with_memory,
            # NOTE: torch_npu.profiler.with_modules is equivalent to
            # torch.profiler.with_stack. The with_stack option in
            # torch_npu.profiler introduces significant time overhead.
            with_modules=profiler_config.torch_profiler_with_stack,
            experimental_config=experimental_config,
            # Use tensorboard_trace_handler directly - NPU profiler expects
            # this specific handler format
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(npu_trace_dir),
        )

    def set_trace_filename(self, filename: str) -> None:
        """Set the trace filename before starting profiling.

        Args:
            filename: Base filename without extension or rank suffix.
                      e.g. "stage_0_llm_1234567890"
                      Can also be a full path (e.g. from diffusion engine).
        """
        self._trace_filename = filename

    def _on_trace_ready(self, prof) -> None:
        """Custom trace handler: export chrome trace with omni naming."""
        rank = self.local_rank
        filename = self._trace_filename or f"omni_{os.getpid()}"
        # If filename already contains a directory, use as-is (e.g. from
        # diffusion engine which builds full path). Otherwise join with trace_dir.
        if os.path.dirname(filename):
            json_file = f"{filename}_rank{rank}.json"
        else:
            json_file = os.path.join(self._trace_dir, f"{filename}_rank{rank}.json")

        os.makedirs(os.path.dirname(json_file), exist_ok=True)

        try:
            prof.export_chrome_trace(json_file)
            logger.info("[Rank %s] Trace exported to %s", rank, json_file)

            if self._use_gzip:
                try:
                    subprocess.Popen(["gzip", "-f", json_file])
                    logger.info(
                        "[Rank %s] Triggered background compression for %s",
                        rank,
                        json_file,
                    )
                    self._trace_path = f"{json_file}.gz"
                except Exception as compress_err:
                    logger.warning(
                        "[Rank %s] Background gzip failed to start: %s",
                        rank,
                        compress_err,
                    )
                    self._trace_path = json_file
            else:
                self._trace_path = json_file

        except Exception as e:
            logger.warning("[Rank %s] Failed to export trace: %s", rank, e)

    @override
    def _start(self) -> None:
        self.profiler.start()

    @override
    def _stop(self) -> None:
        """Stop profiler, export trace via on_trace_ready, and dump table."""
        self.profiler.stop()

        rank = self.local_rank

        # NPU profiler doesn't support key_averages() - trace data needs
        # offline parsing using torch_npu.profiler.profiler.analyse()
        if self._is_npu:
            if rank == 0:
                logger.info(
                    "NPU profiler stopped. Use offline parsing to analyze: "
                    "from torch_npu.profiler.profiler import analyse; "
                    "analyse('%s')",
                    self._trace_path or self._trace_dir,
                )
            return

        if self.profiler_config.torch_profiler_dump_cuda_time_total:
            profiler_dir = self.profiler_config.torch_profiler_dir
            sort_key = "self_cuda_time_total"
            table = self.profiler.key_averages().table(sort_by=sort_key)

            if not _is_uri_path(profiler_dir):
                table_file = os.path.join(profiler_dir, f"profiler_out_{rank}.txt")
                with open(table_file, "w") as f:
                    print(table, file=f)
                self._table_path = table_file

            if rank == 0:
                print(table)

        if self.dump_cpu_time_total and rank == 0:
            logger.info(self.profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=50))

    @override
    def annotate_context_manager(self, name: str):
        return torch.profiler.record_function(name)

    def get_results(self) -> dict:
        """Return collected trace and table paths after stop."""
        return {
            "trace": self._trace_path,
            "table": self._table_path,
        }
