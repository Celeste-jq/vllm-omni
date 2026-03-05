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

logger = init_logger(__name__)

TorchProfilerActivity = Literal["CPU", "CUDA", "XPU"]
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
    """

    def __init__(
        self,
        profiler_config: ProfilerConfig,
        worker_name: str,
        local_rank: int,
        activities: list[TorchProfilerActivity] | None = None,
    ) -> None:
        super().__init__(profiler_config)

        if activities is None:
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
        self.profiler = torch.profiler.profile(
            activities=[TorchProfilerActivityMap[a] for a in activities],
            record_shapes=profiler_config.torch_profiler_record_shapes,
            profile_memory=profiler_config.torch_profiler_with_memory,
            with_stack=profiler_config.torch_profiler_with_stack,
            with_flops=profiler_config.torch_profiler_with_flops,
            on_trace_ready=self._on_trace_ready,
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
