# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Any, Literal

from pydantic import Field
from pydantic.dataclasses import dataclass
from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.utils.hashing import safe_hash

logger = init_logger(__name__)

DiffusionProfilerKind = Literal["torch", "cuda"]


def _is_uri_path(path: str) -> bool:
    """Check if path is a URI (scheme://...), excluding Windows drive letters.

    Supports custom URI schemes like gs://, s3://, hdfs://, etc.
    These paths should not be converted to absolute paths.
    """
    if "://" in path:
        scheme = path.split("://")[0]
        # Windows drive letters are single characters (e.g., C://)
        # Valid URI schemes have more than one character
        return len(scheme) > 1
    return False


@config
@dataclass
class DiffusionProfilerConfig:
    """Dataclass which contains profiler config for the diffusion engine.

    This follows the same pattern as vLLM's ProfilerConfig, providing CLI-based
    configuration instead of relying on environment variables.
    """

    profiler: DiffusionProfilerKind | None = None
    """Which profiler to use. Defaults to None. Options are:

    - 'torch': Use PyTorch profiler.\n
    - 'cuda': Use CUDA profiler."""

    torch_profiler_dir: str = ""
    """Directory to save torch profiler traces. Note that it must be an
    absolute path."""

    torch_profiler_with_stack: bool = True
    """If `True`, enables stack tracing in the torch profiler. Enabled by default."""

    torch_profiler_with_flops: bool = True
    """If `True`, enables FLOPS counting in the torch profiler. Enabled by default
    for diffusion models where FLOPS analysis is important."""

    torch_profiler_use_gzip: bool = True
    """If `True`, saves torch profiler traces in gzip format. Enabled by default."""

    torch_profiler_record_shapes: bool = True
    """If `True`, records tensor shapes in the torch profiler. Enabled by default
    for diffusion models."""

    torch_profiler_with_memory: bool = True
    """If `True`, enables memory profiling in the torch profiler.
    Enabled by default for diffusion models."""

    torch_profiler_active_steps: int = Field(default=100000, ge=1)
    """Number of active profiling steps in the schedule. Defaults to 100000
    for long capture windows."""

    torch_profiler_warmup_steps: int = Field(default=0, ge=0)
    """Number of warmup steps before profiling starts. Defaults to 0."""

    torch_profiler_wait_steps: int = Field(default=0, ge=0)
    """Number of wait steps before warmup. Defaults to 0."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # no factors to consider.
        # this config will not affect the computation graph.
        factors: list[Any] = []
        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self) -> None:
        """Validate and normalize profiler configuration."""
        profiler_dir = self.torch_profiler_dir

        if profiler_dir and self.profiler != "torch":
            raise ValueError("torch_profiler_dir is only applicable when profiler is set to 'torch'")
        if self.profiler == "torch" and not profiler_dir:
            raise ValueError("torch_profiler_dir must be set when profiler is 'torch'")

        # Support any URI scheme (gs://, s3://, hdfs://, etc.)
        # These paths should not be converted to absolute paths
        if profiler_dir and not _is_uri_path(profiler_dir):
            self.torch_profiler_dir = os.path.abspath(os.path.expanduser(profiler_dir))
