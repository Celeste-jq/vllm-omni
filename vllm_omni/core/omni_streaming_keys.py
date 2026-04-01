# SPDX-License-Identifier: Apache-2.0
"""Canonical pooler keys for multi-step upstream streaming signals.

VoxCPM uses these flags on the Stage0 AR latent generator to tell the stage-input
processor whether more latent chunks follow.
"""

from __future__ import annotations

from typing import Any

import torch

OMNI_STREAM_CONTINUE = "omni_stream_continue"
OMNI_STREAM_GEN_EXHAUSTED = "omni_stream_gen_exhausted"


def _tensorish_flag_truthy(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, torch.Tensor):
        if val.numel() == 0:
            return False
        return bool(val.reshape(-1)[0].item() != 0)
    return bool(val)


def pooler_stream_continues(pooler: dict | None) -> bool:
    if not isinstance(pooler, dict):
        return False
    c = pooler.get(OMNI_STREAM_CONTINUE)
    if c is None:
        return False
    if isinstance(c, torch.Tensor):
        if c.numel() == 0:
            return False
        return bool(c.reshape(-1)[0].item() != 0)
    return bool(c)


def pooler_stream_gen_exhausted(pooler: dict | None) -> bool:
    """True when the latent iterator is exhausted (terminal empty step or explicit flag)."""
    if not isinstance(pooler, dict):
        return False
    return _tensorish_flag_truthy(pooler.get(OMNI_STREAM_GEN_EXHAUSTED))
