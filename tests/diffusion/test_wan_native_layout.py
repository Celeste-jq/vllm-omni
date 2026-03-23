# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest

from vllm_omni.diffusion.utils.wan_native import (
    has_wan22_native_t2v_remote_layout,
    is_local_wan22_native_t2v_layout,
    looks_like_wan22_native_t2v,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_looks_like_wan22_native_t2v() -> None:
    assert looks_like_wan22_native_t2v("Wan-AI/Wan2.2-T2V-A14B")
    assert looks_like_wan22_native_t2v("https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B")
    assert not looks_like_wan22_native_t2v("Wan-AI/Wan2.2-T2V-A14B-Diffusers")


def test_is_local_wan22_native_t2v_layout(tmp_path: Path) -> None:
    base = tmp_path / "wan-native"
    (base / "high_noise_model").mkdir(parents=True)
    (base / "low_noise_model").mkdir(parents=True)
    (base / "google/umt5-xxl").mkdir(parents=True)

    (base / "high_noise_model/config.json").write_text("{}\n", encoding="utf-8")
    (base / "low_noise_model/config.json").write_text("{}\n", encoding="utf-8")
    (base / "google/umt5-xxl/config.json").write_text("{}\n", encoding="utf-8")
    (base / "models_t5_umt5-xxl-enc-bf16.pth").write_text("x", encoding="utf-8")
    (base / "Wan2.1_VAE.pth").write_text("x", encoding="utf-8")

    assert is_local_wan22_native_t2v_layout(str(base))


def test_has_wan22_native_t2v_remote_layout_short_circuit_local(tmp_path: Path) -> None:
    base = tmp_path / "wan-native"
    (base / "high_noise_model").mkdir(parents=True)
    (base / "low_noise_model").mkdir(parents=True)
    (base / "google/umt5-xxl").mkdir(parents=True)

    (base / "high_noise_model/config.json").write_text("{}\n", encoding="utf-8")
    (base / "low_noise_model/config.json").write_text("{}\n", encoding="utf-8")
    (base / "google/umt5-xxl/config.json").write_text("{}\n", encoding="utf-8")
    (base / "models_t5_umt5-xxl-enc-bf16.pth").write_text("x", encoding="utf-8")
    (base / "Wan2.1_VAE.pth").write_text("x", encoding="utf-8")

    assert has_wan22_native_t2v_remote_layout(str(base))
