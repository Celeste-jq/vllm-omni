# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest
import torch

from vllm_omni.diffusion.utils.wan_native import (
    _find_local_diffusers_i2v_ref_dir,
    _extract_state_dict,
    has_wan22_native_remote_candidate_layout,
    has_wan22_native_remote_layout,
    infer_wan22_native_model_class,
    is_local_wan22_native_candidate_layout,
    is_local_wan22_native_layout,
    looks_like_wan22_native_checkpoint,
    resolve_local_wan22_native_root,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_looks_like_wan22_native_checkpoint() -> None:
    assert looks_like_wan22_native_checkpoint("Wan-AI/Wan2.2-I2V-A14B")
    assert looks_like_wan22_native_checkpoint("https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B")
    assert not looks_like_wan22_native_checkpoint("Wan-AI/Wan2.2-I2V-A14B-Diffusers")


def test_infer_wan22_native_model_class() -> None:
    assert infer_wan22_native_model_class("Wan-AI/Wan2.2-I2V-A14B") == "WanImageToVideoPipeline"
    assert (
        infer_wan22_native_model_class(
            "Wan-AI/Wan2.2-Unknown",
            transformer_cfg={"image_dim": 1280, "added_kv_proj_dim": 1280},
        )
        == "WanImageToVideoPipeline"
    )


def test_infer_wan22_native_model_class_from_local_i2v_assets(tmp_path: Path) -> None:
    base = tmp_path / "wan-native-local"
    (base / "image_encoder").mkdir(parents=True)
    (base / "image_processor").mkdir(parents=True)
    (base / "image_encoder/config.json").write_text("{}\n", encoding="utf-8")
    assert infer_wan22_native_model_class(str(base)) == "WanImageToVideoPipeline"


def test_is_local_wan22_native_layout(tmp_path: Path) -> None:
    base = tmp_path / "wan-native"
    (base / "high_noise_model").mkdir(parents=True)
    (base / "low_noise_model").mkdir(parents=True)
    (base / "google/umt5-xxl").mkdir(parents=True)

    (base / "high_noise_model/config.json").write_text("{}\n", encoding="utf-8")
    (base / "low_noise_model/config.json").write_text("{}\n", encoding="utf-8")
    (base / "google/umt5-xxl/config.json").write_text("{}\n", encoding="utf-8")
    (base / "models_t5_umt5-xxl-enc-bf16.pth").write_text("x", encoding="utf-8")
    (base / "Wan2.1_VAE.pth").write_text("x", encoding="utf-8")

    assert is_local_wan22_native_layout(str(base))
    assert is_local_wan22_native_candidate_layout(str(base))


def test_is_local_wan22_native_candidate_layout_with_partial_assets(tmp_path: Path) -> None:
    base = tmp_path / "wan-native-partial"
    (base / "high_noise_model").mkdir(parents=True)
    (base / "low_noise_model").mkdir(parents=True)
    (base / "high_noise_model/config.json").write_text("{}\n", encoding="utf-8")
    (base / "low_noise_model/config.json").write_text("{}\n", encoding="utf-8")

    assert is_local_wan22_native_candidate_layout(str(base))
    assert not is_local_wan22_native_layout(str(base))


def test_tokenizer_subdir_without_model_config_json(tmp_path: Path) -> None:
    base = tmp_path / "wan-native-tokenizer-no-config"
    (base / "high_noise_model").mkdir(parents=True)
    (base / "low_noise_model").mkdir(parents=True)
    (base / "google/umt5-xxl").mkdir(parents=True)

    (base / "high_noise_model/config.json").write_text("{}\n", encoding="utf-8")
    (base / "low_noise_model/config.json").write_text("{}\n", encoding="utf-8")
    (base / "google/umt5-xxl/tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    (base / "google/umt5-xxl/spiece.model").write_text("x", encoding="utf-8")
    (base / "models_t5_umt5-xxl-enc-bf16.pth").write_text("x", encoding="utf-8")
    (base / "Wan2.1_VAE.pth").write_text("x", encoding="utf-8")

    assert is_local_wan22_native_layout(str(base))


def test_resolve_local_wan22_native_root_with_stale_adapter_dir(tmp_path: Path) -> None:
    base = tmp_path / "wan-native"
    (base / "high_noise_model").mkdir(parents=True)
    (base / "low_noise_model").mkdir(parents=True)
    (base / ".vllm_omni_wan22_native").mkdir(parents=True)
    (base / "high_noise_model/config.json").write_text("{}\n", encoding="utf-8")
    (base / "low_noise_model/config.json").write_text("{}\n", encoding="utf-8")

    resolved = resolve_local_wan22_native_root(str(base))
    assert resolved is not None
    assert Path(resolved) == base


def test_extract_state_dict_nested_tensor_tree() -> None:
    payload = {
        "model": {
            "encoder": {
                "block": {
                    "0": {
                        "layer": {
                            "0": {
                                "SelfAttention": {
                                    "q": {"weight": torch.zeros((8, 8))},
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    state = _extract_state_dict(payload)
    assert "encoder.block.0.layer.0.SelfAttention.q.weight" in state


def test_find_local_diffusers_i2v_ref_dir_via_env(tmp_path: Path, monkeypatch) -> None:
    native = tmp_path / "Wan2.2-I2V-A14B"
    ref = tmp_path / "Wan2.2-I2V-A14B-Diffusers"
    native.mkdir(parents=True)
    (ref / "text_encoder").mkdir(parents=True)
    (ref / "tokenizer").mkdir(parents=True)
    (ref / "text_encoder/config.json").write_text("{}\n", encoding="utf-8")
    (ref / "text_encoder/model.safetensors").write_text("x", encoding="utf-8")
    (ref / "tokenizer/tokenizer_config.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setenv("VLLM_OMNI_WAN_I2V_DIFFUSERS_REF", str(ref))
    assert _find_local_diffusers_i2v_ref_dir(native) == ref


def test_resolve_local_wan22_native_root_nested_dir(tmp_path: Path) -> None:
    base = tmp_path / "Wan2.2-I2V-A14B"
    nested = base / "snapshots" / "abcdef"
    (nested / "high_noise_model").mkdir(parents=True)
    (nested / "low_noise_model").mkdir(parents=True)
    (nested / "high_noise_model/config.json").write_text("{}\n", encoding="utf-8")
    (nested / "low_noise_model/config.json").write_text("{}\n", encoding="utf-8")

    resolved = resolve_local_wan22_native_root(str(base))
    assert resolved is not None
    assert Path(resolved) == nested


def test_has_wan22_native_remote_layout_short_circuit_local(tmp_path: Path) -> None:
    base = tmp_path / "wan-native"
    (base / "high_noise_model").mkdir(parents=True)
    (base / "low_noise_model").mkdir(parents=True)
    (base / "google/umt5-xxl").mkdir(parents=True)

    (base / "high_noise_model/config.json").write_text("{}\n", encoding="utf-8")
    (base / "low_noise_model/config.json").write_text("{}\n", encoding="utf-8")
    (base / "google/umt5-xxl/config.json").write_text("{}\n", encoding="utf-8")
    (base / "models_t5_umt5-xxl-enc-bf16.pth").write_text("x", encoding="utf-8")
    (base / "Wan2.1_VAE.pth").write_text("x", encoding="utf-8")

    assert has_wan22_native_remote_layout(str(base))
    assert has_wan22_native_remote_candidate_layout(str(base))
