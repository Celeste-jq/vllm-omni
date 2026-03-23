# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.models.wan2_2 import wan2_2_transformer as wan_transformer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestWanFastLayerNormImplMode:
    def test_default_enable_flag_is_true(self, monkeypatch):
        monkeypatch.delenv(wan_transformer._WAN_FAST_LAYERNORM_ENABLE_ENV, raising=False)
        assert wan_transformer._is_wan_fast_layernorm_enabled() is True

    def test_enable_flag_can_disable_fast_path(self, monkeypatch):
        monkeypatch.setenv(wan_transformer._WAN_FAST_LAYERNORM_ENABLE_ENV, "0")
        assert wan_transformer._is_wan_fast_layernorm_enabled() is False

        monkeypatch.setenv(wan_transformer._WAN_FAST_LAYERNORM_ENABLE_ENV, "false")
        assert wan_transformer._is_wan_fast_layernorm_enabled() is False

    def test_default_impl_mode_is_zero(self, monkeypatch):
        monkeypatch.delenv(wan_transformer._WAN_FAST_LAYERNORM_IMPL_MODE_ENV, raising=False)
        assert wan_transformer._get_wan_fast_layernorm_impl_mode() == 0

    def test_valid_impl_mode_from_env(self, monkeypatch):
        monkeypatch.setenv(wan_transformer._WAN_FAST_LAYERNORM_IMPL_MODE_ENV, "2")
        assert wan_transformer._get_wan_fast_layernorm_impl_mode() == 2

    def test_invalid_impl_mode_falls_back_to_zero(self, monkeypatch):
        monkeypatch.setenv(wan_transformer._WAN_FAST_LAYERNORM_IMPL_MODE_ENV, "3")
        assert wan_transformer._get_wan_fast_layernorm_impl_mode() == 0

        monkeypatch.setenv(wan_transformer._WAN_FAST_LAYERNORM_IMPL_MODE_ENV, "invalid")
        assert wan_transformer._get_wan_fast_layernorm_impl_mode() == 0


class TestWanAdaLayerNorm:
    def test_default_enable_flag_is_true(self, monkeypatch):
        monkeypatch.delenv(wan_transformer._WAN_ADALAYERNORM_ENABLE_ENV, raising=False)
        assert wan_transformer._is_wan_adalayernorm_enabled() is True

    def test_enable_flag_can_disable_adalayernorm(self, monkeypatch):
        monkeypatch.setenv(wan_transformer._WAN_ADALAYERNORM_ENABLE_ENV, "0")
        assert wan_transformer._is_wan_adalayernorm_enabled() is False

        monkeypatch.setenv(wan_transformer._WAN_ADALAYERNORM_ENABLE_ENV, "false")
        assert wan_transformer._is_wan_adalayernorm_enabled() is False

    def test_npu_uses_mindiesd_adalayernorm_when_available(self, monkeypatch):
        layernorm = torch.nn.LayerNorm(8, eps=1e-6)
        x = torch.randn(1, 2, 8, dtype=torch.float32)
        scale = torch.randn(1, 8, dtype=torch.float32)
        shift = torch.randn(1, 8, dtype=torch.float32)
        called = {}

        def _mock_adaln(layernorm_arg, x_arg, scale_arg, shift_arg, fused=True):
            called["layernorm"] = layernorm_arg
            called["x"] = x_arg
            called["scale"] = scale_arg
            called["shift"] = shift_arg
            called["fused"] = fused
            return x_arg + 2.0

        monkeypatch.setenv(wan_transformer._WAN_ADALAYERNORM_ENABLE_ENV, "1")
        monkeypatch.setattr(wan_transformer, "_get_mindiesd_layernorm_scale_shift", lambda: _mock_adaln)

        with patch.object(wan_transformer.current_omni_platform, "is_npu", return_value=True):
            out = wan_transformer._wan_layernorm_scale_shift(layernorm, x, scale, shift)

        assert torch.allclose(out, x + 2.0)
        assert called["layernorm"] is layernorm
        assert called["x"] is x
        assert called["scale"].shape == scale.shape
        assert called["shift"].shape == shift.shape
        assert called["fused"] is True

    def test_npu_fallback_when_adalayernorm_shape_not_supported(self, monkeypatch):
        layernorm = torch.nn.LayerNorm(8, eps=1e-6)
        x = torch.randn(1, 2, 8, dtype=torch.float32)
        scale = torch.randn(1, 2, 8, dtype=torch.float32)  # mindiesd does not support BSH scale/shift
        shift = torch.randn(1, 2, 8, dtype=torch.float32)

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("mindiesd adalayernorm should not be called for unsupported scale/shift shape")

        monkeypatch.setattr(wan_transformer, "_get_mindiesd_layernorm_scale_shift", lambda: _fail_if_called)

        with patch.object(wan_transformer.current_omni_platform, "is_npu", return_value=True):
            out = wan_transformer._wan_layernorm_scale_shift(layernorm, x, scale, shift)

        expected = layernorm(x) * (1 + scale) + shift
        assert torch.allclose(out, expected, atol=1e-6, rtol=1e-5)

    def test_npu_fallback_when_adalayernorm_disabled(self, monkeypatch):
        layernorm = torch.nn.LayerNorm(8, eps=1e-6)
        x = torch.randn(1, 2, 8, dtype=torch.float32)
        scale = torch.randn(1, 8, dtype=torch.float32)
        shift = torch.randn(1, 8, dtype=torch.float32)

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("mindiesd adalayernorm should not be called when disabled")

        monkeypatch.setenv(wan_transformer._WAN_ADALAYERNORM_ENABLE_ENV, "0")
        monkeypatch.setattr(wan_transformer, "_get_mindiesd_layernorm_scale_shift", lambda: _fail_if_called)

        with patch.object(wan_transformer.current_omni_platform, "is_npu", return_value=True):
            out = wan_transformer._wan_layernorm_scale_shift(layernorm, x, scale, shift)

        expected = layernorm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        assert torch.allclose(out, expected, atol=1e-6, rtol=1e-5)


class TestWanFP32LayerNorm:
    def test_forward_matches_fp32_layernorm_on_cpu(self):
        norm = wan_transformer.WanFP32LayerNorm(16, eps=1e-6)
        x = torch.randn(2, 4, 16, dtype=torch.float16)

        with patch.object(wan_transformer.current_omni_platform, "is_npu", return_value=False):
            out = norm(x)

        expected = F.layer_norm(
            x.float(),
            norm.normalized_shape,
            norm.weight.float(),
            norm.bias.float(),
            norm.eps,
        ).to(x.dtype)

        assert torch.allclose(out, expected, atol=1e-3, rtol=1e-3)

    def test_npu_uses_mindiesd_fast_layernorm_when_available(self, monkeypatch):
        norm = wan_transformer.WanFP32LayerNorm(8, eps=1e-6)
        x = torch.randn(1, 2, 8, dtype=torch.float32)
        called = {}

        def _mock_fast_layernorm(norm_arg, x_arg, impl_mode=0, fused=True):
            called["norm"] = norm_arg
            called["x"] = x_arg
            called["impl_mode"] = impl_mode
            called["fused"] = fused
            return x_arg + 1.0

        monkeypatch.setenv(wan_transformer._WAN_FAST_LAYERNORM_IMPL_MODE_ENV, "1")
        monkeypatch.setattr(wan_transformer, "_get_mindiesd_fast_layernorm", lambda: _mock_fast_layernorm)

        with patch.object(wan_transformer.current_omni_platform, "is_npu", return_value=True):
            out = norm(x)

        assert torch.allclose(out, x + 1.0)
        assert called["norm"] is norm
        assert called["x"] is x
        assert called["impl_mode"] == 1
        assert called["fused"] is True

    def test_npu_fast_path_can_be_disabled(self, monkeypatch):
        norm = wan_transformer.WanFP32LayerNorm(8, eps=1e-6)
        x = torch.randn(2, 3, 8, dtype=torch.bfloat16)

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("fast_layernorm should not be called when disabled")

        monkeypatch.setenv(wan_transformer._WAN_FAST_LAYERNORM_ENABLE_ENV, "0")
        monkeypatch.setattr(wan_transformer, "_get_mindiesd_fast_layernorm", lambda: _fail_if_called)

        with patch.object(wan_transformer.current_omni_platform, "is_npu", return_value=True):
            out = norm(x)

        expected = F.layer_norm(
            x.float(),
            norm.normalized_shape,
            norm.weight.float(),
            norm.bias.float(),
            norm.eps,
        ).to(x.dtype)
        assert torch.allclose(out, expected, atol=1e-2, rtol=1e-2)

    def test_npu_uses_fast_layernorm_for_bf16_input(self, monkeypatch):
        norm = wan_transformer.WanFP32LayerNorm(8, eps=1e-6)
        x = torch.randn(1, 2, 8, dtype=torch.bfloat16)
        called = {}

        def _mock_fast_layernorm(norm_arg, x_arg, impl_mode=0, fused=True):
            called["norm"] = norm_arg
            called["x"] = x_arg
            called["impl_mode"] = impl_mode
            called["fused"] = fused
            return x_arg + torch.tensor(1.0, dtype=x_arg.dtype)

        monkeypatch.setenv(wan_transformer._WAN_FAST_LAYERNORM_IMPL_MODE_ENV, "0")
        monkeypatch.setattr(wan_transformer, "_get_mindiesd_fast_layernorm", lambda: _mock_fast_layernorm)

        with patch.object(wan_transformer.current_omni_platform, "is_npu", return_value=True):
            out = norm(x)

        assert torch.allclose(out, (x + 1.0).to(x.dtype), atol=1e-2, rtol=1e-2)
        assert out.dtype == torch.bfloat16
        assert called["norm"] is norm
        assert called["x"] is x
        assert called["impl_mode"] == 0
        assert called["fused"] is True

    def test_npu_fallback_to_fp32_when_fast_layernorm_fails(self, monkeypatch):
        norm = wan_transformer.WanFP32LayerNorm(8, eps=1e-6)
        x = torch.randn(2, 3, 8, dtype=torch.float32)

        def _raise_fast_layernorm(*args, **kwargs):
            raise RuntimeError("fast_layernorm failed")

        monkeypatch.setattr(wan_transformer, "_get_mindiesd_fast_layernorm", lambda: _raise_fast_layernorm)

        with patch.object(wan_transformer.current_omni_platform, "is_npu", return_value=True):
            out = norm(x)

        expected = F.layer_norm(
            x.float(),
            norm.normalized_shape,
            norm.weight.float(),
            norm.bias.float(),
            norm.eps,
        ).to(x.dtype)
        assert torch.allclose(out, expected, atol=1e-6, rtol=1e-5)

    def test_npu_skips_fast_path_for_non_3d_input(self, monkeypatch):
        norm = wan_transformer.WanFP32LayerNorm(8, eps=1e-6)
        x = torch.randn(2, 8, dtype=torch.float32)

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("fast_layernorm should not be called for non-3D input")

        monkeypatch.setattr(wan_transformer, "_get_mindiesd_fast_layernorm", lambda: _fail_if_called)

        with patch.object(wan_transformer.current_omni_platform, "is_npu", return_value=True):
            out = norm(x)

        expected = F.layer_norm(
            x.float(),
            norm.normalized_shape,
            norm.weight.float(),
            norm.bias.float(),
            norm.eps,
        ).to(x.dtype)
        assert torch.allclose(out, expected, atol=1e-6, rtol=1e-5)

    def test_npu_skips_fast_path_for_unsupported_dtype(self, monkeypatch):
        norm = wan_transformer.WanFP32LayerNorm(8, eps=1e-6)
        x = torch.randn(2, 3, 8, dtype=torch.float64)

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("fast_layernorm should not be called for float64 input")

        monkeypatch.setattr(wan_transformer, "_get_mindiesd_fast_layernorm", lambda: _fail_if_called)

        with patch.object(wan_transformer.current_omni_platform, "is_npu", return_value=True):
            out = norm(x)

        expected = F.layer_norm(
            x.float(),
            norm.normalized_shape,
            norm.weight.float(),
            norm.bias.float(),
            norm.eps,
        ).to(x.dtype)
        assert torch.allclose(out, expected, atol=1e-2, rtol=1e-2)
