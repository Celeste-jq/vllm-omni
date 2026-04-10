import importlib.util
import sys
import types
import unittest
from pathlib import Path
from typing import NamedTuple

class _FakeTensor:
    def __init__(self, shape, device=None, dtype=None):
        self.shape = tuple(shape)
        self.device = device
        self.dtype = dtype

    def to(self, device=None, dtype=None):
        return _FakeTensor(self.shape, device=device or self.device, dtype=dtype or self.dtype)


class _FakeDevice:
    def __init__(self, device_type: str):
        self.type = device_type
        self.index = None

    def __str__(self):
        return self.type


class _FakeOmniOutput(NamedTuple):
    text_hidden_states: _FakeTensor
    multimodal_outputs: dict | None = None
    intermediate_tensors: object | None = None
    next_token_id: _FakeTensor | None = None


def _ensure_module(name: str, module: types.ModuleType) -> None:
    if name not in sys.modules:
        sys.modules[name] = module


def _load_voxcpm_model_module():
    repo_root = Path(__file__).resolve().parents[4]
    model_path = repo_root / "vllm_omni" / "model_executor" / "models" / "voxcpm" / "voxcpm_model.py"

    fake_numpy = types.ModuleType("numpy")
    fake_numpy.ndarray = object
    fake_numpy.asarray = lambda value, dtype=None: value
    fake_numpy.clip = lambda value, low, high: value
    fake_numpy.isscalar = lambda value: isinstance(value, (int, float, str, bool))
    _ensure_module("numpy", fake_numpy)

    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = "float32"
    fake_torch.int32 = "int32"
    fake_torch.Tensor = _FakeTensor
    fake_torch.dtype = object
    fake_torch.device = _FakeDevice
    fake_torch.zeros = lambda shape, device=None, dtype=None: _FakeTensor(shape, device=device, dtype=dtype)
    fake_torch.tensor = lambda _value, dtype=None: _FakeTensor((1,), dtype=dtype)
    fake_torch.no_grad = lambda: (lambda fn: fn)
    fake_torch.inference_mode = lambda: (lambda fn: fn)
    fake_torch.LongTensor = lambda values: _FakeTensor((len(values),), dtype="int64")
    fake_torch.as_tensor = lambda value: _FakeTensor((len(value),), dtype="float32")
    fake_torch.nn = types.ModuleType("torch.nn")
    fake_torch.nn.Module = object
    _ensure_module("torch", fake_torch)
    _ensure_module("torch.nn", fake_torch.nn)

    _ensure_module("vllm", types.ModuleType("vllm"))

    vllm_config = types.ModuleType("vllm.config")
    vllm_config.VllmConfig = object
    _ensure_module("vllm.config", vllm_config)

    vllm_logger = types.ModuleType("vllm.logger")
    vllm_logger.init_logger = lambda _name: None
    _ensure_module("vllm.logger", vllm_logger)

    vllm_sequence = types.ModuleType("vllm.sequence")
    vllm_sequence.IntermediateTensors = dict
    _ensure_module("vllm.sequence", vllm_sequence)

    vllm_omni_pkg = types.ModuleType("vllm_omni")
    vllm_omni_pkg.__path__ = []  # type: ignore[attr-defined]
    _ensure_module("vllm_omni", vllm_omni_pkg)

    model_executor_pkg = types.ModuleType("vllm_omni.model_executor")
    model_executor_pkg.__path__ = []  # type: ignore[attr-defined]
    _ensure_module("vllm_omni.model_executor", model_executor_pkg)

    models_pkg = types.ModuleType("vllm_omni.model_executor.models")
    models_pkg.__path__ = []  # type: ignore[attr-defined]
    _ensure_module("vllm_omni.model_executor.models", models_pkg)

    output_templates = types.ModuleType("vllm_omni.model_executor.models.output_templates")
    output_templates.OmniOutput = _FakeOmniOutput
    _ensure_module("vllm_omni.model_executor.models.output_templates", output_templates)

    fake_pkg = types.ModuleType("test_voxcpm_pkg")
    fake_pkg.__path__ = []  # type: ignore[attr-defined]
    _ensure_module("test_voxcpm_pkg", fake_pkg)

    fake_utils = types.ModuleType("test_voxcpm_pkg.utils")
    fake_utils._device_to_string = lambda device: str(device)
    fake_utils._load_native_voxcpm_audio_vae = lambda *args, **kwargs: None
    fake_utils._load_native_voxcpm_latent_generator = lambda *args, **kwargs: None
    fake_utils._normalize_dtype_name = lambda dtype: dtype
    fake_utils._resolve_runtime_device = lambda _config: _FakeDevice("cpu")
    _ensure_module("test_voxcpm_pkg.utils", fake_utils)

    spec = importlib.util.spec_from_file_location("test_voxcpm_pkg.voxcpm_model", model_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VoxCPMEmptyOutputTest(unittest.TestCase):
    def test_make_empty_output_keeps_one_hidden_row_per_request(self):
        module = _load_voxcpm_model_module()
        model = module.VoxCPMForConditionalGeneration.__new__(module.VoxCPMForConditionalGeneration)

        output = model._make_empty_output(
            output_key="model_outputs",
            payload_factory=lambda: _FakeTensor((0,), dtype="float32"),
            infos=[{}, {}],
            sample_rate=24000,
            out_device=_FakeDevice("cpu"),
            out_dtype="float32",
        )

        self.assertEqual(tuple(output.text_hidden_states.shape), (2, 1))
