import importlib.util
import sys
import types
import unittest
from pathlib import Path


def _ensure_module(name: str, module: types.ModuleType) -> None:
    if name not in sys.modules:
        sys.modules[name] = module


def _load_utils_module():
    repo_root = Path(__file__).resolve().parents[4]
    voxcpm_dir = repo_root / "vllm_omni" / "model_executor" / "models" / "voxcpm"
    utils_path = repo_root / "vllm_omni" / "model_executor" / "models" / "voxcpm" / "utils.py"

    fake_numpy = types.ModuleType("numpy")
    fake_numpy.ascontiguousarray = lambda value: value
    fake_numpy.ndarray = object
    _ensure_module("numpy", fake_numpy)

    fake_torch = types.ModuleType("torch")
    fake_torch.Tensor = object
    fake_torch.dtype = object
    fake_torch.device = object
    fake_torch.bfloat16 = object()
    fake_torch.float16 = object()
    fake_torch.float32 = object()
    fake_torch.inference_mode = lambda: (lambda fn: fn)
    fake_torch.no_grad = lambda: (lambda fn: fn)
    fake_torch.from_numpy = lambda value: value
    fake_torch.nn = types.ModuleType("torch.nn")
    fake_torch.nn.Module = object
    fake_torch.nn.functional = types.SimpleNamespace(pad=lambda audio, *_args, **_kwargs: audio)
    _ensure_module("torch", fake_torch)
    _ensure_module("torch.nn", fake_torch.nn)

    fake_einops = types.ModuleType("einops")
    fake_einops.rearrange = lambda value, *_args, **_kwargs: value
    _ensure_module("einops", fake_einops)

    fake_tqdm = types.ModuleType("tqdm")
    fake_tqdm.tqdm = lambda value, *_args, **_kwargs: value
    _ensure_module("tqdm", fake_tqdm)

    _ensure_module("vllm", types.ModuleType("vllm"))

    vllm_config = types.ModuleType("vllm.config")
    vllm_config.VllmConfig = object
    _ensure_module("vllm.config", vllm_config)

    vllm_logger = types.ModuleType("vllm.logger")
    vllm_logger.init_logger = lambda _name: None
    _ensure_module("vllm.logger", vllm_logger)

    vllm_omni_pkg = types.ModuleType("vllm_omni")
    vllm_omni_pkg.__path__ = []  # type: ignore[attr-defined]
    _ensure_module("vllm_omni", vllm_omni_pkg)

    model_executor_pkg = types.ModuleType("vllm_omni.model_executor")
    model_executor_pkg.__path__ = []  # type: ignore[attr-defined]
    _ensure_module("vllm_omni.model_executor", model_executor_pkg)

    models_pkg = types.ModuleType("vllm_omni.model_executor.models")
    models_pkg.__path__ = []  # type: ignore[attr-defined]
    _ensure_module("vllm_omni.model_executor.models", models_pkg)

    voxcpm_pkg = types.ModuleType("vllm_omni.model_executor.models.voxcpm")
    voxcpm_pkg.__path__ = [str(voxcpm_dir)]  # type: ignore[attr-defined]
    _ensure_module("vllm_omni.model_executor.models.voxcpm", voxcpm_pkg)

    stage_wrappers = types.ModuleType("vllm_omni.model_executor.models.voxcpm.voxcpm_stage_wrappers")
    stage_wrappers._DirectVoxCPMAudioVAE = object
    stage_wrappers._DirectVoxCPMLatentGenerator = object
    _ensure_module("vllm_omni.model_executor.models.voxcpm.voxcpm_stage_wrappers", stage_wrappers)

    voxcpm_root = types.ModuleType("voxcpm")
    voxcpm_root.__path__ = []  # type: ignore[attr-defined]
    _ensure_module("voxcpm", voxcpm_root)

    voxcpm_model = types.ModuleType("voxcpm.model")
    voxcpm_model.__path__ = []  # type: ignore[attr-defined]
    _ensure_module("voxcpm.model", voxcpm_model)

    voxcpm_model_utils = types.ModuleType("voxcpm.model.utils")
    voxcpm_model_utils.get_dtype = lambda *_args, **_kwargs: "float32"
    _ensure_module("voxcpm.model.utils", voxcpm_model_utils)

    spec = importlib.util.spec_from_file_location(
        "vllm_omni.model_executor.models.voxcpm.utils",
        utils_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class VoxCPMUtilsModuleTest(unittest.TestCase):
    def test_utils_module_exports_core_helpers(self):
        module = _load_utils_module()

        self.assertTrue(hasattr(module, "_make_voxcpm_model_for_omni"))
        self.assertTrue(hasattr(module, "_load_native_voxcpm_latent_generator"))
        self.assertTrue(hasattr(module, "_resolve_runtime_device"))

    def test_build_prompt_cache_falls_back_on_torchcodec_runtime_error(self):
        module = _load_utils_module()

        class FakeBase:
            def build_prompt_cache(self, *_args, **_kwargs):
                raise RuntimeError("Could not load libtorchcodec")

        Wrapped = module._make_voxcpm_model_for_omni(FakeBase)
        wrapped = Wrapped()

        sentinel = {"prompt_text": "hi", "audio_feat": "ok"}
        module._build_prompt_cache_with_soundfile = lambda *_args, **_kwargs: sentinel
        sys.modules[
            "vllm_omni.model_executor.models.voxcpm.voxcpm_import_utils"
        ]._build_prompt_cache_with_soundfile = lambda *_args, **_kwargs: sentinel

        result = wrapped.build_prompt_cache(prompt_text="hi", prompt_wav_path="/tmp/a.wav")

        self.assertIs(result, sentinel)
