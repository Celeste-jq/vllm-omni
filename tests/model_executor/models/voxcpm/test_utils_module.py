import importlib
import unittest


class VoxCPMUtilsModuleTest(unittest.TestCase):
    def test_utils_module_exports_core_helpers(self):
        module = importlib.import_module("vllm_omni.model_executor.models.voxcpm.utils")

        self.assertTrue(hasattr(module, "_import_voxcpm_model_class"))
        self.assertTrue(hasattr(module, "_load_native_voxcpm_latent_generator"))
        self.assertTrue(hasattr(module, "_resolve_runtime_device"))
