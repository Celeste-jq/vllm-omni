import unittest


class WanI2VProfilePatchTests(unittest.TestCase):
    def test_dual_transformer_maps_to_high_and_low_tags(self):
        from tools.wan_i2v_profile_patch import classify_dit_phase

        self.assertEqual(classify_dit_phase("transformer"), "DIT_HIGH")
        self.assertEqual(classify_dit_phase("transformer_2"), "DIT_LOW")

    def test_apply_pipeline_profiling_patch_wraps_expected_methods(self):
        from tools.wan_i2v_profile_patch import PhaseRecorder, apply_pipeline_profiling_patch

        class DummyVAE:
            def encode(self, x):
                return x

            def decode(self, x):
                return x

        class DummyPipeline:
            def __init__(self):
                self.vae = DummyVAE()
                self._calls = []

            def encode_prompt(self, prompt):
                self._calls.append(("encode_prompt", prompt))
                return prompt

            def predict_noise(self, current_model=None, **kwargs):
                name = getattr(current_model, "_profile_name", "transformer")
                self._calls.append(("predict_noise", name))
                return "noise"

        recorder = PhaseRecorder()
        pipeline = DummyPipeline()

        apply_pipeline_profiling_patch(pipeline, recorder)

        high = type("DummyModel", (), {"_profile_name": "transformer"})()
        low = type("DummyModel", (), {"_profile_name": "transformer_2"})()

        pipeline.encode_prompt("hello")
        pipeline.vae.encode("image")
        pipeline.predict_noise(current_model=high)
        pipeline.predict_noise(current_model=low)
        pipeline.vae.decode("latent")

        phase_names = [name for name, _ in recorder.records]
        self.assertIn("TEXT_ENCODER", phase_names)
        self.assertIn("VAE_ENCODE", phase_names)
        self.assertIn("DIT_HIGH", phase_names)
        self.assertIn("DIT_LOW", phase_names)
        self.assertIn("VAE_DECODE", phase_names)


if __name__ == "__main__":
    unittest.main()
