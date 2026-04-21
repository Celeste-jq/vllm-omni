import unittest
from types import SimpleNamespace
from unittest.mock import patch


class WanI2VProfileRuntimeTests(unittest.TestCase):
    def test_maybe_postprocess_output_calls_postprocess_on_rank0(self):
        from tools.profile_wan_i2v import maybe_postprocess_output

        output = SimpleNamespace(output="raw_video")
        request = SimpleNamespace(sampling_params=SimpleNamespace())
        fake_torch = SimpleNamespace(Tensor=type("FakeTensor", (), {}))

        def post_process_func(video, sampling_params=None):
            self.assertEqual(video, "raw_video")
            self.assertIsNotNone(sampling_params)
            return {"video": "processed"}

        with patch.dict("sys.modules", {"torch": fake_torch}):
            processed = maybe_postprocess_output(
                output,
                request,
                post_process_func,
                enable_cpu_offload=False,
                rank=0,
            )

        self.assertEqual(processed, {"video": "processed"})

    def test_maybe_postprocess_output_skips_postprocess_on_nonzero_rank(self):
        from tools.profile_wan_i2v import maybe_postprocess_output

        output = SimpleNamespace(output="raw_video")
        request = SimpleNamespace(sampling_params=SimpleNamespace())
        fake_torch = SimpleNamespace(Tensor=type("FakeTensor", (), {}))

        def post_process_func(video, sampling_params=None):
            raise AssertionError("post_process_func should not be called on nonzero ranks")

        with patch.dict("sys.modules", {"torch": fake_torch}):
            processed = maybe_postprocess_output(
                output,
                request,
                post_process_func,
                enable_cpu_offload=False,
                rank=3,
            )

        self.assertEqual(processed, "raw_video")


if __name__ == "__main__":
    unittest.main()
