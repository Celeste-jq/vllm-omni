import json
import tempfile
import unittest
from pathlib import Path


class WanI2VHSDPBenchmarkTests(unittest.TestCase):
    def test_build_profile_command_includes_hsdp_flags(self):
        from tools.benchmark_wan_i2v_hsdp import build_profile_command

        command = build_profile_command(
            scenario_name="hsdp",
            nproc_per_node=4,
            common_args={
                "model": "/models/wan",
                "image": "/tmp/image.jpg",
                "prompt": "A cat playing with yarn",
                "height": 480,
                "width": 832,
                "num_frames": 81,
                "num_inference_steps": 40,
                "ulysses_degree": 4,
                "ring_degree": 1,
                "tensor_parallel_size": 1,
                "cfg_parallel_size": 1,
                "vae_patch_parallel_size": 1,
                "warmup_runs": 1,
            },
            scenario_args={
                "use_hsdp": True,
                "hsdp_shard_size": 4,
                "hsdp_replicate_size": 1,
            },
            output_dir="profiling_runs/compare/hsdp",
        )

        self.assertEqual(command[:2], ["torchrun", "--nproc_per_node=4"])
        self.assertIn("tools/profile_wan_i2v.py", command)
        self.assertIn("--use-hsdp", command)
        self.assertIn("--hsdp-shard-size", command)
        self.assertIn("--hsdp-replicate-size", command)
        self.assertIn("profiling_runs/compare/hsdp", command)

    def test_classify_failure_marks_oom(self):
        from tools.benchmark_wan_i2v_hsdp import classify_failure

        status = classify_failure(
            returncode=1,
            stdout="",
            stderr="RuntimeError: NPU out of memory while allocating tensor",
        )

        self.assertEqual(status["status"], "oom")
        self.assertTrue(status["oom"])

    def test_aggregate_run_metadata_picks_max_memory(self):
        from tools.benchmark_wan_i2v_hsdp import aggregate_run_metadata_payloads

        payload = aggregate_run_metadata_payloads(
            [
                {
                    "rank": 0,
                    "wall_time_sec": 10.0,
                    "memory": {
                        "peak_allocated_bytes": 100,
                        "peak_reserved_bytes": 120,
                    },
                },
                {
                    "rank": 1,
                    "wall_time_sec": 9.5,
                    "memory": {
                        "peak_allocated_bytes": 150,
                        "peak_reserved_bytes": 180,
                    },
                },
            ]
        )

        self.assertEqual(payload["world_size"], 2)
        self.assertAlmostEqual(payload["max_wall_time_sec"], 10.0)
        self.assertEqual(payload["memory"]["max_peak_allocated_bytes"], 150)
        self.assertEqual(payload["memory"]["max_peak_reserved_bytes"], 180)

    def test_load_scenario_metrics_reads_profile_and_metadata(self):
        from tools.benchmark_wan_i2v_hsdp import load_scenario_metrics

        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            (tmp_dir / "aggregate_summary.json").write_text(
                json.dumps(
                    {
                        "total_profiled_sec": 12.5,
                        "phases": {
                            "DIT_HIGH": {"pct": 35.0, "total_sec": 4.375, "count": 10},
                            "DIT_LOW": {"pct": 55.0, "total_sec": 6.875, "count": 20},
                            "VAE_DECODE": {"pct": 5.0, "total_sec": 0.625, "count": 1},
                        },
                    }
                )
            )
            (tmp_dir / "aggregate_run_metadata.json").write_text(
                json.dumps(
                    {
                        "world_size": 4,
                        "max_wall_time_sec": 18.0,
                        "memory": {
                            "max_peak_allocated_bytes": 1234,
                            "max_peak_reserved_bytes": 2345,
                        },
                    }
                )
            )

            metrics = load_scenario_metrics(tmp_dir)

        self.assertAlmostEqual(metrics["wall_time_sec"], 18.0)
        self.assertAlmostEqual(metrics["total_profiled_sec"], 12.5)
        self.assertEqual(metrics["max_peak_allocated_bytes"], 1234)
        self.assertEqual(metrics["max_peak_reserved_bytes"], 2345)
        self.assertAlmostEqual(metrics["DIT_HIGH_pct"], 35.0)
        self.assertAlmostEqual(metrics["DIT_LOW_pct"], 55.0)
        self.assertAlmostEqual(metrics["VAE_DECODE_pct"], 5.0)

    def test_render_markdown_summary_contains_memory_and_latency(self):
        from tools.benchmark_wan_i2v_hsdp import render_markdown_summary

        summary = render_markdown_summary(
            {
                "baseline": {
                    "status": "ok",
                    "metrics": {
                        "wall_time_sec": 10.0,
                        "max_peak_allocated_bytes": 100,
                        "max_peak_reserved_bytes": 150,
                    },
                },
                "hsdp": {
                    "status": "ok",
                    "metrics": {
                        "wall_time_sec": 11.0,
                        "max_peak_allocated_bytes": 70,
                        "max_peak_reserved_bytes": 100,
                    },
                },
                "deltas": {
                    "wall_time_pct": 10.0,
                    "peak_allocated_pct": 30.0,
                    "peak_reserved_pct": 33.33,
                },
            }
        )

        self.assertIn("wall_time_sec", summary)
        self.assertIn("max_peak_allocated_bytes", summary)
        self.assertIn("peak_allocated_pct", summary)


if __name__ == "__main__":
    unittest.main()
