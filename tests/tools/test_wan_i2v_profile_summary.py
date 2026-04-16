import json
import unittest
from pathlib import Path


class WanI2VProfileSummaryTests(unittest.TestCase):
    def test_summary_aggregates_same_phase_names(self):
        from tools.wan_i2v_profile_patch import build_summary_from_records

        records = [
            ("DIT_LOW", 0.2),
            ("DIT_LOW", 0.3),
            ("VAE_DECODE", 0.5),
        ]

        summary = build_summary_from_records(records)

        self.assertEqual(summary["DIT_LOW"]["count"], 2)
        self.assertAlmostEqual(summary["DIT_LOW"]["total_sec"], 0.5)
        self.assertAlmostEqual(summary["DIT_LOW"]["avg_sec"], 0.25)
        self.assertEqual(summary["VAE_DECODE"]["count"], 1)

    def test_write_summary_outputs_json_and_txt(self):
        from tools.wan_i2v_profile_patch import write_summary_outputs

        temp_dir = Path(self._testMethodName)
        if temp_dir.exists():
            for child in temp_dir.iterdir():
                child.unlink()
            temp_dir.rmdir()
        temp_dir.mkdir()

        try:
            records = [("TEXT_ENCODER", 0.1), ("DIT_HIGH", 0.2)]
            write_summary_outputs(temp_dir, records)

            json_path = temp_dir / "summary.json"
            txt_path = temp_dir / "summary.txt"

            self.assertTrue(json_path.exists())
            self.assertTrue(txt_path.exists())

            data = json.loads(json_path.read_text())
            self.assertIn("TEXT_ENCODER", data["phases"])
            self.assertIn("DIT_HIGH", data["phases"])
            self.assertIn("total_profiled_sec", data)
            self.assertIn("TEXT_ENCODER", txt_path.read_text())
        finally:
            for child in temp_dir.iterdir():
                child.unlink()
            temp_dir.rmdir()

    def test_aggregate_summary_payloads_keeps_totals(self):
        from tools.wan_i2v_profile_patch import aggregate_summary_payloads

        payload = aggregate_summary_payloads(
            [
                {
                    "total_profiled_sec": 0.3,
                    "phases": {
                        "DIT_HIGH": {"count": 1, "total_sec": 0.3, "avg_sec": 0.3, "pct": 100.0},
                    },
                },
                {
                    "total_profiled_sec": 0.7,
                    "phases": {
                        "DIT_HIGH": {"count": 2, "total_sec": 0.5, "avg_sec": 0.25, "pct": 71.4},
                        "VAE_DECODE": {"count": 1, "total_sec": 0.2, "avg_sec": 0.2, "pct": 28.6},
                    },
                },
            ]
        )

        self.assertAlmostEqual(payload["total_profiled_sec"], 1.0)
        self.assertEqual(payload["phases"]["DIT_HIGH"]["count"], 3)
        self.assertAlmostEqual(payload["phases"]["DIT_HIGH"]["total_sec"], 0.8)
        self.assertEqual(payload["phases"]["VAE_DECODE"]["count"], 1)
        self.assertAlmostEqual(payload["phases"]["VAE_DECODE"]["pct"], 20.0)


if __name__ == "__main__":
    unittest.main()
