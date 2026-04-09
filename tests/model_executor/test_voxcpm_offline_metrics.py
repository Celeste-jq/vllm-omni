import importlib.util
from pathlib import Path


def _load_voxcpm_test_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "examples" / "offline_inference" / "voxcpm" / "test.py"
    spec = importlib.util.spec_from_file_location("voxcpm_offline_test_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collect_request_summaries_from_log_includes_ttfp_and_rtf():
    module = _load_voxcpm_test_module()
    log_text = """
INFO [omni_base.py:162] [Summary] {'overall_summary': {'e2e_stage_0_wall_time_ms': 1234.0, 'e2e_stage_1_wall_time_ms': 1250.0}, 'stage_table': [{'request_id': 'req_001', 'stages': []}], 'e2e_table': [{'request_id': 'req_001', 'e2e_total_ms': 1260.0, 'e2e_total_tokens': 43, 'transfers_total_time_ms': 0.0, 'transfers_total_kbytes': 0.0}]}
[OfflineMetrics] {'request_id': 'req_001', 'ttfp_ms': 210.5, 'audio_duration_s': 1.8, 'rtf': 0.7}
"""

    request_summaries = module._collect_request_summaries_from_log(log_text)

    assert len(request_summaries) == 1
    assert request_summaries[0]["request_id"] == "req_001"
    assert request_summaries[0]["ttfp_ms"] == 210.5
    assert request_summaries[0]["audio_duration_s"] == 1.8
    assert request_summaries[0]["rtf"] == 0.7
