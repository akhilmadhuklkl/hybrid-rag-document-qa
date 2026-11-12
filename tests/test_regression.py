import json

import pytest

from evaluation.regression import check_report


def test_quality_gate_accepts_baseline(tmp_path):
    report = {"weight_tuning": {"best": {"recall_at_k": 1.0, "mrr": 0.96}}}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert check_report(path, 0.95, 0.9)["mrr"] == 0.96


def test_quality_gate_rejects_regression(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"hybrid": {"recall_at_k": 0.8, "mrr": 0.7}}), encoding="utf-8")
    with pytest.raises(ValueError, match="Retrieval regression"):
        check_report(path, 0.95, 0.9)
