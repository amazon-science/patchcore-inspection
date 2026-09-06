import json

import pytest

from scripts.compare_experiments import ExperimentComparator


def test_comparator_reports_candidate_minus_baseline(tmp_path):
    baseline = {
        "dataset": "mvtec",
        "class_name": "bottle",
        "seed": 42,
        "i_auroc": 0.9,
        "p_auroc": 0.8,
        "au_pro": 0.7,
        "runtime_seconds": 10.0,
    }
    candidate = {**baseline, "p_auroc": 0.82, "runtime_seconds": 12.0}
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    result = ExperimentComparator().compare(baseline_path, candidate_path)

    assert result["candidate_minus_baseline"]["p_auroc"] == pytest.approx(0.02)
    assert result["candidate_minus_baseline"]["runtime_seconds"] == pytest.approx(2.0)
