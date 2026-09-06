import json

import pytest

from scripts.analyze_robustness import RobustnessAnalyzer


def _run(tmp_path, name, condition, values):
    run_dir = tmp_path / name
    run_dir.mkdir()
    metrics = dict(zip(("i_auroc", "p_auroc", "au_pro"), values))
    metrics["test_condition"] = condition
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return run_dir


def test_robustness_degradation_recovery_and_clean_cost(tmp_path):
    baseline = _run(tmp_path, "baseline", "original", (0.9, 0.8, 0.7))
    augmented = _run(tmp_path, "augmented", "original", (0.89, 0.81, 0.72))
    vanilla_shifted = _run(tmp_path, "vanilla_08", "brightness_0.8", (0.8, 0.6, 0.5))
    augmented_shifted = _run(
        tmp_path, "augmented_08", "brightness_0.8", (0.85, 0.7, 0.6)
    )

    result = RobustnessAnalyzer().run(
        baseline, augmented, [vanilla_shifted], [augmented_shifted], tmp_path / "out"
    )

    image_row = result["comparisons"][0]
    assert image_row["vanilla_degradation"] == pytest.approx(-0.1)
    assert image_row["augmented_degradation"] == pytest.approx(-0.04)
    assert image_row["recovery"] == pytest.approx(0.05)
    assert image_row["clean_cost"] == pytest.approx(-0.01)
    assert (tmp_path / "out" / "robustness_summary.csv").exists()
