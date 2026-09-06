import json

from scripts.generate_summary import ExperimentSummaryGenerator


def test_summary_is_generated_from_artifacts(tmp_path):
    run_dir = tmp_path / "patchcore_baseline_bottle_test"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "dataset": "mvtec",
                "class_name": "bottle",
                "seed": 42,
                "i_auroc": 1.0,
                "p_auroc": 0.98,
                "au_pro": 0.95,
                "runtime_seconds": 100.0,
                "test_condition": "original",
                "augmentation": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "git.json").write_text(
        json.dumps({"commit": "abc", "dirty": False}), encoding="utf-8"
    )
    output = tmp_path / "summary.csv"

    rows = ExperimentSummaryGenerator().generate(tmp_path, output)

    assert len(rows) == 1
    assert rows[0]["test_condition"] == "original"
    assert rows[0]["train_transform"] == "none"
    assert output.read_text(encoding="utf-8").startswith("run_id,method,class")
