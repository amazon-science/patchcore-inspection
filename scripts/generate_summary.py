"""Generate a machine-readable CSV summary from experiment artifacts."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


class ExperimentSummaryGenerator:
    """Collect completed PatchCore runs without manually copying metrics."""

    FIELDS = (
        "run_id",
        "method",
        "class",
        "seed",
        "train_transform",
        "test_condition",
        "i_auroc",
        "p_auroc",
        "au_pro",
        "au_pro_0.05",
        "au_pro_0.30",
        "runtime",
        "git_commit",
        "dirty",
    )

    def generate(self, experiments_root: Path, output: Path) -> List[Dict[str, Any]]:
        rows = []
        for metrics_path in sorted(experiments_root.glob("*/metrics.json")):
            metrics = self._read(metrics_path)
            git_path = metrics_path.parent / "git.json"
            git = self._read(git_path) if git_path.exists() else {}
            augmentation = metrics.get("augmentation", {})
            augmented = augmentation.get(
                "enabled", "illumination_aug" in metrics_path.parent.name
            )
            rows.append(
                {
                    "run_id": metrics_path.parent.name,
                    "method": metrics.get(
                        "experiment_name", self._method(metrics_path.parent.name)
                    ),
                    "class": metrics["class_name"],
                    "seed": metrics["seed"],
                    "train_transform": "illumination_random" if augmented else "none",
                    "test_condition": metrics.get("test_condition", "original"),
                    "i_auroc": metrics["i_auroc"],
                    "p_auroc": metrics["p_auroc"],
                    "au_pro": metrics.get("au_pro"),
                    "au_pro_0.05": metrics.get("au_pro_0.05"),
                    "au_pro_0.30": metrics.get("au_pro_0.30"),
                    "runtime": metrics["runtime_seconds"],
                    "git_commit": git.get("commit"),
                    "dirty": git.get("dirty"),
                }
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=self.FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return rows

    @staticmethod
    def _read(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as input_file:
            return json.load(input_file)

    @staticmethod
    def _method(run_id: str) -> str:
        if "augmented" in run_id or "illumination_aug" in run_id:
            return "patchcore_illumination_aug"
        return "patchcore_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", type=Path, default=Path("experiments"))
    parser.add_argument("--output", type=Path, default=Path("experiments/summary.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = ExperimentSummaryGenerator().generate(args.experiments_root, args.output)
    print("runs: {}".format(len(rows)))
    print("output: {}".format(args.output.resolve()))
    print("PASS")


if __name__ == "__main__":
    main()
