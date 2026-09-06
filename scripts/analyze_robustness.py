"""Compare clean and shifted PatchCore runs and quantify robustness."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


METRICS = ("i_auroc", "p_auroc", "au_pro")


class RobustnessAnalyzer:
    """Calculate degradation and augmentation recovery from run artifacts."""

    def run(
        self,
        baseline_clean: Path,
        augmented_clean: Path,
        vanilla_shifted: List[Path],
        augmented_shifted: List[Path],
        output_dir: Path,
    ) -> Dict[str, Any]:
        baseline = self._load(baseline_clean)
        augmented = self._load(augmented_clean)
        vanilla_by_condition = self._by_condition(vanilla_shifted)
        augmented_by_condition = self._by_condition(augmented_shifted)
        conditions = sorted(set(vanilla_by_condition) & set(augmented_by_condition))
        if not conditions or len(conditions) != len(vanilla_by_condition):
            raise ValueError("Vanilla and augmented shifted conditions must match")

        rows = []
        for condition in conditions:
            vanilla = vanilla_by_condition[condition]
            candidate = augmented_by_condition[condition]
            for metric in METRICS:
                rows.append(
                    {
                        "test_condition": condition,
                        "metric": metric,
                        "baseline_clean": baseline[metric],
                        "vanilla_shifted": vanilla[metric],
                        "vanilla_degradation": vanilla[metric] - baseline[metric],
                        "augmented_clean": augmented[metric],
                        "augmented_shifted": candidate[metric],
                        "augmented_degradation": candidate[metric] - augmented[metric],
                        "recovery": candidate[metric] - vanilla[metric],
                        "clean_cost": augmented[metric] - baseline[metric],
                    }
                )

        result = {
            "definitions": {
                "degradation": "shifted minus clean for the same training method",
                "recovery": "augmented shifted minus vanilla shifted",
                "clean_cost": "augmented clean minus baseline clean",
            },
            "runs": {
                "baseline_clean": baseline_clean.name,
                "augmented_clean": augmented_clean.name,
                "vanilla_shifted": [path.name for path in vanilla_shifted],
                "augmented_shifted": [path.name for path in augmented_shifted],
            },
            "comparisons": rows,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "robustness_summary.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
        with (output_dir / "robustness_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        return result

    @staticmethod
    def _load(run_dir: Path) -> Dict[str, Any]:
        with (run_dir / "metrics.json").open("r", encoding="utf-8") as f:
            return json.load(f)

    def _by_condition(self, paths: List[Path]) -> Dict[str, Dict[str, Any]]:
        runs = {}
        for path in paths:
            metrics = self._load(path)
            condition = metrics.get("test_condition")
            if not condition or condition == "original" or condition in runs:
                raise ValueError(
                    "Each shifted run needs a unique non-original condition"
                )
            runs[condition] = metrics
        return runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-clean", type=Path, required=True)
    parser.add_argument("--augmented-clean", type=Path, required=True)
    parser.add_argument("--vanilla-shifted", type=Path, nargs="+", required=True)
    parser.add_argument("--augmented-shifted", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/robustness")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = RobustnessAnalyzer().run(
        args.baseline_clean,
        args.augmented_clean,
        args.vanilla_shifted,
        args.augmented_shifted,
        args.output_dir,
    )
    print("comparisons: {}".format(len(result["comparisons"])))
    print("output: {}".format(args.output_dir.resolve()))
    print("PASS")


if __name__ == "__main__":
    main()
