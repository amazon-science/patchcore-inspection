"""Compare two PatchCore metric artifacts and save exact metric deltas."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict


class ExperimentComparator:
    """Build a machine-readable baseline-versus-candidate comparison."""

    _METRICS = ("i_auroc", "p_auroc", "au_pro", "runtime_seconds")

    def compare(self, baseline_path: Path, candidate_path: Path) -> Dict[str, Any]:
        baseline = self._load(baseline_path)
        candidate = self._load(candidate_path)
        if baseline["dataset"] != candidate["dataset"]:
            raise ValueError("Experiments use different datasets.")
        if baseline["class_name"] != candidate["class_name"]:
            raise ValueError("Experiments use different classes.")
        if baseline["seed"] != candidate["seed"]:
            raise ValueError("Experiments use different seeds.")
        return {
            "dataset": baseline["dataset"],
            "class_name": baseline["class_name"],
            "seed": baseline["seed"],
            "baseline_path": str(baseline_path.resolve()),
            "candidate_path": str(candidate_path.resolve()),
            "baseline": {key: baseline[key] for key in self._METRICS},
            "candidate": {key: candidate[key] for key in self._METRICS},
            "candidate_minus_baseline": {
                key: candidate[key] - baseline[key] for key in self._METRICS
            },
        }

    @staticmethod
    def _load(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as metrics_file:
            return json.load(metrics_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = ExperimentComparator().compare(args.baseline, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))
    print("PASS")


if __name__ == "__main__":
    main()
