"""Recompute AU-PRO bounds for a saved PatchCore run."""

import argparse
import json
from pathlib import Path

import numpy as np

import patchcore.metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_path = args.run_dir / "predictions.npz"
    metrics_path = args.run_dir / "metrics.json"
    predictions = np.load(predictions_path, allow_pickle=False)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    anomaly_maps = predictions["anomaly_maps"]
    masks = predictions["masks"]
    aupro_005 = patchcore.metrics.compute_aupro(anomaly_maps, masks, fpr_limit=0.05)
    aupro_030 = patchcore.metrics.compute_aupro(anomaly_maps, masks, fpr_limit=0.3)
    metrics["au_pro"] = aupro_030["aupro"]
    metrics["au_pro_0.05"] = aupro_005["aupro"]
    metrics["au_pro_0.30"] = aupro_030["aupro"]
    metrics["au_pro_fpr_limit"] = aupro_030["fpr_limit"]
    metrics["au_pro_implementation"] = aupro_030["implementation"]
    metrics["au_pro_protocol"] = aupro_030["protocol"]
    metrics["au_pro_regions"] = aupro_030["number_of_regions"]
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("au_pro_0.05", metrics["au_pro_0.05"])
    print("au_pro_0.30", metrics["au_pro_0.30"])
    print("PASS")


if __name__ == "__main__":
    main()
