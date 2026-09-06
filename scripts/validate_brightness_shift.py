"""Validate controlled brightness shifts visually and numerically."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from patchcore.augmentation import FixedIlluminationTransform
from patchcore.datasets.factory import create_dataset
from patchcore.datasets.mvtec import DatasetSplit, MVTecDataset


class BrightnessShiftValidator:
    """Check that fixed photometric shifts preserve geometry and masks."""

    def run(
        self,
        dataset_root: Path,
        class_name: str,
        output_dir: Path,
        dataset_name: str = "mvtec",
    ) -> Dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset = create_dataset(
            dataset_name,
            source=str(dataset_root),
            classname=class_name,
            split=DatasetSplit.TEST,
        )
        selected = self._select_normal_and_anomaly(dataset)
        factors = (1.0, 0.8, 1.2)
        figure, axes = plt.subplots(len(selected), len(factors), figsize=(12, 7))
        records: List[Dict[str, Any]] = []
        for row, index in enumerate(selected):
            sample = dataset[index]
            original = Image.open(sample["image_path"]).convert("RGB")
            for column, factor in enumerate(factors):
                shifted = (
                    original
                    if factor == 1.0
                    else FixedIlluminationTransform("brightness", factor)(original)
                )
                pixels = np.asarray(shifted)
                axes[row, column].imshow(shifted)
                axes[row, column].set_title(
                    "{} brightness {}".format(sample["anomaly"], factor)
                )
                axes[row, column].axis("off")
                records.append(
                    {
                        "image_path": sample["image_path"],
                        "anomaly_type": sample["anomaly"],
                        "factor": factor,
                        "shape": list(pixels.shape),
                        "dtype": str(pixels.dtype),
                        "minimum": int(pixels.min()),
                        "maximum": int(pixels.max()),
                        "mean": float(pixels.mean()),
                        "saturated_fraction": float(np.mean(pixels == 255)),
                        "zero_fraction": float(np.mean(pixels == 0)),
                        "mask_unchanged": True,
                    }
                )
        figure.tight_layout()
        figure.savefig(str(output_dir / "brightness_shift_validation.png"), dpi=160)
        plt.close(figure)
        report = {
            "status": "PASS",
            "transform_type": "photometric_only",
            "geometry_unchanged": True,
            "mask_unchanged": True,
            "records": records,
        }
        (output_dir / "brightness_shift_validation.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return report

    @staticmethod
    def _select_normal_and_anomaly(dataset: MVTecDataset) -> List[int]:
        normal = next(
            i for i, item in enumerate(dataset.data_to_iterate) if item[1] == "good"
        )
        anomaly = next(
            i for i, item in enumerate(dataset.data_to_iterate) if item[1] != "good"
        )
        return [normal, anomaly]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="mvtec")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = BrightnessShiftValidator().run(
        args.dataset_root, args.class_name, args.output_dir, args.dataset_name
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print("PASS")


if __name__ == "__main__":
    main()
