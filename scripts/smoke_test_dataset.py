"""Validate one configured MVTec anomaly-detection dataset class."""

import argparse
from pathlib import Path
from typing import Any, Dict

import torch

from patchcore.config import ConfigLoader
from patchcore.datasets.factory import create_dataset
from patchcore.datasets.mvtec import DatasetSplit
from patchcore.datasets.mvtec import MVTecDataset


class DatasetSmokeTester:
    """Validate labels, masks, paths, shapes, and normal-only training data."""

    def run(self, config_path: Path) -> Dict[str, Any]:
        config = ConfigLoader().load(config_path)
        common_args = {
            "source": str(config.dataset.root),
            "classname": config.dataset.class_name,
            "resize": config.dataset.resize,
            "imagesize": config.dataset.image_size,
        }
        train_dataset = create_dataset(
            config.dataset.name, split=DatasetSplit.TRAIN, **common_args
        )
        test_dataset = create_dataset(
            config.dataset.name, split=DatasetSplit.TEST, **common_args
        )

        train_anomalies = [
            item for item in train_dataset.data_to_iterate if item[1] != "good"
        ]
        if train_anomalies:
            raise AssertionError(
                "Anomalous samples found in memory-bank training data."
            )

        train_normal = self._find_sample(train_dataset, is_anomaly=False)
        test_normal = self._find_sample(test_dataset, is_anomaly=False)
        test_anomaly = self._find_sample(test_dataset, is_anomaly=True)
        self._validate_sample(train_normal, config.dataset.image_size, False)
        self._validate_sample(test_normal, config.dataset.image_size, False)
        self._validate_sample(test_anomaly, config.dataset.image_size, True)

        test_normal_count = sum(
            item[1] == "good" for item in test_dataset.data_to_iterate
        )
        test_anomaly_count = len(test_dataset) - test_normal_count
        report = {
            "dataset": config.dataset.name,
            "class": config.dataset.class_name,
            "root": str(config.dataset.root),
            "train_normal_count": len(train_dataset),
            "test_normal_count": test_normal_count,
            "test_anomaly_count": test_anomaly_count,
            "image_dtype": str(test_anomaly["image"].dtype),
            "image_shape": list(test_anomaly["image"].shape),
            "mask_dtype": str(test_anomaly["mask"].dtype),
            "mask_shape": list(test_anomaly["mask"].shape),
            "normal_label": test_normal["is_anomaly"],
            "anomaly_label": test_anomaly["is_anomaly"],
            "anomaly_image_path": test_anomaly["image_path"],
            "anomaly_mask_path": self._mask_path(
                test_dataset, test_anomaly["image_path"]
            ),
        }
        return report

    @staticmethod
    def _find_sample(dataset: MVTecDataset, is_anomaly: bool) -> Dict[str, Any]:
        for index, item in enumerate(dataset.data_to_iterate):
            if (item[1] != "good") == is_anomaly:
                return dataset[index]
        raise AssertionError(
            "Required {} sample was not found.".format(
                "anomalous" if is_anomaly else "normal"
            )
        )

    @staticmethod
    def _validate_sample(
        sample: Dict[str, Any], image_size: int, is_anomaly: bool
    ) -> None:
        expected_shape = (3, image_size, image_size)
        expected_mask_shape = (1, image_size, image_size)
        if sample["image"].shape != expected_shape:
            raise AssertionError(
                "Unexpected image shape: {}".format(sample["image"].shape)
            )
        if sample["mask"].shape != expected_mask_shape:
            raise AssertionError(
                "Unexpected mask shape: {}".format(sample["mask"].shape)
            )
        if (
            sample["image"].dtype != torch.float32
            or sample["mask"].dtype != torch.float32
        ):
            raise AssertionError("Images and masks must be float32 tensors.")
        if sample["is_anomaly"] != int(is_anomaly):
            raise AssertionError("Incorrect anomaly label mapping.")
        if is_anomaly and not torch.any(sample["mask"] > 0):
            raise AssertionError("An anomalous sample has an empty ground-truth mask.")
        if not is_anomaly and torch.any(sample["mask"] != 0):
            raise AssertionError("A normal sample has a non-empty mask.")

    @staticmethod
    def _mask_path(dataset: MVTecDataset, image_path: str) -> str:
        for _, _, candidate_image_path, mask_path in dataset.data_to_iterate:
            if candidate_image_path == image_path:
                if mask_path is None:
                    raise AssertionError("Anomalous sample has no mask path.")
                return mask_path
        raise AssertionError("Sample path was not found in dataset metadata.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = DatasetSmokeTester().run(args.config)
    for key, value in report.items():
        print("{}: {}".format(key, value))
    print("PASS")


if __name__ == "__main__":
    main()
