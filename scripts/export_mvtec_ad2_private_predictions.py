"""Export MVTec AD 2 private anomaly maps for benchmark submission checking."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import patchcore.metrics
import patchcore.patchcore
import patchcore.utils
from patchcore.config import AppConfig, ConfigLoader
from patchcore.datasets.factory import create_dataset
from patchcore.datasets.mvtec import DatasetSplit
from patchcore.device import DeviceManager
from patchcore.experiment import PatchCoreFactory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("experiments"))
    return parser.parse_args()


class MVTecAD2PrivateExporter:
    """Train PatchCore once and export private/mixed anomaly maps."""

    def __init__(self, repository_root: Path) -> None:
        self.repository_root = repository_root.resolve()

    def run(self, config_path: Path, output_root: Path) -> Path:
        config_path = config_path.resolve()
        config = ConfigLoader().load(config_path)
        if config.dataset.name != "mvtec_ad_2":
            raise ValueError("Private export currently supports dataset.name=mvtec_ad_2.")
        device = DeviceManager().resolve(config.device.requested)
        patchcore.utils.fix_seeds(
            config.experiment.seed, with_cuda=device.type == "cuda"
        )
        run_directory = self._create_run_directory(output_root, config)
        shutil.copy2(str(config_path), str(run_directory / "config.yaml"))
        event_log = [
            "timestamp_utc={} event=private_export_started".format(
                datetime.now(timezone.utc).isoformat()
            ),
            "config_path={}".format(config_path),
            "device={}".format(device),
        ]
        train_dataset = self._dataset(config, DatasetSplit.TRAIN)
        public_dataset = self._dataset(config, DatasetSplit.TEST)
        private_dataset = self._dataset(config, "test_private")
        mixed_dataset = self._dataset(config, "test_private_mixed")
        model = PatchCoreFactory().create(config, device)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=2, shuffle=False, num_workers=0
        )
        started = time.perf_counter()
        model.fit(train_loader)
        event_log.append("event=memory_bank_fit_completed seconds={}".format(time.perf_counter() - started))
        threshold, public_metrics = self._public_threshold(model, public_dataset)
        event_log.append("public_threshold={}".format(threshold))
        private_record = self._export_split(model, private_dataset, run_directory, "test_private", threshold)
        mixed_record = self._export_split(model, mixed_dataset, run_directory, "test_private_mixed", threshold)
        metadata = {
            "status": "submission_package_candidate",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": str(config_path),
            "dataset": config.dataset.name,
            "class_name": config.dataset.class_name,
            "seed": config.experiment.seed,
            "device": str(device),
            "threshold_source": "test_public_pixel_f1_optimal_threshold",
            "threshold": threshold,
            "public_metrics_for_threshold_only": public_metrics,
            "splits": {
                "test_private": private_record,
                "test_private_mixed": mixed_record,
            },
            "git": self._git_metadata(),
        }
        (run_directory / "private_export_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        event_log.append("event=private_export_completed")
        (run_directory / "run.log").write_text("\n".join(event_log) + "\n", encoding="utf-8")
        return run_directory

    def _dataset(self, config: AppConfig, split: Any):
        return create_dataset(
            config.dataset.name,
            source=str(config.dataset.root),
            classname=config.dataset.class_name,
            resize=config.dataset.resize,
            imagesize=config.dataset.image_size,
            split=split,
        )

    @staticmethod
    def _public_threshold(model: patchcore.patchcore.PatchCore, dataset) -> Tuple[float, Dict[str, float]]:
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        scores, anomaly_maps, labels, masks = model.predict(loader)
        pixel_metrics = patchcore.metrics.compute_pixelwise_retrieval_metrics(anomaly_maps, masks)
        aupro_005 = patchcore.metrics.compute_aupro(anomaly_maps, masks, fpr_limit=0.05)
        aupro_030 = patchcore.metrics.compute_aupro(anomaly_maps, masks, fpr_limit=0.3)
        return float(pixel_metrics["optimal_threshold"]), {
            "i_auroc": float(patchcore.metrics.compute_imagewise_retrieval_metrics(scores, labels)["auroc"]),
            "p_auroc": float(pixel_metrics["auroc"]),
            "au_pro_0.05": float(aupro_005["aupro"]),
            "au_pro_0.30": float(aupro_030["aupro"]),
        }

    def _export_split(self, model, dataset, run_directory: Path, split_name: str, threshold: float) -> Dict[str, Any]:
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        scores, anomaly_maps, labels, masks = model.predict(loader)
        split_dir = run_directory / split_name
        continuous_dir = split_dir / "anomaly_images"
        thresholded_dir = split_dir / "thresholded_anomaly_images"
        continuous_dir.mkdir(parents=True, exist_ok=True)
        thresholded_dir.mkdir(parents=True, exist_ok=True)
        image_paths = [item[2] for item in dataset.data_to_iterate]
        for image_path, anomaly_map in zip(image_paths, anomaly_maps):
            output_name = Path(image_path).name
            self._save_continuous_map(continuous_dir / output_name, anomaly_map)
            self._save_binary_map(thresholded_dir / output_name, anomaly_map >= threshold)
        np.savez_compressed(
            str(split_dir / "predictions.npz"),
            scores=np.asarray(scores),
            anomaly_maps=np.asarray(anomaly_maps),
            image_paths=np.asarray(image_paths),
            threshold=np.asarray([threshold]),
        )
        return {
            "images": len(image_paths),
            "continuous_dir": str(continuous_dir),
            "thresholded_dir": str(thresholded_dir),
            "predictions_npz": str(split_dir / "predictions.npz"),
        }

    @staticmethod
    def _save_continuous_map(path: Path, anomaly_map: np.ndarray) -> None:
        values = np.asarray(anomaly_map, dtype=np.float32)
        minimum = float(values.min())
        maximum = float(values.max())
        if maximum > minimum:
            values = (values - minimum) / (maximum - minimum)
        else:
            values = np.zeros_like(values)
        Image.fromarray((values * 255).astype(np.uint8)).save(path)

    @staticmethod
    def _save_binary_map(path: Path, mask: np.ndarray) -> None:
        Image.fromarray((np.asarray(mask) * 255).astype(np.uint8)).save(path)

    def _create_run_directory(self, output_root: Path, config: AppConfig) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_directory = output_root.resolve() / "{}_{}_private_export_{}".format(
            config.experiment.name, config.dataset.class_name, timestamp
        )
        run_directory.mkdir(parents=True, exist_ok=False)
        return run_directory

    def _git_metadata(self) -> Dict[str, Any]:
        def git(*arguments: str) -> str:
            return subprocess.check_output(
                ["git", *arguments], cwd=str(self.repository_root), text=True
            ).strip()

        status = git("status", "--porcelain")
        return {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "dirty": bool(status),
            "status": status.splitlines(),
        }


def main() -> None:
    args = parse_args()
    run_directory = MVTecAD2PrivateExporter(Path(__file__).resolve().parents[1]).run(
        args.config, args.output_root
    )
    print("run_directory", run_directory)
    print("PASS")


if __name__ == "__main__":
    main()
