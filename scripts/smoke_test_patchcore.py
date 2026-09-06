"""Run one configured image through PatchCore feature and embedding stages."""

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

import patchcore.backbones
import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils
from patchcore.config import AppConfig
from patchcore.config import ConfigLoader
from patchcore.datasets.factory import create_dataset
from patchcore.datasets.mvtec import DatasetSplit
from patchcore.datasets.mvtec import MVTecDataset
from patchcore.device import DeviceManager


class PatchCoreSmokeTester:
    """Validate backbone features and spatial patch embeddings for one image."""

    def run(self, config_path: Path, fit_memory_bank: bool = False) -> Dict[str, Any]:
        config = ConfigLoader().load(config_path)
        device = DeviceManager().resolve(config.device.requested)
        patchcore.utils.fix_seeds(
            config.experiment.seed, with_cuda=device.type == "cuda"
        )
        dataset = create_dataset(
            config.dataset.name,
            source=str(config.dataset.root),
            classname=config.dataset.class_name,
            resize=config.dataset.resize,
            imagesize=config.dataset.image_size,
            split=DatasetSplit.TRAIN,
        )
        image = dataset[0]["image"].unsqueeze(0).to(device)

        backbone = patchcore.backbones.load(config.patchcore.backbone)
        model = patchcore.patchcore.PatchCore(device)
        coreset_sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(
            percentage=config.patchcore.coreset_ratio,
            device=device,
        )
        model.load(
            backbone=backbone,
            layers_to_extract_from=list(config.patchcore.feature_layers),
            device=device,
            input_shape=[3, config.dataset.image_size, config.dataset.image_size],
            pretrain_embed_dimension=(config.patchcore.pretrain_embedding_dimension),
            target_embed_dimension=config.patchcore.target_embedding_dimension,
            patchsize=config.patchcore.patch_size,
            anomaly_score_num_nn=config.patchcore.nearest_neighbors,
            featuresampler=coreset_sampler,
            nn_method=patchcore.common.FaissNN(on_gpu=False),
        )

        feature_maps = model.forward_modules["feature_aggregator"](image)
        embeddings, patch_shapes = model._embed(image, provide_patch_shapes=True)
        embeddings = np.asarray(embeddings)
        if not np.isfinite(embeddings).all():
            raise AssertionError("Patch embeddings contain NaN or Inf values.")
        if embeddings.shape[0] <= 0:
            raise AssertionError("Patch embedding count must be positive.")
        if embeddings.shape[1] != config.patchcore.target_embedding_dimension:
            raise AssertionError("Patch embedding dimension does not match config.")

        backbone_frozen = all(
            not parameter.requires_grad for parameter in model.backbone.parameters()
        )
        if not backbone_frozen or model.backbone.training:
            raise AssertionError("PatchCore backbone must be frozen in eval mode.")

        report = {
            "device": str(device),
            "input_shape": list(image.shape),
            "layer2_shape": list(
                feature_maps[config.patchcore.feature_layers[0]].shape
            ),
            "layer3_shape": list(
                feature_maps[config.patchcore.feature_layers[1]].shape
            ),
            "patch_embedding_shape": list(embeddings.shape),
            "patch_spatial_shapes": [list(shape) for shape in patch_shapes],
            "patch_count": int(embeddings.shape[0]),
            "embedding_dimension": int(embeddings.shape[1]),
            "nan": bool(np.isnan(embeddings).any()),
            "inf": bool(np.isinf(embeddings).any()),
            "backbone_frozen": backbone_frozen,
            "backbone_eval": not model.backbone.training,
        }
        if fit_memory_bank:
            max_samples = config.dataset.max_train_samples
            if max_samples is None:
                raise ValueError(
                    "Memory-bank smoke tests require dataset.max_train_samples."
                )
            sample_count = min(max_samples, len(dataset))
            subset = torch.utils.data.Subset(dataset, range(sample_count))
            loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)
            patches_before_coreset = sample_count * embeddings.shape[0]
            model.fit(loader)
            memory_bank = model.anomaly_scorer.detection_features
            expected_coreset_size = int(
                patches_before_coreset * config.patchcore.coreset_ratio
            )
            if memory_bank.shape != (
                expected_coreset_size,
                config.patchcore.target_embedding_dimension,
            ):
                raise AssertionError(
                    "Unexpected coreset shape: {}".format(memory_bank.shape)
                )
            if not np.isfinite(memory_bank).all():
                raise AssertionError("Memory bank contains NaN or Inf values.")
            report.update(
                {
                    "normal_training_images": sample_count,
                    "patches_before_coreset": patches_before_coreset,
                    "coreset_ratio": config.patchcore.coreset_ratio,
                    "patches_after_coreset": int(memory_bank.shape[0]),
                    "memory_bank_shape": list(memory_bank.shape),
                }
            )
            test_dataset = create_dataset(
                config.dataset.name,
                source=str(config.dataset.root),
                classname=config.dataset.class_name,
                resize=config.dataset.resize,
                imagesize=config.dataset.image_size,
                split=DatasetSplit.TEST,
            )
            normal_sample = self._find_sample(test_dataset, is_anomaly=False)
            anomaly_sample = self._find_sample(test_dataset, is_anomaly=True)
            report.update(self._score_sample(model, normal_sample, "normal"))
            report.update(self._score_sample(model, anomaly_sample, "anomaly"))
            report.update(self._evaluate_subset(model, test_dataset, config))
        return report

    @staticmethod
    def _find_sample(dataset: MVTecDataset, is_anomaly: bool) -> Dict[str, Any]:
        for index, item in enumerate(dataset.data_to_iterate):
            if (item[1] != "good") == is_anomaly:
                return dataset[index]
        raise AssertionError("Required test sample was not found.")

    @staticmethod
    def _positive_mask_indices(
        dataset: MVTecDataset, anomaly_indices: list, count: int
    ) -> list:
        selected = []
        for index in anomaly_indices:
            if torch.any(dataset[index]["mask"] > 0):
                selected.append(index)
            if len(selected) == count:
                break
        if len(selected) < count:
            raise AssertionError("Not enough anomalous samples with positive masks.")
        return selected

    @staticmethod
    def _score_sample(
        model: patchcore.patchcore.PatchCore,
        sample: Dict[str, Any],
        prefix: str,
    ) -> Dict[str, Any]:
        image = sample["image"].unsqueeze(0).to(model.device)
        embeddings = np.asarray(model._embed(image))
        patch_scores, distances, _ = model.anomaly_scorer.predict([embeddings])
        image_scores, anomaly_maps = model._predict(image)
        anomaly_map = np.asarray(anomaly_maps[0])
        values = [patch_scores, distances, image_scores, anomaly_map]
        if not all(np.isfinite(value).all() for value in values):
            raise AssertionError("{} scoring produced NaN or Inf.".format(prefix))
        return {
            "{}_nn_distance_shape".format(prefix): list(distances.shape),
            "{}_patch_score_min".format(prefix): float(np.min(patch_scores)),
            "{}_patch_score_max".format(prefix): float(np.max(patch_scores)),
            "{}_image_score".format(prefix): float(image_scores[0]),
            "{}_anomaly_map_shape".format(prefix): list(anomaly_map.shape),
        }

    @staticmethod
    def _evaluate_subset(
        model: patchcore.patchcore.PatchCore,
        dataset: MVTecDataset,
        config: AppConfig,
    ) -> Dict[str, Any]:
        max_samples = config.dataset.max_test_samples
        if max_samples is None or max_samples < 2:
            raise ValueError(
                "Evaluation smoke tests require dataset.max_test_samples >= 2."
            )
        normal_indices = [
            index
            for index, item in enumerate(dataset.data_to_iterate)
            if item[1] == "good"
        ]
        anomaly_indices = [
            index
            for index, item in enumerate(dataset.data_to_iterate)
            if item[1] != "good"
        ]
        normal_count = min(max_samples // 2, len(normal_indices))
        anomaly_count = min(max_samples - normal_count, len(anomaly_indices))
        selected_indices = normal_indices[:normal_count]
        selected_indices.extend(
            PatchCoreSmokeTester._positive_mask_indices(
                dataset, anomaly_indices, anomaly_count
            )
        )
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, selected_indices),
            batch_size=1,
            shuffle=False,
        )
        scores, anomaly_maps, labels, masks = model.predict(loader)
        image_metrics = patchcore.metrics.compute_imagewise_retrieval_metrics(
            scores, labels
        )
        pixel_metrics = patchcore.metrics.compute_pixelwise_retrieval_metrics(
            anomaly_maps, masks
        )
        aupro_005_metrics = (
            patchcore.metrics.compute_aupro(anomaly_maps, masks, fpr_limit=0.05)
            if config.evaluation.au_pro
            else None
        )
        aupro_030_metrics = (
            patchcore.metrics.compute_aupro(anomaly_maps, masks, fpr_limit=0.3)
            if config.evaluation.au_pro
            else None
        )
        metric_values = [image_metrics["auroc"], pixel_metrics["auroc"]]
        if not np.isfinite(metric_values).all():
            raise AssertionError("Debug evaluation produced NaN or Inf metrics.")
        return {
            "debug_metrics_not_for_reporting": True,
            "evaluation_normal_images": normal_count,
            "evaluation_anomaly_images": anomaly_count,
            "debug_image_auroc": float(image_metrics["auroc"]),
            "debug_pixel_auroc": float(pixel_metrics["auroc"]),
            "debug_au_pro_0.05": (
                aupro_005_metrics["aupro"] if aupro_005_metrics else None
            ),
            "debug_au_pro_0.30": (
                aupro_030_metrics["aupro"] if aupro_030_metrics else None
            ),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fit-memory-bank", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = PatchCoreSmokeTester().run(args.config, args.fit_memory_bank)
    for key, value in report.items():
        print("{}: {}".format(key, value))
    print("PASS")


if __name__ == "__main__":
    main()
