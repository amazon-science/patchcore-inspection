"""Generate paired baseline-versus-candidate PatchCore failure analysis."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class PairedFailureAnalyzer:
    """Analyze score and localization changes for exactly paired test images."""

    def run(
        self, baseline_dir: Path, candidate_dir: Path, output_dir: Path
    ) -> Dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        baseline = self._load_predictions(baseline_dir)
        candidate = self._load_predictions(candidate_dir)
        self._validate_pairing(baseline, candidate)

        paths = baseline["image_paths"]
        labels = baseline["labels"].astype(bool)
        masks = baseline["masks"]
        baseline_scores = baseline["scores"]
        candidate_scores = candidate["scores"]
        baseline_maps = baseline["anomaly_maps"]
        candidate_maps = candidate["anomaly_maps"]
        delta_scores = candidate_scores - baseline_scores
        delta_maps = candidate_maps - baseline_maps

        normal_indices = np.flatnonzero(~labels)
        anomaly_indices = np.flatnonzero(labels)
        top_baseline_normal = normal_indices[
            np.argsort(baseline_scores[normal_indices])[-6:][::-1]
        ]
        top_candidate_normal = normal_indices[
            np.argsort(candidate_scores[normal_indices])[-6:][::-1]
        ]
        low_baseline_anomaly = anomaly_indices[
            np.argsort(baseline_scores[anomaly_indices])[:6]
        ]
        low_candidate_anomaly = anomaly_indices[
            np.argsort(candidate_scores[anomaly_indices])[:6]
        ]
        largest_score_change = np.argsort(np.abs(delta_scores))[-6:][::-1]
        mean_absolute_map_change = np.mean(np.abs(delta_maps), axis=(1, 2))
        largest_map_change = np.argsort(mean_absolute_map_change)[-6:][::-1]

        self._score_grid(
            output_dir / "baseline_highest_normal_scores.png",
            paths,
            baseline_maps,
            baseline_scores,
            top_baseline_normal,
            "Baseline highest-scoring normal samples",
        )
        self._score_grid(
            output_dir / "augmented_highest_normal_scores.png",
            paths,
            candidate_maps,
            candidate_scores,
            top_candidate_normal,
            "Augmented highest-scoring normal samples",
        )
        self._score_grid(
            output_dir / "baseline_lowest_anomaly_scores.png",
            paths,
            baseline_maps,
            baseline_scores,
            low_baseline_anomaly,
            "Baseline lowest-scoring anomalous samples",
        )
        self._score_grid(
            output_dir / "augmented_lowest_anomaly_scores.png",
            paths,
            candidate_maps,
            candidate_scores,
            low_candidate_anomaly,
            "Augmented lowest-scoring anomalous samples",
        )
        self._change_grid(
            output_dir / "largest_image_score_changes.png",
            paths,
            baseline_maps,
            candidate_maps,
            delta_maps,
            largest_score_change,
            delta_scores,
            "Largest absolute image-score changes",
        )
        self._change_grid(
            output_dir / "largest_anomaly_map_changes.png",
            paths,
            baseline_maps,
            candidate_maps,
            delta_maps,
            largest_map_change,
            delta_scores,
            "Largest mean absolute anomaly-map changes",
        )

        inside_delta, outside_delta = self._inside_outside_delta(delta_maps, masks)
        summary = {
            "baseline_run": baseline_dir.name,
            "candidate_run": candidate_dir.name,
            "paired_images": int(len(paths)),
            "terminology": "ranking_without_classification_threshold",
            "image_score_delta": {
                "mean_normal": float(np.mean(delta_scores[normal_indices])),
                "mean_anomaly": float(np.mean(delta_scores[anomaly_indices])),
                "largest_absolute_changes": self._records(
                    paths, labels, largest_score_change, delta_scores
                ),
            },
            "pixel_map_delta": {
                "mean_inside_gt_anomaly_images": float(np.nanmean(inside_delta)),
                "mean_outside_gt_all_images": float(np.mean(outside_delta)),
                "mean_absolute_per_image": float(np.mean(mean_absolute_map_change)),
                "largest_map_changes": self._records(
                    paths, labels, largest_map_change, mean_absolute_map_change
                ),
            },
            "rankings": {
                "baseline_highest_normal": self._records(
                    paths, labels, top_baseline_normal, baseline_scores
                ),
                "candidate_highest_normal": self._records(
                    paths, labels, top_candidate_normal, candidate_scores
                ),
                "baseline_lowest_anomaly": self._records(
                    paths, labels, low_baseline_anomaly, baseline_scores
                ),
                "candidate_lowest_anomaly": self._records(
                    paths, labels, low_candidate_anomaly, candidate_scores
                ),
            },
        }
        (output_dir / "failure_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (output_dir / "failure_analysis.md").write_text(
            self._markdown(summary), encoding="utf-8"
        )
        return summary

    @staticmethod
    def _load_predictions(run_dir: Path) -> Dict[str, np.ndarray]:
        path = run_dir / "predictions.npz"
        if not path.exists():
            raise FileNotFoundError("Missing raw predictions: {}".format(path))
        with np.load(str(path)) as archive:
            return {key: archive[key] for key in archive.files}

    @staticmethod
    def _validate_pairing(
        baseline: Dict[str, np.ndarray], candidate: Dict[str, np.ndarray]
    ) -> None:
        if not np.array_equal(baseline["image_paths"], candidate["image_paths"]):
            raise ValueError(
                "Prediction artifacts do not contain the same ordered image paths."
            )
        if not np.array_equal(baseline["labels"], candidate["labels"]):
            raise ValueError("Prediction artifacts do not contain the same labels.")
        if not np.array_equal(baseline["masks"], candidate["masks"]):
            raise ValueError(
                "Prediction artifacts do not contain the same ground-truth masks."
            )

    @staticmethod
    def _inside_outside_delta(delta_maps: np.ndarray, masks: np.ndarray):
        masks = np.asarray(masks)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        masks = masks > 0.5
        inside = np.full(len(masks), np.nan)
        outside = np.zeros(len(masks))
        for index, mask in enumerate(masks):
            if mask.any():
                inside[index] = float(np.mean(delta_maps[index][mask]))
            outside[index] = float(np.mean(delta_maps[index][~mask]))
        return inside, outside

    @staticmethod
    def _records(paths, labels, indices: Iterable[int], values) -> List[Dict[str, Any]]:
        return [
            {
                "index": int(index),
                "image_path": str(paths[index]),
                "is_anomaly": bool(labels[index]),
                "value": float(values[index]),
            }
            for index in indices
        ]

    @staticmethod
    def _score_grid(
        path, image_paths, maps, scores, indices: Sequence[int], title: str
    ) -> None:
        figure, axes = plt.subplots(2, len(indices), figsize=(3 * len(indices), 6))
        for column, index in enumerate(indices):
            axes[0, column].imshow(Image.open(str(image_paths[index])).convert("RGB"))
            axes[0, column].set_title(
                "idx={} score={:.4f}".format(index, scores[index])
            )
            axes[0, column].axis("off")
            axes[1, column].imshow(maps[index], cmap="inferno")
            axes[1, column].axis("off")
        figure.suptitle(title)
        figure.tight_layout()
        figure.savefig(str(path), dpi=140)
        plt.close(figure)

    @staticmethod
    def _change_grid(
        path,
        image_paths,
        baseline_maps,
        candidate_maps,
        delta_maps,
        indices,
        deltas,
        title,
    ):
        figure, axes = plt.subplots(4, len(indices), figsize=(3 * len(indices), 10))
        for column, index in enumerate(indices):
            axes[0, column].imshow(Image.open(str(image_paths[index])).convert("RGB"))
            axes[0, column].set_title(
                "idx={} delta={:+.4f}".format(index, deltas[index])
            )
            axes[1, column].imshow(baseline_maps[index], cmap="inferno")
            axes[2, column].imshow(candidate_maps[index], cmap="inferno")
            maximum = max(float(np.max(np.abs(delta_maps[index]))), 1e-12)
            axes[3, column].imshow(
                delta_maps[index], cmap="coolwarm", vmin=-maximum, vmax=maximum
            )
            for row in range(4):
                axes[row, column].axis("off")
        axes[1, 0].set_ylabel("baseline")
        axes[2, 0].set_ylabel("augmented")
        axes[3, 0].set_ylabel("delta")
        figure.suptitle(title)
        figure.tight_layout()
        figure.savefig(str(path), dpi=140)
        plt.close(figure)

    @staticmethod
    def _markdown(summary: Dict[str, Any]) -> str:
        image = summary["image_score_delta"]
        pixel = summary["pixel_map_delta"]
        return """# Paired failure analysis

## Finding 1 — normal image-score movement

Observation:
The mean candidate-minus-baseline score on normal images is {normal:+.6f}.

Evidence:
Scores are paired by exact ordered image path; no classification threshold is used.

Hypothesis:
Positive movement may indicate additional normal-sample activation; negative movement may indicate suppression.

Alternative explanation:
Small score movements can arise from the changed memory-bank representation without a meaningful failure change.

Decision:
Treat this as descriptive evidence only.

Next experiment:
Repeat the paired analysis on controlled shifted tests.

## Finding 2 — anomalous image-score movement

Observation:
The mean candidate-minus-baseline score on anomalous images is {anomaly:+.6f}.

Evidence:
The analysis includes every paired anomalous test image.

Hypothesis:
Positive movement may increase anomaly confidence, while negative movement may weaken it.

Alternative explanation:
AUROC depends on ranking rather than absolute score scale.

Decision:
Do not call samples false negatives without a documented threshold policy.

Next experiment:
Inspect score distributions and shifted-test rankings.

## Finding 3 — localization inside ground truth

Observation:
The mean anomaly-map delta inside ground-truth regions is {inside:+.6f}.

Evidence:
The value is averaged over anomalous images with non-empty masks.

Hypothesis:
Positive values may indicate stronger activation on defects.

Alternative explanation:
A global map-scale increase can raise both useful and spurious activation.

Decision:
Interpret jointly with outside-mask movement.

Next experiment:
Compare inside/outside changes under brightness 0.8 and 1.2.

## Finding 4 — localization outside ground truth

Observation:
The mean anomaly-map delta outside ground-truth regions is {outside:+.6f}.

Evidence:
This includes background pixels from normal and anomalous images.

Hypothesis:
Positive values may indicate diffuse localization noise.

Alternative explanation:
Raw anomaly-map scales are model-relative and threshold-free.

Decision:
Use AU-PRO and pixel AUROC as the primary aggregate evidence.

Next experiment:
Examine the saved largest-map-change grid.
""".format(
            normal=image["mean_normal"],
            anomaly=image["mean_anomaly"],
            inside=pixel["mean_inside_gt_anomaly_images"],
            outside=pixel["mean_outside_gt_all_images"],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = PairedFailureAnalyzer().run(
        args.baseline_dir, args.candidate_dir, args.output_dir
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("PASS")


if __name__ == "__main__":
    main()
