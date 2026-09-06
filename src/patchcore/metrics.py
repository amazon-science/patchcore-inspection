"""Anomaly metrics."""

import numpy as np
from skimage import measure
from sklearn import metrics


def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights, anomaly_ground_truth_labels
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
    fpr, tpr, thresholds = metrics.roc_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    return {"auroc": auroc, "fpr": fpr, "tpr": tpr, "threshold": thresholds}


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.

    Args:
        anomaly_segmentations: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    fpr, tpr, thresholds = metrics.roc_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    precision, recall, thresholds = metrics.precision_recall_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    optimal_threshold = thresholds[np.argmax(F1_scores)]
    predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
    }


def compute_aupro(anomaly_segmentations, ground_truth_masks, fpr_limit=0.3):
    """Compute normalized area under the per-region-overlap curve.

    This independent NumPy implementation follows the public MVTec protocol
    documented by Anomalib (Apache-2.0): connected ground-truth regions have
    equal weight, background pixels define global FPR, and the curve is
    integrated through ``fpr_limit``.
    """
    if not 0 < fpr_limit <= 1:
        raise ValueError("fpr_limit must be in the interval (0, 1].")
    predictions = np.asarray(anomaly_segmentations, dtype=np.float64)
    targets = np.asarray(ground_truth_masks)
    if targets.ndim == predictions.ndim + 1 and targets.shape[1] == 1:
        targets = targets[:, 0]
    if predictions.shape != targets.shape or predictions.ndim != 3:
        raise ValueError("AU-PRO inputs must have matching N x H x W shapes.")
    if not np.isfinite(predictions).all():
        raise ValueError("AU-PRO predictions must contain only finite values.")
    targets = targets > 0.5

    region_labels = np.zeros(targets.shape, dtype=np.int64)
    region_sizes = [0]
    next_label = 1
    for image_index, target in enumerate(targets):
        image_labels = measure.label(target, connectivity=2)
        for local_label in range(1, int(image_labels.max()) + 1):
            region = image_labels == local_label
            region_labels[image_index, region] = next_label
            region_sizes.append(int(region.sum()))
            next_label += 1
    number_of_regions = next_label - 1
    if number_of_regions == 0:
        raise ValueError("AU-PRO requires at least one anomalous ground-truth region.")

    flat_labels = region_labels.reshape(-1)
    flat_predictions = predictions.reshape(-1)
    background = flat_labels == 0
    background_pixels = int(background.sum())
    if background_pixels == 0:
        raise ValueError("AU-PRO requires at least one background pixel.")

    false_positive_change = background.astype(np.float64)
    pro_change = np.zeros(flat_predictions.shape, dtype=np.float64)
    foreground = ~background
    sizes = np.asarray(region_sizes, dtype=np.float64)
    pro_change[foreground] = 1.0 / sizes[flat_labels[foreground]]

    order = np.argsort(-flat_predictions, kind="stable")
    sorted_predictions = flat_predictions[order]
    cumulative_fpr = np.cumsum(false_positive_change[order]) / background_pixels
    cumulative_pro = np.cumsum(pro_change[order]) / number_of_regions
    # Tied scores belong to one threshold and must be included together.
    group_end = np.r_[sorted_predictions[:-1] != sorted_predictions[1:], True]
    fpr = np.r_[0.0, cumulative_fpr[group_end]]
    pro = np.r_[0.0, cumulative_pro[group_end]]

    last_index = int(np.flatnonzero(fpr <= fpr_limit)[-1])
    clipped_fpr = fpr[: last_index + 1]
    clipped_pro = pro[: last_index + 1]
    if clipped_fpr[-1] < fpr_limit:
        next_index = last_index + 1
        if next_index >= len(fpr):
            limit_pro = clipped_pro[-1]
        else:
            limit_pro = np.interp(
                fpr_limit,
                [fpr[last_index], fpr[next_index]],
                [pro[last_index], pro[next_index]],
            )
        clipped_fpr = np.r_[clipped_fpr, fpr_limit]
        clipped_pro = np.r_[clipped_pro, limit_pro]

    return {
        "aupro": float(np.trapz(clipped_pro, clipped_fpr) / fpr_limit),
        "fpr_limit": float(fpr_limit),
        "number_of_regions": number_of_regions,
        "implementation": "robustvisionad_numpy_v1",
        "protocol": "connected_components_equal_weight_normalized_auc",
    }
