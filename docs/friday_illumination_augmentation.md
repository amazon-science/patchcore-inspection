# Friday illumination augmentation report — 2026-09-04

## Objective

Verify baseline reproducibility, then evaluate a simple illumination-aware
PatchCore training baseline on MVTec AD `bottle`.

## Experimental control

The baseline and candidate use the same dataset, class, split, seed, backbone,
feature layers, image size, embedding dimensions, coreset ratio, patch size,
nearest-neighbor count, and evaluation metrics. The candidate changes only the
normal-training image transform:

```text
brightness: uniform [0.8, 1.2]
contrast:   uniform [0.8, 1.2]
gamma:      uniform [0.8, 1.2]
```

The test set is never augmented. The number of training images remains 209, so
the comparison is not confounded by duplicating training data or enlarging the
memory bank solely through extra samples. Seed 42, deterministic sample order,
and zero DataLoader workers make the sampled transform sequence reproducible.

## Implementation

- Added typed augmentation configuration and validation.
- Added one composable `RandomIlluminationTransform`; no dataset subclasses were
  duplicated.
- Integrated the transform into the existing MVTec dataset construction only
  for the training split.
- Added exact per-image factor logging for all 209 training images.
- Added raw prediction persistence (`predictions.npz`).
- Added stage timestamps and fit/inference durations to `run.log`.
- Added Git diff capture for runs performed from an uncommitted implementation.
- Added a machine-readable baseline-versus-candidate comparison command.

## Commands

```bash
python scripts/run_patchcore.py \
  --config configs/experiments/illumination_aug.yaml

python scripts/compare_experiments.py \
  --baseline experiments/patchcore_baseline_bottle_20260904T104716Z/metrics.json \
  --candidate experiments/patchcore_illumination_aug_bottle_20260904T105453Z/metrics.json \
  --output experiments/patchcore_illumination_aug_bottle_20260904T105453Z/comparison.json

pytest -q
```

## Results

| Metric | Reproduced baseline | Illumination augmentation | Delta |
|---|---:|---:|---:|
| I-AUROC | 1.000000 | 1.000000 | 0.000000 |
| P-AUROC | 0.985153 | 0.984672 | -0.000482 |
| AU-PRO@0.30 | 0.947059 | 0.948729 | +0.001670 |
| Runtime (seconds) | 113.193 | 104.856 | -8.337 |

The augmentation preserves perfect image-level detection and slightly improves
region-balanced localization, but slightly reduces pixel AUROC. With one class
and one seed, this is mixed preliminary evidence rather than proof that the
augmentation is generally better. Runtime variation is not interpreted as an
algorithmic speed improvement.

## Verification

- Full augmented run: PASS.
- Unit and integration tests: 36 passed, 0 failed.
- Training images: 209.
- Test images: 83.
- Ground-truth anomaly regions: 68.
- Saved anomaly maps: 83.
- Logged augmentation samples: 209.

## Artifacts

```text
experiments/patchcore_illumination_aug_bottle_20260904T105453Z/
├── anomaly_maps/                 # 83 rendered maps
├── augmentation_samples.json     # 209 exact factor triplets and image paths
├── comparison.json               # exact baseline/candidate deltas
├── config.yaml                   # frozen candidate configuration
├── environment.json              # runtime environment
├── git.diff.patch                # uncommitted implementation used by this run
├── git.json                      # commit, branch, and dirty status
├── metrics.json                  # scientific results and timing
├── predictions.npz               # raw scores, maps, labels, and masks
└── run.log                       # stage-by-stage execution log
```

## Next gate

Do not tune augmentation ranges based on this single result. The next runbook
scope is automatic summary generation and false-positive/false-negative failure
analysis using the saved raw predictions.
