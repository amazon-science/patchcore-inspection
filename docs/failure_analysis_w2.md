# W2 controlled robustness and failure analysis — 2026-09-04

## Scope

This report closes the W2 P0 gate for MVTec AD `bottle`, seed 42. It compares
the frozen PatchCore baseline with the illumination-augmented training baseline
on the original test set and on controlled brightness factors 0.8 and 1.2.
All runs use the same architecture, data split, and evaluation protocol.

## Shift validation

The fixed brightness transform changes photometry only. Visual and automated
checks confirm unchanged geometry, masks, dtype/range, and recognizable defects.
The original images contain a large saturated white background; brightness 1.2
increases saturation slightly, while brightness 0.8 removes it. This limits how
strongly the bright-shift result can be generalized, but no artificial geometry
or vanished defect was observed.

## Results

| Train method | Test condition | I-AUROC | P-AUROC | AU-PRO@0.30 |
|---|---:|---:|---:|---:|
| Vanilla | Original | 1.000000 | 0.985153 | 0.947059 |
| Augmented | Original | 1.000000 | 0.984672 | 0.948729 |
| Vanilla | Brightness 0.8 | 1.000000 | 0.980274 | 0.923716 |
| Augmented | Brightness 0.8 | 1.000000 | 0.984500 | 0.949523 |
| Vanilla | Brightness 1.2 | 1.000000 | 0.985040 | 0.946115 |
| Augmented | Brightness 1.2 | 1.000000 | 0.984679 | 0.948449 |

Degradation is shifted minus clean for the same training method. Recovery is
augmented shifted minus vanilla shifted under the same test condition.

| Condition | Metric | Vanilla degradation | Augmented degradation | Recovery |
|---|---|---:|---:|---:|
| Brightness 0.8 | I-AUROC | +0.000000 | +0.000000 | +0.000000 |
| Brightness 0.8 | P-AUROC | -0.004879 | -0.000171 | +0.004226 |
| Brightness 0.8 | AU-PRO | -0.023343 | +0.000793 | +0.025806 |
| Brightness 1.2 | I-AUROC | +0.000000 | +0.000000 | +0.000000 |
| Brightness 1.2 | P-AUROC | -0.000113 | +0.000008 | -0.000361 |
| Brightness 1.2 | AU-PRO | -0.000945 | -0.000280 | +0.002335 |

The clean cost of augmentation is 0.000000 I-AUROC, -0.000482 P-AUROC, and
+0.001670 AU-PRO. The controlled evidence therefore shows a meaningful dark
shift localization failure and recovery, but no repeated image-level failure
and only negligible degradation under brightness 1.2.

## Paired clean-test failure analysis

All 83 images were matched by image path. No classification threshold was
defined, so samples are described as highest-scoring normal and lowest-scoring
anomalous images, not false positives or false negatives.

- Mean augmented-minus-baseline image score: +0.060057 for normal images and
  -0.029401 for anomalous images.
- Mean anomaly-map delta inside ground-truth regions: -0.012985.
- Mean anomaly-map delta outside ground truth over all images: +0.069449.
- Mean absolute map change per image: 0.090109.
- Both methods rank `good/001.png` as the highest-scoring normal and
  `contamination/003.png` as the lowest-scoring anomaly.

### Finding 1 — clean localization becomes more diffuse

**Observation:** augmentation raises activation outside ground truth while
slightly lowering it inside anomaly regions.

**Evidence:** outside-GT mean delta is +0.069449 and inside-GT mean delta is
-0.012985; P-AUROC also falls by 0.000482 on clean data.

**Hypothesis:** the broader illumination memory bank makes local descriptors
less selective and spreads anomaly activation into normal bottle structure.

**Alternative explanation:** one seed and one class may expose ordinary
coreset-sampling variation rather than a stable augmentation effect.

**Decision:** retain this as descriptive evidence, not a causal conclusion.

**Next experiment:** repeat across seeds and inspect class-specific map deltas.

### Finding 2 — darkness is the observed nuisance

**Observation:** vanilla localization degrades at brightness 0.8, while the
same model is nearly unchanged at brightness 1.2.

**Evidence:** vanilla AU-PRO changes by -0.023343 at 0.8 and only -0.000945 at
1.2; I-AUROC remains 1.0 in both cases.

**Hypothesis:** darker pixels disturb local feature matching more than the
bright transform because the original background is already near saturation.

**Alternative explanation:** the selected severities are not perceptually
symmetric and the `bottle` class may be unusually easy at image level.

**Decision:** the current protocol demonstrates a directional localization
failure, not a general illumination-robustness failure.

**Next experiment:** calibrate perceptual severity, then test contrast/gamma or
a harder class without tuning against these test results.

### Finding 3 — simple augmentation recovers the dark-shift loss

**Observation:** augmentation recovers localization performance under 0.8.

**Evidence:** recovery is +0.004226 P-AUROC and +0.025806 AU-PRO, with negligible
clean cost. Augmented AU-PRO at 0.8 is slightly above its own clean result.

**Hypothesis:** train-time illumination diversity already covers this simple
global brightness nuisance.

**Alternative explanation:** the candidate combines brightness, contrast, and
gamma, so the recovery cannot be attributed to brightness augmentation alone.

**Decision:** use augmentation as the W2 robustness baseline; do not claim a
new calibration method is needed from this evidence.

**Next experiment:** isolate augmentation components only after defining a
stronger, externally justified shifted-test protocol.

## GO / NO-GO decision

**NO-GO for nuisance-shift calibration under the current W2 evidence.**

The required repeated degradation is absent: brightness 0.8 affects pixel and
region localization, but brightness 1.2 is nearly neutral and image AUROC never
changes. The paired clean analysis shows score/map movement, but does not prove
a shared illumination-induced feature-space drift. Simple augmentation also
recovers the meaningful 0.8 degradation with negligible clean cost.

This decision applies only to calibration as the immediate W3 direction. A
future GO requires a validated shift protocol that causes repeatable degradation
across severities/classes, evidence of shared nuisance drift in feature space,
and incomplete recovery from the simple augmentation baseline.

## Reproducible artifacts

- `experiments/summary.csv`: automatic index of 11 completed runs.
- `experiments/robustness/robustness_summary.{csv,json}`: exact degradation,
  recovery, clean-cost definitions, and run IDs.
- `experiments/failure_analysis/`: six visual grids plus JSON/Markdown analysis.
- `experiments/shift_validation/`: visual grid and machine-readable validation.
- Every selected run directory contains frozen config, logs, metadata, metrics,
  anomaly maps, and raw predictions.

Final verification: 42 tests passed; `git diff --check` reported no whitespace
errors. Torchvision emitted only the already-known deprecated `pretrained`
argument warnings.
