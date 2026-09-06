# E0 PatchCore baseline candidate — MVTec AD 2 sheet_metal

Date: 2026-09-05 UTC  
Dataset: MVTec AD 2 `sheet_metal`  
Dataset root: `/mnt/e/Downloads/USTH_ICTLab/datasets/sheet_metal`  
Run status: completed, pending baseline reference check

## Scope

This run follows `PAFA_Experiment_Roadmap.md` E0 for a pure PatchCore baseline on `sheet_metal`. No PAFA adapter, augmentation, scorer change, backbone change, or metric change was introduced.

Because the repository does not contain a known `sheet_metal` baseline reference (`baseline_memory.json`, `baseline_config.yaml`, or prior saved metrics), this run is recorded as a local baseline candidate only. It must not be used to advance to E1 until an expected reference metric is supplied or accepted.

## Pre-run validation

Command:

```bash
python scripts/smoke_test_dataset.py --config configs/patchcore_mvtec_ad2_sheet_metal_cpu_smoke.yaml
python scripts/smoke_test_patchcore.py --config configs/patchcore_mvtec_ad2_sheet_metal_cpu_smoke.yaml --fit-memory-bank
```

Result: PASS.

Validated counts:

| Split | Count |
| --- | ---: |
| train/good | 137 |
| validation/good | 19 |
| test_public/good | 24 |
| test_public/bad | 90 |
| test_public/ground_truth/bad | 90 |

Smoke-test metrics are debug-only and are not reported as scientific baseline results.

## E0 candidate run

Command:

```bash
python scripts/run_patchcore.py --config configs/patchcore_mvtec_ad2_sheet_metal_baseline.yaml
```

Run directory:

```text
experiments/patchcore_mvtec_ad2_sheet_metal_baseline_sheet_metal_20260905T191841Z
```

Metrics:

| Metric | Value |
| --- | ---: |
| image AUROC | 0.7041666666666667 |
| pixel AUROC | 0.9463972797590605 |
| AU-PRO | 0.7387977001460587 |
| AU-PRO regions | 102 |
| train images | 137 |
| test images | 114 |
| fit seconds | 52.782662542000025 |
| inference seconds | 80.744987202 |
| runtime seconds | 133.52775428900003 |

Artifacts saved by the runner:

```text
config.yaml
environment.json
git.json
git.diff.patch
metrics.json
predictions.npz
anomaly_maps/
run.log
```

## Decision gate

E0 is not marked PASS for the roadmap yet. The run completed successfully, but the roadmap requires E0 metrics to match a known baseline within tolerance. No known MVTec AD 2 `sheet_metal` PatchCore reference was found in the repository docs or experiment artifacts.

Next allowed action: provide or define the expected `sheet_metal` baseline reference/tolerance, then compare this run before moving to E1.
