# AU-PRO evaluation protocol

The project uses `patchcore.metrics.compute_aupro` for pixel-level anomaly
localization evaluation.

- Protocol: equal-weight per-region overlap over connected ground-truth regions.
- False-positive rate: computed globally over all ground-truth background pixels.
- Integration range: FPR 0.00 through 0.30.
- Reported score: trapezoidal area divided by 0.30, yielding a normalized value
  in the interval 0 through 1.
- Connected-component convention: 8-connectivity in two dimensions.
- Implementation identifier: `robustvisionad_numpy_v1`.

The implementation was written for this repository using NumPy and scikit-image.
Its algorithm and metadata conventions follow the public Anomalib AUPRO
implementation, Copyright (C) 2022-2026 Intel Corporation, available under the
Apache License 2.0:

https://github.com/open-edge-platform/anomalib/blob/main/src/anomalib/metrics/aupro.py

The original Amazon PatchCore repository reports PRO results but explicitly does
not distribute its own PRO metric implementation for license reasons. No code
from that unavailable implementation is included here.
