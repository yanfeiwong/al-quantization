# Reproducibility environment capture

Run this script from the same Python environment used for the paper experiments:

```bash
python scripts/reproducibility/capture_environment.py
```

The default outputs in this artifact repository are:

- `reports_md/environment.md`
- `reports_md/environment.json`

An explicit destination can be supplied when needed:

```bash
python scripts/reproducibility/capture_environment.py \
  --repo-root . \
  --output reports_md/environment.md
```

Use `--extra-package DISTRIBUTION_NAME` to record another result-relevant
Python distribution. The option may be repeated.

The generated report deliberately omits local installation directories. For a
package installed from a local wheel, it retains only the wheel filename and
the digest stored in `direct_url.json`. A machine-readable JSON companion is
written alongside the Markdown report.

## Experiment source anchor

The analyzed paper experiments currently use implementation commit
`4f544d8b8eebaf50053a4e8a27096e79b049b480` (Adafactor8Bit v0.4.3) as
their verified source anchor. This is the revision that introduced independent
confidence-state quantization through `conf_quant_type`.

Later commits that only change author metadata, documentation, manuscript
sources, or figure-generation scripts do not replace the experiment anchor.
The artifact distinguishes the implementation revision used by the experiments
from the revision used to assemble this repository. See
`EXPERIMENT_SNAPSHOT.md` and fill its assembly fields after the initial clean
artifact commit.

## CUDA numerical reproducibility

The Python fallback is the semantic reference. Full-rank CUDA paths that do
not use factored accumulation are tested for strict repeatability. Some 2D
factored CUDA paths compute row and column statistics with `atomicAdd`.
Floating-point addition is non-associative, and the order in which CUDA threads
reach an atomic accumulation is not fixed. Repeated runs can therefore differ
in the last bits even when library-level deterministic options are enabled.

This effect is not treated as an algorithmic mismatch, but it is not hidden as
bitwise determinism either. The smoke suite reports two separate policies:

- **Reference Algorithm Alignment** uses the Python/reference path and a
  strict parameter-difference threshold of `1e-5`.
- **CUDA Numerical Consistency** reports `PASS` below `1e-5`, `WARN` from
  `1e-5` to below `5e-3`, and `FAIL` at or above `5e-3` for paths with
  parallel accumulation-order effects.

`CUBLAS_WORKSPACE_CONFIG` and PyTorch deterministic settings remain useful for
library operations, but they do not impose a fixed accumulation order on a
custom CUDA `atomicAdd` kernel. See `reports_md/smoke_test.md` for the measured
run-to-run and CUDA-versus-Python differences.
