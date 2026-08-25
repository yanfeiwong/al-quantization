# Paper artifact manifest

This repository contains the experiment evidence and generation tools for
*Beyond Dense Adam States*. It is separate from both the installable optimizer
implementation and the manuscript LaTeX source.

## Source boundary

- Experiment implementation: Adafactor8Bit v0.4.3, commit
  `4f544d8b8eebaf50053a4e8a27096e79b049b480`.
- The implementation commit anchors the optimizer code used by the reported
  experiments. It does not contain this manuscript or the paper-figure sources.
- Later paper-only, documentation, or author-metadata commits do not replace the
  experiment anchor.

## Claim-to-artifact map

| Evidence | Checked intermediate | Generator or validator |
|---|---|---|
| State topology and AL encoding | `figures/fig01_*`, `figures/fig02_*` | matching scripts under `scripts/paper_figures/` |
| State ranges and exact zeros | `scripts/paper_figures/data/fig03_trace_components.csv` | `scripts/paper_figures/export_fig03_trace_data.py` and raw state traces |
| Controlled fidelity | `scripts/paper_figures/data/fig04_*.csv` | executed notebook cells F7, A1, and D4 |
| Training curves and benchmark tables | `reports_md/tb_analysis_report.md` | TensorBoard events and `scripts/analyze_tb.py` |
| Completed training records | `benchmarks/` | frozen TensorBoard event archive |
| State-range source snapshots | `state_traces/` | `scripts/trace_states.py` and `scripts/analyze_traces.py` |
| Reference/CUDA validation | `reports_md/smoke_test.md` and `.json` | `scripts/smoke_test.py` |
| Environment | `reports_md/environment.md` and `.json` | `scripts/reproducibility/capture_environment.py` |

The release-ready figure assets are PDF/SVG/PNG triples under `figures/`.
Review candidates, caches, and obsolete versioned wireframes are not scientific
inputs and will be excluded from the public release.

Raw TensorBoard events and state-trace snapshots are included as ancillary
artifacts; they are not manuscript or LaTeX inputs. The public repository keeps
the frozen TensorBoard event files byte-for-byte unchanged. Windows peak memory
uses PyTorch allocation counters; optimizer-state storage is computed from live
CUDA tensors. Neither is an operating-system measure of total resident VRAM.
