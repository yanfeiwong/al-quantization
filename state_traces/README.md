# State-trace archive

This directory contains 52 `.pt` snapshots from four 10K-step TinyLlama runs,
plus their TensorBoard event files. The snapshots store summary tensors and
statistics used to derive `scripts/paper_figures/data/fig03_trace_components.csv`;
they do not contain model checkpoints, optimizer objects, source paths, or
training text.

The checked summaries are:

- `reports_md/state_trace_report.md` — compact selected-state report;
- `reports_md/state_trace_report_full.md` — full per-parameter/per-state report.

Rebuild the Figure 3 input table from the repository root with:

```bash
python scripts/paper_figures/export_fig03_trace_data.py --base-dir state_traces
```

As with all pickle-backed PyTorch files, load `.pt` files only from a trusted
artifact source. The release audit verified that these files are valid PyTorch
ZIP serializations and reference only tensor reconstruction primitives,
`FloatStorage`, and `OrderedDict`.
