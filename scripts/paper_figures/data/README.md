# Figure 3 derived inputs

`fig03_trace_components.csv` is a lossless tidy export of the summary
statistics stored in the local state-trace snapshots. The older hand-
transcribed range and zero-fraction CSVs were removed after this table became
the single source for Figure 3.

The optimizer traces use the run identifiers stored in each row. Figure 3 is
evidence of state/tensor heterogeneity, not a controlled ranking of optimizers:
the representative paths do not all share the same optimizer hyperparameters.

To regenerate the checked table, run
`scripts/paper_figures/export_fig03_trace_data.py` against the local
`state_traces/` directory. It produces `fig03_trace_components.csv` with one
row per parameter and saved state component. Current snapshots contain
block-range median/p95/max, not individual block samples; any distribution
plot must therefore identify parameters/state components—not blocks—as its
sampling unit.

## Exported component table

The checked `fig03_trace_components.csv` contains:

- 18,564 component rows from 52 snapshots;
- four optimizer runs and 13 traced steps from 50 through 10K;
- 10,452 optimizer/step/parameter identities;
- no missing fields and no duplicate component identities;
- SHA-256 `56e4f54d68b6e17751329ff0e7793f36e7f3ce97d675136900dc24b7a34941aa`.

At step 10K, Attention and MLP provide 88 and 66 parameter-state tensors per
applicable component, respectively. Embedding and LM head provide one tensor
per component, so they may appear as individual heatmap values or trace lines
but must not be rendered as parameter-level violin distributions.

The full table remains the checked source for the frozen Figure 3 and for its
reproducibility artifact. A later repository-release pass may derive a compact
plot-only table, but the full export should not be deleted until that compact
table and its provenance have been verified.

## Figure 4 controlled inputs

The three `fig04_*.csv` files are compact transcriptions of executed notebook
cells, with the source cell recorded in every row:

- `fig04_state_sensitivity.csv`: F7 trace-calibrated update sensitivity;
- `fig04_fixed_range.csv`: A1 adaptive-versus-fixed update error;
- `fig04_pareto.csv`: D4 5K EMA memory--drift Pareto data.

Values printed as `>100` or `0.000` by F7 are stored as explicitly censored
observations rather than invented point estimates. The figure renders them as
`>100` and `<0.001`, respectively.
