# Paper figure scripts

Publication figure code lives here, separate from training, trace, and report
analysis scripts. Stable output names use figure numbers rather than internal
design revisions.

```text
scripts/paper_figures/
├─ style.py
├─ fig01_state_landscape.py
├─ fig02_al_encoding.py
├─ fig03_state_heterogeneity.py      # frozen publication figure
├─ fig04_controlled_fidelity.py      # controlled fidelity figure
├─ fig05_training_curves.py          # 20K-step validation trajectories
├─ export_fig03_trace_data.py        # derive tidy data from local traces
├─ data/                              # Figure-specific derived inputs
└─ build_all.py
```

From the repository root:

```bash
python scripts/paper_figures/build_all.py
```

The scripts resolve the artifact root from their own location. Stable PDF, SVG,
and PNG assets are written together to `figures/`. Temporary comparisons and
unapproved review candidates go to `tmp/figure_reviews/` and are not release
artifacts.

To compare Figure 2 marker densities temporarily:

```bash
python scripts\paper_figures\fig02_al_encoding.py --review-densities
```

The comparison PNGs are written to `tmp/figure_reviews/fig02_density/`. The
main Figure 2 uses 64 illustrative markers. This visual count is not the
quantizer code count; the exact AL8/AL16 counts are stated in the figure and
caption.

Figure 3 is generated as a stable publication asset:

```bash
python scripts\paper_figures\fig03_state_heterogeneity.py
```

Its input is the checked component table under `scripts/paper_figures/data/`.
The script contains only the selected unified-grouped design and writes
`figures/fig03_state_heterogeneity.{pdf,svg,png}`. Its manual tuning controls
are grouped at the beginning of the file: canvas/layout, shared axis,
box/point styling, singleton marker styling, colors, and text positions.

The design places selected 2D state paths, Embedding, Attention, MLP, and LM
head in one coordinate system with a linear 0--90 bit axis. Attention/MLP use
boxes plus raw parameter-state observations; Embedding and LM head use small,
independently sized diamonds for singleton tensors rather than implying a
distribution. Only the global maximum and minimum singleton observations are
annotated. Superseded heatmap, violin, separate-facet, log-range-axis, and
dumbbell variants have been removed from the publication script.

To reproduce the per-parameter/per-state summary table from local traces, run
either from the paper root or this script directory:

```bash
python scripts\paper_figures\export_fig03_trace_data.py --base-dir state_traces
```

Relative `--base-dir` and `--output` paths are resolved from the paper root,
so the command does not depend on the terminal's current directory. From
`scripts\paper_figures`, `py export_fig03_trace_data.py` is sufficient.

This writes `scripts/paper_figures/data/fig03_trace_components.csv`. Current
trace snapshots do not contain individual block ranges, so the CSV supports
parameter-level strip/box/violin comparisons but not a true block-level
distribution. The project-root `data/` directory remains reserved for local
datasets and is not used for figure artifacts.

Figure 4 combines executed controlled experiments F7, A1, and D4 from
`theory_and_ablation_final.ipynb`. Figure 5 parses the 20K-step evaluation
trajectories from `reports_md/tb_analysis_report.md`, retaining the report as
the auditable intermediate derived from the TensorBoard event files.
