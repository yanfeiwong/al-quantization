# Adaptive Log-Space Quantization: Paper Artifacts

Reproducibility artifacts for
**[Beyond Dense Adam States: Adaptive Log-Space Quantization for Memory-Efficient Optimizers](https://arxiv.org/abs/2608.22322)**
([PDF](https://arxiv.org/pdf/2608.22322)).

This repository contains the frozen experiment launchers, TensorBoard records,
state-trace summaries, controlled-study notebook, analysis scripts, and figure
sources used for the paper. The installable optimizer implementation lives in
[Adafactor8Bit](https://github.com/yanfeiwong/adafactor-8bit); manuscript LaTeX
sources are intentionally outside this artifact repository.

## Contents

| Path | Contents |
|---|---|
| `benchmarks/` | 108 TensorBoard event files for the reported 1K, 10K, 20K, and 100K runs |
| `state_traces/` | 52 statistics-only PyTorch snapshots and four matching TensorBoard event files |
| `theory_and_ablation_final.ipynb` | Executed controlled fidelity and ablation notebook |
| `scripts/` | Training, tracing, analysis, validation, and figure-generation code |
| `scripts/paper_figures/data/` | Checked machine-readable inputs for Figures 3 and 4 |
| `reports_md/` | Environment, smoke-test, state-trace, and TensorBoard analysis reports |
| `figures/` | Publication figures in PDF, SVG, and PNG formats |
| `data/`, `models/` | Acquisition links and expected local directory layout; payloads are not vendored |

See [ARTIFACT_MANIFEST.md](ARTIFACT_MANIFEST.md) for the claim-to-artifact map
and [scripts/reproducibility/EXPERIMENT_SNAPSHOT.md](scripts/reproducibility/EXPERIMENT_SNAPSHOT.md)
for the source boundary.

## Source and environment

The reported experiments use **Adafactor8Bit v0.4.3** at implementation commit
[`4f544d8b8eebaf50053a4e8a27096e79b049b480`](https://github.com/yanfeiwong/adafactor-8bit/commit/4f544d8b8eebaf50053a4e8a27096e79b049b480).
Later README or author-metadata commits in the implementation repository do not
change this experimental anchor.

The captured software, CUDA, compiler, and GPU environment is recorded in
[`reports_md/environment.md`](reports_md/environment.md). The experiments were
run on an NVIDIA GeForce RTX 3090 Ti with 24 GiB of memory.

Model and dataset payloads must be downloaded separately:

- [TinyLlama v1.1](https://huggingface.co/TinyLlama/TinyLlama_v1.1) into `models/TinyLlama-1.1B/`
- [GPT-2](https://huggingface.co/openai-community/gpt2) into `models/GPT2-124M/`
- [WikiText-103 raw v1](https://huggingface.co/datasets/Salesforce/wikitext/tree/main/wikitext-103-raw-v1) into `data/wikitext-103-raw-v1/`

## Reproduction entry points

From the repository root:

```bash
python scripts/smoke_test.py
python scripts/analyze_tb.py
python scripts/analyze_traces.py
python scripts/paper_figures/build_all.py
```

The numbered benchmark launchers are frozen, staged experiment queues rather
than a single polished CLI. Their roles and the VRAM supplement are documented
in [`scripts/README.md`](scripts/README.md). Commented job lists preserve the
historical queue state; the event directories and checked reports are the
authoritative inventory of completed runs.

## Citation

```bibtex
@misc{wang2026denseadamstatesadaptive,
      title={Beyond Dense Adam States: Adaptive Log-Space Quantization for Memory-Efficient Optimizers},
      author={Yan Wang},
      year={2026},
      eprint={2608.22322},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2608.22322},
}
```

## License

Released under the [MIT License](LICENSE).
