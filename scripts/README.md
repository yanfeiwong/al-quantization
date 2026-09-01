# Experiment and analysis scripts

The training launchers are frozen records of staged experiment queues. During
the original campaign, completed entries were commented out and later entries
were enabled in place. Therefore the current uncommented lists are **not** a
complete run manifest and should not be executed blindly. Completed runs are
identified by `benchmarks/` and summarized in `reports_md/tb_analysis_report.md`.

## Launcher map

| Script | Role |
|---|---|
| `run_benchmark_llm_pretrain_0.py` | 10K-step TinyLlama learning-rate sweep at batch size 4, using the common `1e-3` preset times `{0.1, 1, 10}` |
| `run_benchmark_llm_pretrain_1.py` | 10K-step TinyLlama batch-size sweep at nominal per-path learning rates; batch 4 comes from the LR sweep and this queue adds 8 and 16 |
| `run_benchmark_llm_pretrain_2.py` | 20K-step TinyLlama core and ablation runs with validation every 1K steps |
| `run_benchmark_llm_pretrain_3.py` | 100K-step GPT-2 long-horizon runs at batch size 16 |
| `run_benchmark_llm_pretrain_seed_sup.py` | Repeats the headline G0 AdamW and CAME 20K-step configurations under two additional seeds |
| `run_benchmark_llm_pretrain_vram_sup.py` | 1K-step baseline reruns that recollect PyTorch peak-allocation counters for runs whose earlier Windows adapter-memory observations were unsuitable |
| `run_trace.py` / `trace_states.py` | 10K-step state-statistics capture used for the range and exact-zero analysis |
| `train_llm.py` | Shared training entry point and TensorBoard logging |

The common learning-rate preset is `1e-3`. Thus AdamW, CAME, and APOLLO
nominal runs use `lr_mult=0.1`, while Adafactor nominal runs use `lr_mult=1.0`.

## Analysis and validation

- `analyze_tb.py` builds the checked benchmark report, summarizes the paired
  repeated-run fidelity checks, and performs the 1K VRAM proxy matching
  described above.
- `analyze_traces.py` builds compact or full state-trace reports.
- `smoke_test.py` validates Python-reference and CUDA optimizer paths.
- `paper_figures/` contains the publication figure builders and checked input
  tables.
- `reproducibility/` records the environment and artifact source boundary.

All paths resolve relative to this repository. Local model and dataset payloads
must be placed under `models/` and `data/` as described in the root README.
