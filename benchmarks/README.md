# Benchmark event archive

This directory contains 111 TensorBoard event files used by
`scripts/analyze_tb.py` and `reports_md/tb_analysis_report.md`.

The directory schema is:

```text
pretrain/<model>/<dataset>/<step-budget>/<run-name>/events.out.tfevents.*
```

Run names encode grouping (`G0`/`G1`), optimizer path, quantization choices,
learning-rate multiplier, batch size, and sequence length. The 1K-step runs are
PyTorch-memory-counter supplements; 10K runs contain learning-rate and batch
sensitivity; 20K runs provide the main TinyLlama comparisons and evaluation
trajectories; 100K runs provide the GPT-2 long-horizon comparison.
The 20K archive additionally repeats the headline G0 AdamW and CAME
configurations under two additional seeds; `analyze_tb.py` summarizes these
runs with same-seed differences from the matching full-precision reference.

The public artifact keeps these TensorBoard files byte-for-byte unchanged,
including TensorBoard's generated hostname and process-ID filename components
and the archived command-line text summary. These runtime details are retained
as provenance; they are not used as experimental variables or reported metrics.

## Memory metrics

The runs used Windows 11 and PyTorch `2.12.1+cu132`. Their `mem_dynamic/*`
series combines device-level NVML usage with Windows PDH shared-memory
counters, so it is sensitive to allocator reservations, fragmentation, and
other activity on the device. The experimental build also predates two relevant
PyTorch allocator changes:

- [Windows support for `expandable_segments`](https://github.com/pytorch/pytorch/commit/d3f1493b)
- [Fix for a `no_split_pools` leak during memory-pool release](https://github.com/pytorch/pytorch/commit/752e3d89)

These later changes are context, not a causal explanation for any individual
run. Accordingly, `mem_dynamic/*` is retained only as a diagnostic record and
is not used for cross-run comparisons. The paper reports `Peak Alloc` from
`mem_torch/max_allocated_mb` (`torch.cuda.max_memory_allocated()`); missing 20K
baseline counters are filled from the matching 1K supplement runs.
Optimizer-state memory is computed separately from live CUDA tensors.
For the three APOLLO `ours` configurations, matching v0.4.4 1K supplements
refresh `Peak Alloc` and optimizer-state memory after removal of an unconsumed
legacy buffer; their reported training metrics remain from the v0.4.3 20K runs.
