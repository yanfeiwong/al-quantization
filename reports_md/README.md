# Checked reports

This directory contains the human-readable and machine-readable intermediates
used to audit the paper artifacts:

- `environment.{md,json}` — captured software, CUDA, compiler, and GPU stack;
- `smoke_test.{md,json}` — reference/CUDA optimizer validation;
- `tb_analysis_report.md` — benchmark inventory, tables, and training metrics;
- `state_trace_report.md` — compact selected state-range summary;
- `state_trace_report_full.md` — expanded per-parameter/per-state trace report.

The compact and full trace reports come from the same snapshots and serve
different reading needs; the full report is intentionally retained rather than
treated as a duplicate.
