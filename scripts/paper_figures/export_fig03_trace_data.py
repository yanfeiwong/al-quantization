"""Export tidy Figure 3 inputs from saved state-trace snapshots.

This exporter reads the statistics already stored in ``state_traces/**/step_*.pt``.
It does not reconstruct samples that were discarded by ``trace_states.py``.
In particular, current snapshots contain block-range median/p95/max per state
tensor, but not the individual range of every block.

Example (Windows):

    python scripts\\paper_figures\\export_fig03_trace_data.py ^
        --base-dir state_traces

The default output is
``scripts/paper_figures/data/fig03_trace_components.csv``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Iterable

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_ROOT = SCRIPT_DIR.parents[1]
ANALYSIS_DIR = SCRIPT_DIR.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from analyze_traces import classify_param  # noqa: E402


COMPONENTS = (
    ("v", "second_moment", "full"),
    ("v_row", "second_moment", "row"),
    ("v_col", "second_moment", "col"),
    ("res_row", "confidence", "row"),
    ("res_col", "confidence", "col"),
)

FIELDS = (
    "run_path",
    "optimizer",
    "step",
    "parameter_name",
    "tensor_type",
    "parameter_shape",
    "parameter_numel",
    "state_family",
    "state_component",
    "state_numel",
    "zero_fraction",
    "log2_median",
    "log2_p5",
    "log2_p95",
    "log2_p99",
    "log2_min",
    "log2_max",
    "block_logrange_median",
    "block_logrange_p95",
    "block_logrange_max",
    "n_valid_blocks",
    "n_total_blocks",
    "source_file",
)


def parse_step(path: Path) -> int | None:
    try:
        return int(path.stem.removeprefix("step_"))
    except ValueError:
        return None


def resolve_from_paper_root(path: Path) -> Path:
    """Resolve CLI paths consistently, independent of the current directory."""
    if path.is_absolute():
        return path.resolve()
    return (PAPER_ROOT / path).resolve()


def load_snapshot(path: Path) -> dict[str, Any]:
    """Load tensor-only snapshots without enabling arbitrary pickle objects."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Compatibility with older PyTorch versions. These trace files must be
        # locally generated/trusted before using the legacy loader.
        return torch.load(path, map_location="cpu")


def iter_rows(base_dir: Path) -> Iterable[dict[str, Any]]:
    for path in sorted(base_dir.rglob("step_*.pt")):
        step = parse_step(path)
        if step is None:
            continue
        snapshot = load_snapshot(path)
        run_path = path.parent.relative_to(base_dir).as_posix()
        optimizer = path.parent.name
        for parameter_name, pdata in snapshot.get("params", {}).items():
            shape = pdata.get("shape", [])
            for key, family, component in COMPONENTS:
                stats = pdata.get(key)
                if not isinstance(stats, dict):
                    continue
                yield {
                    "run_path": run_path,
                    "optimizer": optimizer,
                    "step": step,
                    "parameter_name": parameter_name,
                    "tensor_type": classify_param(parameter_name),
                    "parameter_shape": "x".join(str(v) for v in shape),
                    "parameter_numel": pdata.get("numel", 0),
                    "state_family": family,
                    "state_component": component,
                    "state_numel": stats.get("numel", 0),
                    "zero_fraction": stats.get("zero_frac", 0.0),
                    "log2_median": stats.get("log2_median", 0.0),
                    "log2_p5": stats.get("log2_p5", 0.0),
                    "log2_p95": stats.get("log2_p95", 0.0),
                    "log2_p99": stats.get("log2_p99", 0.0),
                    "log2_min": stats.get("log2_min", 0.0),
                    "log2_max": stats.get("log2_max", 0.0),
                    "block_logrange_median": stats.get("block_logrange_median", 0.0),
                    "block_logrange_p95": stats.get("block_logrange_p95", 0.0),
                    "block_logrange_max": stats.get("block_logrange_max", 0.0),
                    "n_valid_blocks": stats.get("n_valid_blocks", 0),
                    "n_total_blocks": stats.get("n_total_blocks", 0),
                    "source_file": path.relative_to(base_dir).as_posix(),
                }


def export(base_dir: Path, output: Path) -> tuple[int, int]:
    rows = list(iter_rows(base_dir))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    n_files = len({row["source_file"] for row in rows})
    return len(rows), n_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=PAPER_ROOT / "state_traces",
        help=(
            "Root containing **/step_*.pt trace snapshots. Relative paths are "
            "resolved from the paper root, not the current terminal directory."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "data" / "fig03_trace_components.csv",
        help=(
            "Tidy per-parameter/per-component CSV destination. Relative paths "
            "are resolved from the paper root."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = resolve_from_paper_root(args.base_dir)
    output = resolve_from_paper_root(args.output)
    if not base_dir.exists():
        raise SystemExit(f"Trace directory not found: {base_dir}")
    n_rows, n_files = export(base_dir, output)
    if n_rows == 0:
        raise SystemExit(f"No state-component rows found under: {base_dir}")
    print(f"Exported {n_rows} component rows from {n_files} snapshots")
    print(output)
    print(
        "NOTE: current snapshots contain per-state block-range summaries, "
        "not individual block samples."
    )


if __name__ == "__main__":
    main()
