"""
analyze_traces.py — Compile state trace data into a single markdown report.

Scans state_traces/ directory, reads all .pt files, and generates
a comprehensive markdown report for LLM-assisted analysis.

Usage:
    python analyze_traces.py
    python analyze_traces.py --output state_trace_report.md
"""

import os
import argparse
import torch
from pathlib import Path
from collections import defaultdict


# ==========================================
# Parameter Classification
# ==========================================
def classify_param(name):
    name_lower = name.lower()
    if "embed" in name_lower or "wte" in name_lower:
        return "emb"
    if "self_attn" in name or "attn" in name:
        if "norm" in name_lower or "layernorm" in name_lower:
            return "norm"
        if "bias" in name_lower:
            return "bias"
        return "attn"
    if "mlp" in name:
        if "bias" in name_lower:
            return "bias"
        return "mlp"
    if "norm" in name_lower or "layernorm" in name_lower:
        return "norm"
    if "lm_head" in name_lower:
        return "lm_head"
    if "bias" in name_lower:
        return "bias"
    return "other"


TYPE_ORDER = ["emb", "attn", "mlp", "norm", "lm_head", "bias", "other"]


# ==========================================
# Formatting Helpers
# ==========================================
def fmt_numel(n):
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


def fmt_pct(x):
    return f"{x*100:.2f}%"


def fmt_val(x, decimals=3):
    return f"{x:.{decimals}f}"


# ==========================================
# Trace Loading
# ==========================================
def find_traces_base():
    script_dir = Path(__file__).resolve().parent
    base = script_dir.parent / "state_traces"
    if not base.exists():
        base = script_dir / "state_traces"
    return base


def load_all_traces(base_dir):
    """
    Returns:
    {
        optimizer_name: {
            "meta": {...},
            "steps": {step_int: snapshot_dict, ...}
        }
    }
    """
    results = {}
    base = Path(base_dir)
    if not base.exists():
        print(f"[ERROR] state_traces directory not found: {base}")
        return results

    # Walk: state_traces/{task}/{model}/{dataset}/{optimizer}/step_*.pt
    for pt_file in sorted(base.rglob("step_*.pt")):
        # Extract optimizer name (parent directory)
        optimizer = pt_file.parent.name
        # Extract step number
        step_str = pt_file.stem.replace("step_", "")
        try:
            step = int(step_str)
        except ValueError:
            continue

        if optimizer not in results:
            results[optimizer] = {"meta": None, "steps": {}}

        try:
            snapshot = torch.load(pt_file, map_location="cpu", weights_only=False)
            results[optimizer]["steps"][step] = snapshot
            if results[optimizer]["meta"] is None:
                results[optimizer]["meta"] = snapshot.get("hist_params", {})
        except Exception as e:
            print(f"[WARN] Failed to load {pt_file}: {e}")

    # Sort steps
    for opt in results:
        results[opt]["steps"] = dict(sorted(results[opt]["steps"].items()))

    return results


# ==========================================
# Statistics Extraction
# ==========================================
def extract_v_summary(v_stats):
    """Extract key V statistics into a flat dict."""
    return {
        "numel": v_stats.get("numel", 0),
        "zero_frac": v_stats.get("zero_frac", 0.0),
        "log2_median": v_stats.get("log2_median", 0.0),
        "log2_p5": v_stats.get("log2_p5", 0.0),
        "log2_p95": v_stats.get("log2_p95", 0.0),
        "log2_p99": v_stats.get("log2_p99", 0.0),
        "log2_min": v_stats.get("log2_min", 0.0),
        "log2_max": v_stats.get("log2_max", 0.0),
        "br_median": v_stats.get("block_logrange_median", 0.0),
        "br_p95": v_stats.get("block_logrange_p95", 0.0),
        "br_max": v_stats.get("block_logrange_max", 0.0),
        "n_valid_blocks": v_stats.get("n_valid_blocks", 0),
        "n_total_blocks": v_stats.get("n_total_blocks", 0),
    }


def extract_m_summary(m_stats):
    """Extract key M statistics into a flat dict."""
    return {
        "numel": m_stats.get("numel", 0),
        "absmax": m_stats.get("absmax", 0.0),
        "l2_norm": m_stats.get("l2_norm", 0.0),
    }


def parse_snapshot(snapshot):
    """
    Parse a snapshot into structured per-param data.
    Returns list of dicts with keys: name, type, shape, numel, ndim, param_rms, v_stats, m_stats
    For factored V, v_stats is a dict with 'row' and 'col' keys.
    """
    params_data = []
    for name, pdata in snapshot.get("params", {}).items():
        entry = {
            "name": name,
            "type": classify_param(name),
            "shape": pdata.get("shape", []),
            "numel": pdata.get("numel", 0),
            "ndim": pdata.get("ndim", 0),
            "param_rms": pdata.get("param_rms", 0.0),
        }

        # V: could be "v" (full-rank) or "v_row"/"v_col" (factored)
        if "v" in pdata:
            entry["v"] = extract_v_summary(pdata["v"])
            entry["v_mode"] = "full"
        elif "v_row" in pdata and "v_col" in pdata:
            entry["v_row"] = extract_v_summary(pdata["v_row"])
            entry["v_col"] = extract_v_summary(pdata["v_col"])
            entry["v_mode"] = "factored"

        # M
        if "m" in pdata:
            entry["m"] = extract_m_summary(pdata["m"])

        # Residual (CAME)
        if "res_row" in pdata and "res_col" in pdata:
            entry["res_row"] = extract_v_summary(pdata["res_row"])
            entry["res_col"] = extract_v_summary(pdata["res_col"])
            entry["has_res"] = True
        else:
            entry["has_res"] = False

        params_data.append(entry)

    return params_data


# ==========================================
# Aggregation
# ==========================================
def _select_display_params(params_data):
    """Compact mode: select representative params for display."""
    selected = []
    for p in params_data:
        name = p["name"]
        if p["type"] in ("emb", "lm_head"):
            selected.append(p)
        elif "layers.0." in name or "layers.21." in name:
            selected.append(p)
        elif name == "model.norm.weight":
            selected.append(p)
    # For each type, add the param with highest BR max
    for ptype in ("attn", "mlp", "norm"):
        candidates = [p for p in params_data if p["type"] == ptype]
        if not candidates:
            continue
        if "v" in candidates[0]:
            v_key = "v"
        elif "v_row" in candidates[0]:
            v_key = "v_row"
        else:
            continue
        candidates_with_v = [p for p in candidates if v_key in p]
        if candidates_with_v:
            best = max(candidates_with_v, key=lambda p: p[v_key].get("br_max", 0))
            if best not in selected:
                selected.append(best)
    return selected


def aggregate_by_type(params_data):
    """Aggregate V stats by parameter type (numel-weighted)."""
    type_groups = defaultdict(list)
    for p in params_data:
        type_groups[p["type"]].append(p)

    agg = {}
    for ptype, plist in type_groups.items():
        total_numel = sum(p["numel"] for p in plist)
        n_params = len(plist)

        # Collect V stats
        v_entries = []
        for p in plist:
            if p.get("v_mode") == "full" and "v" in p:
                v_entries.append(("full", p["v"]))
            elif p.get("v_mode") == "factored":
                v_entries.append(("row", p["v_row"]))
                v_entries.append(("col", p["v_col"]))

        if v_entries:
            # Weighted averages for key stats
            total_v_numel = sum(v["numel"] for _, v in v_entries)
            if total_v_numel > 0:
                w_zero = sum(v["numel"] * v["zero_frac"] for _, v in v_entries) / total_v_numel
                w_med = sum(v["numel"] * v["log2_median"] for _, v in v_entries) / total_v_numel
                w_p5 = sum(v["numel"] * v["log2_p5"] for _, v in v_entries) / total_v_numel
                w_p95 = sum(v["numel"] * v["log2_p95"] for _, v in v_entries) / total_v_numel
                w_p99 = sum(v["numel"] * v["log2_p99"] for _, v in v_entries) / total_v_numel
                w_br_med = sum(v["numel"] * v["br_median"] for _, v in v_entries) / total_v_numel
                w_br_p95 = sum(v["numel"] * v["br_p95"] for _, v in v_entries) / total_v_numel
                # min/max: take extremes
                g_min = min(v["log2_min"] for _, v in v_entries)
                g_max = max(v["log2_max"] for _, v in v_entries)
                br_max = max(v["br_max"] for _, v in v_entries)
            else:
                w_zero = w_med = w_p5 = w_p95 = w_p99 = w_br_med = w_br_p95 = 0
                g_min = g_max = br_max = 0
        else:
            w_zero = w_med = w_p5 = w_p95 = w_p99 = w_br_med = w_br_p95 = 0
            g_min = g_max = br_max = 0

        agg[ptype] = {
            "n_params": n_params,
            "total_numel": total_numel,
            "v_zero_frac": w_zero,
            "v_log2_median": w_med,
            "v_log2_p5": w_p5,
            "v_log2_p95": w_p95,
            "v_log2_p99": w_p99,
            "v_log2_min": g_min,
            "v_log2_max": g_max,
            "v_br_median": w_br_med,
            "v_br_p95": w_br_p95,
            "v_br_max": br_max,
        }

    return agg


def compute_global_stats(params_data):
    """Compute global V stats across all parameters."""
    all_v = []
    for p in params_data:
        if p.get("v_mode") == "full" and "v" in p:
            all_v.append(p["v"])
        elif p.get("v_mode") == "factored":
            all_v.append(p["v_row"])
            all_v.append(p["v_col"])

    if not all_v:
        return None

    total_numel = sum(v["numel"] for v in all_v)
    if total_numel == 0:
        return None

    return {
        "total_v_numel": total_numel,
        "zero_frac": sum(v["numel"] * v["zero_frac"] for v in all_v) / total_numel,
        "log2_median": sum(v["numel"] * v["log2_median"] for v in all_v) / total_numel,
        "log2_p5": sum(v["numel"] * v["log2_p5"] for v in all_v) / total_numel,
        "log2_p95": sum(v["numel"] * v["log2_p95"] for v in all_v) / total_numel,
        "log2_p99": sum(v["numel"] * v["log2_p99"] for v in all_v) / total_numel,
        "log2_min": min(v["log2_min"] for v in all_v),
        "log2_max": max(v["log2_max"] for v in all_v),
        "br_median": sum(v["numel"] * v["br_median"] for v in all_v) / total_numel,
        "br_p95": sum(v["numel"] * v["br_p95"] for v in all_v) / total_numel,
        "br_max": max(v["br_max"] for v in all_v),
    }


# ==========================================
# Markdown Generation
# ==========================================
def generate_report(traces, output_path, compact=False):
    lines = []
    lines.append("# Optimizer State Trace Report\n")
    lines.append(f"Generated by `analyze_traces.py`\n")

    for opt_name, opt_data in traces.items():
        meta = opt_data["meta"] or {}
        steps_dict = opt_data["steps"]
        if not steps_dict:
            continue

        steps_sorted = sorted(steps_dict.keys())

        lines.append(f"\n---\n\n## Optimizer: `{opt_name}`\n")
        lines.append(f"- Trace steps: {steps_sorted}")
        lines.append(f"- V block size: {meta.get('v_block_size', 'N/A')}")
        lines.append(f"- Log2 hist range: [{meta.get('log2_hist_min', '?')}, "
                     f"{meta.get('log2_hist_max', '?')}], step={meta.get('log2_hist_step', '?')}")
        lines.append(f"- M hist bins: {meta.get('m_hist_bins', 'N/A')}")

        # Parse all steps
        parsed_steps = {}
        for step in steps_sorted:
            parsed_steps[step] = parse_snapshot(steps_dict[step])

        # === Section 1: Cross-step evolution ===
        lines.append(f"\n### Cross-Step Evolution (Global)\n")
        lines.append("| Step | Zero% | log2 median | log2 p5 | log2 p95 | log2 p99 | "
                     "log2 min | log2 max | BR med | BR p95 | BR max |")
        lines.append("|------|-------|-------------|---------|----------|----------|"
                     "----------|----------|--------|--------|--------|")
        for step in steps_sorted:
            gs = compute_global_stats(parsed_steps[step])
            if gs is None:
                continue
            lines.append(
                f"| {step} | {fmt_pct(gs['zero_frac'])} | {fmt_val(gs['log2_median'])} | "
                f"{fmt_val(gs['log2_p5'])} | {fmt_val(gs['log2_p95'])} | {fmt_val(gs['log2_p99'])} | "
                f"{fmt_val(gs['log2_min'])} | {fmt_val(gs['log2_max'])} | "
                f"{fmt_val(gs['br_median'])} | {fmt_val(gs['br_p95'])} | {fmt_val(gs['br_max'])} |"
            )

        # === Section 2: Per-type at final step ===
        final_step = steps_sorted[-1]
        final_params = parsed_steps[final_step]
        type_agg = aggregate_by_type(final_params)

        lines.append(f"\n### Per-Type Summary (Step {final_step})\n")
        lines.append("| Type | #Params | Numel | Zero% | log2 med | log2 p5 | log2 p95 | log2 p99 | "
                     "BR med | BR p95 | BR max |")
        lines.append("|------|---------|-------|-------|----------|---------|----------|----------|"
                     "--------|--------|--------|")
        for ptype in TYPE_ORDER:
            if ptype not in type_agg:
                continue
            a = type_agg[ptype]
            lines.append(
                f"| {ptype} | {a['n_params']} | {fmt_numel(a['total_numel'])} | "
                f"{fmt_pct(a['v_zero_frac'])} | {fmt_val(a['v_log2_median'])} | "
                f"{fmt_val(a['v_log2_p5'])} | {fmt_val(a['v_log2_p95'])} | "
                f"{fmt_val(a['v_log2_p99'])} | {fmt_val(a['v_br_median'])} | "
                f"{fmt_val(a['v_br_p95'])} | {fmt_val(a['v_br_max'])} |"
            )

        # === Section 3: Per-param detail at final step ===
        lines.append(f"\n### Per-Parameter Detail (Step {final_step})\n")

        # Determine V mode
        sample = final_params[0] if final_params else None
        is_factored = sample and sample.get("v_mode") == "factored"
        display_params = _select_display_params(final_params) if compact else final_params

        if is_factored:
            lines.append("V is **factored** (row/col). Showing row and col separately.\n")
            lines.append("#### Row V\n")
            lines.append("| Param | Type | Shape | Numel | Zero% | log2 med | log2 p95 | log2 p99 | "
                         "BR med | BR p95 |")
            lines.append("|-------|------|-------|-------|-------|----------|----------|----------|"
                         "--------|--------|")
            for p in sorted(display_params, key=lambda x: (TYPE_ORDER.index(x["type"]), x["name"])):
                if "v_row" not in p:
                    continue
                v = p["v_row"]
                shape_str = str(p["shape"])
                lines.append(
                    f"| {p['name']} | {p['type']} | {shape_str} | {fmt_numel(v['numel'])} | "
                    f"{fmt_pct(v['zero_frac'])} | {fmt_val(v['log2_median'])} | "
                    f"{fmt_val(v['log2_p95'])} | {fmt_val(v['log2_p99'])} | "
                    f"{fmt_val(v['br_median'])} | {fmt_val(v['br_p95'])} |"
                )
            lines.append("\n#### Col V\n")
            lines.append("| Param | Type | Shape | Numel | Zero% | log2 med | log2 p95 | log2 p99 | "
                         "BR med | BR p95 |")
            lines.append("|-------|------|-------|-------|-------|----------|----------|----------|"
                         "--------|--------|")
            for p in sorted(display_params, key=lambda x: (TYPE_ORDER.index(x["type"]), x["name"])):
                if "v_col" not in p:
                    continue
                v = p["v_col"]
                shape_str = str(p["shape"])
                lines.append(
                    f"| {p['name']} | {p['type']} | {shape_str} | {fmt_numel(v['numel'])} | "
                    f"{fmt_pct(v['zero_frac'])} | {fmt_val(v['log2_median'])} | "
                    f"{fmt_val(v['log2_p95'])} | {fmt_val(v['log2_p99'])} | "
                    f"{fmt_val(v['br_median'])} | {fmt_val(v['br_p95'])} |"
                )
            
            # CAME Residual Tables
            has_res = any(p.get("has_res") for p in display_params)
            if has_res:
                lines.append("\n#### Residual Row (CAME)\n")
                lines.append("| Param | Type | Shape | Numel | Zero% | log2 med | log2 p95 | log2 p99 | "
                             "BR med | BR p95 |")
                lines.append("|-------|------|-------|-------|-------|----------|----------|----------|"
                             "--------|--------|")
                for p in sorted(display_params, key=lambda x: (TYPE_ORDER.index(x["type"]), x["name"])):
                    if not p.get("has_res"):
                        continue
                    v = p["res_row"]
                    shape_str = str(p["shape"])
                    lines.append(
                        f"| {p['name']} | {p['type']} | {shape_str} | {fmt_numel(v['numel'])} | "
                        f"{fmt_pct(v['zero_frac'])} | {fmt_val(v['log2_median'])} | "
                        f"{fmt_val(v['log2_p95'])} | {fmt_val(v['log2_p99'])} | "
                        f"{fmt_val(v['br_median'])} | {fmt_val(v['br_p95'])} |"
                    )
                    
                lines.append("\n#### Residual Col (CAME)\n")
                lines.append("| Param | Type | Shape | Numel | Zero% | log2 med | log2 p95 | log2 p99 | "
                             "BR med | BR p95 |")
                lines.append("|-------|------|-------|-------|-------|----------|----------|----------|"
                             "--------|--------|")
                for p in sorted(display_params, key=lambda x: (TYPE_ORDER.index(x["type"]), x["name"])):
                    if not p.get("has_res"):
                        continue
                    v = p["res_col"]
                    shape_str = str(p["shape"])
                    lines.append(
                        f"| {p['name']} | {p['type']} | {shape_str} | {fmt_numel(v['numel'])} | "
                        f"{fmt_pct(v['zero_frac'])} | {fmt_val(v['log2_median'])} | "
                        f"{fmt_val(v['log2_p95'])} | {fmt_val(v['log2_p99'])} | "
                        f"{fmt_val(v['br_median'])} | {fmt_val(v['br_p95'])} |"
                    )
        else:
            lines.append("V is **full-rank** (per-element).\n")
            lines.append("| Param | Type | Shape | Numel | Zero% | log2 med | log2 p5 | log2 p95 | "
                         "log2 p99 | BR med | BR p95 | BR max |")
            lines.append("|-------|------|-------|-------|-------|----------|---------|----------|"
                         "----------|--------|--------|--------|")
            for p in sorted(display_params, key=lambda x: (TYPE_ORDER.index(x["type"]), x["name"])):
                if "v" not in p:
                    continue
                v = p["v"]
                shape_str = str(p["shape"])
                lines.append(
                    f"| {p['name']} | {p['type']} | {shape_str} | {fmt_numel(v['numel'])} | "
                    f"{fmt_pct(v['zero_frac'])} | {fmt_val(v['log2_median'])} | "
                    f"{fmt_val(v['log2_p5'])} | {fmt_val(v['log2_p95'])} | "
                    f"{fmt_val(v['log2_p99'])} | "
                    f"{fmt_val(v['br_median'])} | {fmt_val(v['br_p95'])} | {fmt_val(v['br_max'])} |"
                )

        # === Section 4: M summary (if available) ===
        has_m = any("m" in p for p in final_params)
        if has_m:
            if compact:
                lines.append(f"\n### M Summary by Type (Step {final_step})\n")
                lines.append("| Type | #Params | Total Numel | absmax min | absmax max | L2 min | L2 max |")
                lines.append("|------|---------|-------------|------------|------------|--------|--------|")
                m_by_type = defaultdict(list)
                for p in final_params:
                    if "m" in p:
                        m_by_type[p["type"]].append(p["m"])
                for ptype in TYPE_ORDER:
                    if ptype not in m_by_type:
                        continue
                    ms = m_by_type[ptype]
                    total_numel = sum(m["numel"] for m in ms)
                    absmax_min = min(m["absmax"] for m in ms)
                    absmax_max = max(m["absmax"] for m in ms)
                    l2_min = min(m["l2_norm"] for m in ms)
                    l2_max = max(m["l2_norm"] for m in ms)
                    lines.append(
                        f"| {ptype} | {len(ms)} | {fmt_numel(total_numel)} | "
                        f"{absmax_min:.4e} | {absmax_max:.4e} | "
                        f"{l2_min:.4e} | {l2_max:.4e} |"
                    )
            else:
                lines.append(f"\n### M Summary (Step {final_step})\n")
                lines.append("| Param | Type | Numel | absmax | L2 norm |")
                lines.append("|-------|------|-------|--------|---------|")
                for p in sorted(final_params, key=lambda x: (TYPE_ORDER.index(x["type"]), x["name"])):
                    if "m" not in p:
                        continue
                    m = p["m"]
                    lines.append(
                        f"| {p['name']} | {p['type']} | {fmt_numel(m['numel'])} | "
                        f"{m['absmax']:.4e} | {m['l2_norm']:.4e} |"
                    )

        # === Section 5: Per-type at earliest step (cold-start comparison) ===
        earliest_step = steps_sorted[0]
        if earliest_step != final_step:
            earliest_params = parsed_steps[earliest_step]
            type_agg_early = aggregate_by_type(earliest_params)

            lines.append(f"\n### Per-Type Summary (Step {earliest_step}, cold-start)\n")
            lines.append("| Type | #Params | Zero% | log2 med | log2 p95 | log2 p99 | BR med | BR p95 |")
            lines.append("|------|---------|-------|----------|----------|----------|--------|--------|")
            for ptype in TYPE_ORDER:
                if ptype not in type_agg_early:
                    continue
                a = type_agg_early[ptype]
                lines.append(
                    f"| {ptype} | {a['n_params']} | {fmt_pct(a['v_zero_frac'])} | "
                    f"{fmt_val(a['v_log2_median'])} | {fmt_val(a['v_log2_p95'])} | "
                    f"{fmt_val(a['v_log2_p99'])} | {fmt_val(a['v_br_median'])} | "
                    f"{fmt_val(a['v_br_p95'])} |"
                )

    # Write output
    report = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[DONE] Report written to: {output_path}")
    print(f"       Total size: {len(report)} chars, {len(lines)} lines")


# ==========================================
# CLI
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Compile state traces into markdown report")
    parser.add_argument("--base_dir", type=str, default=None,
                        help="Base directory of state_traces. Auto-detected if not specified.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output markdown path. Defaults to reports_md/state_trace_report.md")
    parser.add_argument("--full", action="store_true",
                        help="Full report: all per-param rows, M per-param. Default is compact.")
    args = parser.parse_args()

    if args.base_dir:
        base_dir = Path(args.base_dir)
    else:
        base_dir = find_traces_base()

    if not base_dir.exists():
        print(f"[ERROR] Cannot find state_traces directory at: {base_dir}")
        print("        Please specify with --base_dir")
        return

    if args.output:
        output_path = Path(args.output)
    else:
        reports_dir = base_dir.parent / "reports_md"
        reports_dir.mkdir(parents=True, exist_ok=True)
        output_path = reports_dir / "state_trace_report.md"

    print(f"[INFO] Scanning: {base_dir}")
    traces = load_all_traces(base_dir)

    if not traces:
        print("[ERROR] No trace files found.")
        return

    for opt, data in traces.items():
        print(f"[INFO] Found '{opt}': {len(data['steps'])} steps "
              f"({sorted(data['steps'].keys())})")

    generate_report(traces, output_path, compact=not args.full)


if __name__ == "__main__":
    main()