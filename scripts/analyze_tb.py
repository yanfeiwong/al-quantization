import os
import glob
import re
import math
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator

# ==========================================
# 1. Configuration & Lexical Parser
# ==========================================
KNOWN_ALGOS = ["adamw", "apollo", "came", "calm", "adafactor", "rmsprop"]
KNOWN_CLASSES = ["torch", "hf", "ours", "8bit_bnb", "bnb", "off"]

# Sorting weights for logical report ordering
ALGO_ORDER = {"adamw": 0, "came": 1, "adafactor": 2, "apollo": 3, "rmsprop": 4, "calm": 5, "unknown": 9}
CLASS_ORDER = {"torch": 0, "hf": 0, "8bit_bnb": 1, "bnb": 1, "ours": 2, "off": 3, "unknown": 4}
V_QUANT_RANK = {"al8": 0, "al16": 1, "bnb_u8": 2, "-": 3}
C_QUANT_RANK = {"-": 0, "al8": 0, "al16": 1, "fp32": 2, "bnb_u8": 2}

HP_PATTERN = re.compile(r'_x(?P<lr_mult>[\d\.]+)_bs(?P<batch_size>\d+)_seq(?P<seq_len>\d+)$')
STEPS_PATTERN = re.compile(r'(\d+)steps')

def parse_run_name(run_name):
    hp_match = HP_PATTERN.search(run_name)
    if not hp_match:
        return {"run_name": run_name, "algorithm": "unknown", "class": "unknown", "grouping": "Unknown"}
    
    hp_dict = hp_match.groupdict()
    prefix = run_name[:hp_match.start()]
    parts = prefix.split('_')
    
    grouping = parts[0] if parts else "Unknown"
    algo, cls, variant = "unknown", "unknown", ""
    
    algo_idx = -1
    for i, p in enumerate(parts[1:], 1):
        if p in KNOWN_ALGOS:
            algo = p
            algo_idx = i
            break
            
    if algo_idx != -1:
        cls_idx = -1
        for i in range(algo_idx + 1, len(parts)):
            if parts[i] in KNOWN_CLASSES:
                cls = parts[i]
                cls_idx = i
                break
            if i + 1 < len(parts) and f"{parts[i]}_{parts[i+1]}" in KNOWN_CLASSES:
                cls = f"{parts[i]}_{parts[i+1]}"
                cls_idx = i + 1
                break
                
        if cls_idx != -1:
            variant = "_".join(parts[cls_idx + 1:])
        else:
            variant = "_".join(parts[algo_idx + 1:])

    # Robust Quantization Parsing based on naming convention: {mtype}_{vtype}_vblk{vblk}[_c_{ctype}]
    m_quant, v_quant, v_blk, c_quant = "-", "-", "-", "-"
    
    # Inject BNB physical defaults and explicit variant for fair comparison
    if cls == "8bit_bnb":
        m_quant, v_quant, v_blk = "d8", "bnb_u8", "256"
        variant = "d8_bnb_u8_vblk256"
    elif variant:
        blk_match = re.search(r'vblk(\d+)', variant)
        if blk_match:
            v_blk = blk_match.group(1)
            var_parts = variant.split('_')
            vblk_idx = -1
            for i, p in enumerate(var_parts):
                if p.startswith('vblk'):
                    vblk_idx = i
                    break
            
            if vblk_idx > 0:
                v_quant = var_parts[vblk_idx - 1]
                if vblk_idx > 1:
                    m_quant = var_parts[vblk_idx - 2]
                    
            # Extract C-Quant (e.g., _c_fp32, _c_al16)
            c_match = re.search(r'_c_([a-zA-Z0-9]+)', variant)
            if c_match:
                c_quant = c_match.group(1)

    # C-Quant: for CAME, C follows V when no explicit _c_ suffix is present
    if c_quant == "-" and algo == "came" and v_quant != "-":
        c_quant = v_quant

    return {
        "run_name": run_name, "grouping": grouping,
        "algorithm": algo, "class": cls, "variant": variant,
        "m_quant": m_quant, "v_quant": v_quant, "c_quant": c_quant, "v_blk": v_blk,
        **{k: float(v) if k == "lr_mult" else int(v) for k, v in hp_dict.items()}
    }

# ==========================================
# 2. Core Data Processing & Metrics
# ==========================================
def get_ts(ea, tag_candidates):
    for t in tag_candidates:
        if t in ea.Tags().get('scalars', []):
            events = ea.Scalars(t)
            return np.array([e.step for e in events]), np.array([e.value for e in events])
    return np.array([]), np.array([])

def get_scalar(ea, tag_candidates):
    for t in tag_candidates:
        if t in ea.Tags().get('scalars', []):
            events = ea.Scalars(t)
            return events[-1].value if events else None
    return None

def calc_ppl(loss):
    if loss is None or np.isnan(loss) or loss > 20:
        return float('inf')
    return math.exp(loss)

def nearest_sample_indices(steps, max_step, interval):
    """Select nearest logged points without repeating the same event."""
    if len(steps) == 0:
        return []

    indices = []
    for target in range(0, max_step + 1, interval):
        idx = int(np.abs(steps - target).argmin())
        if not indices or idx != indices[-1]:
            indices.append(idx)
    return indices

def process_run(log_dir):
    info = parse_run_name(os.path.basename(log_dir))
    
    # Extract intended_steps from parent directory name (e.g., "10000steps" -> 10000)
    parent_dir = os.path.basename(os.path.dirname(log_dir))
    steps_match = STEPS_PATTERN.search(parent_dir)
    intended_steps = int(steps_match.group(1)) if steps_match else 0
    
    ea = event_accumulator.EventAccumulator(log_dir, size_guidance={'scalars': 0})
    try:
        ea.Reload()
    except Exception:
        return None

    train_steps, train_values = get_ts(ea, ['train/loss'])
    grad_steps, grad_values = get_ts(ea, ['train/grad_norm'])
    eval_steps, eval_values = get_ts(ea, ['eval/loss'])
    
    tokens_per_sec = get_scalar(ea, ['train/train_tokens_per_second', 'train/tokens_per_second'])
    opt_mem = get_scalar(ea, ['mem_static/optimizer_mb'])
    peak_alloc = get_scalar(ea, ['mem_torch/max_allocated_mb'])
    avg_train_loss = get_scalar(ea, ['train/train_loss'])
    runtime_sec = get_scalar(ea, ['train/train_runtime'])
    runtime_hrs = round(runtime_sec / 3600.0, 2) if runtime_sec else 0.0

    # Divergence Detection
    nan_mask = np.isnan(grad_values) | np.isinf(grad_values) if len(grad_values) > 0 else np.array([], dtype=bool)
    nan_idx = np.argmax(nan_mask) if np.any(nan_mask) else len(grad_values)
    
    zero_mask = (train_values == 0) | np.isnan(train_values) if len(train_values) > 0 else np.array([], dtype=bool)
    zero_idx = len(train_values)
    for i in range(len(zero_mask) - 2):
        if np.all(zero_mask[i:i+3]):
            zero_idx = i
            break
            
    collapse_idx = min(nan_idx, zero_idx)
    is_collapsed = collapse_idx < len(train_values)
    collapse_step = int(train_steps[collapse_idx]) if is_collapsed and collapse_idx < len(train_steps) else None

    if is_collapsed:
        train_values = train_values[:collapse_idx]
        train_steps = train_steps[:collapse_idx]
        if len(grad_steps) > 0:
            healthy_grad_mask = grad_steps <= (collapse_step if collapse_step else grad_steps[-1])
            grad_values = grad_values[healthy_grad_mask]
        eval_values = eval_values[eval_steps <= (collapse_step if collapse_step else eval_steps[-1])]
        eval_steps = eval_steps[eval_steps <= (collapse_step if collapse_step else eval_steps[-1])]

    max_step = int(train_steps[-1]) if len(train_steps) > 0 else 0

    # Core Metrics
    final_eval_loss = eval_values[-1] if len(eval_values) > 0 else None
    final_ppl = calc_ppl(final_eval_loss)
    
    # Late Train Loss: mean of last 20% of train/loss curve (steady-state convergence)
    late_train_loss = np.nanmean(train_values[int(len(train_values)*0.8):]) if len(train_values) > 10 else None
    
    auc = np.sum((train_values[:-1] + train_values[1:]) * np.diff(train_steps)) / (2.0 * (max_step - train_steps[0])) if len(train_steps) > 1 and max_step > train_steps[0] else 0.0
    
    t95_step = None
    if len(train_values) > 10:
        target_loss = train_values[0] - 0.95 * (train_values[0] - np.nanmin(train_values))
        below_target = np.where(train_values <= target_loss)[0]
        if len(below_target) > 0:
            t95_step = int(train_steps[below_target[0]])

    volatility = np.nanstd(np.diff(train_values)) if len(train_values) > 1 else 0.0
    
    grad_spike_freq = 0.0
    late_grad_cv = 0.0
    if len(grad_values) > 10:
        median_grad = np.nanmedian(grad_values)
        mad = np.nanmedian(np.abs(grad_values - median_grad))
        threshold = median_grad + 3 * (1.4826 * mad + 1e-8) 
        grad_spike_freq = np.nansum(grad_values > threshold) / len(grad_values)
        
        late_stage = grad_values[int(len(grad_values) * 0.8):]
        late_mean = np.nanmean(late_stage)
        if late_mean > 1e-8:
            late_grad_cv = np.nanstd(late_stage) / late_mean

    # Adaptive Trajectories
    interval = max(1000, max_step // 20) if max_step > 0 else 1000
    train_traj, ppl_traj = [], []
    
    for idx in nearest_sample_indices(train_steps, max_step, interval):
        train_traj.append(round(train_values[idx], 4))
            
    for idx in nearest_sample_indices(eval_steps, max_step, interval):
        ppl_traj.append(round(calc_ppl(eval_values[idx]), 2))

    if is_collapsed: 
        train_traj.append(f"NaN({collapse_step})")
        ppl_traj.append(f"NaN({collapse_step})")

    return {
        **info,
        "intended_steps": intended_steps,
        "max_step": max_step,
        "opt_mem_mb": round(opt_mem, 2) if opt_mem else None,
        "peak_alloc_mb": round(peak_alloc, 2) if peak_alloc else None,
        "tokens_per_sec": round(tokens_per_sec, 1) if tokens_per_sec else None,
        "runtime_hrs": runtime_hrs,
        "final_eval_loss": round(final_eval_loss, 4) if final_eval_loss is not None else None,
        "final_ppl": round(final_ppl, 2) if final_ppl != float('inf') else None,
        "avg_train_loss": round(avg_train_loss, 4) if avg_train_loss is not None else None,
        "late_train_loss": round(late_train_loss, 4) if late_train_loss is not None else None,
        "auc": round(auc, 4),
        "t95_step_k": round(t95_step / 1000, 1) if t95_step is not None else None,
        "volatility": round(volatility, 5),
        "spike_freq": round(grad_spike_freq, 4),
        "late_grad_cv": round(late_grad_cv, 4),
        "is_collapsed": is_collapsed, "collapse_step": collapse_step,
        "train_traj": train_traj,
        "ppl_traj": ppl_traj,
        "train_steps": train_steps, "train_values": train_values
    }

# ==========================================
# 3. Report Generation
# ==========================================
def fmt(val, decimals=2, suffix=""):
    if val is None: return "-"
    if isinstance(val, str): return val
    return f"{val:.{decimals}f}{suffix}"

def compute_train_mae(base_run, target_run):
    """Compute train loss MAE between two runs on common steps."""
    if base_run is None or target_run is None:
        return None
    if len(base_run['train_steps']) == 0 or len(target_run['train_steps']) == 0:
        return None
    base_map = dict(zip(base_run['train_steps'], base_run['train_values']))
    tgt_map = dict(zip(target_run['train_steps'], target_run['train_values']))
    common_steps = sorted(list(set(base_map.keys()).intersection(set(tgt_map.keys()))))
    if len(common_steps) < 10:
        return None
    base_vals = np.array([base_map[s] for s in common_steps])
    tgt_vals = np.array([tgt_map[s] for s in common_steps])
    return float(np.mean(np.abs(base_vals - tgt_vals)))

def compute_staged_mae(base_run, target_run):
    """Compute staged MAE (4 equal quartiles of intended steps) between two runs."""
    if base_run is None or target_run is None:
        return None, None, None, None
    if len(base_run['train_steps']) == 0 or len(target_run['train_steps']) == 0:
        return None, None, None, None
        
    base_map = dict(zip(base_run['train_steps'], base_run['train_values']))
    tgt_map = dict(zip(target_run['train_steps'], target_run['train_values']))
    common_steps = sorted(list(set(base_map.keys()).intersection(set(tgt_map.keys()))))
    
    if len(common_steps) < 10:
        return None, None, None, None
        
    base_vals = np.array([base_map[s] for s in common_steps])
    tgt_vals = np.array([tgt_map[s] for s in common_steps])
    diffs = np.abs(base_vals - tgt_vals)
    steps_arr = np.array(common_steps)
    
    max_s = int(steps_arr[-1])
    q = max_s // 4
    mae_q1 = float(np.mean(diffs[(steps_arr >= 0) & (steps_arr < q)])) if np.any((steps_arr >= 0) & (steps_arr < q)) else None
    mae_q2 = float(np.mean(diffs[(steps_arr >= q) & (steps_arr < 2*q)])) if np.any((steps_arr >= q) & (steps_arr < 2*q)) else None
    mae_q3 = float(np.mean(diffs[(steps_arr >= 2*q) & (steps_arr < 3*q)])) if np.any((steps_arr >= 2*q) & (steps_arr < 3*q)) else None
    mae_q4 = float(np.mean(diffs[steps_arr >= 3*q])) if np.any(steps_arr >= 3*q) else None
    
    return mae_q1, mae_q2, mae_q3, mae_q4

def find_baseline(summaries, algorithm, lr_mult, intended_steps=None, batch_size=None):
    """Find the full-precision baseline for a given algorithm."""
    candidates = [
        s for s in summaries
        if s['algorithm'] == algorithm
        and s['class'] in ['torch', 'hf']
        and float(s.get('lr_mult', 0)) == float(lr_mult)
        and (intended_steps is None or s.get('intended_steps') == intended_steps)
        and (batch_size is None or int(s.get('batch_size', 0)) == int(batch_size))
    ]
    return candidates[0] if candidates else None

def generate_report(summaries, output_path):
    summaries = [s for s in summaries if s is not None]
    
    # Logical Sorting: Algo -> Class (Baseline first) -> Group -> Quant Configs
    def sort_key(s):
        algo = ALGO_ORDER.get(s.get('algorithm', 'unknown'), 9)
        cls = CLASS_ORDER.get(s.get('class', 'unknown'), 9)
        grp = s.get('grouping', 'Z')
        mq = s.get('m_quant', '-')
        vq_rank = V_QUANT_RANK.get(s.get('v_quant', '-'), 3)
        cq_rank = C_QUANT_RANK.get(s.get('c_quant', '-'), 0)
        vb = int(s.get('v_blk', 0)) if s.get('v_blk', '-').isdigit() else 0
        lr = float(s.get('lr_mult', 0))
        bs = int(s.get('batch_size', 0))
        intended_steps = int(s.get('intended_steps', 0))
        run_name = s.get('run_name', '')
        return (algo, cls, grp, mq, vq_rank, cq_rank, vb, lr, bs,
                intended_steps, run_name)
        
    summaries.sort(key=sort_key)
    
    total_runtime = sum(s.get('runtime_hrs', 0.0) for s in summaries)
    lines = [
        f"# TensorBoard Analysis Report",
        f"**Runs**: {len(summaries)} | **Total Compute Time**: {total_runtime:.1f} Hrs", ""
    ]
    
    # ==========================================
    # Section 1: Core Benchmark (20K Steps)
    # ==========================================
    core_runs = [s for s in summaries if s['intended_steps'] == 20000]
    if core_runs:
        lines.append("## 1. Core Benchmark (20K Steps, TinyLlama-1.1B)")
        
        # 1.1 Evaluation Perplexity, Convergence & Memory
        lines.append("### 1.1 Evaluation Perplexity, Convergence & Memory")
        h1 = ["Group", "Algo", "Class", "M-Quant", "V-Quant", "C-Quant", "V-Blk",
              "Final PPL (↓)", "Final Eval Loss (↓)", "Avg Train Loss (↓)", "Late Train Loss (↓)",
              "Peak Alloc (MiB)", "Opt State (MiB)", "Throughput (tok/s)"]
        lines.append("| " + " | ".join(h1) + " |")
        lines.append("|" + "|".join(["---"] * len(h1)) + "|")
        for s in core_runs:
            # Fallback for Peak Alloc: use 1K proxy run if 20K run is missing it
            peak = s['peak_alloc_mb']
            if peak is None:
                proxy = next((p for p in summaries 
                              if p['intended_steps'] == 1000
                              and p['algorithm'] == s['algorithm']
                              and p['grouping'] == s['grouping']
                              and p['class'] == s['class']
                              and p['batch_size'] == s['batch_size']
                              and p['peak_alloc_mb'] is not None), None)
                if proxy:
                    peak = proxy['peak_alloc_mb']

            row = [s['grouping'], s['algorithm'], s['class'], s['m_quant'], s['v_quant'], s['c_quant'], s['v_blk'],
                   fmt(s['final_ppl']), fmt(s['final_eval_loss'], 4),
                   fmt(s['avg_train_loss'], 4), fmt(s['late_train_loss'], 4),
                   fmt(peak, 1), fmt(s['opt_mem_mb'], 1),
                   fmt(s['tokens_per_sec'], 0)]
            lines.append("| " + " | ".join(map(str, row)) + " |")

        # 1.2 Convergence Dynamics
        lines.append("\n### 1.2 Convergence Dynamics")
        h2 = ["Group", "Algo", "Class", "Variant", "LR Mult", "AUC (↓)", "T-95% (K)", "Volatility", "Spike Freq", "Collapse"]
        lines.append("| " + " | ".join(h2) + " |")
        lines.append("|" + "|".join(["---"] * len(h2)) + "|")
        for s in core_runs:
            row = [s['grouping'], s['algorithm'], s['class'], s['variant'] or '-', s['lr_mult'],
                   fmt(s['auc'], 4), f"{s['t95_step_k']}" if s['t95_step_k'] is not None else "-",
                   fmt(s['volatility'], 5), fmt(s['spike_freq'], 4),
                   f"⚠️ {s['collapse_step']}" if s['is_collapsed'] else "-"]
            lines.append("| " + " | ".join(map(str, row)) + " |")

        # 1.3 Fidelity to Full-Precision Baseline
        lines.append("\n### 1.3 Fidelity to Full-Precision Baseline")
        lines.append("| Group | Algo | LR Mult | Target (Class) | Final PPL Delta (↑ worse) | Train Trajectory MAE (↓) |")
        lines.append("|---|---|---|---|---|---|")
        
        target_groups = defaultdict(list)
        for s in summaries: 
            if s['intended_steps'] == 20000 and s['class'] not in ['torch', 'hf']:
                target_groups[(s['grouping'], s['algorithm'], s['lr_mult'], s['seq_len'])].append(s)
        
        for key, targets in target_groups.items():
            grp, algo, lr, seq = key
            
            for tgt in targets:
                base = find_baseline(summaries, algo, lr, intended_steps=20000, batch_size=tgt['batch_size'])
                if base:
                    base_ppl = base.get('final_ppl')
                    tgt_ppl = tgt.get('final_ppl')
                    delta = (tgt_ppl - base_ppl) if (base_ppl and tgt_ppl) else None
                    mae_val = compute_train_mae(base, tgt)
                    mae = f"{mae_val:.4f}" if mae_val is not None else "N/A"
                else:
                    delta = None
                    mae = "No baseline"
                    
                tgt_name = f"{tgt['class']}" + (f"_{tgt['variant']}" if tgt['variant'] else "")
                lines.append(f"| {grp} | {algo} | {lr} | {tgt_name} | {fmt(delta, 2)} | {mae} |")

    # ==========================================
    # Section 2: Hyperparameter Sensitivity (10K Sweeps)
    # ==========================================
    sweep_runs = [s for s in summaries if s['intended_steps'] == 10000]
    if sweep_runs:
        lines.append("\n## 2. Hyperparameter Sensitivity (10K Sweeps, TinyLlama-1.1B)")
        
        # 2.1 LR Sensitivity
        lr_groups = defaultdict(list)
        for s in sweep_runs:
            lr_groups[(s['grouping'], s['algorithm'], s['class'], s['variant'], s['batch_size'])].append(s)
            
        # Check if we have any valid LR sweeps or extreme single-point tests
        has_lr_sweep = False
        for runs in lr_groups.values():
            is_sweep = len(runs) >= 2
            is_extreme_single = len(runs) == 1 and abs(float(runs[0]['lr_mult']) - 10.0) < 1e-6
            if is_sweep or is_extreme_single:
                has_lr_sweep = True
                break
                
        if has_lr_sweep:
            lines.append("\n### 2.1 Learning Rate Sensitivity")
            lines.append("| Group | Algo | Class | Variant | BS | LR CV (↓) | Collapse | MAE@x0.1 (↓) | MAE@x1.0 (↓) | MAE@x10.0 (↓) |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|")
            
            lr_rows = []
            for key, runs in lr_groups.items():
                # Filter: keep full sweeps (>=2 runs) OR single extreme lr*10.0 tests
                is_sweep = len(runs) >= 2
                is_extreme_single = len(runs) == 1 and abs(float(runs[0]['lr_mult']) - 10.0) < 1e-6
                if not (is_sweep or is_extreme_single):
                    continue
                    
                grp, algo, cls, var, bs = key
                n_total = len(runs)
                n_collapse = sum(1 for r in runs if r['is_collapsed'])
                
                # Annotate specific collapsed LRs (e.g., 1/3 (lr*10.0))
                collapse_str = f"{n_collapse}/{n_total}"
                if n_collapse > 0:
                    collapsed_lrs = sorted(list(set(float(r['lr_mult']) for r in runs if r['is_collapsed'])))
                    lr_labels = [f"lr*{lr}" for lr in collapsed_lrs]
                    collapse_str += f" ({', '.join(lr_labels)})"
                
                healthy_losses = [r['late_train_loss'] for r in runs 
                                  if not r['is_collapsed'] and r['late_train_loss'] is not None]
                cv = np.std(healthy_losses) / (np.mean(healthy_losses) + 1e-8) if len(healthy_losses) >= 2 else np.nan
                
                mae_cols = []
                for lr_target in [0.1, 1.0, 10.0]:
                    if cls in ['torch', 'hf']:
                        mae_cols.append(None)
                    else:
                        tgt_run = next((r for r in runs if abs(float(r['lr_mult']) - lr_target) < 1e-6), None)
                        if tgt_run is None:
                            mae_cols.append(None)
                        elif tgt_run['is_collapsed']:
                            mae_cols.append(f"NaN(step={tgt_run['collapse_step']})")
                        else:
                            base = find_baseline(summaries, algo, lr_target, intended_steps=10000, batch_size=bs)
                            mae_val = compute_train_mae(base, tgt_run) if base else None
                            mae_cols.append(mae_val)
                
                lr_rows.append((grp, algo, cls, var, bs, cv, collapse_str, mae_cols))
                
            lr_rows.sort(key=lambda x: (
                ALGO_ORDER.get(x[1], 9),
                CLASS_ORDER.get(x[2], 9),
                x[5] if not np.isnan(x[5]) else 999
            ))
            
            for grp, algo, cls, var, bs, cv, collapse_str, mae_cols in lr_rows:
                cv_str = f"{cv:.4f}" if not np.isnan(cv) else "-"
                mae_strs = [fmt(m, 4) if m is not None else "-" for m in mae_cols]
                lines.append(f"| {grp} | {algo} | {cls} | {var or '-'} | {bs} | {cv_str} | {collapse_str} | {mae_strs[0]} | {mae_strs[1]} | {mae_strs[2]} |")

        # 2.2 Batch Size Scaling
        bs_groups = defaultdict(list)
        for s in sweep_runs:
            bs_groups[(s['grouping'], s['algorithm'], s['class'], s['variant'], s['lr_mult'])].append(s)
            
        has_bs_sweep = any(len(runs) >= 2 for runs in bs_groups.values())
        if has_bs_sweep:
            lines.append("\n### 2.2 Batch Size Scaling")
            lines.append("| Group | Algo | Class | Variant | LR Mult | BS | AUC (↓) | T-95% (K) | MAE vs Base (↓) |")
            lines.append("|---|---|---|---|---|---|---|---|---|")
            
            bs_rows = []
            for key, runs in bs_groups.items():
                if len(runs) < 2:
                    continue
                grp, algo, cls, var, lr = key
                
                for s in runs:
                    if s['class'] in ['torch', 'hf']:
                        mae_str = "-"
                    elif s['is_collapsed']:
                        mae_str = f"NaN(step={s['collapse_step']})"
                    else:
                        base = find_baseline(summaries, algo, lr, intended_steps=10000, batch_size=s['batch_size'])
                        mae_val = compute_train_mae(base, s) if base else None
                        mae_str = fmt(mae_val, 4) if mae_val is not None else "No baseline"
                    
                    bs_rows.append((grp, algo, cls, var, lr, s['batch_size'], s['auc'], s['t95_step_k'], mae_str))
                    
            bs_rows.sort(key=lambda x: (
                ALGO_ORDER.get(x[1], 9),
                CLASS_ORDER.get(x[2], 9),
                x[4], # lr_mult
                x[5]  # batch_size
            ))
            
            for grp, algo, cls, var, lr, bs, auc, t95, mae in bs_rows:
                t95_str = f"{t95}" if t95 is not None else "-"
                lines.append(f"| {grp} | {algo} | {cls} | {var or '-'} | {lr} | {bs} | {fmt(auc, 4)} | {t95_str} | {mae} |")

    # ==========================================
    # Section 3: Long-Horizon Training (100K Steps)
    # ==========================================
    long_runs = [s for s in summaries if s['intended_steps'] >= 50000]
    if long_runs:
        lines.append("\n## 3. Long-Horizon Training (100K Steps, GPT2-124M)")
        
        # 3.1 Stability & Steady-State Quality
        lines.append("\n### 3.1 Stability & Steady-State Quality")
        lines.append("| Group | Algo | Class | Variant | Late Train Loss (↓) | Δ vs Base (↓) | Late Grad CV | Collapse |")
        lines.append("|---|---|---|---|---|---|---|---|")
        
        for s in long_runs:
            if s['class'] in ['torch', 'hf']:
                delta_str = "-"
            else:
                base = find_baseline(long_runs, s['algorithm'], s['lr_mult'], batch_size=s['batch_size'])
                if base and base['late_train_loss'] is not None and s['late_train_loss'] is not None:
                    delta = s['late_train_loss'] - base['late_train_loss']
                    delta_str = f"{delta:+.4f}"
                else:
                    delta_str = "No base"
            collapse_str = f"⚠️ step={s['collapse_step']}" if s['is_collapsed'] else "-"
            lines.append(f"| {s['grouping']} | {s['algorithm']} | {s['class']} | {s['variant'] or '-'} | "
                         f"{fmt(s['late_train_loss'], 4)} | {delta_str} | "
                         f"{fmt(s['late_grad_cv'], 4)} | {collapse_str} |")

        # 3.2 Error Evolution (Staged MAE, 4 equal quartiles)
        lines.append("\n### 3.2 Error Evolution (Staged MAE, 4 equal quartiles)")
        lines.append("| Group | Algo | Class | Variant | MAE Q1 (0-25%) (↓) | MAE Q2 (25-50%) (↓) | MAE Q3 (50-75%) (↓) | MAE Q4 (75-100%) (↓) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        
        for s in long_runs:
            if s['class'] in ['torch', 'hf']:
                lines.append(f"| {s['grouping']} | {s['algorithm']} | {s['class']} | {s['variant'] or '-'} | - | - | - | - |")
            else:
                base = find_baseline(long_runs, s['algorithm'], s['lr_mult'], batch_size=s['batch_size'])
                if base:
                    q1, q2, q3, q4 = compute_staged_mae(base, s)
                    lines.append(f"| {s['grouping']} | {s['algorithm']} | {s['class']} | {s['variant'] or '-'} | "
                                 f"{fmt(q1, 4)} | {fmt(q2, 4)} | {fmt(q3, 4)} | {fmt(q4, 4)} |")
                else:
                    lines.append(f"| {s['grouping']} | {s['algorithm']} | {s['class']} | {s['variant'] or '-'} | No base | No base | No base | No base |")

    # ==========================================
    # Section 4: Trajectories (20K Steps)
    # ==========================================
    lines.append("\n## 4. Sampled Trajectories (20K Steps, TinyLlama-1.1B)")
    lines.append("Values are sampled at approximately 1K-step intervals; evaluation starts at the first available checkpoint.")
    for s in summaries:
        if s['intended_steps'] == 20000:
            name = f"{s['grouping']}_{s['algorithm']}_{s['class']}" + (f"_{s['variant']}" if s['variant'] else "")
            lines.append(f"\n**{name} (x{s['lr_mult']}, bs{s['batch_size']})**")
            lines.append(f"- Train Loss: {' ➔ '.join(map(str, s['train_traj']))}")
            if s['ppl_traj']:
                lines.append(f"- Eval PPL:   {' ➔ '.join(map(str, s['ppl_traj']))}")

    # ==========================================
    # Section 5: Metric Glossary
    # ==========================================
    lines.append("\n## 5. Metric Glossary")
    lines.append("- **Final PPL**: Perplexity on the evaluation set at the final step. Calculated as `exp(eval_loss)`.")
    lines.append("- **Avg Train Loss**: Global average of `train/loss` over the entire training run (HF Trainer's `train/train_loss`). Captures early-stage instability that Late Train Loss may miss.")
    lines.append("- **Late Train Loss**: Mean of the last 20% of the `train/loss` curve. Represents steady-state convergence quality, unaffected by early high loss.")
    lines.append("- **AUC**: Normalized Area Under the Training Loss Curve. Lower indicates faster overall convergence.")
    lines.append("- **T-95% (K)**: Steps (in thousands) required to achieve 95% of the total training loss reduction.")
    lines.append("- **Collapse**: Number of runs that experienced divergence (NaN/Inf gradients or zero-loss collapse) out of total runs in the sweep group.")
    lines.append("- **Late Grad Norm CV**: Coefficient of variation of gradient norms in the final 20% of training. Lower indicates smoother late-stage optimization.")
    lines.append("- **Peak Alloc (MiB)**: Maximum memory allocated by PyTorch (`max_allocated_mb`), excluding allocator fragmentation. If a 20K run is missing this data, the value is backfilled from a matching 1K-step proxy run (peak allocation is reached early and remains stable).")
    lines.append("- **Throughput (tok/s)**: Training tokens per second (`train_tokens_per_second`). Eval overhead is equal across all 20K runs.")
    lines.append("- **Final PPL Delta**: Positive values indicate perplexity degradation relative to the full-precision baseline.")
    lines.append("- **Train MAE vs Base**: Mean absolute error of training loss trajectory compared to the full-precision baseline on common steps.")
    lines.append("- **MAE Q1-Q4**: Staged MAE of training loss trajectory vs baseline, split into 4 equal quartiles of total steps. Reveals whether quantization error converges or accumulates over time.")
    lines.append("- **C-Quant**: Quantization type for Confidence/Residual states. When no explicit `_c_` suffix is present, C follows V-Quant. Displayed as `-` for non-CAME optimizers.")
    
    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f: f.write(content)
    print(f"\n[SUCCESS] Report saved to: {output_path}")

# ==========================================
# 4. CLI Entry Point
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Compile TensorBoard events into a structured markdown report.")
    parser.add_argument("--base_dir", type=str, default=None, help="Base directory containing TB event files.")
    parser.add_argument("--output", type=str, default=None, help="Output markdown path.")
    args = parser.parse_args()

    if args.base_dir:
        base_dir = Path(args.base_dir)
    else:
        script_dir = Path(__file__).resolve().parent
        base_dir = script_dir.parent / "benchmarks"
        if not base_dir.exists():
            base_dir = script_dir / "benchmarks"

    if not base_dir.exists():
        print(f"[ERROR] Cannot find benchmarks directory at: {base_dir}")
        return

    if args.output:
        output_path = Path(args.output)
    else:
        reports_dir = base_dir.parent / "reports_md"
        reports_dir.mkdir(parents=True, exist_ok=True)
        output_path = reports_dir / "tb_analysis_report.md"

    print(f"[INFO] Scanning: {base_dir}")
    event_files = glob.glob(os.path.join(base_dir, "**", "events.out.tfevents.*"), recursive=True)
    run_dirs = sorted(set(os.path.dirname(f) for f in event_files))
    
    all_summaries = [process_run(d) for d in run_dirs if os.path.exists(d)]
    generate_report(all_summaries, output_path)

if __name__ == "__main__":
    main()
