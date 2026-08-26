"""
trace_states.py — Lightweight optimizer state trace runner.

Based on train_llm.py. Captures optimizer state statistics at fixed
checkpoints without eval, checkpoint saving, or full benchmark logging.

Purpose: Validate whether real LLM optimizer states fall within the
distribution ranges covered by our synthetic theory experiments.

Usage:
    python trace_states.py --optimizer=adamw_torch --batch_size=4
    python trace_states.py --optimizer=adafactor_hf --batch_size=4
"""

import os
import argparse
import random
import numpy as np
import torch
import time
from torch.utils.tensorboard import SummaryWriter
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers.integrations import TensorBoardCallback

from utils.gpu_memory_monitor import get_gpu_memory_details
from utils.data_utils_llm import get_streaming_dataset, get_data_collator
from utils.model_presets_llm import get_model_and_tokenizer
from utils.grouping_utils import get_param_groups, print_optimizer_groups

# ==========================================
# Trace Configuration
# ==========================================
TRACE_STEPS = {50, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000}

# V histogram: log2 space, bins from -130 to +10, step 0.5
LOG2_HIST_MIN = -130.0
LOG2_HIST_MAX = 10.0
LOG2_HIST_STEP = 0.5
LOG2_N_BINS = int((LOG2_HIST_MAX - LOG2_HIST_MIN) / LOG2_HIST_STEP)

# M histogram: normalized by absmax, -1 to +1, 200 bins
M_HIST_BINS = 200

# Block size for block-wise log-range (matches our optimizer's default)
V_BLOCK_SIZE = 2048

# Trace-run watchdog settings
OOM_SHARED_MEM_THRESHOLD_MB = 1024
WATCHDOG_TIMEOUT_SEC = 300
MAX_TOTAL_TRAINING_SEC = 106060


# ==========================================
# Helper Functions
# ==========================================
def tokenize_function(examples, tokenizer=None, max_length=512):
    return tokenizer(examples["text"], truncation=True, max_length=max_length)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_tensor_memory_mb(obj):
    total_bytes = 0
    if hasattr(obj, 'parameters'):
        for p in obj.parameters():
            if p.is_cuda: total_bytes += p.element_size() * p.nelement()
    else:
        for state in obj.state.values():
            for v in state.values():
                if torch.is_tensor(v) and v.is_cuda:
                    total_bytes += v.element_size() * v.nelement()
    return total_bytes / (1024 ** 2)


# ==========================================
# GPU Memory Callback (same as train_llm.py)
# ==========================================
class GPUMemoryCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.opt_mem_logged = False
        self.oom_triggered = False
        self.oom_step = -1
        self.last_log_time = time.time()
        self.start_time = None

    def _get_tb_writer(self):
        for cb in self.trainer.callback_handler.callbacks:
            if isinstance(cb, TensorBoardCallback) and cb.tb_writer is not None:
                return cb.tb_writer
        return None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.last_log_time = time.time()
        tb_writer = self._get_tb_writer()
        if tb_writer:
            try:
                model_mem = get_tensor_memory_mb(self.trainer.model)
                tb_writer.add_scalar("mem_static/model_mb", model_mem, 0)
            except Exception as e:
                print(f"Model memory logging error: {e}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        tb_writer = self._get_tb_writer()
        if not tb_writer: return

        # Global Watchdog (Total Time Limit)
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed > MAX_TOTAL_TRAINING_SEC:
                print(f"\n[FATAL TIMEOUT] Total training time {elapsed/3600:.2f} Hrs exceeds limit. Killing.")
                os._exit(1)

        step = state.global_step
        current_time = time.time()
        step_duration = current_time - self.last_log_time
        self.last_log_time = current_time

        # Local Watchdog (Swap Thrashing Detection)
        if step > 0 and step_duration > WATCHDOG_TIMEOUT_SEC:
            print(f"\n[FATAL WATCHDOG] Step interval took {step_duration:.1f}s. Killing.")
            os._exit(1)

        try:
            mem = get_gpu_memory_details()
            shared_mb = mem["shared_gpu_memory_mb"]
            if not self.oom_triggered and shared_mb > OOM_SHARED_MEM_THRESHOLD_MB:
                self.oom_triggered = True
                self.oom_step = step
                print(f"\n[WARNING] Shared memory exceeded {OOM_SHARED_MEM_THRESHOLD_MB}MB at step {step}.")
                tb_writer.add_scalar("mem_dynamic/oom_triggered", 1, step)
            tb_writer.add_scalar("mem_dynamic/dedicated_mb", mem["dedicated_gpu_memory_mb"], step)
            tb_writer.add_scalar("mem_dynamic/shared_mb", shared_mb, step)
            tb_writer.add_scalar("mem_dynamic/total_used_mb", mem["total_used_memory_mb"], step)
            if not self.opt_mem_logged and step > 0:
                opt_mem = get_tensor_memory_mb(self.trainer.optimizer)
                tb_writer.add_scalar("mem_static/optimizer_mb", opt_mem, step)
                self.opt_mem_logged = True
        except Exception as e:
            print(f"Dynamic memory logging error: {e}")


# ==========================================
# State Trace Callback
# ==========================================
class StateTraceCallback(TrainerCallback):
    """
    Captures optimizer state statistics at predefined steps.
    
    For each parameter, records:
      V: log2 histogram, zero fraction, percentiles, block-wise log-range
      M: signed histogram (normalized), absmax, L2 norm
      param_rms: parameter RMS (for Adafactor scale_parameter analysis)
    """

    def __init__(self, trace_dir, trace_steps, v_block_size=2048):
        self.trace_dir = trace_dir
        self.trace_steps = trace_steps
        self.v_block_size = v_block_size
        self._optimizer = None
        self._model = None

    def on_train_begin(self, args, state, control, **kwargs):
        # Cache optimizer and model references (version-safe)
        self._optimizer = kwargs.get("optimizer")
        self._model = kwargs.get("model")

        # Diagnostic: confirm optimizer is accessible
        opt_available = self._optimizer is not None
        print(f"[TRACE] optimizer available at train_begin: {opt_available}")
        if not opt_available:
            print("[TRACE] FATAL: optimizer not found in on_train_begin kwargs. "
                  "Trace cannot proceed. Exiting.")
            os._exit(1)

        # Create output directory once
        os.makedirs(self.trace_dir, exist_ok=True)
        print(f"[TRACE] Output directory: {self.trace_dir}")

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step not in self.trace_steps:
            return

        # Use cached optimizer; fallback to kwargs if cache failed
        optimizer = self._optimizer or kwargs.get("optimizer")
        model = self._model or kwargs.get("model")
        if optimizer is None or model is None:
            print(f"[TRACE] WARNING: Cannot access optimizer/model at step {step}. Skipping.")
            return

        snapshot = {
            "step": step,
            "hist_params": {
                "log2_hist_min": LOG2_HIST_MIN,
                "log2_hist_max": LOG2_HIST_MAX,
                "log2_hist_step": LOG2_HIST_STEP,
                "log2_n_bins": LOG2_N_BINS,
                "m_hist_bins": M_HIST_BINS,
                "v_block_size": self.v_block_size,
            },
            "params": {},
        }

        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if param not in optimizer.state:
                    continue
                opt_state = optimizer.state[param]
                param_stats = self._extract_param_stats(param, opt_state, name)
                if param_stats:
                    snapshot["params"][name] = param_stats

        # Save
        save_path = os.path.join(self.trace_dir, f"step_{step}.pt")
        torch.save(snapshot, save_path)
        n_params = len(snapshot["params"])
        print(f"[TRACE] Step {step:>6d}: captured {n_params} params -> {save_path}")

    def _extract_param_stats(self, param, opt_state, name):
        stats = {
            "shape": list(param.shape),
            "numel": param.numel(),
            "ndim": param.ndim,
            "param_rms": (param.float().norm() / (param.numel() ** 0.5)).item(),
        }

        # --- Extract V ---
        v_data = self._get_v(opt_state)
        if v_data is not None:
            if isinstance(v_data, dict):
                for key, val in v_data.items():
                    stats[f"v_{key}"] = self._compute_v_stats(val)
            else:
                stats["v"] = self._compute_v_stats(v_data)

        # --- Extract M ---
        m_data = self._get_m(opt_state)
        if m_data is not None:
            stats["m"] = self._compute_m_stats(m_data)

        # --- Extract Residual (CAME) ---
        res_data = self._get_res(opt_state)
        if res_data is not None:
            stats["res_row"] = self._compute_v_stats(res_data["row"])
            stats["res_col"] = self._compute_v_stats(res_data["col"])

        return stats

    def _get_v(self, opt_state):
        """Extract V (second moment) from optimizer state."""
        # AdamW / torch native
        if "exp_avg_sq" in opt_state:
            return opt_state["exp_avg_sq"]
        # HF Adafactor (factored)
        if "exp_avg_sq_row" in opt_state and "exp_avg_sq_col" in opt_state:
            return {
                "row": opt_state["exp_avg_sq_row"],
                "col": opt_state["exp_avg_sq_col"],
            }
        # Our Adafactor8Bit (factored, quantize=False)
        if "row_var" in opt_state and "col_var" in opt_state:
            return {
                "row": opt_state["row_var"],
                "col": opt_state["col_var"],
            }
        # Our Adafactor8Bit (non-factored, quantize=False)
        if "variance" in opt_state:
            return opt_state["variance"]
        return None

    def _get_m(self, opt_state):
        """Extract M (first moment) from optimizer state."""
        if "exp_avg" in opt_state:
            return opt_state["exp_avg"]
        # Our Adafactor8Bit (quantize=False)
        if "m" in opt_state and torch.is_tensor(opt_state["m"]):
            return opt_state["m"]
        return None

    def _get_res(self, opt_state):
        """Extract Residual/Confidence EMA from optimizer state (CAME specific)."""
        # Official CAME
        if "exp_avg_res_row" in opt_state and "exp_avg_res_col" in opt_state:
            return {
                "row": opt_state["exp_avg_res_row"],
                "col": opt_state["exp_avg_res_col"],
            }
        # Our Adafactor8Bit (CAME path, quantize=False)
        if "conf_row" in opt_state and "conf_col" in opt_state:
            return {
                "row": opt_state["conf_row"],
                "col": opt_state["conf_col"],
            }
        return None

    def _compute_v_stats(self, v):
        """Compute statistics for a non-negative V tensor."""
        v_flat = v.flatten().float()
        numel = v_flat.numel()

        # Zero fraction
        zero_frac = (v_flat == 0).float().mean().item()

        # Log2 statistics (non-zero elements only)
        v_nz = v_flat[v_flat > 0]
        if v_nz.numel() > 0:
            log_v = torch.log2(v_nz)
            hist = torch.histc(log_v, bins=LOG2_N_BINS,
                               min=LOG2_HIST_MIN, max=LOG2_HIST_MAX)
            MAX_Q_SAMPLES = 1_000_000
            if log_v.numel() > MAX_Q_SAMPLES:
                stride = log_v.numel() // MAX_Q_SAMPLES
                log_v_sample = log_v[::stride]
            else:
                log_v_sample = log_v
            median = log_v_sample.median().item()
            p5 = log_v_sample.quantile(0.05).item()
            p95 = log_v_sample.quantile(0.95).item()
            p99 = log_v_sample.quantile(0.99).item()
            min_val = log_v.min().item()
            max_val = log_v.max().item()
        else:
            hist = torch.zeros(LOG2_N_BINS)
            median = p5 = p95 = p99 = min_val = max_val = 0.0

        # Block-wise log-range
        block_size = self.v_block_size
        pad = (block_size - numel % block_size) % block_size
        v_padded = torch.nn.functional.pad(v_flat, (0, pad))
        blocks = v_padded.view(-1, block_size)
        is_zero = (blocks == 0)
        v_safe = blocks.clamp(min=1e-38)
        log_blocks = torch.log2(v_safe)

        # Max per block (zeros -> -inf, won't affect max)
        log_for_max = log_blocks.clone()
        log_for_max[is_zero] = float("-inf")
        block_max = log_for_max.max(dim=1).values

        # Min per block (zeros -> +inf, won't affect min)
        log_for_min = log_blocks.clone()
        log_for_min[is_zero] = float("inf")
        block_min = log_for_min.min(dim=1).values

        # Range (only blocks with >= 2 non-zero elements)
        n_nz = (~is_zero).sum(dim=1)
        valid = n_nz >= 2
        block_range = torch.where(valid, block_max - block_min,
                                  torch.zeros_like(block_max))
        valid_ranges = block_range[valid]

        if valid_ranges.numel() > 0:
            br_median = valid_ranges.median().item()
            br_p95 = valid_ranges.quantile(0.95).item()
            br_max = valid_ranges.max().item()
        else:
            br_median = br_p95 = br_max = 0.0

        return {
            "numel": numel,
            "zero_frac": zero_frac,
            "log2_hist": hist.cpu(),
            "log2_median": median,
            "log2_p5": p5,
            "log2_p95": p95,
            "log2_p99": p99,
            "log2_min": min_val,
            "log2_max": max_val,
            "block_logrange_median": br_median,
            "block_logrange_p95": br_p95,
            "block_logrange_max": br_max,
            "n_valid_blocks": int(valid.sum().item()),
            "n_total_blocks": int(blocks.shape[0]),
        }

    def _compute_m_stats(self, m):
        """Compute statistics for a signed M tensor."""
        m_flat = m.flatten().float()
        numel = m_flat.numel()

        absmax = m_flat.abs().max().item()
        l2_norm = m_flat.norm().item()

        # Normalized histogram (-1 to +1)
        if absmax > 1e-30:
            m_norm = m_flat / absmax
            hist = torch.histc(m_norm, bins=M_HIST_BINS, min=-1.0, max=1.0)
        else:
            hist = torch.zeros(M_HIST_BINS)

        return {
            "numel": numel,
            "absmax": absmax,
            "l2_norm": l2_norm,
            "hist": hist.cpu(),
        }


# ==========================================
# Core Trace Runner
# ==========================================
def run_trace(args):
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_name = f"{args.grouping}_{args.optimizer}_x{args.lr_mult}_bs{args.batch_size}_seq{args.seq_len}"
    trace_dir = os.path.join(
        parent_dir, "state_traces",
        args.task, args.model, args.dataset, f"{args.max_steps}steps", run_name
    )

    set_seed(args.seed)

    # Load optimizer preset
    if args.task == "pretrain":
        from optimizer_presets_pretrain import OPTIMIZER_PRESETS
    else:
        raise ValueError(f"Unknown task: {args.task}")

    if args.optimizer not in OPTIMIZER_PRESETS:
        raise ValueError(f"Optimizer '{args.optimizer}' not found in presets.")

    # Model & data
    model, tokenizer = get_model_and_tokenizer(args.model)
    raw_train_dataset = get_streaming_dataset(args.dataset, split="train", seed=args.seed)
    tokenized_train_dataset = raw_train_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"],
        fn_kwargs={"tokenizer": tokenizer, "max_length": args.seq_len}
    )
    data_collator = get_data_collator(tokenizer)

    # Optimizer setup (identical to train_llm.py)
    opt_cfg = OPTIMIZER_PRESETS[args.optimizer]
    opt_class = opt_cfg["class"]
    kwargs = opt_cfg["kwargs"].copy()
    apollo_addit_kwargs = opt_cfg.get("apollo_addit_kwargs", {})
    max_grad_norm = opt_cfg.get("max_grad_norm", 0.0)

    param_groups = get_param_groups(
        model, args.grouping, opt_class,
        nd_factored=opt_cfg.get("nd_factored", False),
        apollo_addit_kwargs=apollo_addit_kwargs,
        low_rank_proj_targets=opt_cfg.get("low_rank_proj_targets", None),
        fallback_behavior=opt_cfg.get("fallback_behavior", None),
    )

    optimizer = opt_class(param_groups, **kwargs)
    for param_group in optimizer.param_groups:
        param_group["lr"] *= args.lr_mult

    print_optimizer_groups(optimizer)

    # Training arguments (no eval, no checkpoint)
    training_args = TrainingArguments(
        output_dir=trace_dir,
        run_name=f"trace_{args.optimizer}",
        per_device_train_batch_size=args.batch_size,
        max_steps=args.max_steps,
        warmup_steps=1000,
        lr_scheduler_type="constant_with_warmup",
        logging_steps=100,
        save_strategy="no",
        report_to="tensorboard",
        include_num_input_tokens_seen=True,
        seed=args.seed,
        bf16=True,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        max_grad_norm=max_grad_norm,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        data_collator=data_collator,
        optimizers=(optimizer, None),
    )

    # Tensorboard (for loss curve + memory monitoring)
    tb_writer = SummaryWriter(log_dir=trace_dir)
    trainer.pop_callback(TensorBoardCallback)
    trainer.add_callback(TensorBoardCallback(tb_writer=tb_writer))

    # GPU memory monitoring + watchdog (same as train_llm.py)
    trainer.add_callback(GPUMemoryCallback(trainer))

    # State trace callback
    trainer.add_callback(StateTraceCallback(
        trace_dir=trace_dir,
        trace_steps=TRACE_STEPS,
        v_block_size=V_BLOCK_SIZE,
    ))

    print(f"\n[TRACE] Optimizer: {args.optimizer}")
    print(f"[TRACE] Trace steps: {sorted(TRACE_STEPS)}")
    print(f"[TRACE] Starting...\n")

    trainer.train()
    tb_writer.close()
    print(f"\n[TRACE] Finished. Results in: {trace_dir}")


# ==========================================
# CLI Entry Point
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Optimizer State Trace")
    parser.add_argument("--task", type=str, default="pretrain",
                        choices=["pretrain", "finetune"])
    parser.add_argument("--model", type=str, default="TinyLlama-1.1B")
    parser.add_argument("--dataset", type=str, default="wikitext-103-raw-v1")
    parser.add_argument("--optimizer", type=str, default="adamw_torch")
    parser.add_argument("--grouping", type=str, default="G0",
                        choices=["G0", "G1", "G2"])
    parser.add_argument("--lr_mult", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=921)
    args = parser.parse_args()
    run_trace(args)


if __name__ == "__main__":
    main()
