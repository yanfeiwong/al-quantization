import os
import argparse
import random
import re
import numpy as np
import torch
import time
from torch.utils.tensorboard import SummaryWriter
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers.integrations import TensorBoardCallback

from utils.gpu_memory_monitor import get_gpu_memory_details
from utils.data_utils_llm import get_streaming_dataset, get_data_collator, get_eval_dataset
from utils.model_presets_llm import get_model_and_tokenizer
from utils.grouping_utils import get_param_groups, print_optimizer_groups

# ==========================================
# Global Configurations
# ==========================================
SKIP_EXISTING = True
OOM_SHARED_MEM_THRESHOLD_MB = 1024

# Dual Watchdog System
WATCHDOG_TIMEOUT_SEC = 300       # Local: If 20 steps take > 5 mins, system is thrashing in swap.
MAX_TOTAL_TRAINING_SEC = 10*60*60   # Global: Hard limit of 5 hours per experiment.

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

def normalize_run_suffix(run_suffix: str) -> str:
    """Return a filesystem-safe optional suffix without changing legacy names."""
    if not run_suffix:
        return ""
    if re.fullmatch(r"[A-Za-z0-9_.-]+", run_suffix) is None:
        raise ValueError(
            "run_suffix may contain only letters, digits, '.', '_' and '-'."
        )
    return run_suffix if run_suffix.startswith(("_", "-")) else f"_{run_suffix}"

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
# Custom Callbacks
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
        
        # 1. Global Watchdog (Total Time Limit)
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed > MAX_TOTAL_TRAINING_SEC:
                print(f"\n[FATAL TIMEOUT] Total training time {elapsed/3600:.2f} Hrs exceeds {MAX_TOTAL_TRAINING_SEC/3600:.1f} Hrs limit. Killing to save queue.")
                os._exit(1)

        step = state.global_step
        current_time = time.time()
        step_duration = current_time - self.last_log_time
        self.last_log_time = current_time
        
        # 2. Local Watchdog (Swap Thrashing Detection)
        if step > 0 and step_duration > WATCHDOG_TIMEOUT_SEC: 
            print(f"\n[FATAL WATCHDOG] Step interval took {step_duration:.1f}s. System is thrashing in swap. Killing process to save the queue.")
            os._exit(1) 
        
        try:
            mem = get_gpu_memory_details()
            shared_mb = mem["shared_gpu_memory_mb"]
            
            # Check OOM Threshold (Mark it, but DO NOT stop logging)
            if not self.oom_triggered and shared_mb > OOM_SHARED_MEM_THRESHOLD_MB:
                self.oom_triggered = True
                self.oom_step = step
                print(f"\n[WARNING] Shared memory exceeded {OOM_SHARED_MEM_THRESHOLD_MB}MB at step {step}. Performance degradation expected.")
                tb_writer.add_scalar("mem_dynamic/oom_triggered", 1, step)
            
            # Always log memory to capture the full degradation curve
            tb_writer.add_scalar("mem_dynamic/dedicated_mb", mem["dedicated_gpu_memory_mb"], step)
            tb_writer.add_scalar("mem_dynamic/shared_mb", shared_mb, step)
            tb_writer.add_scalar("mem_dynamic/total_used_mb", mem["total_used_memory_mb"], step)
            tb_writer.add_scalar("mem_torch/allocated_mb", torch.cuda.memory_allocated() / (1024 ** 2), step)
            tb_writer.add_scalar("mem_torch/max_allocated_mb", torch.cuda.max_memory_allocated() / (1024 ** 2), step)
            tb_writer.add_scalar("mem_torch/reserved_mb", torch.cuda.memory_reserved() / (1024 ** 2), step)
            tb_writer.add_scalar("mem_torch/max_reserved_mb", torch.cuda.max_memory_reserved() / (1024 ** 2), step)
            
            if not self.opt_mem_logged and step > 0:
                opt_mem = get_tensor_memory_mb(self.trainer.optimizer)
                tb_writer.add_scalar("mem_static/optimizer_mb", opt_mem, step)
                self.opt_mem_logged = True
        except Exception as e:
            print(f"Dynamic memory logging error: {e}")

# ==========================================
# Core Experiment Runner
# ==========================================
def run_experiment(args):
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(parent_dir, "benchmarks", args.task, args.model, args.dataset, f"{args.max_steps}steps")
    run_name = (
        f"{args.grouping}_{args.optimizer}_x{args.lr_mult}_bs{args.batch_size}_seq{args.seq_len}"
        f"{normalize_run_suffix(args.run_suffix)}"
    )
    log_dir = os.path.join(output_dir, run_name)

    if SKIP_EXISTING and os.path.exists(log_dir) and os.listdir(log_dir):
        print(f"[SKIP] Run '{run_name}' already exists. Skipping...")
        return

    set_seed(args.seed)
    if args.task == "pretrain":
        from optimizer_presets_pretrain import OPTIMIZER_PRESETS
    elif args.task == "finetune":
        # from optimizer_presets_finetune import OPTIMIZER_PRESETS
        raise NotImplementedError("Finetune task is not implemented yet. Please add the preset file.")
    else:
        raise ValueError(f"Unknown task: {args.task}")

    model, tokenizer = get_model_and_tokenizer(args.model)

    raw_train_dataset = get_streaming_dataset(args.dataset, split="train", seed=args.seed)
    tokenized_train_dataset = raw_train_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"],
        fn_kwargs={"tokenizer": tokenizer, "max_length": args.seq_len}
    )
    data_collator = get_data_collator(tokenizer)

    if args.optimizer not in OPTIMIZER_PRESETS:
        raise ValueError(f"Optimizer '{args.optimizer}' not found.")
        
    opt_cfg = OPTIMIZER_PRESETS[args.optimizer]
    opt_class = opt_cfg["class"]
    kwargs = opt_cfg["kwargs"].copy()
    apollo_addit_kwargs = opt_cfg.get("apollo_addit_kwargs", {})
    
    if "lr" not in kwargs:
        raise ValueError(f"Preset '{args.optimizer}' must explicitly define 'lr' in kwargs.")
        
    max_grad_norm = opt_cfg.get("max_grad_norm", 0.0)
    
    param_groups = get_param_groups(
        model, args.grouping, opt_class, 
        nd_factored=opt_cfg.get("nd_factored", False),
        apollo_addit_kwargs=apollo_addit_kwargs,
        low_rank_proj_targets=opt_cfg.get("low_rank_proj_targets", None),
        fallback_behavior=opt_cfg.get("fallback_behavior", None)
    )
    
    optimizer = opt_class(param_groups, **kwargs)

    for param_group in optimizer.param_groups:
        param_group["lr"] *= args.lr_mult
        
    print_optimizer_groups(optimizer)

    eval_dataset = None
    eval_args = {}
    if args.do_eval:
        raw_eval_dataset = get_eval_dataset(args.dataset)
        eval_dataset = raw_eval_dataset.map(
            tokenize_function, batched=True, remove_columns=["text"],
            fn_kwargs={"tokenizer": tokenizer, "max_length": args.seq_len}
        )
        eval_args = {
            "eval_strategy": "steps",
            "eval_steps": args.eval_steps,
            "per_device_eval_batch_size": args.batch_size,
        }

    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        per_device_train_batch_size=args.batch_size,
        max_steps=args.max_steps,
        warmup_steps=1000,         
        lr_scheduler_type="constant_with_warmup",     
        logging_steps=20,
        save_strategy="no",
        report_to="tensorboard",
        include_num_input_tokens_seen=True,
        seed=args.seed,
        bf16=True,  
        dataloader_drop_last=True,
        dataloader_num_workers=0, 
        max_grad_norm=max_grad_norm,
        **eval_args,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator, 
        optimizers=(optimizer, None),
    )
    
    tb_writer = SummaryWriter(log_dir=log_dir)
    
    trainer.pop_callback(TensorBoardCallback)
    trainer.add_callback(TensorBoardCallback(tb_writer=tb_writer))
    trainer.add_callback(GPUMemoryCallback(trainer))

    print(f"[{run_name}] Starting training...")
    trainer.train()
    tb_writer.close()
    print(f"[{run_name}] Finished.")

# ==========================================
# CLI Entry Point
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="LLM Benchmark")
    parser.add_argument("--task", type=str, default="pretrain", choices=["pretrain", "finetune"], 
                        help="Task type: pretrain or finetune")
    parser.add_argument("--model", type=str, default="TinyLlama-1.1B")
    parser.add_argument("--dataset", type=str, default="wikitext-103-raw-v1")
    parser.add_argument("--optimizer", type=str, default="adamw_torch")
    parser.add_argument("--grouping", type=str, default="G0", choices=["G0", "G1", "G2"])
    parser.add_argument("--lr_mult", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=921)
    parser.add_argument(
        "--run_suffix",
        type=str,
        default="",
        help="Optional suffix for the run directory (for example, seed922).",
    )
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--eval_steps", type=int, default=2000)
    args = parser.parse_args()
    run_experiment(args)

if __name__ == '__main__':
    main()
