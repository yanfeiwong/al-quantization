import subprocess
import sys
from datetime import datetime

FAILED_LOG = "benchmark_failed.log"

COMMON_ARGS = {
    "task": "pretrain",
    "model": "TinyLlama-1.1B",
    "dataset": "wikitext-103-raw-v1",
    "max_steps": 10000,
    "seq_len": 512,
    "seed": 921,
}

# Batch size sweep: 8 / 16 (bs=4 already covered in LR sweep)
BATCH_SIZES = [8, 16]

# Per-optimizer lr_mult: AdamW-style uses 0.1x, RMS-based uses 1.0x
OPTIMIZERS = [
    # ==========================================
    # Baselines
    # ==========================================
    # {"optimizer": "adamw_torch",      "lr_mult": 0.1},
    # {"optimizer": "adamw_8bit_bnb",   "lr_mult": 0.1},
    # {"optimizer": "adafactor_hf",     "lr_mult": 1.0},
    # {"optimizer": "came_torch",       "lr_mult": 0.1},
    # {"optimizer": "apollo_torch",     "lr_mult": 0.1},

    # ==========================================
    # Ours (uf8 M + al8 V + vblk2048)
    # ==========================================
    # {"optimizer": "adamw_ours_uf8_al8_vblk2048",     "lr_mult": 0.1},
    # {"optimizer": "adafactor_ours_al8_vblk2048",      "lr_mult": 1.0},
    # {"optimizer": "came_ours_uf8_al8_vblk2048",       "lr_mult": 0.1},
    # {"optimizer": "apollo_ours_uf8_al8_vblk2048",     "lr_mult": 0.1},
]

def run_benchmark():
    jobs = []
    for entry in OPTIMIZERS:
        for bs in BATCH_SIZES:
            jobs.append({
                "optimizer": entry["optimizer"],
                "lr_mult": entry["lr_mult"],
                "batch_size": bs,
            })

    print(f"Benchmark started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Optimizers: {[e['optimizer'] for e in OPTIMIZERS]}")
    print(f"Batch sizes: {BATCH_SIZES}")
    print(f"Total jobs: {len(jobs)}\n")

    for i, job in enumerate(jobs, 1):
        cmd = [sys.executable, "train_llm.py"]
        for k, v in COMMON_ARGS.items():
            if isinstance(v, bool):
                if v:
                    cmd.append(f"--{k}")
            else:
                cmd.append(f"--{k}={v}")
        cmd.append(f"--optimizer={job['optimizer']}")
        cmd.append(f"--lr_mult={job['lr_mult']}")
        cmd.append(f"--batch_size={job['batch_size']}")
        cmd.append("--grouping=G0")

        tag = f"G0 | {job['optimizer']} | lr={job['lr_mult']} | bs={job['batch_size']}"
        print(f"[{i}/{len(jobs)}] {tag}")
        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"[WARNING] Failed (exit {result.returncode}). Continuing...")
            with open(FAILED_LOG, "a", encoding="utf-8") as f:
                f.write(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                    f"[{i}/{len(jobs)}] | "
                    f"grouping=G0 | "
                    f"optimizer={job['optimizer']} | "
                    f"lr_mult={job['lr_mult']} | "
                    f"batch_size={job['batch_size']} | "
                    f"exit={result.returncode}\n"
                )
        else:
            print(f"[SUCCESS] Completed.")
        print("-" * 50)

    print(f"\nBenchmark finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    run_benchmark()