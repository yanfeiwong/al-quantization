# Learning rate sweep: 0.1x / 1.0x / 10.0x
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
    "batch_size": 4,
    "seed": 921,
}

OPTIMIZERS = [
    # ==========================================
    # Baselines
    # ==========================================
    # {"optimizer": "adamw_torch",      "lr_mults": [0.1, 1.0, 10.0]},
    # {"optimizer": "adamw_8bit_bnb",   "lr_mults": [0.1, 1.0, 10.0]},
    # {"optimizer": "adafactor_hf",     "lr_mults": [0.1, 1.0, 10.0]},
    # {"optimizer": "came_torch",       "lr_mults": [0.1, 1.0, 10.0]},
    # {"optimizer": "apollo_torch",     "lr_mults": [0.1, 1.0, 10.0]},

    # ==========================================
    # Ours (uf8 M + al8 V + vblk2048)
    # ==========================================
    # {"optimizer": "adamw_ours_uf8_al8_vblk2048",     "lr_mults": [0.1, 1.0, 10.0]},
    # {"optimizer": "adafactor_ours_al8_vblk2048",      "lr_mults": [0.1, 1.0, 10.0]},
    # {"optimizer": "came_ours_uf8_al8_vblk2048",       "lr_mults": [0.1, 1.0, 10.0]},
    # {"optimizer": "apollo_ours_uf8_al8_vblk2048",     "lr_mults": [0.1, 1.0, 10.0]},
    
    # ==========================================
    # Ours - Supplemental
    # ==========================================
    # {"optimizer": "came_ours_uf8_al8_vblk2048_c_fp32", "lr_mults": [0.1, 1.0, 10.0]},
    # {"optimizer": "came_ours_uf8_al16_vblk2048",       "lr_mults": [0.1, 1.0, 10.0]},
]

def run_benchmark():
    jobs = []
    for entry in OPTIMIZERS:
        for lr in entry["lr_mults"]:
            jobs.append({"optimizer": entry["optimizer"], "lr_mult": lr})

    print(f"Benchmark started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Optimizers: {[e['optimizer'] for e in OPTIMIZERS]}")
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
        cmd.append("--grouping=G0")

        tag = f"G0 | {job['optimizer']} | lr={job['lr_mult']}"
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
                    f"exit={result.returncode}\n"
                )
        else:
            print(f"[SUCCESS] Completed.")
        print("-" * 50)

    print(f"\nBenchmark finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    run_benchmark()