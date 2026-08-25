import subprocess
import sys
from datetime import datetime

FAILED_LOG = "benchmark_failed.log"

COMMON_ARGS = {
    "task": "pretrain",
    "model": "TinyLlama-1.1B",
    "dataset": "wikitext-103-raw-v1",
    "max_steps": 1000,
    "seq_len": 512,
    "batch_size": 4,
    "seed": 921,
}

JOBS = [
    # ==========================================
    # Baselines
    # ==========================================
    {"optimizer": "adamw_torch",      "lr_mult": 0.1, "grouping": "G0"},
    {"optimizer": "adamw_8bit_bnb",   "lr_mult": 0.1, "grouping": "G0"},
    {"optimizer": "adamw_8bit_bnb",   "lr_mult": 0.1, "grouping": "G1"},
    {"optimizer": "came_torch",       "lr_mult": 0.1, "grouping": "G0"},
    {"optimizer": "adafactor_hf",     "lr_mult": 1.0, "grouping": "G0"},
    {"optimizer": "apollo_torch",     "lr_mult": 0.1, "grouping": "G0"},
]

def run_benchmark():
    expanded = []
    for job in JOBS:
        groupings = job.get("grouping", "G0")
        if isinstance(groupings, str):
            groupings = [groupings]
        for g in groupings:
            entry = {k: v for k, v in job.items() if k != "grouping"}
            entry["grouping"] = g
            expanded.append(entry)

    print(f"Benchmark started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total jobs: {len(expanded)}\n")

    for i, job in enumerate(expanded, 1):
        cmd = [sys.executable, "train_llm.py"]
        for k, v in COMMON_ARGS.items():
            if isinstance(v, bool):
                if v:
                    cmd.append(f"--{k}")
            else:
                cmd.append(f"--{k}={v}")
        for k, v in job.items():
            cmd.append(f"--{k}={v}")

        tag = f"{job['grouping']} | {job['optimizer']} | lr={job['lr_mult']}"
        print(f"[{i}/{len(expanded)}] {tag}")
        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"[WARNING] Failed (exit {result.returncode}). Continuing...")
            with open(FAILED_LOG, "a", encoding="utf-8") as f:
                f.write(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                    f"[{i}/{len(expanded)}] | "
                    f"grouping={job['grouping']} | "
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