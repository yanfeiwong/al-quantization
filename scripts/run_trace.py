"""run_trace.py — Batch runner for state trace experiments."""

import subprocess
import sys
from datetime import datetime

COMMON_ARGS = {
    "task": "pretrain",
    "model": "TinyLlama-1.1B",
    "dataset": "wikitext-103-raw-v1",
    "max_steps": 10000,
    "seq_len": 512,
    "batch_size": 4,
    "seed": 921,
}

JOBS = [
    # {"optimizer": "adamw_torch",      "lr_mult": 0.1, "grouping": "G0"},
    # {"optimizer": "adafactor_hf",     "lr_mult": 1.0, "grouping": "G0"},
    {"optimizer": "came_torch",       "lr_mult": 0.1, "grouping": "G0"},
]


def run_trace():
    print(f"State trace started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total jobs: {len(JOBS)}\n")

    for i, job in enumerate(JOBS, 1):
        cmd = [sys.executable, "trace_states.py"]
        for k, v in COMMON_ARGS.items():
            cmd.append(f"--{k}={v}")
        for k, v in job.items():
            cmd.append(f"--{k}={v}")

        tag = f"{job['optimizer']} | lr_mult={job['lr_mult']} | {job['grouping']}"
        print(f"[{i}/{len(JOBS)}] {tag}")
        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"[WARNING] Failed (exit {result.returncode}). Continuing...")
        else:
            print(f"[SUCCESS] Completed.")
        print("-" * 50)

    print(f"\nState trace finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    run_trace()