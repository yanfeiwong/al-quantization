import subprocess
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
FAILED_LOG = SCRIPT_DIR / "benchmark_seed_sup_failed.log"

SEEDS = [0, 1997]

COMMON_ARGS = {
    "task": "pretrain",
    "model": "TinyLlama-1.1B",
    "dataset": "wikitext-103-raw-v1",
    "max_steps": 20000,
    "seq_len": 512,
    "batch_size": 4,
    "do_eval": True,
    "eval_steps": 1000,
}

JOBS = [
    # AdamW
    {"optimizer": "adamw_torch", "lr_mult": 0.1, "grouping": "G0"},
    {"optimizer": "adamw_8bit_bnb", "lr_mult": 0.1, "grouping": "G0"},
    {"optimizer": "adamw_ours_uf8_al8_vblk2048", "lr_mult": 0.1, "grouping": "G0"},

    # CAME
    {"optimizer": "came_torch", "lr_mult": 0.1, "grouping": "G0"},
    {"optimizer": "came_ours_uf8_al8_vblk2048", "lr_mult": 0.1, "grouping": "G0"},
    {"optimizer": "came_ours_uf8_al16_vblk2048", "lr_mult": 0.1, "grouping": "G0"},
]


def run_benchmark():
    expanded = []
    for seed in SEEDS:
        for job in JOBS:
            expanded.append({**job, "seed": seed, "run_suffix": f"seed{seed}"})

    print(f"Benchmark started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seeds: {SEEDS}")
    print(f"Total jobs: {len(expanded)}\n")

    for i, job in enumerate(expanded, 1):
        cmd = [sys.executable, str(SCRIPT_DIR / "train_llm.py")]
        for key, value in COMMON_ARGS.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.append(f"--{key}={value}")
        for key, value in job.items():
            cmd.append(f"--{key}={value}")

        tag = (
            f"seed={job['seed']} | {job['grouping']} | "
            f"{job['optimizer']} | lr={job['lr_mult']}"
        )
        print(f"[{i}/{len(expanded)}] {tag}")
        result = subprocess.run(cmd, cwd=SCRIPT_DIR)

        if result.returncode != 0:
            print(f"[WARNING] Failed (exit {result.returncode}). Continuing...")
            with FAILED_LOG.open("a", encoding="utf-8") as f:
                f.write(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                    f"[{i}/{len(expanded)}] | seed={job['seed']} | "
                    f"grouping={job['grouping']} | optimizer={job['optimizer']} | "
                    f"lr_mult={job['lr_mult']} | exit={result.returncode}\n"
                )
        else:
            print("[SUCCESS] Completed.")
        print("-" * 50)

    print(f"\nBenchmark finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    run_benchmark()
