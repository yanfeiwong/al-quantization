import subprocess
import sys
from datetime import datetime

FAILED_LOG = "benchmark_failed.log"

COMMON_ARGS = {
    "task": "pretrain",
    "model": "TinyLlama-1.1B",
    "dataset": "wikitext-103-raw-v1",
    "max_steps": 20000,
    "seq_len": 512,
    "batch_size": 4,
    "seed": 921,
    "do_eval": True,
    "eval_steps": 1000,
}

JOBS = [
    # # ==========================================
    # # Baselines
    # # ==========================================
    # # {"optimizer": "adamw_torch",      "lr_mult": 0.1, "grouping": "G0"},
    # # {"optimizer": "adamw_8bit_bnb",   "lr_mult": 0.1, "grouping": "G0"},
    # # {"optimizer": "adamw_8bit_bnb",   "lr_mult": 0.1, "grouping": "G1"},
    # # {"optimizer": "came_torch",       "lr_mult": 0.1, "grouping": "G0"},
    # # {"optimizer": "adafactor_hf",     "lr_mult": 1.0, "grouping": "G0"},
    # # {"optimizer": "apollo_torch",     "lr_mult": 0.1, "grouping": "G0"},

    # # ==========================================
    # # Adafactor Path
    # # ==========================================

    # # Full Quantization
    # # {"optimizer": "adafactor_ours_al8_vblk2048",       "lr_mult": 1.0, "grouping": ["G0", "G1"]},
    # # {"optimizer": "adafactor_ours_al8_vblk2048_fb",       "lr_mult": 1.0, "grouping": ["G0", "G1"]},
    # {"optimizer": "adafactor_ours_al8_vblk256",        "lr_mult": 1.0, "grouping": ["G0", "G1"]},
    # # {"optimizer": "adafactor_ours_al8_vblk256_fb",     "lr_mult": 1.0, "grouping": ["G0", "G1"]},


    # # ==========================================
    # # AdamW Path
    # # ==========================================
    # # V-Only Ablation (M=fp32, V=al8)
    # {"optimizer": "adamw_ours_fp32_al8_vblk2048",     "lr_mult": 0.1, "grouping": "G0"},
    # # {"optimizer": "adamw_ours_fp32_al8_vblk256",      "lr_mult": 0.1, "grouping": "G0"},

    # # Full Quantization (8-bit M)
    # {"optimizer": "adamw_ours_uf8_al8_vblk2048",       "lr_mult": 0.1, "grouping": "G0"},
    # {"optimizer": "adamw_ours_uf8_al8_vblk256",        "lr_mult": 0.1, "grouping": "G0"},
    # {"optimizer": "adamw_ours_d8_al8_vblk2048",        "lr_mult": 0.1, "grouping": "G0"},
    # # {"optimizer": "adamw_ours_d8_al8_vblk256",         "lr_mult": 0.1, "grouping": "G0"},
    
    # # 4-bit M (Lower priority)
    # {"optimizer": "adamw_ours_uf4_al8_vblk2048",     "lr_mult": 0.1, "grouping": "G0"},
    # # {"optimizer": "adamw_ours_uf4_al8_vblk256",      "lr_mult": 0.1, "grouping": "G0"},

    # # ==========================================
    # # CAME Path
    # # ==========================================
    # # V-Only Ablation (M=fp32, V=al8)
    # {"optimizer": "came_ours_fp32_al8_vblk2048",      "lr_mult": 0.1, "grouping": "G0"},
    # # {"optimizer": "came_ours_fp32_al8_vblk256",       "lr_mult": 0.1, "grouping": ["G0", "G1"]},

    # # Full Quantization (8-bit M)
    # {"optimizer": "came_ours_uf8_al8_vblk2048",        "lr_mult": 0.1, "grouping": ["G0", "G1"]}, # "G0"
    # {"optimizer": "came_ours_uf8_al8_vblk256",          "lr_mult": 0.1, "grouping": "G1"},
    # {"optimizer": "came_ours_d8_al8_vblk2048",         "lr_mult": 0.1, "grouping": "G0"}, # ["G0", "G1"]
    
    # # Bias Correction variants (Lower priority)
    # # {"optimizer": "came_ours_uf8_al8_vblk2048_bc",   "lr_mult": 0.1, "grouping": "G0"},
    # # {"optimizer": "came_ours_d8_al8_vblk2048_bc",    "lr_mult": 0.1, "grouping": "G0"},
    
    # # 4-bit M
    # # {"optimizer": "came_ours_uf4_al8_vblk2048",      "lr_mult": 0.1, "grouping": "G0"},
    # # {"optimizer": "came_ours_uf4_al8_vblk256",       "lr_mult": 0.1, "grouping": "G0"},
    # # {"optimizer": "came_ours_uf4_al8_vblk2048_bc",   "lr_mult": 0.1, "grouping": "G0"},
    # # {"optimizer": "came_ours_uf4_al8_vblk256_bc",    "lr_mult": 0.1, "grouping": "G0"},

    # # 16-bit M
    # {"optimizer": "came_ours_uf8_al16_vblk2048",          "lr_mult": 0.1, "grouping": "G0"},

    # # FP32 C
    # {"optimizer": "came_ours_uf8_al8_vblk2048_c_fp32",       "lr_mult": 0.1, "grouping": "G0"},

    # # ==========================================
    # # APOLLO Path
    # # ==========================================
    # # V-Only Ablation (M=fp32, V=al8)
    # {"optimizer": "apollo_ours_fp32_al8_vblk2048",    "lr_mult": 0.1, "grouping": "G0"},
    # # Full Quantization
    # {"optimizer": "apollo_ours_uf8_al8_vblk2048",      "lr_mult": 0.1, "grouping": "G0"},
    # # {"optimizer": "apollo_ours_uf8_al8_vblk256",     "lr_mult": 0.1, "grouping": "G0"},
    # # {"optimizer": "apollo_ours_d8_al8_vblk2048",     "lr_mult": 0.1, "grouping": "G0"},
    # # {"optimizer": "apollo_ours_d8_al8_vblk256",      "lr_mult": 0.1, "grouping": "G0"},
    # {"optimizer": "apollo_ours_uf4_al8_vblk2048",      "lr_mult": 0.1, "grouping": "G0"},
    # # {"optimizer": "apollo_ours_uf4_al8_vblk256",     "lr_mult": 0.1, "grouping": "G0"},
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