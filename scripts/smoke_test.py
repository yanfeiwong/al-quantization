"""
Adafactor8Bit Smoke Test
========================
Part 1: Official algorithm alignment (fp32, quantize=False)
Part 2: Backend consistency (CUDA kernel vs Python fallback, quantized)
Part 3: Full quantization matrix + Memory sanity (merged)
Part 4: Edge cases & robustness
Part 5: Loss tracking (quantized vs fp32)
Part 6: Checkpoint integrity
Part 7: Determinism (whitelist-based)
Part 8: Hot-swap & state migration (placed last to isolate warning output)

Run:  python smoke_test.py [--part 1|2|3|4|5|6|7|8|all]
"""

import torch
import torch.nn as nn
import json
import math
import platform
import sys,os
import traceback
import argparse
import logging
import contextlib
from datetime import datetime
from pathlib import Path

from transformers.optimization import Adafactor
from adafactor8bit import Adafactor8Bit

try:
    from came_pytorch import CAME
    HAS_CAME = True
except ImportError:
    HAS_CAME = False

try:
    from apollo_torch import APOLLOAdamW
    HAS_APOLLO = True
except ImportError:
    HAS_APOLLO = False


# ════════════════════════════════════════════════════════════
# Section 0: Configuration (review-friendly, all knobs here)
# ════════════════════════════════════════════════════════════

# 0a: Device & Seed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# 0b: Model dimensions
ALIGN_DIM = 256          # Part 1/2: alignment & backend consistency
MATRIX_DIM = 2048         # Part 3: quantization matrix + memory ordering

# 0c: Steps per part
STEPS_ALIGN = 30         # Part 1: official alignment
STEPS_BACKEND = 30       # Part 2: backend consistency
STEPS_SANITY = 5         # Part 3: matrix sanity check
STEPS_TRACK = 30         # Part 5: loss tracking
STEPS_MEM_STABLE = 10    # Part 3: memory stability (extra steps)

# 0d: Tolerances
ALIGN_TOL = 1e-5         # Part 1: fp32 alignment
BACKEND_FAIL = 5e-3      # Part 2: backend consistency fail threshold
BACKEND_WARN = 1e-5      # Part 2: backend consistency warn threshold
LOSS_TOL_8BIT = 0.10     # Part 5: 8-bit loss tracking (relative)
LOSS_TOL_4BIT = 0.25     # Part 5: 4-bit loss tracking (relative)
CKPT_TOL = 1e-3          # Part 6: checkpoint roundtrip loss tolerance

# 0e: Determinism whitelist
# CUDA 2D factored paths use atomicAdd in compute_factored_sums_kernel,
# introducing floating-point non-determinism across runs.
# Only 1D full-rank CUDA paths are treated as strict-repeatability paths in
# this suite. Python fallback paths are tested under the same strict policy.
DETERMINISM_STRICT = {
    "cuda": ["adamw", "rmsprop"],
    "py":   ["adamw", "adafactor", "rmsprop"],
}

# 0f: Named algorithm configs (referenced by Part 1/2/3/5)
ADAFACTOR_KW = dict(
    lr=1e-3, relative_step=False, scale_parameter=True,
    weight_decay=0.0,
)
ADAMW_KW = dict(
    lr=1e-3, beta1=0.9, beta2=0.999, factored=False,
    relative_step=False, scale_parameter=False, d=0,
    eps=(1e-8, 1e-3), weight_decay=0.0,
)
CAME_KW = dict(
    lr=1e-3, beta1=0.9, beta2=0.999, beta3=0.9999,
    eps_came=1e-16, d=1.0, relative_step=False,
    scale_parameter=False, weight_decay=0.0,
)
RMSPROP_KW = dict(
    lr=1e-3, beta1=None, beta2=0.99, factored=False,
    relative_step=False, scale_parameter=False, d=0,
    eps=(1e-8, 1e-3), weight_decay=0.0,
)
APOLLO_BASE_KW = dict(
    lr=1e-3, beta1=0.9, beta2=0.999,
    relative_step=False, scale_parameter=False, d=0,
    apollo_eps=1e-6, eps=(1e-6, 1e-3),
    weight_decay=0.0,
)

# 0g: Quantization type configs (referenced by Part 3)
V_TYPES = {
    "al8":   {},
    "al16":  {"v_quant_type": "al16"},
    "vfp32": {"v_quant_type": "fp32"},
}
C_TYPES = {
    "cal8":   {"c_quant_type": "al8"},
    "cal16":  {"c_quant_type": "al16"},
    "cfp32":  {"c_quant_type": "fp32"},
}

M_TYPES = {
    "uf4":   {"m_quant_type": "uf4", "m_block_size": 128},
    "uf8":   {"m_quant_type": "uf8", "m_block_size": 256},
    "d4":    {"m_quant_type": "d4",  "m_block_size": 128},
    "d8":    {"m_quant_type": "d8",  "m_block_size": 256},
    "mfp32": {"m_quant_type": "fp32"},
}

# 0h: Matrix path definitions (which M types each path supports)
MATRIX_PATHS = {
    "adafactor": {
        "base": ADAFACTOR_KW,
        "m_types": {"noM": {}},
    },
    "adamw": {
        "base": ADAMW_KW,
        "m_types": M_TYPES,
    },
    "came": {
        "base": CAME_KW,
        "m_types": {k: v for k, v in M_TYPES.items()
                    if k in ("uf8", "d8", "mfp32")},
    },
}


# ════════════════════════════════════════════════════════════
# Section 1: Helpers
# ════════════════════════════════════════════════════════════

PASS = 0
FAIL = 0
WARN = 0
SKIP = 0
FAILURES = []
WARNINGS = []
RESULTS = []


def record(name, ok, msg="", policy="Functional / Regression"):
    global PASS, FAIL
    status = "PASS" if ok else "FAIL"
    RESULTS.append({
        "name": name, "status": status, "policy": policy, "msg": msg,
    })
    if ok:
        PASS += 1
        print(f"  ✅  {name}" + (f"  [{msg}]" if msg else ""))
    else:
        FAIL += 1
        FAILURES.append((name, msg))
        print(f"  ❌  {name}" + (f"  [{msg}]" if msg else ""))


def warn(name, msg="", policy="Functional / Regression"):
    global WARN
    RESULTS.append({
        "name": name, "status": "WARN", "policy": policy, "msg": msg,
    })
    WARN += 1
    WARNINGS.append((name, msg))
    print(f"  ⚠️  {name}" + (f"  [{msg}]" if msg else ""))


def skip(name, msg="", policy="Functional / Regression"):
    global SKIP
    RESULTS.append({
        "name": name, "status": "SKIP", "policy": policy, "msg": msg,
    })
    SKIP += 1
    print(f"  ⏭️  {name}" + (f"  [{msg}]" if msg else ""))


def check_diff(name, diff, worst="", *, pass_tol, fail_tol,
               metric="diff", note="", policy="Numerical Consistency"):
    """Classify a numerical difference using explicit test-specific limits.

    ``diff < pass_tol`` is PASS, ``pass_tol <= diff < fail_tol`` is WARN,
    and ``diff >= fail_tol`` is FAIL. Setting both limits equal creates a
    strict PASS/FAIL check with no warning band.
    """
    if pass_tol > fail_tol:
        raise ValueError(
            f"pass_tol ({pass_tol}) must be <= fail_tol ({fail_tol})"
        )

    detail_parts = [f"{metric}={diff:.2e}"]
    if worst:
        detail_parts.append(f"worst={worst}")
    if pass_tol == fail_tol:
        detail_parts.append(f"tol={pass_tol:.0e}")
    else:
        detail_parts.append(f"pass<{pass_tol:.0e}")
        detail_parts.append(f"fail>={fail_tol:.0e}")
    if note:
        detail_parts.append(note)
    detail = " ".join(detail_parts)

    if not math.isfinite(diff):
        record(name, False, detail, policy=policy)
        return "FAIL"
    if diff < pass_tol:
        record(name, True, detail, policy=policy)
        return "PASS"
    if diff < fail_tol:
        warn(name, detail, policy=policy)
        return "WARN"
    record(name, False, detail, policy=policy)
    return "FAIL"


def check_reference_alignment(name, diff, worst=""):
    """Strict Part 1 comparison against an official/reference algorithm."""
    return check_diff(
        name, diff, worst,
        pass_tol=ALIGN_TOL,
        fail_tol=ALIGN_TOL,
        policy="Reference Algorithm Alignment",
    )


def check_cuda_numerical_consistency(name, diff, worst="", note=""):
    """CUDA numerical comparison with an explicit warning band."""
    return check_diff(
        name, diff, worst,
        pass_tol=BACKEND_WARN,
        fail_tol=BACKEND_FAIL,
        metric="Δp",
        note=note,
        policy="CUDA Numerical Consistency",
    )


def fresh_model(d_in=None, d_out=None, bias=False):
    if d_in is None:
        d_in = ALIGN_DIM
    if d_out is None:
        d_out = ALIGN_DIM
    torch.manual_seed(SEED)
    return nn.Linear(d_in, d_out, bias=bias).to(DEVICE)


def fresh_model_mixed():
    """Model with both 2D and 1D params (for CAME 1D / Adafactor mixed tests)."""
    torch.manual_seed(SEED)
    return nn.Sequential(
        nn.Linear(128, 64, bias=True),
        nn.LayerNorm(64),
        nn.Linear(64, 32, bias=False),
    ).to(DEVICE)


def make_input(batch=8, d=None):
    if d is None:
        d = ALIGN_DIM
    torch.manual_seed(SEED + 777)
    return torch.randn(batch, d, device=DEVICE)


def make_input_mixed(batch=8):
    torch.manual_seed(SEED + 777)
    return torch.randn(batch, 128, device=DEVICE)


def param_diff(m1, m2):
    return max((p1.data - p2.data).abs().max().item()
               for p1, p2 in zip(m1.parameters(), m2.parameters()))


def worst_param_name(m1, m2):
    worst_n, worst_d = "", 0.0
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        d = (p1.data - p2.data).abs().max().item()
        if d > worst_d:
            worst_d, worst_n = d, n1
    return worst_n, worst_d


def param_norm(m):
    return sum(p.data.norm().item() ** 2 for p in m.parameters()) ** 0.5


def train(model, opt, x, steps):
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = model(x).sum()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def build_opt(params, backend, **kw):
    return Adafactor8Bit(params, use_cuda_kernel=(backend == "cuda"), **kw)


def get_state_bytes(opt):
    total = 0
    for st in opt.state.values():
        for v in st.values():
            if isinstance(v, torch.Tensor):
                total += v.numel() * v.element_size()
    return total


def make_apollo_groups(model, rank=4):
    """Simplified APOLLO grouping: all 2D → APOLLO, rest → regular."""
    apollo_params, regular_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 2:
            apollo_params.append(param)
        else:
            regular_params.append(param)
    groups = []
    if regular_params:
        groups.append({"params": regular_params})
    if apollo_params:
        groups.append({"params": apollo_params, "apollo_rank": rank})
    return groups


def make_apollo_groups_official(model, rank=4):
    """Official APOLLO grouping (uses 'rank' key, not 'apollo_rank')."""
    apollo_params, regular_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 2:
            apollo_params.append(param)
        else:
            regular_params.append(param)
    groups = []
    if regular_params:
        groups.append({"params": regular_params})
    if apollo_params:
        groups.append({"params": apollo_params, "rank": rank,
                       "update_proj_gap": 200, "scale_type": "channel",
                       "proj": "random", "scale": 1.0, "proj_type": "std"})
    return groups


def backend_available():
    if DEVICE.type != "cuda":
        return False
    try:
        m = nn.Linear(4, 4, bias=False).to(DEVICE)
        opt = Adafactor8Bit(m.parameters(), lr=1e-3, use_cuda_kernel=True,
                            quantize=True, min_8bit_size=0)
        x = torch.randn(2, 4, device=DEVICE)
        opt.zero_grad(); m(x).sum().backward(); opt.step()
        return True
    except Exception:
        return False


@contextlib.contextmanager
def suppress_adafactor_warnings():
    """Best-effort suppression of expected hot-swap warnings.
    If ineffective, warnings appear in Part 8 output (placed last)."""
    logger = logging.getLogger('adafactor8bit')
    old_level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(old_level)


HAS_CUDA_KERNEL = backend_available()
BACKENDS = ["py"] + (["cuda"] if HAS_CUDA_KERNEL else [])


# ════════════════════════════════════════════════════════════
# Part 1: Official Algorithm Alignment (fp32, quantize=False)
# ════════════════════════════════════════════════════════════

def part1():
    print(f"\n{'─' * 60}")
    print("Part 1: Official Algorithm Alignment (fp32, quantize=False)")
    print(f"{'─' * 60}")

    for backend in BACKENDS:
        print(f"\n  [{backend}]")

        # 1.1 Adafactor vs HF-Adafactor (2D only)
        m1, m2 = fresh_model(), fresh_model()
        x = make_input()
        o1 = Adafactor(m1.parameters(), lr=1e-3, relative_step=False,
                       scale_parameter=True, warmup_init=False, weight_decay=0.0)
        o2 = build_opt(m2.parameters(), backend, quantize=False, **ADAFACTOR_KW)
        l1 = train(m1, o1, x, STEPS_ALIGN)
        l2 = train(m2, o2, x, STEPS_ALIGN)
        d = param_diff(m1, m2)
        wn, wd = worst_param_name(m1, m2)
        check_reference_alignment(f"[{backend}] Adafactor vs HF", d, wn)

        # 1.2 Adafactor mixed (2D + 1D)
        m1, m2 = fresh_model_mixed(), fresh_model_mixed()
        x = make_input_mixed()
        o1 = Adafactor(m1.parameters(), lr=1e-3, relative_step=False,
                       scale_parameter=True, warmup_init=False, weight_decay=0.0)
        o2 = build_opt(m2.parameters(), backend, quantize=False, **ADAFACTOR_KW)
        train(m1, o1, x, STEPS_ALIGN)
        train(m2, o2, x, STEPS_ALIGN)
        d = param_diff(m1, m2)
        wn, wd = worst_param_name(m1, m2)
        check_reference_alignment(f"[{backend}] Adafactor mixed", d, wn)

        # 1.3 AdamW vs torch-AdamW
        m1, m2 = fresh_model(), fresh_model()
        x = make_input()
        o1 = torch.optim.AdamW(m1.parameters(), lr=1e-3, betas=(0.9, 0.999),
                               eps=1e-8, weight_decay=0.0)
        o2 = build_opt(m2.parameters(), backend, quantize=False,
                       bias_correction=True, **ADAMW_KW)
        train(m1, o1, x, STEPS_ALIGN)
        train(m2, o2, x, STEPS_ALIGN)
        d = param_diff(m1, m2)
        check_reference_alignment(f"[{backend}] AdamW vs torch", d)

        # 1.4 RMSprop vs torch-RMSprop
        m1, m2 = fresh_model(), fresh_model()
        x = make_input()
        o1 = torch.optim.RMSprop(m1.parameters(), lr=1e-3, alpha=0.99,
                                 eps=1e-8, weight_decay=0.0, momentum=0)
        o2 = build_opt(m2.parameters(), backend, quantize=False, **RMSPROP_KW)
        train(m1, o1, x, STEPS_ALIGN)
        train(m2, o2, x, STEPS_ALIGN)
        d = param_diff(m1, m2)
        check_reference_alignment(f"[{backend}] RMSprop vs torch", d)

        # 1.5 CAME vs CAME-official (2D)
        if HAS_CAME:
            m1, m2 = fresh_model(), fresh_model()
            x = make_input()
            o1 = CAME(m1.parameters(), lr=1e-3, eps=(1e-30, 1e-16),
                      clip_threshold=1.0, betas=(0.9, 0.999, 0.9999),
                      weight_decay=0.0)
            o2 = build_opt(m2.parameters(), backend, quantize=False, **CAME_KW)
            train(m1, o1, x, STEPS_ALIGN)
            train(m2, o2, x, STEPS_ALIGN)
            d = param_diff(m1, m2)
            wn, wd = worst_param_name(m1, m2)
            check_reference_alignment(f"[{backend}] CAME vs official (2D)", d, wn)

            # 1.6 CAME mixed model (2D + 1D)
            # Python checks strict reference semantics. The factored CUDA path
            # uses atomicAdd reductions, so its multi-step comparison is instead
            # classified under the explicit CUDA numerical-consistency policy.
            m1, m2 = fresh_model_mixed(), fresh_model_mixed()
            x = make_input_mixed()
            o1 = CAME(m1.parameters(), lr=1e-3, eps=(1e-30, 1e-16),
                      clip_threshold=1.0, betas=(0.9, 0.999, 0.9999),
                      weight_decay=0.0)
            o2 = build_opt(m2.parameters(), backend, quantize=False, **CAME_KW)
            train(m1, o1, x, STEPS_ALIGN)
            train(m2, o2, x, STEPS_ALIGN)
            d = param_diff(m1, m2)
            wn, wd = worst_param_name(m1, m2)
            test_name = f"[{backend}] CAME mixed model (2D + 1D)"
            if backend == "py":
                check_reference_alignment(test_name, d, wn)
            else:
                check_cuda_numerical_consistency(
                    test_name, d, wn,
                    note=(
                        "expected accumulation-order variation from factored "
                        "CUDA atomicAdd reductions"
                    ),
                )
            if not math.isfinite(d) or d >= ALIGN_TOL:
                _km = {'exp_avg_sq': 'variance', 'exp_avg': 'm',
                       'exp_avg_sq_row': 'row_var', 'exp_avg_sq_col': 'col_var',
                       'exp_avg_res_row': 'conf_row', 'exp_avg_res_col': 'conf_col'}
                _m1, _m2 = fresh_model_mixed(), fresh_model_mixed()
                _x = make_input_mixed()
                _o1 = CAME(_m1.parameters(), lr=1e-3, eps=(1e-30, 1e-16),
                           clip_threshold=1.0, betas=(0.9, 0.999, 0.9999),
                           weight_decay=0.0)
                _o2 = build_opt(_m2.parameters(), backend, quantize=False, **CAME_KW)
                _o1.zero_grad(); _m1(_x).sum().backward(); _o1.step()
                _o2.zero_grad(); _m2(_x).sum().backward(); _o2.step()
                print(f"    [1-step probe]")
                for (_n1, _p1), (_n2, _p2) in zip(_m1.named_parameters(), _m2.named_parameters()):
                    _dd = (_p1.data - _p2.data).abs().max().item()
                    print(f"      param {_n1}: {_dd:.2e}")
                    _s1, _s2 = _o1.state[_p1], _o2.state[_p2]
                    for _k1, _v1 in _s1.items():
                        if not isinstance(_v1, torch.Tensor):
                            continue
                        _k2 = _km.get(_k1, _k1)
                        _v2 = _s2.get(_k2)
                        if isinstance(_v2, torch.Tensor) and _v2.numel() == _v1.numel():
                            _ds = (_v1.flatten() - _v2.flatten()).abs().max().item()
                            print(f"      state {_n1}.{_k1}->{_k2}: {_ds:.2e}")
        else:
            skip(f"[{backend}] CAME", "came_pytorch not installed")

        # 1.7 APOLLO vs official APOLLO
        if HAS_APOLLO:
            m1, m2 = fresh_model(), fresh_model()
            x = make_input()
            g_ref = make_apollo_groups_official(m1, rank=4)
            g_ours = make_apollo_groups(m2, rank=4)
            o1 = APOLLOAdamW(g_ref, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                             weight_decay=0.0)
            o2 = build_opt(g_ours, backend, quantize=False,
                           bias_correction=True, **APOLLO_BASE_KW)
            train(m1, o1, x, STEPS_ALIGN)
            train(m2, o2, x, STEPS_ALIGN)
            d = param_diff(m1, m2)
            wn, wd = worst_param_name(m1, m2)
            check_reference_alignment(f"[{backend}] APOLLO vs official", d, wn)
            if not math.isfinite(d) or d >= ALIGN_TOL:
                print(f"    Note: projection seed may differ (ours=0, official=1).")
                print(f"    Worst param: {wn} (diff={wd:.2e})")
        else:
            skip(f"[{backend}] APOLLO", "apollo_torch not installed")


# ════════════════════════════════════════════════════════════
# Part 2: Backend Consistency (CUDA vs Python, quantized)
# ════════════════════════════════════════════════════════════

def part2():
    print(f"\n{'─' * 60}")
    print("Part 2: Backend Consistency (cuda vs py, quantized)")
    print(f"  Fail: {BACKEND_FAIL:.0e}  Warn: {BACKEND_WARN:.0e}")
    print(f"{'─' * 60}")

    if not HAS_CUDA_KERNEL:
        skip("Backend consistency", "CUDA kernel not available")
        return

    configs = [
        ("adafactor al8",
         dict(**ADAFACTOR_KW, quantize=True, min_8bit_size=0)),
        ("adamw al8+uf8",
         dict(**ADAMW_KW, quantize=True, m_quant_type="uf8", min_8bit_size=0)),
        ("adamw al8+d8",
         dict(**ADAMW_KW, quantize=True, m_quant_type="d8", min_8bit_size=0)),
        ("adamw al16+uf8",
         dict(**ADAMW_KW, quantize=True, v_quant_type="al16",
              m_quant_type="uf8", min_8bit_size=0)),
        ("adamw vfp32+uf8",
         dict(**ADAMW_KW, quantize=True, v_quant_type="fp32",
              m_quant_type="uf8", min_8bit_size=0)),
        ("came al8+uf8",
         dict(**CAME_KW, quantize=True, m_quant_type="uf8", min_8bit_size=0)),
        ("came al8+d8",
         dict(**CAME_KW, quantize=True, m_quant_type="d8", min_8bit_size=0)),
        ("came al8+uf8+cal16",
         dict(**CAME_KW, quantize=True, m_quant_type="uf8", c_quant_type="al16", min_8bit_size=0)),
        ("came al8+uf8+cfp32",
         dict(**CAME_KW, quantize=True, m_quant_type="uf8", c_quant_type="fp32", min_8bit_size=0)),
    ]

    for name, kw in configs:
        try:
            torch.manual_seed(SEED)
            m_py = fresh_model()
            torch.manual_seed(SEED)
            m_cu = fresh_model()
            x = make_input()
            opt_py = build_opt(m_py.parameters(), "py", **kw)
            opt_cu = build_opt(m_cu.parameters(), "cuda", **kw)
            l_py = train(m_py, opt_py, x, STEPS_BACKEND)
            l_cu = train(m_cu, opt_cu, x, STEPS_BACKEND)
            d = param_diff(m_py, m_cu)
            wn, wd = worst_param_name(m_py, m_cu)
            check_cuda_numerical_consistency(f"cuda≈py {name}", d, wn)
        except Exception as e:
            record(f"cuda≈py {name}", False, str(e)[:120])

    # APOLLO backend consistency
    for m_name, m_kw in [("uf8", dict(m_quant_type="uf8", m_block_size=256)),
                          ("d8", dict(m_quant_type="d8", m_block_size=256))]:
        try:
            torch.manual_seed(SEED)
            m_py = fresh_model()
            torch.manual_seed(SEED)
            m_cu = fresh_model()
            x = make_input()
            opt_py = build_opt(make_apollo_groups(m_py), "py",
                               quantize=True, min_8bit_size=0,
                               **APOLLO_BASE_KW, **m_kw)
            opt_cu = build_opt(make_apollo_groups(m_cu), "cuda",
                               quantize=True, min_8bit_size=0,
                               **APOLLO_BASE_KW, **m_kw)
            l_py = train(m_py, opt_py, x, STEPS_SANITY)
            l_cu = train(m_cu, opt_cu, x, STEPS_SANITY)
            d = param_diff(m_py, m_cu)
            check_cuda_numerical_consistency(
                f"cuda≈py apollo M={m_name}", d,
                note="projection seed-dependent",
            )
        except Exception as e:
            record(f"cuda≈py apollo M={m_name}", False, str(e)[:120])


# ════════════════════════════════════════════════════════════
# Part 3: Full Quantization Matrix + Memory Sanity (merged)
# ════════════════════════════════════════════════════════════

def part3():
    print(f"\n{'─' * 60}")
    print(f"Part 3: Quantization Matrix + Memory (dim={MATRIX_DIM})")
    print(f"{'─' * 60}")

    memory_records = {}

    for backend in BACKENDS:
        print(f"\n  [{backend}]")

        # 3a: Full matrix
        for path_name, pcfg in MATRIX_PATHS.items():
            if path_name == "came" and not HAS_CAME:
                skip(f"[{backend}] CAME matrix", "came_pytorch not installed")
                continue
            for v_name, v_kw in V_TYPES.items():
                for m_name, m_kw in pcfg["m_types"].items():
                    label = f"[{backend}] {path_name} V={v_name} M={m_name}"
                    try:
                        m = fresh_model(MATRIX_DIM, MATRIX_DIM)
                        x = make_input(8, MATRIX_DIM)
                        kw = {**pcfg["base"], **v_kw, **m_kw,
                              "quantize": True, "min_8bit_size": 0}
                        opt = build_opt(m.parameters(), backend, **kw)
                        snap = {n: p.data.clone() for n, p in m.named_parameters()}
                        losses = train(m, opt, x, STEPS_SANITY)
                        finite = all(math.isfinite(l) for l in losses)
                        changed = any(not torch.equal(snap[n], p.data)
                                      for n, p in m.named_parameters())
                        ok = finite and changed
                        mem = get_state_bytes(opt)
                        memory_records[(path_name, v_name, m_name)] = mem
                        record(label, ok,
                               f"loss {losses[0]:.2f}→{losses[-1]:.2f} mem={mem//1024}KB"
                               if ok else "NaN or frozen")
                    except Exception as e:
                        record(label, False, str(e)[:120])

        # 3b: APOLLO V variants (full: al8/al16/fp32 × uf8/d8)
        for v_name, v_kw in V_TYPES.items():
            for m_name, m_kw in [("uf8", M_TYPES["uf8"]), ("d8", M_TYPES["d8"])]:
                label = f"[{backend}] apollo V={v_name} M={m_name}"
                try:
                    m = fresh_model(MATRIX_DIM, MATRIX_DIM)
                    x = make_input(8, MATRIX_DIM)
                    groups = make_apollo_groups(m, rank=4)
                    kw = {**APOLLO_BASE_KW, **v_kw, **m_kw,
                          "quantize": True, "min_8bit_size": 0}
                    opt = build_opt(groups, backend, **kw)
                    losses = train(m, opt, x, STEPS_SANITY)
                    finite = all(math.isfinite(l) for l in losses)
                    mem = get_state_bytes(opt)
                    memory_records[("apollo", v_name, m_name)] = mem
                    record(label, finite,
                           f"loss {losses[0]:.2f}→{losses[-1]:.2f} mem={mem//1024}KB")
                except Exception as e:
                    record(label, False, str(e)[:120])

        # 3b2: CAME C variants (c_quant_type: al8/al16/fp32)
        for c_name, c_kw in C_TYPES.items():
            label = f"[{backend}] came C={c_name}"
            try:
                m = fresh_model(MATRIX_DIM, MATRIX_DIM)
                x = make_input(8, MATRIX_DIM)
                kw = {**CAME_KW, "quantize": True, "m_quant_type": "uf8", "min_8bit_size": 0, **c_kw}
                opt = build_opt(m.parameters(), backend, **kw)
                losses = train(m, opt, x, STEPS_SANITY)
                finite = all(math.isfinite(l) for l in losses)
                mem = get_state_bytes(opt)
                memory_records[("came_c", c_name)] = mem
                record(label, finite,
                       f"loss {losses[0]:.2f}→{losses[-1]:.2f} mem={mem//1024}KB")
            except Exception as e:
                record(label, False, str(e)[:120])

    # 3c: Memory ordering assertions
    print(f"\n  Memory ordering assertions:")

    # V ordering: al8 < al16 < fp32 (adamw path, M=uf8)
    v_keys = [("adamw", "al8", "uf8"), ("adamw", "al16", "uf8"), ("adamw", "vfp32", "uf8")]
    if all(k in memory_records for k in v_keys):
        b_al8, b_al16, b_fp32 = [memory_records[k] for k in v_keys]
        ok = b_al8 < b_al16 < b_fp32
        record("V ordering: al8 < al16 < fp32", ok,
               f"{b_al8//1024}KB < {b_al16//1024}KB < {b_fp32//1024}KB")
    else:
        skip("V ordering", "missing configs in matrix")

    # M ordering: uf4 < uf8 < mfp32 (adamw path, V=al8)
    m_keys = [("adamw", "al8", "uf4"), ("adamw", "al8", "uf8"), ("adamw", "al8", "mfp32")]
    if all(k in memory_records for k in m_keys):
        b_uf4, b_uf8, b_mfp = [memory_records[k] for k in m_keys]
        ok = b_uf4 < b_uf8 < b_mfp
        record("M ordering: uf4 < uf8 < fp32", ok,
               f"{b_uf4//1024}KB < {b_uf8//1024}KB < {b_mfp//1024}KB")
    else:
        skip("M ordering", "missing configs in matrix")

    # C ordering: cal8 < cal16 < cfp32 (came path)
    c_keys = [("came_c", "cal8"), ("came_c", "cal16"), ("came_c", "cfp32")]
    if all(k in memory_records for k in c_keys):
        b_cal8, b_cal16, b_cfp32 = [memory_records[k] for k in c_keys]
        ok = b_cal8 < b_cal16 < b_cfp32
        record("C ordering: cal8 < cal16 < cfp32", ok,
               f"{b_cal8//1024}KB < {b_cal16//1024}KB < {b_cfp32//1024}KB")
    else:
        skip("C ordering", "missing configs in matrix")

    # Overall: quantized < 60% of fp32 (adamw path, V=al8, M=uf8 vs all fp32)
    q_key = ("adamw", "al8", "uf8")
    fp_key = ("adamw", "vfp32", "mfp32")
    if q_key in memory_records and fp_key in memory_records:
        b_q, b_fp = memory_records[q_key], memory_records[fp_key]
        ok = b_q < b_fp * 0.6
        record("Overall: quantized < 60% fp32", ok,
               f"{b_q//1024}KB vs {b_fp//1024}KB ({b_q/b_fp*100:.1f}%)")
    else:
        skip("Overall memory", "missing configs")

    # 3d: APOLLO rank ordering
    try:
        torch.manual_seed(SEED)
        m1 = fresh_model(MATRIX_DIM, MATRIX_DIM)
        x = make_input(8, MATRIX_DIM)
        opt1 = build_opt(make_apollo_groups(m1, rank=4), "py",
                         quantize=True, min_8bit_size=0, **APOLLO_BASE_KW)
        train(m1, opt1, x, STEPS_SANITY)
        mem_r4 = get_state_bytes(opt1)

        torch.manual_seed(SEED)
        m2 = fresh_model(MATRIX_DIM, MATRIX_DIM)
        opt2 = build_opt(make_apollo_groups(m2, rank=64), "py",
                         quantize=True, min_8bit_size=0, **APOLLO_BASE_KW)
        train(m2, opt2, x, STEPS_SANITY)
        mem_r64 = get_state_bytes(opt2)

        ok = mem_r4 < mem_r64
        record("APOLLO rank: r4 < r64", ok,
               f"{mem_r4//1024}KB < {mem_r64//1024}KB")
    except Exception as e:
        record("APOLLO rank ordering", False, str(e)[:120])

    # 3e: State stability (no growth over extra steps)
    try:
        torch.manual_seed(SEED)
        m = fresh_model(MATRIX_DIM, MATRIX_DIM)
        x = make_input(8, MATRIX_DIM)
        opt = build_opt(m.parameters(), "py", quantize=True,
                        min_8bit_size=0, **ADAMW_KW)
        train(m, opt, x, STEPS_SANITY)
        mem_warm = get_state_bytes(opt)
        train(m, opt, x, STEPS_MEM_STABLE)
        mem_after = get_state_bytes(opt)
        ok = mem_after <= mem_warm * 1.02
        record("State stability (10 extra steps)", ok,
               f"{mem_warm//1024}KB → {mem_after//1024}KB")
    except Exception as e:
        record("State stability", False, str(e)[:120])


# ════════════════════════════════════════════════════════════
# Part 4: Edge Cases & Robustness
# ════════════════════════════════════════════════════════════

def part4():
    print(f"\n{'─' * 60}")
    print("Part 4: Edge Cases & Robustness")
    print(f"{'─' * 60}")

    for backend in BACKENDS:
        print(f"\n  [{backend}]")

        # block_size > numel
        try:
            m = nn.Linear(16, 16, bias=False).to(DEVICE)
            x = torch.randn(4, 16, device=DEVICE)
            opt = build_opt(m.parameters(), backend, lr=1e-3, beta1=0.9,
                            beta2=0.999, factored=False, relative_step=False,
                            scale_parameter=False, d=0, quantize=True,
                            block_size=2048, min_8bit_size=0, weight_decay=0.0)
            losses = train(m, opt, x, 5)
            record(f"[{backend}] block_size > numel",
                   all(math.isfinite(l) for l in losses))
        except Exception as e:
            record(f"[{backend}] block_size > numel", False, str(e))

        # frozen params
        try:
            model = nn.Sequential(
                nn.Linear(64, 64, bias=False),
                nn.Linear(64, 32, bias=False),
            ).to(DEVICE)
            for p in model[0].parameters():
                p.requires_grad_(False)
            snap0 = model[0].weight.data.clone()
            snap1 = model[1].weight.data.clone()
            x = torch.randn(4, 64, device=DEVICE)
            opt = build_opt(
                filter(lambda p: p.requires_grad, model.parameters()),
                backend, lr=1e-3, beta1=0.9, beta2=0.999, factored=False,
                relative_step=False, scale_parameter=False, d=0,
                quantize=True, min_8bit_size=0, weight_decay=0.0)
            train(model, opt, x, 5)
            frozen_ok = torch.equal(snap0, model[0].weight.data)
            moved_ok = not torch.equal(snap1, model[1].weight.data)
            record(f"[{backend}] frozen params", frozen_ok and moved_ok)
        except Exception as e:
            record(f"[{backend}] frozen params", False, str(e))

        # grad accumulation
        try:
            m = fresh_model()
            x = make_input()
            opt = build_opt(m.parameters(), backend, quantize=True,
                            min_8bit_size=0, **ADAMW_KW)
            for _ in range(2):
                m(x).sum().backward()
            opt.step()
            opt.zero_grad()
            m(x).sum().backward()
            opt.step()
            record(f"[{backend}] grad accumulation", True)
        except Exception as e:
            record(f"[{backend}] grad accumulation", False, str(e))

        # NaN grad recovery
        try:
            m = fresh_model()
            x = make_input()
            opt = build_opt(m.parameters(), backend, quantize=True,
                            min_8bit_size=0, **ADAMW_KW)
            train(m, opt, x, 1)
            opt.zero_grad(); m(x).sum().backward()
            for p in m.parameters():
                if p.grad is not None:
                    p.grad.data.fill_(float("nan"))
            opt.step()
            train(m, opt, x, 1)
            record(f"[{backend}] NaN grad recovery", True)
        except Exception as e:
            record(f"[{backend}] NaN grad recovery", False, str(e))

        # weight decay modes
        for wd_mode, wd_kw in [("decoupled", dict(weight_decay=0.01,
                                                    scale_weight_decay=False)),
                                ("scaled", dict(weight_decay=0.01,
                                                scale_weight_decay=True))]:
            try:
                m = fresh_model()
                x = make_input()
                # Remove weight_decay from ADAMW_KW to avoid duplicate
                adamw_kw_no_wd = {k: v for k, v in ADAMW_KW.items() if k != "weight_decay"}
                opt = build_opt(m.parameters(), backend, quantize=True,
                                min_8bit_size=0, **adamw_kw_no_wd, **wd_kw)
                losses = train(m, opt, x, 5)
                record(f"[{backend}] wd {wd_mode}",
                       all(math.isfinite(l) for l in losses))
            except Exception as e:
                record(f"[{backend}] wd {wd_mode}", False, str(e))

        # bf16 model
        try:
            torch.manual_seed(SEED)
            m = nn.Linear(128, 64, bias=False).to(DEVICE, dtype=torch.bfloat16)
            x = torch.randn(4, 128, device=DEVICE, dtype=torch.bfloat16)
            opt = build_opt(m.parameters(), backend, quantize=True,
                            min_8bit_size=0, **ADAMW_KW)
            losses = train(m, opt, x, 5)
            record(f"[{backend}] bf16 model",
                   all(math.isfinite(l) for l in losses))
        except Exception as e:
            record(f"[{backend}] bf16 model", False, str(e))

        # embedding routing
        try:
            torch.manual_seed(SEED)
            emb = nn.Embedding(100, 32).to(DEVICE)
            fc = nn.Linear(32, 10, bias=False).to(DEVICE)
            groups = [
                {"params": [emb.weight], "factored": False,
                 "scale_parameter": False, "d": 0},
                {"params": list(fc.parameters())},
            ]
            x_idx = torch.randint(0, 100, (4, 8), device=DEVICE)
            opt = build_opt(groups, backend, lr=1e-3,
                            relative_step=False, quantize=True,
                            min_8bit_size=0, weight_decay=0.0)
            losses = []
            for _ in range(5):
                opt.zero_grad()
                loss = fc(emb(x_idx).mean(dim=1)).sum()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            record(f"[{backend}] embedding routing",
                   all(math.isfinite(l) for l in losses))
        except Exception as e:
            record(f"[{backend}] embedding routing", False, str(e))


# ════════════════════════════════════════════════════════════
# Part 5: Loss Tracking (quant vs fp32)
# ════════════════════════════════════════════════════════════

def part5():
    print(f"\n{'─' * 60}")
    print(f"Part 5: Loss Tracking ({STEPS_TRACK} steps)")
    print(f"  8-bit tol: {LOSS_TOL_8BIT*100:.0f}%  4-bit tol: {LOSS_TOL_4BIT*100:.0f}%")
    print(f"{'─' * 60}")

    configs = [
        ("Adafactor V=al8", LOSS_TOL_8BIT,
         dict(**ADAFACTOR_KW, quantize=True, min_8bit_size=0)),
        ("AdamW V=al8 M=uf8", LOSS_TOL_8BIT,
         dict(**ADAMW_KW, quantize=True, m_quant_type="uf8", min_8bit_size=0)),
        ("AdamW V=al8 M=d8", LOSS_TOL_8BIT,
         dict(**ADAMW_KW, quantize=True, m_quant_type="d8", min_8bit_size=0)),
        ("AdamW V=al8 M=uf4", LOSS_TOL_4BIT,
         dict(**ADAMW_KW, quantize=True, m_quant_type="uf4",
              m_block_size=128, min_8bit_size=0)),
        ("AdamW V=al8 M=d4", LOSS_TOL_4BIT,
         dict(**ADAMW_KW, quantize=True, m_quant_type="d4",
              m_block_size=128, min_8bit_size=0)),
        ("AdamW V=al16 M=uf8", LOSS_TOL_8BIT,
         dict(**ADAMW_KW, quantize=True, v_quant_type="al16",
              m_quant_type="uf8", min_8bit_size=0)),
        ("AdamW V=fp32 M=uf8", LOSS_TOL_8BIT,
         dict(**ADAMW_KW, quantize=True, v_quant_type="fp32",
              m_quant_type="uf8", min_8bit_size=0)),
        ("AdamW V=al8 M=fp32", LOSS_TOL_8BIT,
         dict(**ADAMW_KW, quantize=True, m_quant_type="fp32", min_8bit_size=0)),
        ("CAME V=al8 M=uf8", LOSS_TOL_8BIT,
         dict(**CAME_KW, quantize=True, m_quant_type="uf8", min_8bit_size=0)),
        ("CAME V=al8 M=d8", LOSS_TOL_8BIT,
         dict(**CAME_KW, quantize=True, m_quant_type="d8", min_8bit_size=0)),
        ("CAME V=al8 M=uf8 C=al16", LOSS_TOL_8BIT,
         dict(**CAME_KW, quantize=True, m_quant_type="uf8", c_quant_type="al16", min_8bit_size=0)),
        ("CAME V=al8 M=uf8 C=fp32", LOSS_TOL_8BIT,
         dict(**CAME_KW, quantize=True, m_quant_type="uf8", c_quant_type="fp32", min_8bit_size=0)),
    ]

    for backend in BACKENDS:
        print(f"\n  [{backend}]")
        for name, tol, kw in configs:
            if "CAME" in name and not HAS_CAME:
                skip(f"[{backend}] {name}", "came_pytorch not installed")
                continue
            try:
                torch.manual_seed(SEED)
                m_fp = fresh_model()
                torch.manual_seed(SEED)
                m_q = fresh_model()
                x = make_input()
                fp_kw = {k: v for k, v in kw.items()
                         if k not in ("quantize", "m_quant_type",
                                      "v_quant_type", "m_block_size")}
                opt_fp = build_opt(m_fp.parameters(), backend,
                                   quantize=False, **fp_kw)
                opt_q = build_opt(m_q.parameters(), backend, **kw)
                l_fp = train(m_fp, opt_fp, x, STEPS_TRACK)
                l_q = train(m_q, opt_q, x, STEPS_TRACK)
                ref = abs(l_fp[-1]) if abs(l_fp[-1]) > 1e-8 else 1.0
                rel = abs(l_q[-1] - l_fp[-1]) / ref
                finite = all(math.isfinite(l) for l in l_q)
                ok = rel < tol and finite
                record(f"[{backend}] {name}", ok,
                       f"fp32={l_fp[-1]:.4f} quant={l_q[-1]:.4f} rel={rel:.4f}")
            except Exception as e:
                record(f"[{backend}] {name}", False, str(e)[:120])


# ════════════════════════════════════════════════════════════
# Part 6: Checkpoint Integrity
# ════════════════════════════════════════════════════════════

def part6():
    print(f"\n{'─' * 60}")
    print("Part 6: Checkpoint Integrity")
    print(f"{'─' * 60}")

    for backend in BACKENDS:
        print(f"\n  [{backend}]")

        # 6.1 state_dict roundtrip (model weights synced)
        try:
            m1 = fresh_model()
            x = make_input()
            opt1 = build_opt(m1.parameters(), backend, quantize=True,
                             min_8bit_size=0, **ADAMW_KW)
            train(m1, opt1, x, 5)
            sd = opt1.state_dict()
            m2 = fresh_model()
            m2.load_state_dict(m1.state_dict())
            opt2 = build_opt(m2.parameters(), backend, quantize=True,
                             min_8bit_size=0, **ADAMW_KW)
            opt2.load_state_dict(sd)
            l1 = train(m1, opt1, x, 5)
            l2 = train(m2, opt2, x, 5)
            md = max(abs(a - b) for a, b in zip(l1, l2))
            record(f"[{backend}] roundtrip", md < CKPT_TOL,
                   f"max_loss_diff={md:.2e}")
        except Exception as e:
            record(f"[{backend}] roundtrip", False, str(e))

        # 6.1b disk roundtrip (torch.save / torch.load)
        try:
            import tempfile
            m1 = fresh_model()
            x = make_input()
            opt1 = build_opt(m1.parameters(), backend, quantize=True,
                             v_quant_type="al16", min_8bit_size=0, **ADAMW_KW)
            train(m1, opt1, x, 5)
            sd = opt1.state_dict()
            tmp = os.path.join(tempfile.gettempdir(), f"_a8b_smoke_{backend}.pt")
            torch.save(sd, tmp)
            sd_loaded = torch.load(tmp, weights_only=False)
            os.remove(tmp)
            m2 = fresh_model()
            m2.load_state_dict(m1.state_dict())
            opt2 = build_opt(m2.parameters(), backend, quantize=True,
                             v_quant_type="al16", min_8bit_size=0, **ADAMW_KW)
            opt2.load_state_dict(sd_loaded)
            ok_vqt = all(st.get("v_quant_type") == "al16"
                         for st in opt2.state.values())
            l1 = train(m1, opt1, x, 5)
            l2 = train(m2, opt2, x, 5)
            md = max(abs(a - b) for a, b in zip(l1, l2))
            record(f"[{backend}] disk roundtrip (al16)",
                   md < CKPT_TOL and ok_vqt,
                   f"max_loss_diff={md:.2e} vqt_preserved={ok_vqt}")
        except Exception as e:
            record(f"[{backend}] disk roundtrip (al16)", False, str(e))

        # 6.2 Resume trajectory comparison
        try:
            torch.manual_seed(SEED)
            m1 = fresh_model()
            x = make_input()
            opt1 = build_opt(m1.parameters(), backend, quantize=True,
                             min_8bit_size=0, **ADAMW_KW)
            train(m1, opt1, x, 5)
            sd = opt1.state_dict()
            import copy
            sd_isolated = copy.deepcopy(sd)
            params_snap = [p.detach().clone() for p in m1.parameters()]
            losses_cont = train(m1, opt1, x, 5)

            torch.manual_seed(SEED)
            m2 = fresh_model()
            with torch.no_grad():
                for p_snap, p2 in zip(params_snap, m2.parameters()):
                    p2.copy_(p_snap)
            opt2 = build_opt(m2.parameters(), backend, quantize=True,
                             min_8bit_size=0, **ADAMW_KW)
            opt2.load_state_dict(sd_isolated)
            losses_resumed = train(m2, opt2, x, 5)

            md = max(abs(a - b) for a, b in zip(losses_cont, losses_resumed))
            record(f"[{backend}] resume trajectory", md < CKPT_TOL,
                   f"max_loss_diff={md:.2e}")
        except Exception as e:
            record(f"[{backend}] resume trajectory", False, str(e))

        # 6.3 m_quant_type preserved
        try:
            m1 = fresh_model()
            x = make_input()
            opt1 = build_opt(m1.parameters(), backend, quantize=True,
                             m_quant_type="d8", min_8bit_size=0, **ADAMW_KW)
            train(m1, opt1, x, 5)
            sd = opt1.state_dict()
            m2 = fresh_model()
            m2.load_state_dict(m1.state_dict())
            opt2 = build_opt(m2.parameters(), backend, quantize=True,
                             m_quant_type="d8", min_8bit_size=0, **ADAMW_KW)
            opt2.load_state_dict(sd)
            ok = all(st.get("m_quant_type") == "d8" for st in opt2.state.values())
            record(f"[{backend}] m_quant_type preserved", ok)
        except Exception as e:
            record(f"[{backend}] m_quant_type preserved", False, str(e))

        # 6.4 v_quant_type preserved (al16)
        try:
            m1 = fresh_model()
            x = make_input()
            opt1 = build_opt(m1.parameters(), backend, quantize=True,
                             v_quant_type="al16", min_8bit_size=0, **ADAMW_KW)
            train(m1, opt1, x, 5)
            sd = opt1.state_dict()
            m2 = fresh_model()
            m2.load_state_dict(m1.state_dict())
            opt2 = build_opt(m2.parameters(), backend, quantize=True,
                             v_quant_type="al16", min_8bit_size=0, **ADAMW_KW)
            opt2.load_state_dict(sd)
            ok = all(st.get("v_quant_type") == "al16" for st in opt2.state.values())
            record(f"[{backend}] v_quant_type al16 preserved", ok)
        except Exception as e:
            record(f"[{backend}] v_quant_type al16", False, str(e))

        # 6.5 v_quant_type preserved (fp32)
        try:
            m1 = fresh_model()
            x = make_input()
            opt1 = build_opt(m1.parameters(), backend, quantize=True,
                             v_quant_type="fp32", min_8bit_size=0, **ADAMW_KW)
            train(m1, opt1, x, 5)
            sd = opt1.state_dict()
            m2 = fresh_model()
            m2.load_state_dict(m1.state_dict())
            opt2 = build_opt(m2.parameters(), backend, quantize=True,
                             v_quant_type="fp32", min_8bit_size=0, **ADAMW_KW)
            opt2.load_state_dict(sd)
            ok = all(st.get("v_quant_type") == "fp32" for st in opt2.state.values())
            record(f"[{backend}] v_quant_type fp32 preserved", ok)
        except Exception as e:
            record(f"[{backend}] v_quant_type fp32", False, str(e))

        # 6.6 c_quant_type preserved (al16)
        try:
            m1 = fresh_model()
            x = make_input()
            opt1 = build_opt(m1.parameters(), backend, quantize=True,
                             c_quant_type="al16", min_8bit_size=0, **CAME_KW)
            train(m1, opt1, x, 5)
            sd = opt1.state_dict()
            m2 = fresh_model()
            m2.load_state_dict(m1.state_dict())
            opt2 = build_opt(m2.parameters(), backend, quantize=True,
                             c_quant_type="al16", min_8bit_size=0, **CAME_KW)
            opt2.load_state_dict(sd)
            ok = all(st.get("c_quant_type") == "al16" for st in opt2.state.values())
            record(f"[{backend}] c_quant_type al16 preserved", ok)
        except Exception as e:
            record(f"[{backend}] c_quant_type al16", False, str(e))

        # 6.7 APOLLO seed counter preserved
        try:
            m1 = fresh_model()
            x = make_input()
            opt1 = build_opt(make_apollo_groups(m1), backend, quantize=True,
                             min_8bit_size=0, **APOLLO_BASE_KW)
            train(m1, opt1, x, 5)
            sd = opt1.state_dict()
            seed_before = sd.get('_apollo_seed_counter', None)
            m2 = fresh_model()
            m2.load_state_dict(m1.state_dict())
            opt2 = build_opt(make_apollo_groups(m2), backend, quantize=True,
                             min_8bit_size=0, **APOLLO_BASE_KW)
            opt2.load_state_dict(sd)
            seed_after = opt2._apollo_seed_counter
            ok = seed_before == seed_after
            record(f"[{backend}] APOLLO seed preserved", ok,
                   f"{seed_before} → {seed_after}")
        except Exception as e:
            record(f"[{backend}] APOLLO seed", False, str(e))

        # 6.8 Empty state save/load
        try:
            m1 = fresh_model()
            opt1 = build_opt(m1.parameters(), backend, quantize=True,
                             min_8bit_size=0, **ADAMW_KW)
            sd = opt1.state_dict()
            m2 = fresh_model()
            opt2 = build_opt(m2.parameters(), backend, quantize=True,
                             min_8bit_size=0, **ADAMW_KW)
            opt2.load_state_dict(sd)
            x = make_input()
            train(m2, opt2, x, 3)
            record(f"[{backend}] empty state save/load", True)
        except Exception as e:
            record(f"[{backend}] empty state", False, str(e))


# ════════════════════════════════════════════════════════════
# Part 7: Determinism (whitelist-based)
# ════════════════════════════════════════════════════════════

def part7():
    print(f"\n{'─' * 60}")
    print("Part 7: Determinism (same seed → same result)")
    print(f"{'─' * 60}")

    det_configs = {
        "adamw": dict(**ADAMW_KW, quantize=True, m_quant_type="uf8",
                      min_8bit_size=0),
        "adafactor": dict(**ADAFACTOR_KW, quantize=True, min_8bit_size=0),
        "rmsprop": dict(**RMSPROP_KW, quantize=True, min_8bit_size=0),
    }

    for backend in BACKENDS:
        print(f"\n  [{backend}]")
        strict = DETERMINISM_STRICT.get(backend, [])

        for name, kw in det_configs.items():
            try:
                results = []
                for run in range(2):
                    torch.manual_seed(SEED)
                    m = fresh_model()
                    x = make_input()
                    opt = build_opt(m.parameters(), backend, **kw)
                    losses = train(m, opt, x, STEPS_ALIGN)
                    results.append((losses, param_norm(m)))
                loss_diff = max(abs(a - b) for a, b in
                                zip(results[0][0], results[1][0]))
                norm_diff = abs(results[0][1] - results[1][1])
                detail = f"Δloss={loss_diff:.2e} Δ|p|={norm_diff:.2e}"
                if name in strict:
                    ok = loss_diff < 1e-6 and norm_diff < 1e-6
                    record(f"[{backend}] determinism {name}", ok, detail)
                else:
                    if loss_diff < 1e-6 and norm_diff < 1e-6:
                        record(f"[{backend}] determinism {name}", True,
                               detail + " (non-strict)")
                    else:
                        warn(f"[{backend}] determinism {name}",
                             detail + " (expected: atomicAdd)")
            except Exception as e:
                record(f"[{backend}] determinism {name}", False, str(e))


# ════════════════════════════════════════════════════════════
# Part 8: Hot-swap & State Migration
# (Placed LAST: hot-swap triggers expected library warnings.
#  Isolated here to keep Parts 1-7 output clean.)
# ════════════════════════════════════════════════════════════

def part8():
    print(f"\n{'─' * 60}")
    print("Part 8: Hot-swap & State Migration")
    print("  (Expected warnings may appear below)")
    print(f"{'─' * 60}")

    for backend in BACKENDS:
        print(f"\n  [{backend}]")

        # 8.1 quantize toggle (True → False → True)
        try:
            m = fresh_model()
            x = make_input()
            opt = build_opt(m.parameters(), backend, quantize=True,
                            min_8bit_size=0, **ADAMW_KW)
            train(m, opt, x, STEPS_SANITY)
            for g in opt.param_groups:
                g["quantize"] = False
            train(m, opt, x, STEPS_SANITY)
            for g in opt.param_groups:
                g["quantize"] = True
            train(m, opt, x, STEPS_SANITY)
            record(f"[{backend}] quantize toggle", True)
        except Exception as e:
            record(f"[{backend}] quantize toggle", False, str(e))

        # 8.2 M hot-swap (uf8 → d8)
        with suppress_adafactor_warnings():
            try:
                m = fresh_model()
                x = make_input()
                opt = build_opt(m.parameters(), backend, quantize=True,
                                m_quant_type="uf8", min_8bit_size=0, **ADAMW_KW)
                train(m, opt, x, STEPS_SANITY)
                for g in opt.param_groups:
                    g["m_quant_type"] = "d8"
                losses = train(m, opt, x, STEPS_SANITY)
                finite = all(math.isfinite(l) for l in losses)
                record(f"[{backend}] M hot-swap (uf8→d8)", finite)
            except Exception as e:
                record(f"[{backend}] M hot-swap", False, str(e))

        # 8.3 V hot-swap (al8 → al16)
        with suppress_adafactor_warnings():
            try:
                m = fresh_model()
                x = make_input()
                opt = build_opt(m.parameters(), backend, quantize=True,
                                v_quant_type="al8", min_8bit_size=0, **ADAMW_KW)
                train(m, opt, x, STEPS_SANITY)
                for g in opt.param_groups:
                    g["v_quant_type"] = "al16"
                losses = train(m, opt, x, STEPS_SANITY)
                finite = all(math.isfinite(l) for l in losses)
                record(f"[{backend}] V hot-swap (al8→al16)", finite)
            except Exception as e:
                record(f"[{backend}] V hot-swap", False, str(e))

        # 8.4 C hot-swap (al8 → al16 → fp32)
        with suppress_adafactor_warnings():
            try:
                m = fresh_model()
                x = make_input()
                opt = build_opt(m.parameters(), backend, quantize=True,
                                c_quant_type="al8", min_8bit_size=0, **CAME_KW)
                train(m, opt, x, STEPS_SANITY)
                for g in opt.param_groups:
                    g["c_quant_type"] = "al16"
                train(m, opt, x, STEPS_SANITY)
                for g in opt.param_groups:
                    g["c_quant_type"] = "fp32"
                losses = train(m, opt, x, STEPS_SANITY)
                finite = all(math.isfinite(l) for l in losses)
                record(f"[{backend}] C hot-swap (al8→al16→fp32)", finite)
            except Exception as e:
                record(f"[{backend}] C hot-swap", False, str(e))


# ════════════════════════════════════════════════════════════
# Report Generation
# ════════════════════════════════════════════════════════════

def _report_environment():
    cuda_device = None
    if DEVICE.type == "cuda":
        try:
            cuda_device = torch.cuda.get_device_name(DEVICE)
        except Exception:
            cuda_device = "unknown"

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": str(DEVICE),
        "cuda_device": cuda_device,
        "cuda_kernel": HAS_CUDA_KERNEL,
        "backends": list(BACKENDS),
        "came_reference_available": HAS_CAME,
        "apollo_reference_available": HAS_APOLLO,
    }


def _report_configuration(selected_part):
    return {
        "selected_part": selected_part,
        "seed": SEED,
        "align_dim": ALIGN_DIM,
        "matrix_dim": MATRIX_DIM,
        "steps": {
            "alignment": STEPS_ALIGN,
            "backend": STEPS_BACKEND,
            "sanity": STEPS_SANITY,
            "tracking": STEPS_TRACK,
            "memory_stability": STEPS_MEM_STABLE,
        },
        "tolerances": {
            "reference_alignment": ALIGN_TOL,
            "backend_pass": BACKEND_WARN,
            "backend_fail": BACKEND_FAIL,
            "loss_8bit": LOSS_TOL_8BIT,
            "loss_4bit": LOSS_TOL_4BIT,
            "checkpoint": CKPT_TOL,
        },
    }


def _markdown_escape(value):
    return str(value).replace("|", r"\|").replace("\r", "").replace("\n", "<br>")


def _atomic_write_text(path, text):
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(path)


def generate_reports(report_dir, selected_part, started_at, finished_at):
    report_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if selected_part == "all" else f"_part{selected_part}"
    json_path = report_dir / f"smoke_test{suffix}.json"
    md_path = report_dir / f"smoke_test{suffix}.md"

    summary = {
        "pass": PASS,
        "fail": FAIL,
        "warn": WARN,
        "skip": SKIP,
    }
    environment = _report_environment()
    configuration = _report_configuration(selected_part)
    payload = {
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "environment": environment,
        "configuration": configuration,
        "summary": summary,
        "results": RESULTS,
    }

    json_text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    lines = [
        "# Adafactor8Bit Smoke Test Report",
        "",
        f"**Started**: {payload['started_at']}",
        f"**Finished**: {payload['finished_at']}",
        f"**Selected part**: `{selected_part}`",
        f"**Device**: `{environment['device']}`",
        f"**CUDA device**: `{environment['cuda_device'] or 'N/A'}`",
        f"**CUDA kernel**: {'Available' if HAS_CUDA_KERNEL else 'N/A'}",
        "",
        "## Summary",
        "",
        f"- ✅ Passed: {PASS}",
        f"- ❌ Failed: {FAIL}",
        f"- ⚠️ Warnings: {WARN}",
        f"- ⏭️ Skipped: {SKIP}",
        "",
        "## Configuration",
        "",
        f"- Seed: `{SEED}`",
        f"- ALIGN_DIM: `{ALIGN_DIM}`",
        f"- MATRIX_DIM: `{MATRIX_DIM}`",
        f"- Reference alignment tolerance: `{ALIGN_TOL:.0e}` (strict)",
        f"- CUDA/backend consistency PASS threshold: `< {BACKEND_WARN:.0e}`",
        f"- CUDA/backend consistency FAIL threshold: `>= {BACKEND_FAIL:.0e}`",
        "",
        "## Result Policies",
        "",
        "- **Reference Algorithm Alignment**: strict semantic/reference check; "
        "no warning band.",
        "- **CUDA Numerical Consistency**: permits the documented warning band "
        "for CUDA accumulation-order and backend numerical effects.",
        "- **Functional / Regression**: behavioral, state, checkpoint, memory, "
        "and API regression checks.",
        "",
        "## Detailed Results",
        "",
        "| Status | Policy | Test Name | Details |",
        "|---|---|---|---|",
    ]
    icons = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️", "SKIP": "⏭️"}
    for result in RESULTS:
        status = result["status"]
        lines.append(
            f"| {icons.get(status, '')} {status} | "
            f"{_markdown_escape(result.get('policy', ''))} | "
            f"{_markdown_escape(result['name'])} | "
            f"{_markdown_escape(result['msg']) if result['msg'] else ''} |"
        )
    md_text = "\n".join(lines) + "\n"

    _atomic_write_text(json_path, json_text)
    _atomic_write_text(md_path, md_text)
    return json_path, md_path


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Adafactor8Bit Smoke Test")
    parser.add_argument("--part", default="all",
                        choices=["1", "2", "3", "4", "5", "6", "7", "8", "all"])
    parser.add_argument(
        "--report-dir", default=None,
        help=(
            "Directory for JSON/Markdown reports "
            "(default: <project>/reports_md)"
        ),
    )
    parser.add_argument(
        "--no-report", action="store_true",
        help="Disable JSON/Markdown report generation",
    )
    args = parser.parse_args()

    started_at = datetime.now().astimezone()

    print(f"{'═' * 60}")
    print("Adafactor8Bit Smoke Test")
    print(f"Device: {DEVICE}  |  Seed: {SEED}")
    print(f"ALIGN_DIM={ALIGN_DIM}  MATRIX_DIM={MATRIX_DIM}")
    print(f"CUDA kernel: {'available' if HAS_CUDA_KERNEL else 'NOT available'}")
    print(f"Backends: {BACKENDS}")
    print(f"Started: {started_at.strftime('%H:%M:%S')}")
    print(f"{'═' * 60}")

    parts = {"1": part1, "2": part2, "3": part3, "4": part4,
             "5": part5, "6": part6, "7": part7, "8": part8}

    if args.part == "all":
        for fn in parts.values():
            try:
                fn()
            except Exception:
                print(f"  💥  {fn.__name__} crashed:")
                trace = traceback.format_exc()
                print(trace, end="")
                record(
                    f"{fn.__name__} crashed", False,
                    trace.strip().splitlines()[-1],
                )
    else:
        try:
            parts[args.part]()
        except Exception:
            print(f"  💥  Part {args.part} crashed:")
            trace = traceback.format_exc()
            print(trace, end="")
            record(
                f"Part {args.part} crashed", False,
                trace.strip().splitlines()[-1],
            )

    finished_at = datetime.now().astimezone()
    report_paths = None
    if not args.no_report:
        try:
            if args.report_dir:
                report_dir = Path(args.report_dir).expanduser().resolve()
            else:
                report_dir = Path(__file__).resolve().parent.parent / "reports_md"
            report_paths = generate_reports(
                report_dir,
                args.part,
                started_at,
                finished_at,
            )
        except Exception as exc:
            warn("Report generation", f"{type(exc).__name__}: {exc}")

    print(f"\n{'═' * 60}")
    status = "🎉 ALL PASSED" if FAIL == 0 else f"{FAIL} FAILED"
    print(f"  {PASS} passed, {FAIL} failed, {WARN} warnings, {SKIP} skipped  →  {status}")
    if WARNINGS:
        print(f"\n  Warnings:")
        for name, msg in WARNINGS:
            print(f"    ⚠️ {name}: {msg}")
    if FAILURES:
        print(f"\n  Failures:")
        for name, msg in FAILURES:
            print(f"    ❌ {name}: {msg}")
    if report_paths is not None:
        print(f"\n  Reports:")
        print(f"    JSON: {report_paths[0]}")
        print(f"    Markdown: {report_paths[1]}")
    print(f"{'═' * 60}")
    print(f"Finished: {finished_at.strftime('%H:%M:%S')}")
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
