import torch
import bitsandbytes as bnb
from transformers.optimization import Adafactor
from adafactor8bit import Adafactor8Bit
from came_pytorch import CAME
from apollo_torch import APOLLOAdamW

# ==========================================
# Optimizer Presets (pretrain)
# ==========================================
# Naming: {path}_ours_{mtype}_{vtype}_vblk{vblk}[_c_{ctype}][_{extras}]
# mtype : fp32 / uf4 / uf8 / d4 / d8  (M quantization; fp32 = no M quant)
# vtype : al8 / al16                   (V quantization, always explicit)
# vblk  : 2048 / 256                   (V block_size, always explicit)
# ctype : fp32 / al16 / al8            (C quantization, optional; None = follows vtype)
# extras: bc = bias_correction, fb = fixed_beta2, full = quantize=False, nc = no_cuda_kernel
# Adafactor/RMSprop paths have no M, so mtype is omitted.
# ==========================================

DEFAULT_V_QUANT = "al8"
V_BLOCK_SIZES = (2048, 256)

M_QUANT_TABLE = {
    "fp32": ("fp32", None),
    "uf4":  ("uf4", 128),
    "uf8":  ("uf8", 256),
    "d4":   ("d4", 128),
    "d8":   ("d8", 256),
}

ADAMW_MTYPES  = ("fp32", "uf4", "uf8", "d4", "d8")
CAME_MTYPES   = ("fp32", "uf4", "uf8", "d4", "d8")
APOLLO_MTYPES = ("fp32", "uf4", "uf8", "d4", "d8")

EXTRAS_KWARGS = {
    "bc": {"bias_correction": True},
    "fb": {"beta2": 0.999, "beta2_decay": None},
}

# ==========================================
# Per-path base kwargs
# ==========================================
ADAMW_BASE = {
    "lr": 1e-3,
    "beta1": 0.9,
    "beta2": 0.999,
    "weight_decay": 0.0,
    "scale_weight_decay": False,
    "scale_parameter": False,
    "d": 0,
    "relative_step": False,
    "factored": False,
    "eps": (1e-8, 1e-3),
}

ADAFACTOR_BASE = {
    "lr": 1e-3,
    "relative_step": False,
    "weight_decay": 0.0,
}

RMSPROP_BASE = {
    "lr": 1e-3,
    "beta1": None,
    "beta2": 0.99,
    "weight_decay": 0.0,
    "scale_weight_decay": False,
    "scale_parameter": False,
    "d": 0,
    "relative_step": False,
    "factored": False,
    "eps": (1e-8, 1e-3),
}

CAME_BASE = {
    "lr": 1e-3,
    "beta1": 0.9,
    "beta2": 0.999,
    "beta3": 0.9999,
    "eps_came": 1e-16,
    "scale_weight_decay": False,
    "scale_parameter": False,
    "d": 1.0,
    "relative_step": False,
    "weight_decay": 0.0,
}

APOLLO_BASE = {
    "lr": 1e-3,
    "beta1": 0.9,
    "beta2": 0.999,
    "scale_weight_decay": False,
    "scale_parameter": False,
    "d": 0,
    "relative_step": False,
    "apollo_rank": 256,
    "apollo_scale_type": "channel",
    "apollo_update_proj_gap": 200,
    "weight_decay": 0.0,
    "apollo_eps": 1e-6,
    "eps": (1e-6, 1e-3),
    "apollo_cache_proj": True,
}

# ==========================================
# Grid generators
# ==========================================
def _ours_cfg(**top_level):
    cfg = {"class": Adafactor8Bit, "max_grad_norm": 0.0}
    cfg.update(top_level)
    return cfg


def _apply_m(kwargs, mtype):
    m_quant, m_block = M_QUANT_TABLE[mtype]
    kwargs["m_quant_type"] = m_quant
    if m_block is not None:
        kwargs["m_block_size"] = m_block


def _gen_grid(presets, path, base_kwargs, mtypes, vblks=V_BLOCK_SIZES,
              vtype=DEFAULT_V_QUANT, ctype=None, extras=("",), top_cfg=None):
    for mtype in mtypes:
        for vblk in vblks:
            for extra in extras:
                kwargs = dict(base_kwargs)
                if extra:
                    kwargs.update(EXTRAS_KWARGS[extra])
                if mtype is not None:
                    _apply_m(kwargs, mtype)
                kwargs["block_size"] = vblk
                if vtype != DEFAULT_V_QUANT:
                    kwargs["v_quant_type"] = vtype
                if ctype is not None:
                    kwargs["c_quant_type"] = ctype
                
                name = f"{path}_ours"
                if mtype is not None:
                    name += f"_{mtype}"
                name += f"_{vtype}_vblk{vblk}"
                if ctype is not None:
                    name += f"_c_{ctype}"
                if extra:
                    name += f"_{extra}"
                presets[name] = {**_ours_cfg(**(top_cfg or {})), "kwargs": kwargs}


# ==========================================
# Presets
# ==========================================
OPTIMIZER_PRESETS = {
    # ==============================================================
    # Baselines
    # ==============================================================
    "adamw_torch": {
        "class": torch.optim.AdamW,
        "max_grad_norm": 0.0,
        "kwargs": {
            "lr": 1e-3,
            "weight_decay": 0.0,
        }
    },
    "adamw_8bit_bnb": {
        "class": bnb.optim.AdamW8bit,
        "max_grad_norm": 0.0,
        "kwargs": {
            "lr": 1e-3,
            "weight_decay": 0.0,
        }
    },
    "adafactor_hf": {
        "class": Adafactor,
        "max_grad_norm": 0.0,
        "kwargs": {
            "lr": 1e-3,
            "scale_parameter": True,
            "relative_step": False,
            "warmup_init": False,
            "weight_decay": 0.0,
        }
    },
    "rmsprop_torch": {
        "class": torch.optim.RMSprop,
        "max_grad_norm": 0.0,
        "kwargs": {
            "lr": 1e-3,
            "alpha": 0.99,
            "eps": 1e-8,
            "weight_decay": 0.0,
            "momentum": 0,
            "centered": False,
        }
    },
    "came_torch": {
        "class": CAME,
        "max_grad_norm": 0.0,
        "kwargs": {
            "lr": 1e-3,
            "eps": (1e-30, 1e-16),
            "clip_threshold": 1.0,
            "betas": (0.9, 0.999, 0.9999),
            "weight_decay": 0.0,
        }
    },
    "apollo_torch": {
        "class": APOLLOAdamW,
        "max_grad_norm": 0.0,
        "low_rank_proj_targets": ["attn", "mlp"],
        "kwargs": {
            "lr": 1e-3,
            "weight_decay": 0.0,
        },
        "apollo_addit_kwargs": {
            "rank": 256,
            "update_proj_gap": 200,
            "scale_type": "channel",
            "proj": "random",
            "scale": 1.0,
            "proj_type": "std",
        }
    },
}

# ==============================================================
# Ours - AdamW path
# ==============================================================
_gen_grid(OPTIMIZER_PRESETS, "adamw", ADAMW_BASE, ADAMW_MTYPES)
_gen_grid(OPTIMIZER_PRESETS, "adamw", ADAMW_BASE, ("uf8",), vblks=(2048,), vtype="al16")

OPTIMIZER_PRESETS["adamw_ours_full"] = {
    "class": Adafactor8Bit,
    "max_grad_norm": 0.0,
    "kwargs": {
        **ADAMW_BASE,
        "quantize": False,
    }
}

# ==============================================================
# Ours - Adafactor path (no M)
# ==============================================================
_gen_grid(OPTIMIZER_PRESETS, "adafactor", ADAFACTOR_BASE, (None,), extras=("", "fb"))
_gen_grid(OPTIMIZER_PRESETS, "adafactor", ADAFACTOR_BASE, (None,), vtype="al16")

OPTIMIZER_PRESETS["adafactor_ours_full"] = {
    "class": Adafactor8Bit,
    "max_grad_norm": 0.0,
    "kwargs": {
        **ADAFACTOR_BASE,
        "quantize": False,
    }
}

# ==============================================================
# Ours - RMSprop path (no M)
# ==============================================================
_gen_grid(OPTIMIZER_PRESETS, "rmsprop", RMSPROP_BASE, (None,))

# ==============================================================
# Ours - CAME path
# ==============================================================
# Standard CAME grid (c_quant_type=None, follows v_quant_type)
_gen_grid(OPTIMIZER_PRESETS, "came", CAME_BASE, CAME_MTYPES, extras=("", "bc"))

# CAME with c_quant_type="fp32" (for ablation study)
_gen_grid(OPTIMIZER_PRESETS, "came", CAME_BASE, ("uf8",), vblks=(2048,), ctype="fp32")

# CAME with c_quant_type="al16" (for ablation study)
_gen_grid(OPTIMIZER_PRESETS, "came", CAME_BASE, ("uf8",), vblks=(2048,), ctype="al16")

OPTIMIZER_PRESETS["came_ours_uf8_al8_vblk2048_nc"] = {
    "class": Adafactor8Bit,
    "max_grad_norm": 0.0,
    "kwargs": {
        **CAME_BASE,
        "m_quant_type": "uf8",
        "m_block_size": 256,
        "block_size": 2048,
        "use_cuda_kernel": False,
        "c_quant_type": None,
    }
}

_gen_grid(OPTIMIZER_PRESETS, "came", CAME_BASE, ("uf8",), vblks=(2048,), vtype="al16")

OPTIMIZER_PRESETS["came_ours_full"] = {
    "class": Adafactor8Bit,
    "max_grad_norm": 0.0,
    "kwargs": {
        **CAME_BASE,
        "quantize": False,
    }
}

# ==============================================================
# Ours - APOLLO path
# ==============================================================
_gen_grid(OPTIMIZER_PRESETS, "apollo", APOLLO_BASE, APOLLO_MTYPES,
          top_cfg={"low_rank_proj_targets": ["attn", "mlp"], "fallback_behavior": "adam"})
_gen_grid(OPTIMIZER_PRESETS, "apollo", APOLLO_BASE, ("uf8",), vblks=(2048,), vtype="al16",
          top_cfg={"low_rank_proj_targets": ["attn", "mlp"], "fallback_behavior": "adam"})

OPTIMIZER_PRESETS["apollo_ours_full"] = {
    "class": Adafactor8Bit,
    "max_grad_norm": 0.0,
    "low_rank_proj_targets": ["attn", "mlp"],
    "fallback_behavior": "adam",
    "kwargs": {
        **APOLLO_BASE,
        "quantize": False,
    }
}

# ==============================================================
# Ours - CALM (experimental, deferred)
# ==============================================================
OPTIMIZER_PRESETS["calm_ours_rms_off"] = {
    "class": Adafactor8Bit,
    "max_grad_norm": 0.0,
    "nd_factored": False,
    "low_rank_proj_targets": ["attn", "mlp"],
    "fallback_behavior": "adafactor",
    "kwargs": {
        "lr": 1e-3,
        "beta1": 0.9,
        "beta2": 0.999,
        "beta3": 0.9999,
        "scale_weight_decay": False,
        "scale_parameter": False,
        "d": 0,
        "relative_step": False,
        "apollo_rank": 256,
        "apollo_scale_type": "channel",
        "apollo_update_proj_gap": 200,
        "enable_fira_for_adafactor": True,
        "weight_decay": 0.0,
        "eps_came": 1e-16,
        "m_quant_type": "uf4",
        "m_block_size": 128,
        "block_size": 2048,
        "c_quant_type": None,
    }
}
OPTIMIZER_PRESETS["calm_ours_rms_on_fira_off"] = {
    "class": Adafactor8Bit,
    "max_grad_norm": 0.0,
    "nd_factored": False,
    "low_rank_proj_targets": ["attn", "mlp"],
    "fallback_behavior": "adafactor",
    "kwargs": {
        "lr": 1e-3,
        "beta1": 0.9,
        "beta2": 0.999,
        "beta3": 0.9999,
        "scale_weight_decay": True,
        "scale_parameter": True,
        "d": 1.0,
        "relative_step": False,
        "apollo_rank": 256,
        "apollo_scale_type": "channel",
        "apollo_update_proj_gap": 200,
        "enable_fira_for_adafactor": False,
        "weight_decay": 0.0,
        "eps_came": 1e-16,
        "m_quant_type": "uf4",
        "m_block_size": 128,
        "block_size": 2048,
        "c_quant_type": None,
    }
}