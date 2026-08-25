# ==========================================
# Parameter Grouping Strategies
# ==========================================
# G0 (Baseline/Bare-metal):
#   Default routing with no parameter protection. 
#   Exceptions are applied only for optimizers requiring specialized routing (e.g., APOLLO).
#
# G1 (Basic Protection):
#   Protects sensitive parameters (1D, Bias, Norms, Embeddings) from weight decay 
#   and quantization (for 8-bit optimizers).
#
# G2 (Advanced/Custom):
#   Exclusively designed for `Adafactor8Bit`. Provides fine-grained control over 
#   factorization, momentum, and learning rates for different layer dimensions.
# ==========================================
# [1] Parameter Classification (Source Nodes):
# ├── [p_1d]        : 1D tensors, Bias, Norms
# ├── [p_emb]       : Token Embeddings
# ├── [p_lm_head]   : LM Head / output projection (2D, vocab-sized)
# ├── [p_2d_apollo] : 2D Weights matching targets (e.g., 'attn', 'mlp')
# ├── [p_2d]        : Other 2D Weights
# └── [p_nd]        : >2D Weights (e.g., Conv)
#
# [2] Routing Topology (Sink Nodes):
#
# [G0] Baseline / Bare-metal
# ├── No Special Config (Regular/Ours) ──────────> G0_All (All parameters)
# └── Has Special Config (APOLLO/Ours)
#     ├── p_2d_apollo ───────────────────────────> G0_Apollo
#     └── Remainder (1D+Emb+LMHead+2D) + ND
#         ├── Official APOLLO ───────────────────> G0_Others
#         └── Ours (Adafactor8Bit)
#             ├── fallback="adam" ───────────────> G0_Adam (All remainder + ND)
#             └── fallback="adafactor"
#                 ├── Remainder ─────────────────> G0_Adafactor
#                 └── ND ────────────────────────> G0_Adafactor (if nd_factored) OR G0_Adam
#
# [G1] Basic Protection
# ├── p_1d + p_emb + p_lm_head ─────────────────> G1_Sensitive (WD=0, Q=False, BNB FP32)
# └── Remaining Dense (2D + ND)
#     ├── No Special Config ─────────────────────> G1_Others
#     └── Has Special Config
#         ├── p_2d_apollo ───────────────────────> G1_Apollo
#         └── Remainder (2D) + ND
#             ├── Official APOLLO ───────────────> G1_Others
#             └── Ours
#                 ├── fallback="adam" ───────────> G1_Adam
#                 └── fallback="adafactor"
#                     ├── Remainder (2D) ────────> G1_Adafactor
#                     └── ND ────────────────────> G1_Adafactor OR G1_Adam
#
# [G2] Advanced / Custom (Adafactor8Bit Only)
# ├── p_1d ─────────────────────────────────────> G2_1D_Sensitive (WD=0, Q=False)
# ├── p_emb ────────────────────────────────────> G2_Embeddings (Adam-style, Lower_LR, No Momentum, Q=False)
# ├── p_lm_head ────────────────────────────────> G2_LM_Head (Adam-style, WD=0, Q=True, keep user beta1)
# ├── p_2d_apollo ──────────────────────────────> G2_2D_Apollo
# ├── p_2d ─────────────────────────────────────> G2_2D / G2_2D_Adam / G2_2D_Adafactor
# └── p_nd ─────────────────────────────────────> G2_ND_Weights (Adam-style, Lower_LR)
# ==========================================

import torch
from adafactor8bit import Adafactor8Bit

LR_ADAM_STYLE = 1e-4


def get_adam_style_config(params, group_name, lr=None):
    # For ours only, enforce this group behave like adam
    return_dict = {
        "params": params,
        "group_name": group_name,
        "factored": False,
        "scale_parameter": False,
        "d": 0,
        "beta3": None,
        "apollo_rank": 0,
    }
    if lr is not None:
        return_dict["lr"] = lr
    return return_dict

def get_adafactor_style_config(params, group_name, lr=None):
    # For ours only, this group behave like adafactor based (eg. adafactor, CAME)
    # Set only factored to True and pop apollo_rank, and keep outhers
    return_dict = {
        "params": params,
        "group_name": group_name,
        "factored": True,
        "apollo_rank": 0
    }
    if lr is not None:
        return_dict["lr"] = lr
    return return_dict

def get_param_groups(
        model,
        grouping,
        optimizer_class,
        nd_factored=False,
        apollo_addit_kwargs=None,
        low_rank_proj_targets=None,  # eg. [None,"attn", "mlp"]
        fallback_behavior=None  # none apollo goes here [None, "adam", "adafactor"]
):

    is_ours = (optimizer_class == Adafactor8Bit)
    has_special_config = bool(apollo_addit_kwargs) or bool(low_rank_proj_targets)

    # if apollo_addit_kwargs is None:
    #     apollo_addit_kwargs = {}
    if low_rank_proj_targets is None:
        low_rank_proj_targets = []

    # ==========================================
    # Prepare Parameters
    # ==========================================
    p_1d, p_emb, p_lm_head, p_2d_apollo, p_2d, p_nd = [], [], [], [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_1d = param.ndim <= 1 or "bias" in name or "norm" in name
        is_emb = (("embed" in name.lower() or "wte" in name.lower())
                  and "position" not in name.lower()
                  and "pos_embed" not in name.lower()
                  and "wpe" not in name.lower()
                  and "time" not in name.lower())
        is_lm_head = param.ndim == 2 and "lm_head" in name.lower()
        is_2d_target = param.ndim == 2 and any(t in name for t in low_rank_proj_targets)

        if is_1d:
            p_1d.append(param)
        elif is_emb:
            p_emb.append(param)
        elif is_lm_head:
            p_lm_head.append(param)
        elif is_2d_target:
            p_2d_apollo.append(param)
        elif param.ndim == 2:
            p_2d.append(param)
        else:
            p_nd.append(param)

    # ==========================================
    # Grouping
    # ==========================================
    if grouping == "G0":

        # Regular & Ours 
        if not has_special_config:
            return [{"params": model.parameters(), "group_name": "G0_All"}]
        

        # Offical APOLLO & Ours Special
        groups = []
        if p_2d_apollo:
            g_apollo = {"params": p_2d_apollo, "group_name": "G0_Apollo"}
            if apollo_addit_kwargs:
                g_apollo.update(apollo_addit_kwargs)
            groups.append(g_apollo)

        p_remain = p_1d + p_emb + p_lm_head + p_2d

        # apollo offical (does not take any our custom configs)
        if apollo_addit_kwargs:
            if p_remain or p_nd:
                groups.append({"params": p_remain + p_nd, "group_name": "G0_Others"})

        # ours special
        else:
            g0_adafactor = None
            g0_adam = None
            if fallback_behavior == "adam":
                g0_adam = p_remain + p_nd
            elif fallback_behavior == "adafactor":
                if nd_factored:
                    g0_adafactor = p_remain + p_nd
                else:
                    g0_adafactor = p_remain
                    g0_adam = p_nd
            else:
                raise ValueError(f"UNsupported fallback_behavior {fallback_behavior}.")
            if g0_adam:
                groups.append(get_adam_style_config(g0_adam, "G0_Adam"))
            if g0_adafactor:
                groups.append(get_adafactor_style_config(g0_adafactor, "G0_Adafactor"))

        return groups

    elif grouping == "G1":
        groups = []
        sensitive = p_1d + p_emb + p_lm_head

        # Protect sensitive parameters from quantization and weight decay
        if sensitive:
            if is_ours and has_special_config and fallback_behavior:
                if fallback_behavior == "adam":
                    g_sens = get_adam_style_config(sensitive, "G1_Sensitive")
                else:
                    g_sens = get_adafactor_style_config(sensitive, "G1_Sensitive")
            else:
                g_sens = {"params": sensitive, "group_name": "G1_Sensitive"}
            g_sens["weight_decay"] = 0.0
            if is_ours:
                g_sens["quantize"] = False
            groups.append(g_sens)

        if not has_special_config:
            p_remain = p_2d + p_nd
            if p_remain:
                groups.append({"params": p_remain, "group_name": "G1_Others"})
            
            # BNB Specific for G1
            if "bitsandbytes" in str(optimizer_class.__module__).lower():
                from bitsandbytes.optim import GlobalOptimManager
                mng = GlobalOptimManager.get_instance()
                for module in model.modules():
                    if isinstance(module, torch.nn.Embedding):
                        mng.register_module_override(module, 'weight', {'optim_bits': 32})
            return groups

        # Offical APOLLO & Ours Special
        if p_2d_apollo:
            g_apollo = {"params": p_2d_apollo, "group_name": "G1_Apollo"}
            if apollo_addit_kwargs:
                g_apollo.update(apollo_addit_kwargs)
            groups.append(g_apollo)

        p_remain = p_2d

        # apollo offical (does not take any our custom configs)
        if apollo_addit_kwargs:
            if p_remain or p_nd:
                groups.append({"params": p_remain + p_nd, "group_name": "G1_Others"})

        # ours special
        else:
            g1_adafactor = None
            g1_adam = None
            if fallback_behavior == "adam":
                g1_adam = p_remain + p_nd
            elif fallback_behavior == "adafactor":
                if nd_factored:
                    g1_adafactor = p_remain + p_nd
                else:
                    g1_adafactor = p_remain
                    g1_adam = p_nd
            else:
                raise ValueError(f"UNsupported fallback_behavior {fallback_behavior}.")
                
            if g1_adam:
                groups.append(get_adam_style_config(g1_adam, "G1_Adam"))
            if g1_adafactor:
                groups.append(get_adafactor_style_config(g1_adafactor, "G1_Adafactor"))

        # BNB Specific for G1
        if "bitsandbytes" in str(optimizer_class.__module__).lower():
            from bitsandbytes.optim import GlobalOptimManager
            mng = GlobalOptimManager.get_instance()
            for name, module in model.named_modules():
                if not hasattr(module, 'weight') or module.weight is None:
                    continue
                is_emb_module = (("embed" in name.lower() or "wte" in name.lower())
                                 and "position" not in name.lower()
                                 and "pos_embed" not in name.lower()
                                 and "wpe" not in name.lower()
                                 and "time" not in name.lower())
                is_lm_head_module = "lm_head" in name.lower()
                if is_emb_module or is_lm_head_module:
                    mng.register_module_override(module, 'weight', {'optim_bits': 32})

        return groups

    elif grouping == "G2":
        if not is_ours:
            raise ValueError("G2 grouping is only supported for Adafactor8Bit.")

        groups = []

        # 1. 1D / Sensitive
        if p_1d:
            g1d = {
                "params": p_1d,
                "group_name": "G2_1D_Sensitive",
                "weight_decay": 0.0,
                "quantize": False,
                "apollo_rank": 0,
            }
            groups.append(g1d)

        # 2. Token Embeddings (Momentum-free Adam style) 
        if p_emb:
            gemb = get_adam_style_config(p_emb, "G2_Embeddings", lr=LR_ADAM_STYLE)
            gemb["weight_decay"] = 0.0
            gemb["quantize"] = False
            gemb["beta1"] = None
            groups.append(gemb)

        # 2b. LM Head (full-rank V, quantized, keep user beta1)
        if p_lm_head:
            glm = get_adam_style_config(p_lm_head, "G2_LM_Head")
            glm["weight_decay"] = 0.0
            groups.append(glm)

        # 3. 2D Weights (Apollo)
        if p_2d_apollo:
            g2d_apollo = {
                "params": p_2d_apollo,
                "group_name": "G2_2D_Apollo"
            }
            if apollo_addit_kwargs:
                g2d_apollo.update(apollo_addit_kwargs)
            groups.append(g2d_apollo)

        # 4. 2D Weights (Fallback)
        if p_2d:
            if not fallback_behavior:
                g2d_fallback = {
                    "params": p_2d,
                    "group_name": "G2_2D",
                }
            elif fallback_behavior == "adam":
                g2d_fallback = get_adam_style_config(p_2d, "G2_2D_Adam")
            elif fallback_behavior == "adafactor":
                g2d_fallback = get_adafactor_style_config(p_2d, "G2_2D_Adafactor")
            else:
                raise ValueError(f"UNsupported fallback_behavior {fallback_behavior}.")
            groups.append(g2d_fallback)

        # 5. >2D Weights (Force un-factorized, AdamW style)
        if p_nd:
            gnd = get_adam_style_config(p_nd, "G2_ND_Weights", lr=LR_ADAM_STYLE)
            groups.append(gnd)

        return groups

    else:
        raise ValueError(f"Unknown grouping strategy: {grouping}")


def print_optimizer_groups(optimizer):
    """Prints a concise summary of optimizer parameter groups for verification."""
    print("\n--- Optimizer Groups Summary ---")
    total_tensors = sum(len(g['params']) for g in optimizer.param_groups)
    total_elements = sum(p.numel() for g in optimizer.param_groups for p in g['params'])

    def get_beta(param_group, index, default='N/A'):
        key = f'beta{index}'
        if key in param_group:
            return param_group[key]
        betas = param_group.get('betas')
        if isinstance(betas, (tuple, list)) and len(betas) >= index:
            return betas[index-1]
        return default

    for i, g in enumerate(optimizer.param_groups):
        num_tensors = len(g['params'])
        if num_tensors == 0:
            continue

        num_elements = sum(p.numel() for p in g['params'])
        pct_tensors = 100 * num_tensors / total_tensors if total_tensors > 0 else 0
        pct_elements = 100 * num_elements / total_elements if total_elements > 0 else 0

        name = g.get('group_name', f'Group_{i}')
        lr = g.get('lr', 0.0)
        wd = g.get('weight_decay', 0.0)

        quantize = g.get('quantize', 'N/A')
        factored = g.get('factored', 'N/A')
        scale_p = g.get('scale_parameter', 'N/A')
        apollo = g.get('apollo_rank', g.get('rank', 'N/A'))

        b1 = get_beta(g, 1)
        b2 = get_beta(g, 2)
        b3 = get_beta(g, 3)

        print(f"[{name}] Tensors: {num_tensors} ({pct_tensors:.1f}%) | Params: {num_elements:,} ({pct_elements:.1f}%) | LR: {lr:.1e} | WD: {wd} | Q: {quantize} | Factored: {factored} | Scale: {scale_p} | Apollo: {apollo} | Beta1: {b1} | Beta2: {b2} | Beta3: {b3}")