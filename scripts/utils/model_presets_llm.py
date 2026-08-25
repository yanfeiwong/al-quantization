from pathlib import Path

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ==========================================
# Model Presets
# ==========================================
MODEL_PRESETS = {
    "TinyLlama-1.1B": {
        "type": "llm",
        "dir": str(PROJECT_ROOT / "models" / "TinyLlama-1.1B"),
    },
    "GPT2-124M": {
        "type": "llm",
        "dir": str(PROJECT_ROOT / "models" / "GPT2-124M"),
    },
}

# ==========================================
# Model Factory
# ==========================================
def get_model_and_tokenizer(model_name, attn_implementation="flash_attention_2"):
    if model_name not in MODEL_PRESETS:
        raise ValueError(f"Model '{model_name}' not found. Available: {list(MODEL_PRESETS.keys())}")
    
    cfg = MODEL_PRESETS[model_name]
    model_dir = cfg["dir"]
    
    config = AutoConfig.from_pretrained(model_dir)
    config.use_cache = False 
    config.dtype = "float32"
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_config(
        config, 
        attn_implementation=attn_implementation
    )
    model.gradient_checkpointing_enable()
    
    return model, tokenizer
