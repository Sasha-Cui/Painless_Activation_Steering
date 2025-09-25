# ─────────────────────────────── Model loading ───────────────────────────────
# Initialise the configurations
import torch
from preambles import *

def load_lm(name: str, device: str, quantization: str = "4bit"):
    if not device.startswith("cuda"):
        raise ValueError(f"🚫 This loader only supports GPU execution. You passed: {device}")

    quantization = quantization.lower()
    print(f"\n\n>>> Loading model from {name} on {device} with quantization: {quantization}")
    
    max_memory = {i: "80GiB" for i in range(torch.cuda.device_count())}
    max_memory["cpu"] = "0GiB"  # prevents CPU offload

    if quantization == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            name,
            cache_dir = os.environ.get('HF_HOME'),
            device_map="auto",  # will spread across all visible GPUs
            max_memory=max_memory,  # will spread across all visible GPUs
            quantization_config=quant_config,
            torch_dtype=torch.float16
        )
    elif quantization == "8bit":
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            name,
            cache_dir = os.environ.get('HF_HOME'),
            device_map="auto",  # will spread across all visible GPUs
            max_memory=max_memory,  # will spread across all visible GPUs
            quantization_config=quant_config,
            torch_dtype=torch.float16
        )
    elif quantization == "none":
        model = AutoModelForCausalLM.from_pretrained(
            name,
            cache_dir = os.environ.get('HF_HOME'),
            device_map="auto",  # will spread across all visible GPUs
            max_memory=max_memory,  # will spread across all visible GPUs
            torch_dtype=torch.float16,
        )
    else:
        raise ValueError("quantization must be one of: '4bit', '8bit', or 'none'")

    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": tok.eos_token})
        model.resize_token_embeddings(len(tok))
    tok.padding_side = "left"

    print("✅ Model and tokenizer fully loaded.")
    return model, tok

