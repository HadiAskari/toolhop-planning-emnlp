import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Resolves ${FORTE_ROOT} from the environment (default: repo root).
_ROOT = os.environ.get("FORTE_ROOT", ".")
adapter_path = os.path.expandvars(f"{_ROOT}/judge_finetuning/models/judge-toolhop-perfectonly/final")
output_path  = os.path.expandvars(f"{_ROOT}/judge_finetuning/models/judge-toolhop-perfectonly/merged")

with open(f"{adapter_path}/adapter_config.json") as f:
    base_model_name = json.load(f)["base_model_name_or_path"]

print(f"Base model: {base_model_name}")
print("Loading base model on CPU (no bitsandbytes)...")

# Load base model directly — no quantization, so bitsandbytes is never touched
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

print("Applying LoRA adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)

print("Merging and unloading adapter...")
model = model.merge_and_unload()

print(f"Saving merged model to {output_path} ...")
model.save_pretrained(output_path)

tokenizer = AutoTokenizer.from_pretrained(adapter_path)
tokenizer.save_pretrained(output_path)

print(f"✅ Done — merged model saved to {output_path}")