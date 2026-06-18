"""
Distill 4 role-specialized LoRA students from the teacher pairs.

Trains a separate LoRA adapter on Qwen2.5-7B-Instruct per role using PEFT +
TRL's SFTTrainer. Each adapter is small (~16M trainable params on a 7B base)
and saves as a single safetensors file that vLLM hot-swaps at inference.

Runs on Modal A100 — 4 roles trained sequentially in one Modal call so the
volume + GPU lifecycle is amortized.

    python3 -m modal run scripts/train_lora_students.py
"""

import os
import sys

import modal

GPU = os.environ.get("LORA_GPU", "A100-40GB")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers>=4.45",
        "peft>=0.13",
        "trl>=0.11",
        "accelerate>=1.0",
        "datasets>=3.0",
        "safetensors>=0.4",
        "sentencepiece",
    )
    .add_local_dir(
        "data_real/distill/pairs", "/root/pairs", copy=True,
        ignore=["**/__pycache__", "**/*.pyc"],
    )
)

app = modal.App("netsta-lora-distill")
vol = modal.Volume.from_name("netsta-lora-adapters", create_if_missing=True)


@app.function(gpu=GPU, image=image, volumes={"/vol": vol}, timeout=60 * 60 * 5)
def train_one_lora(role: str, base_model: str, epochs: int, lr: float):
    """Train one LoRA adapter for one role on its (system, user, assistant) pairs."""
    import json
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
    )
    from trl import SFTTrainer, SFTConfig

    print(f"\n=== [LoRA student: {role}] ===")
    pair_path = f"/root/pairs/{role}.jsonl"
    if not os.path.exists(pair_path):
        print(f"  no pairs for {role} — skipping")
        return None

    pairs = []
    with open(pair_path) as f:
        for line in f:
            d = json.loads(line)
            pairs.append({
                "messages": [
                    {"role": "system", "content": d["system"]},
                    {"role": "user", "content": d["user"]},
                    {"role": "assistant", "content": d["assistant"]},
                ]
            })
    print(f"  loaded {len(pairs)} pairs")
    if len(pairs) < 8:
        print(f"  too few pairs ({len(pairs)}) — skipping")
        return None

    ds = Dataset.from_list(pairs)
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype="bfloat16", device_map="auto",
    )
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {trainable:,}")

    out_dir = f"/vol/{role}_lora"
    cfg = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        max_length=4096,
        packing=False,
        dataset_kwargs={"skip_prepare_dataset": False},
    )
    trainer = SFTTrainer(
        model=model, args=cfg, train_dataset=ds,
        processing_class=tok,
    )
    trainer.train()
    # Save the adapter only (peft saves the adapter, not the base weights).
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    vol.commit()
    print(f"  saved adapter -> {out_dir}")
    return out_dir


@app.local_entrypoint()
def main(
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    epochs: int = 3,
    lr: float = 2e-4,
    roles: str = "supervisor,timing,drc,optimization",
):
    selected = [r.strip() for r in roles.split(",") if r.strip()]
    for role in selected:
        train_one_lora.remote(role, base_model, epochs, lr)
    print("\nAll role adapters trained.")
