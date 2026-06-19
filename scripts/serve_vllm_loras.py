"""
Serve Qwen2.5-7B + 4 role LoRA adapters via vLLM on a Modal A100.

Exposes an OpenAI-compatible chat-completions endpoint with hot-swap LoRA
routing: each diagnosis agent specifies its own adapter in the request via
the `model` field (vLLM resolves "{base}@{adapter}" to a LoRA request).

    python3 -m modal deploy scripts/serve_vllm_loras.py
    # then POST to https://<account>--netsta-vllm-serve.modal.run/v1/chat/completions
"""

import os

import modal

GPU = os.environ.get("VLLM_GPU", "A100-40GB")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.6.4",
        # Pin transformers to a 4.x release — vllm 0.6.4 still calls
        # `tokenizer.all_special_tokens_extended` which transformers 5.x removed.
        "transformers>=4.45,<5",
        # Pin huggingface_hub similarly; vllm 0.6.4 was built before 1.x.
        "huggingface_hub>=0.25,<1",
    )
)

app = modal.App("netsta-vllm-serve")
adapter_vol = modal.Volume.from_name("netsta-lora-adapters", create_if_missing=True)

ROLES = ["supervisor", "timing", "drc", "optimization"]


@app.function(
    gpu=GPU,
    image=image,
    volumes={"/adapters": adapter_vol},
    timeout=60 * 60 * 4,
    scaledown_window=60 * 5,
)
@modal.web_server(port=8000, startup_timeout=300)
def vllm_serve():
    """Launch vLLM OpenAI-compatible server with all role LoRAs mounted."""
    import subprocess

    base = os.environ.get("VLLM_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    lora_modules = []
    for role in ROLES:
        adapter_dir = f"/adapters/{role}_lora"
        if os.path.exists(adapter_dir):
            # `name=path` form is the vLLM CLI convention for named LoRAs.
            lora_modules.append(f"{role}={adapter_dir}")
    print(f"Starting vLLM with base={base}, loras={lora_modules}")

    cmd = [
        "vllm", "serve", base,
        "--port", "8000",
        "--host", "0.0.0.0",
        "--enable-lora",
        "--max-loras", "4",
        "--max-lora-rank", "16",
        "--max-model-len", "8192",
    ]
    if lora_modules:
        cmd += ["--lora-modules"] + lora_modules
    subprocess.Popen(cmd)
