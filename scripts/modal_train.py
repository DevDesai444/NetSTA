"""
Train NetSTA on a GPU via Modal.

Uploads the locally-built real-netlist dataset to a Modal Volume, trains the
model on a cloud GPU, and pulls the checkpoint + results back to
checkpoints_real/. One command does the whole round trip:

    python3 -m modal run scripts/modal_train.py --epochs 250 --num-layers 8

The dataset is built separately and cheaply on CPU (build_real_dataset.py);
only the training itself needs the GPU.
"""

import os
import sys

import modal

GPU = os.environ.get("NETSTA_GPU", "A10G")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("torch_geometric>=2.5", "numpy", "scipy", "scikit-learn")
    .add_local_dir("netsta", "/root/netsta", copy=True)
)

app = modal.App("netsta-train")
vol = modal.Volume.from_name("netsta-train-data", create_if_missing=True)


@app.function(gpu=GPU, image=image, volumes={"/vol": vol}, timeout=60 * 60 * 5)
def train_remote(tasks, epochs, num_layers, batch_size, lr, raw_feature_residual, seed):
    sys.path.insert(0, "/root")
    import torch

    from netsta.real_dataset import (
        InMemoryGraphDataset,
        circuit_level_split,
        load_dataset,
        summarize,
    )
    from netsta.train import train

    print("CUDA:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

    graphs, sources, meta = load_dataset("/vol/graphs.pt")
    print("dataset summary:", summarize(graphs, sources))
    ds = InMemoryGraphDataset(graphs)
    split = circuit_level_split(sources, seed=seed)
    print("split sizes (train/val/test graphs):", [len(s) for s in split])

    ckpt_dir = "/vol/ckpt"
    train(
        dataset=ds, split=split, checkpoint_dir=ckpt_dir,
        epochs=epochs, batch_size=batch_size, lr=lr, num_layers=num_layers,
        tasks=tuple(tasks), raw_feature_residual=raw_feature_residual,
        seed=seed, augment=True, device="auto",
    )
    vol.commit()

    out = {}
    for fn in ("results.json", "best_model.pt", "dataset_stats.json"):
        p = os.path.join(ckpt_dir, fn)
        if os.path.exists(p):
            with open(p, "rb") as f:
                out[fn] = f.read()
    return out


@app.local_entrypoint()
def main(
    epochs: int = 250,
    num_layers: int = 8,
    batch_size: int = 32,
    lr: float = 1e-3,
    data: str = "data_real/graphs.pt",
):
    if not os.path.exists(data):
        raise SystemExit(f"dataset not found: {data} (run build_real_dataset.py first)")

    print(f"Uploading {data} -> volume netsta-train-data:/graphs.pt")
    with vol.batch_upload(force=True) as b:
        b.put_file(data, "graphs.pt")

    tasks = ["slack", "arrival_time", "required_time", "critical_path", "congestion", "drc"]
    out = train_remote.remote(tasks, epochs, num_layers, batch_size, lr, True, 42)

    os.makedirs("checkpoints_real", exist_ok=True)
    for fn, blob in out.items():
        dest = os.path.join("checkpoints_real", fn)
        with open(dest, "wb") as f:
            f.write(blob)
        print(f"wrote {dest} ({len(blob)} bytes)")
    print("done.")
