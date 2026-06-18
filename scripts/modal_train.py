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

GPU = os.environ.get("NETSTA_GPU", "A100-40GB")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("torch_geometric>=2.5", "numpy", "scipy", "scikit-learn")
    # Ignore bytecode so recompilation mid-build doesn't trip the copy hash check.
    .add_local_dir(
        "netsta", "/root/netsta", copy=True,
        ignore=["**/__pycache__", "**/*.pyc", "**/*.pyo"],
    )
)

app = modal.App("netsta-train")
vol = modal.Volume.from_name("netsta-train-data", create_if_missing=True)


@app.function(gpu=GPU, image=image, volumes={"/vol": vol}, timeout=60 * 60 * 5)
def train_remote(tasks, epochs, num_layers, batch_size, lr, raw_feature_residual,
                 seed, split_mode, hidden_channels, backbone_kind):
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
    # circuit: hold out whole source circuits (cross-topology generalization).
    # random: 70/15/15 over all graphs (in-distribution surrogate quality).
    split = circuit_level_split(sources, seed=seed) if split_mode == "circuit" else None
    print(f"split_mode={split_mode} hidden={hidden_channels} layers={num_layers} backbone={backbone_kind}")
    if split is not None:
        print("split sizes (train/val/test graphs):", [len(s) for s in split])

    ckpt_dir = "/vol/ckpt"
    train(
        dataset=ds, split=split, checkpoint_dir=ckpt_dir,
        epochs=epochs, batch_size=batch_size, lr=lr, num_layers=num_layers,
        hidden_channels=hidden_channels,
        tasks=tuple(tasks), raw_feature_residual=raw_feature_residual,
        seed=seed, augment=True, device="auto",
        backbone_kind=backbone_kind,
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
    epochs: int = 100,
    num_layers: int = 8,
    batch_size: int = 16,
    lr: float = 5e-4,
    hidden: int = 64,
    backbone: str = "graphgps_sta",
    split_mode: str = "circuit",
    data: str = "data_real/graphs.pt",
    out_subdir: str = "",
    skip_upload: bool = False,
):
    if not os.path.exists(data):
        raise SystemExit(f"dataset not found: {data} (run build_real_dataset.py first)")

    if not skip_upload:
        print(f"Uploading {data} -> volume netsta-train-data:/graphs.pt")
        with vol.batch_upload(force=True) as b:
            b.put_file(data, "graphs.pt")

    tasks = ["slack", "arrival_time", "required_time", "critical_path", "congestion", "drc"]
    out = train_remote.remote(
        tasks, epochs, num_layers, batch_size, lr, True, 42, split_mode, hidden,
        backbone,
    )

    dest_dir = os.path.join("checkpoints_real", out_subdir) if out_subdir else "checkpoints_real"
    os.makedirs(dest_dir, exist_ok=True)
    for fn, blob in out.items():
        dest = os.path.join(dest_dir, fn)
        with open(dest, "wb") as f:
            f.write(blob)
        print(f"wrote {dest} ({len(blob)} bytes)")
    print("done.")
