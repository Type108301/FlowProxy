#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flowproxy.dataset import FlowSequenceDataset, FlowSnapshotDataset  # noqa: E402
from flowproxy.models import build_model  # noqa: E402
from flowproxy.paths import load_config, resolve_under_root  # noqa: E402
from flowproxy.train import evaluate, pick_device  # noqa: E402
from flowproxy.visualize import animate_compare, plot_triple, tensors_to_numpy  # noqa: E402


@torch.no_grad()
def write_sequence_gif(model, ds, device, path: Path, stride: int = 2) -> Path:
    """future_t=1 时按真实历史逐帧预测，把一个算例串成动画。"""
    import numpy as np

    cid = ds.samples[0][0]
    group = ds._h5()["cases"][cid]
    fields = np.asarray(group["fields"][()], dtype=np.float32)
    geom = np.asarray(group["geom"][()], dtype=np.float32)
    mask = np.asarray(group["mask"][()], dtype=np.float32)
    k = ds.history_k
    height, width = fields.shape[-2], fields.shape[-1]
    truths, preds = [], []
    model.eval()
    for t in range(k, fields.shape[0], stride):
        hist = fields[t - k : t].reshape(-1, height, width)
        x = np.concatenate([geom, hist], axis=0)
        pred = model(torch.from_numpy(x).unsqueeze(0).to(device))[0, :3].cpu().numpy()
        truths.append(fields[t])
        preds.append(pred)
    animate_compare(np.stack(truths), np.stack(preds), path, mask=mask, channel=0)
    print(f"gif case {cid} frames {len(truths)}")
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--gif", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    pack = torch.load(args.ckpt, map_location="cpu")
    h5 = args.data or (resolve_under_root(cfg, "data_processed") / "dataset.h5")
    task = pack["task"]
    if task == "snapshot":
        ds = FlowSnapshotDataset(h5, args.split)
    else:
        ds = FlowSequenceDataset(h5, args.split, cfg["data"]["history_k"], cfg["data"]["future_t"])
    device = pick_device(str(cfg["train"]["device"]))
    model = build_model(pack["model_name"], pack["in_channels"], pack["out_channels"], cfg).to(device)
    model.load_state_dict(pack["model"])
    loader = DataLoader(ds, batch_size=int(cfg["train"]["batch_size"]))
    domain = cfg["domain"]
    dx = (domain["x_max"] - domain["x_min"]) / max(cfg["grid"]["nx"] - 1, 1)
    dy = (domain["y_max"] - domain["y_min"]) / max(cfg["grid"]["ny"] - 1, 1)
    metrics = evaluate(model, loader, device, dx, dy)
    out = resolve_under_root(cfg, "outputs") / "eval" / pack["model_name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    n_show = min(int(cfg["vis"]["n_show"]), len(ds))
    model.eval()
    with torch.no_grad():
        for i in range(n_show):
            sample = ds[i]
            pred = model(sample["x"].unsqueeze(0).to(device))[0].cpu()
            y = sample["y"]
            if y.shape[0] > 3:
                y, pred = y[:3], pred[:3]
            plot_triple(
                tensors_to_numpy(y),
                tensors_to_numpy(pred),
                tensors_to_numpy(sample["mask"]),
                out / f"triple_{i}.png",
                title=f"{sample['case_id']} Re={sample['Re']}",
            )
    if args.gif and task == "sequence" and len(ds) > 0:
        gif_path = write_sequence_gif(model, ds, device, out / "pred.gif")
        print(f"gif -> {gif_path}")


if __name__ == "__main__":
    main()
