#!/usr/bin/env python3
"""同一测试集上对比 UNetEx 与 FNO，写出误差表（组会方案要求的对照实验）。"""

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


def eval_ckpt(cfg, ckpt: Path, h5: Path) -> dict:
    pack = torch.load(ckpt, map_location="cpu")
    task = pack["task"]
    ds = (
        FlowSnapshotDataset(h5, "test")
        if task == "snapshot"
        else FlowSequenceDataset(h5, "test", cfg["data"]["history_k"], cfg["data"]["future_t"])
    )
    device = pick_device(str(cfg["train"]["device"]))
    model = build_model(pack["model_name"], pack["in_channels"], pack["out_channels"], cfg).to(device)
    model.load_state_dict(pack["model"])
    loader = DataLoader(ds, batch_size=int(cfg["train"]["batch_size"]))
    domain = cfg["domain"]
    dx = (domain["x_max"] - domain["x_min"]) / max(cfg["grid"]["nx"] - 1, 1)
    dy = (domain["y_max"] - domain["y_min"]) / max(cfg["grid"]["ny"] - 1, 1)
    metrics = evaluate(model, loader, device, dx, dy)
    metrics["model"] = pack["model_name"]
    metrics["task"] = task
    metrics["n_test"] = len(ds)
    metrics["ckpt"] = str(ckpt)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--unetex", type=Path, default=None)
    p.add_argument("--fno", type=Path, default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    h5 = args.data or (resolve_under_root(cfg, "data_processed") / "dataset.h5")
    out_root = resolve_under_root(cfg, "outputs")
    unetex = args.unetex or out_root / "checkpoints" / "unetex_snapshot" / "best.pt"
    fno = args.fno or out_root / "checkpoints" / "fno_sequence" / "best.pt"
    rows = []
    for ckpt in (unetex, fno):
        if Path(ckpt).exists():
            rows.append(eval_ckpt(cfg, Path(ckpt), h5))
        else:
            print(f"missing {ckpt}")
    table = out_root / "eval" / "compare.json"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"{'model':10s} {'task':10s} {'relL2':8s} {'ux':8s} {'uy':8s} {'p':8s} n")
    for r in rows:
        print(
            f"{r['model']:10s} {r['task']:10s} {r.get('rel_l2_mean', 0):8.4f} "
            f"{r.get('rel_l2_ux', 0):8.4f} {r.get('rel_l2_uy', 0):8.4f} "
            f"{r.get('rel_l2_p', 0):8.4f} {r['n_test']}"
        )
    print(f"wrote {table}")


if __name__ == "__main__":
    main()
