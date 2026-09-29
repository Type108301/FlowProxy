#!/usr/bin/env python3
"""本机冒烟：合成涡街 → UNetEx / FNO 过拟合 → 三视图。不需要 OpenFOAM / GPU。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flowproxy.dataset import FlowSequenceDataset, FlowSnapshotDataset, save_cases_h5  # noqa: E402
from flowproxy.paths import load_config, resolve_under_root  # noqa: E402
from flowproxy.splits import default_split  # noqa: E402
from flowproxy.synthetic import make_synthetic_case  # noqa: E402
from flowproxy.train import train_loop  # noqa: E402
from flowproxy.visualize import plot_triple, tensors_to_numpy  # noqa: E402
from flowproxy.models import build_model  # noqa: E402


def make_toy_h5(cfg, path: Path, n_times: int = 24) -> Path:
    phys = cfg["physics"]
    times = np.linspace(40.0, 70.0, n_times)
    cases = []
    for Re in [80, 100, 150, 200]:
        for cy in [-0.4, 0.0, 0.4]:
            cid = f"cylinder_Re{Re:g}_cy{cy:g}_D{phys['D']}"
            cases.append(
                make_synthetic_case(
                    cfg, cid, Re=Re, times=times, cx=phys["cx"], cy=cy, D=phys["D"], shape="cylinder"
                )
            )
    ids = [c["case_id"] for c in cases]
    split = default_split(ids, cfg)
    save_cases_h5(path, cases, split)
    print(f"synthetic dataset {len(cases)} cases -> {path}")
    return path


def show_checkpoint(cfg, ckpt_path: Path, h5_path: Path, task: str, img_path: Path) -> None:
    pack = torch.load(ckpt_path, map_location="cpu")
    ds = FlowSnapshotDataset(h5_path, "val") if task == "snapshot" else FlowSequenceDataset(
        h5_path, "val", cfg["data"]["history_k"], cfg["data"]["future_t"]
    )
    sample = ds[0]
    model = build_model(pack["model_name"], pack["in_channels"], pack["out_channels"], cfg)
    model.load_state_dict(pack["model"])
    model.eval()
    with torch.no_grad():
        pred = model(sample["x"].unsqueeze(0))[0]
    y = sample["y"]
    if y.shape[0] > 3:
        y, pred = y[:3], pred[:3]
    plot_triple(
        tensors_to_numpy(y),
        tensors_to_numpy(pred),
        tensors_to_numpy(sample["mask"]),
        img_path,
        title=f"{pack['model_name']} {task}  Re={sample['Re']}",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--skip-train", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    cfg["train"]["epochs"] = args.epochs
    cfg["train"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["train"]["batch_size"] = 4
    cfg["train"]["patience"] = args.epochs
    cfg["train"]["amp"] = False
    cfg["train"]["num_workers"] = 0
    out = resolve_under_root(cfg, "outputs") / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    h5 = out / "synthetic.h5"
    make_toy_h5(cfg, h5)
    if args.skip_train:
        return
    snap_tr = FlowSnapshotDataset(h5, "train")
    snap_va = FlowSnapshotDataset(h5, "val")
    ck_u = train_loop(cfg, "unetex", "snapshot", snap_tr, snap_va, tag="smoke_unetex")
    seq_tr = FlowSequenceDataset(h5, "train", cfg["data"]["history_k"], cfg["data"]["future_t"])
    seq_va = FlowSequenceDataset(h5, "val", cfg["data"]["history_k"], cfg["data"]["future_t"])
    ck_f = train_loop(cfg, "fno", "sequence", seq_tr, seq_va, tag="smoke_fno")
    show_checkpoint(cfg, ck_u, h5, "snapshot", out / "unetex_triple.png")
    show_checkpoint(cfg, ck_f, h5, "sequence", out / "fno_triple.png")
    print(f"smoke plots -> {out}")


if __name__ == "__main__":
    main()
