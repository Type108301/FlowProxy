#!/usr/bin/env python3
"""同一居中圆柱、同一时刻，按 Re 画出 UNetEx 与 FNO 对 CFD 的对比图。"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flowproxy.metrics import relative_l2  # noqa: E402
from flowproxy.models import build_model  # noqa: E402
from flowproxy.paths import load_config, resolve_under_root  # noqa: E402
from flowproxy.train import pick_device  # noqa: E402
from flowproxy.visualize import plot_triple  # noqa: E402

RES = (80, 100, 150, 200)


def load_model(cfg, ckpt: Path, device):
    pack = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = build_model(pack["model_name"], pack["in_channels"], pack["out_channels"], cfg).to(device)
    model.load_state_dict(pack["model"])
    model.eval()
    return model


def show(img, mask):
    return np.where(mask > 0.5, img, np.nan)


def sheet(rows, path: Path, pred_name: str) -> None:
    """每个 Re 一行：CFD | 预测 | 误差，通道为 Ux。"""
    fig, axes = plt.subplots(len(rows), 3, figsize=(10, 2.6 * len(rows)), constrained_layout=True)
    fig.suptitle(f"{pred_name}  cy=0  D=1  Ux")
    for ax_row, row in zip(axes, rows):
        truth, pred, mask = row["truth"], row["pred"], row["mask"]
        err = np.abs(pred - truth)
        vmin, vmax = float(np.min(truth[0])), float(np.max(truth[0]))
        panels = (
            (truth[0], f"CFD  Re={row['Re']:.0f}", vmin, vmax),
            (pred[0], f"{pred_name}  relL2 {row['mean']:.3f}", vmin, vmax),
            (err[0], f"|err|  ux {row['ux']:.3f}  uy {row['uy']:.3f}  p {row['p']:.3f}", None, None),
        )
        for ax, (img, title, lo, hi) in zip(ax_row, panels):
            im = ax.imshow(show(img, mask), origin="lower", cmap="jet", aspect="auto", vmin=lo, vmax=hi)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    device = pick_device(str(cfg["train"]["device"]))
    out_root = resolve_under_root(cfg, "outputs") / "checkpoints"
    unet = load_model(cfg, out_root / "unetex_snapshot" / "best.pt", device)
    fno = load_model(cfg, out_root / "fno_sequence" / "best.pt", device)
    h5_path = resolve_under_root(cfg, "data_processed") / "dataset.h5"
    visible = ROOT / "vis" / "figures" / "compare"
    hidden = resolve_under_root(cfg, "outputs") / "eval" / "compare_re"
    k = int(cfg["data"]["history_k"])
    by_model = {"UNetEx": [], "FNO": []}

    with h5py.File(h5_path, "r") as f:
        for re in RES:
            group = f["cases"][f"cylinder_Re{re}_cy0_D1"]
            fields = np.asarray(group["fields"][()], dtype=np.float32)
            geom = np.asarray(group["geom"][()], dtype=np.float32)
            mask = np.asarray(group["mask"][()], dtype=np.float32)
            t = fields.shape[0] // 2
            truth_t = torch.from_numpy(fields[t])
            mask_t = torch.from_numpy(mask)
            with torch.no_grad():
                pred_u = unet(torch.from_numpy(geom).unsqueeze(0).to(device))[0].cpu()
                hist = fields[t - k : t].reshape(-1, *fields.shape[-2:])
                x = np.concatenate([geom, hist], axis=0)
                pred_f = fno(torch.from_numpy(x).unsqueeze(0).to(device))[0, :3].cpu()
            for name, pred in (("UNetEx", pred_u), ("FNO", pred_f)):
                score = relative_l2(pred.unsqueeze(0), truth_t.unsqueeze(0), mask_t).numpy()
                truth = fields[t]
                pred_np = pred.numpy()
                row = {
                    "Re": re,
                    "truth": truth,
                    "pred": pred_np,
                    "mask": mask,
                    "ux": float(score[0]),
                    "uy": float(score[1]),
                    "p": float(score[2]),
                    "mean": float(score.mean()),
                }
                by_model[name].append(row)
                title = (
                    f"{name}  Re={re}  cy=0  D=1  "
                    f"ux {row['ux']:.3f}  uy {row['uy']:.3f}  p {row['p']:.3f}"
                )
                for folder in (visible / name.lower(), hidden / name.lower()):
                    plot_triple(truth, pred_np, mask, folder / f"Re{re}.png", title=title)
                print(title)

    for name, rows in by_model.items():
        for folder in (visible / name.lower(), hidden / name.lower()):
            sheet(rows, folder / "by_re_ux.png", name)
            print(f"sheet -> {folder / 'by_re_ux.png'}")


if __name__ == "__main__":
    main()
