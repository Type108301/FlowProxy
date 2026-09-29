#!/usr/bin/env python3
"""在同一批圆柱中间帧上比较续训前后的 UNetEx，并给出三角形留出集的误差。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flowproxy.metrics import relative_l2  # noqa: E402
from flowproxy.models import build_model  # noqa: E402
from flowproxy.paths import load_config  # noqa: E402


def load_model(ckpt: Path, cfg: dict, device: torch.device) -> torch.nn.Module:
    model = build_model("unetex", 4, 3, cfg).to(device)
    saved = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    model.eval()
    return model


def case_scores(model, geom, field, mask, device) -> dict[str, float]:
    x = torch.from_numpy(geom).unsqueeze(0).to(device)
    y = torch.from_numpy(field).unsqueeze(0).to(device)
    m = torch.from_numpy(mask).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x)
        rel = relative_l2(pred, y, m)
    values = rel.detach().cpu().numpy()
    return {
        "ux": float(values[0]),
        "uy": float(values[1]),
        "p": float(values[2]),
        "mean": float(values.mean()),
    }


def summarize(rows: list[dict]) -> dict[str, float]:
    keys = ("ux", "uy", "p", "mean")
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def eval_h5(model, path: Path, case_ids: list[str] | None, device) -> list[dict]:
    rows = []
    with h5py.File(path, "r") as handle:
        ids = case_ids or list(handle["cases"].keys())
        for cid in ids:
            group = handle["cases"][cid]
            fields = group["fields"]
            mid = fields.shape[0] // 2
            scores = case_scores(
                model,
                np.asarray(group["geom"][()], dtype=np.float32),
                np.asarray(fields[mid], dtype=np.float32),
                np.asarray(group["mask"][()], dtype=np.float32),
                device,
            )
            scores["case_id"] = cid
            rows.append(scores)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "finetune_triangle.yaml"))
    parser.add_argument("--old", type=Path, default=ROOT / "outputs/checkpoints/unetex_snapshot/best.pt")
    parser.add_argument("--new", type=Path, default=ROOT / "outputs/checkpoints/unetex_snapshot_triangle/best.pt")
    parser.add_argument("--cylinders", type=Path, default=ROOT / "data/processed/dataset.h5")
    parser.add_argument("--merged", type=Path, default=ROOT / "data/processed/dataset_triangle.h5")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/eval/triangle_resume.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    old = load_model(args.old, cfg, device)
    new = load_model(args.new, cfg, device)
    with h5py.File(args.cylinders, "r") as handle:
        cylinder_ids = list(handle["cases"].keys())
    old_rows = eval_h5(old, args.cylinders, cylinder_ids, device)
    new_rows = eval_h5(new, args.cylinders, cylinder_ids, device)
    report = {
        "cylinder_midframes": {
            "n": len(cylinder_ids),
            "old": summarize(old_rows),
            "new": summarize(new_rows),
        }
    }
    if args.merged.is_file():
        with h5py.File(args.merged, "r") as handle:
            tri_test = [
                x.decode() if isinstance(x, bytes) else str(x)
                for x in handle["splits"]["test"][()]
                if str(x.decode() if isinstance(x, bytes) else x).startswith("triangle_")
            ]
            tri_all = [cid for cid in handle["cases"] if cid.startswith("triangle_")]
        report["triangle_all_midframes"] = {
            "n": len(tri_all),
            "new": summarize(eval_h5(new, args.merged, tri_all, device)),
        }
        report["triangle_test_midframes"] = {
            "n": len(tri_test),
            "ids": tri_test,
            "new": summarize(eval_h5(new, args.merged, tri_test, device)),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    cyl = report["cylinder_midframes"]
    print(
        f"cylinders n={cyl['n']}  "
        f"old mean {cyl['old']['mean']:.4f} (ux {cyl['old']['ux']:.4f} uy {cyl['old']['uy']:.4f} p {cyl['old']['p']:.4f})  "
        f"new mean {cyl['new']['mean']:.4f} (ux {cyl['new']['ux']:.4f} uy {cyl['new']['uy']:.4f} p {cyl['new']['p']:.4f})"
    )
    if "triangle_test_midframes" in report:
        tri = report["triangle_test_midframes"]
        print(
            f"triangle test n={tri['n']}  "
            f"new mean {tri['new']['mean']:.4f} (ux {tri['new']['ux']:.4f} uy {tri['new']['uy']:.4f} p {tri['new']['p']:.4f})"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
