#!/usr/bin/env python3
"""UNetEx 几何分支：换圆柱位置和直径，给出近似速度场。"""

from __future__ import annotations

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
from flowproxy.paths import load_config, resolve_under_root  # noqa: E402
from flowproxy.train import pick_device  # noqa: E402
from flowproxy.visualize import animate_geometry  # noqa: E402


def main():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    ckpt = resolve_under_root(cfg, "outputs") / "checkpoints" / "unetex_snapshot" / "best.pt"
    h5_path = resolve_under_root(cfg, "data_processed") / "dataset.h5"
    pack = torch.load(ckpt, map_location="cpu", weights_only=False)
    device = pick_device(str(cfg["train"]["device"]))
    model = build_model(pack["model_name"], pack["in_channels"], pack["out_channels"], cfg).to(device)
    model.load_state_dict(pack["model"])
    model.eval()

    rows = []
    with h5py.File(h5_path, "r") as f:
        for cid in f["cases"].keys():
            group = f["cases"][cid]
            fields = np.asarray(group["fields"][()], dtype=np.float32)
            geom = np.asarray(group["geom"][()], dtype=np.float32)
            mask = np.asarray(group["mask"][()], dtype=np.float32)
            truth = fields[fields.shape[0] // 2]
            with torch.no_grad():
                pred = model(torch.from_numpy(geom).unsqueeze(0).to(device))[0].cpu()
            score = relative_l2(pred.unsqueeze(0), torch.from_numpy(truth).unsqueeze(0), torch.from_numpy(mask))
            rows.append(
                {
                    "case_id": cid,
                    "Re": float(group.attrs["Re"]),
                    "cy": float(group.attrs["cy"]),
                    "D": float(group.attrs["D"]),
                    "rel_l2": [float(x) for x in score],
                    "truth": truth,
                    "pred": pred.numpy(),
                    "mask": mask,
                }
            )
    rows.sort(key=lambda r: (r["Re"], r["cy"], r["D"]))
    truth = np.stack([r["truth"] for r in rows])
    pred = np.stack([r["pred"] for r in rows])
    masks = np.stack([r["mask"] for r in rows])
    titles = [
        f"Re={r['Re']:.0f}  cy={r['cy']:+.1f}  D={r['D']:.1f}  relL2={sum(r['rel_l2']) / 3:.3f}"
        for r in rows
    ]
    out = resolve_under_root(cfg, "outputs") / "eval" / "unetex"
    visible = ROOT / "vis" / "figures" / "unetex"
    out.mkdir(parents=True, exist_ok=True)
    visible.mkdir(parents=True, exist_ok=True)
    gif = out / "geometry.gif"
    animate_geometry(truth, pred, masks, titles, gif, channel=0, fps=2)
    summary = [
        {
            "case_id": r["case_id"],
            "Re": r["Re"],
            "cy": r["cy"],
            "D": r["D"],
            "rel_l2_ux": r["rel_l2"][0],
            "rel_l2_uy": r["rel_l2"][1],
            "rel_l2_p": r["rel_l2"][2],
        }
        for r in rows
    ]
    (out / "geometry.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for path in (gif, out / "geometry.json"):
        target = visible / path.name
        target.write_bytes(path.read_bytes())
    mean = np.mean([[r["rel_l2_ux"], r["rel_l2_uy"], r["rel_l2_p"]] for r in summary], axis=0)
    print(f"gif -> {visible / 'geometry.gif'}")
    print(f"cases {len(rows)}  mean ux {mean[0]:.3f}  uy {mean[1]:.3f}  p {mean[2]:.3f}")


if __name__ == "__main__":
    main()
