#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flowproxy.dataset import FlowSequenceDataset, FlowSnapshotDataset  # noqa: E402
from flowproxy.paths import load_config, resolve_under_root  # noqa: E402
from flowproxy.train import train_loop  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--model", choices=["unetex", "fno", "unet"], default="unetex")
    p.add_argument("--task", choices=["snapshot", "sequence"], default=None)
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--resume", type=Path, default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    task = args.task or ("snapshot" if args.model in {"unetex", "unet"} else "sequence")
    h5 = args.data or (resolve_under_root(cfg, "data_processed") / "dataset.h5")
    if not Path(h5).exists():
        raise SystemExit(f"missing dataset {h5}. Run scripts/build_dataset.py or scripts/smoke_test.py first.")
    if task == "snapshot":
        train_set = FlowSnapshotDataset(h5, "train")
        val_set = FlowSnapshotDataset(h5, "val")
    else:
        k, t = cfg["data"]["history_k"], cfg["data"]["future_t"]
        train_set = FlowSequenceDataset(h5, "train", k, t)
        val_set = FlowSequenceDataset(h5, "val", k, t)
    tag = args.tag or f"{args.model}_{task}"
    ckpt = train_loop(cfg, args.model, task, train_set, val_set, tag=tag, resume=args.resume)
    print(f"best checkpoint -> {ckpt}")


if __name__ == "__main__":
    main()
