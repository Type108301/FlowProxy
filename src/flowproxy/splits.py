from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def default_split(case_ids: list[str], cfg: dict) -> dict[str, list[str]]:
    holdout_re = {float(x) for x in cfg["data"].get("holdout_re", [])}
    hold_geo = set(cfg["data"].get("holdout_geometry", []))
    test, rest = [], []
    for cid in case_ids:
        meta = _meta_from_id(cid)
        geo_key = f"{meta['shape']}_cy{meta['cy']}_D{meta['D']}"
        if meta["Re"] in holdout_re or geo_key in hold_geo:
            test.append(cid)
        else:
            rest.append(cid)
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    rng.shuffle(rest)
    n_train = int(len(rest) * float(cfg["data"]["train_ratio"]))
    n_val = int(len(rest) * float(cfg["data"]["val_ratio"]))
    train = rest[:n_train]
    val = rest[n_train : n_train + n_val]
    leftover = rest[n_train + n_val :]
    test.extend(leftover)
    if not val and train:
        val = [train.pop()]
    if not test and val:
        test = [val[-1]]
    return {"train": train, "val": val, "test": test}


def _meta_from_id(cid: str) -> dict:
    meta = {"shape": "cylinder", "Re": 100.0, "cy": 0.0, "D": 1.0}
    for p in cid.split("_"):
        if p.startswith("Re"):
            try:
                meta["Re"] = float(p[2:])
            except ValueError:
                pass
        elif p.startswith("cy"):
            try:
                meta["cy"] = float(p[2:])
            except ValueError:
                pass
        elif p.startswith("D"):
            try:
                meta["D"] = float(p[1:])
            except ValueError:
                pass
        elif p in {"cylinder", "square", "triangle"}:
            meta["shape"] = p
    return meta


def write_split(path: str | Path, split: dict[str, list[str]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(split, indent=2), encoding="utf-8")
