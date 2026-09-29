from __future__ import annotations

import os
from pathlib import Path

# 父进程若一直握着同一个 HDF5，DataLoader 子进程再打开会被文件锁堵住。
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def save_cases_h5(path: str | Path, cases: list[dict], split: dict[str, list[str]] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        grp = f.create_group("cases")
        for case in cases:
            g = grp.create_group(case["case_id"])
            g.create_dataset("geom", data=case["geom"], compression="gzip")
            g.create_dataset("fields", data=case["fields"], compression="gzip")
            g.create_dataset("times", data=case["times"])
            g.create_dataset("mask", data=case["mask"], compression="gzip")
            g.attrs["Re"] = case["Re"]
            g.attrs["cx"] = case["cx"]
            g.attrs["cy"] = case["cy"]
            g.attrs["D"] = case["D"]
            g.attrs["shape"] = case["shape"]
            if "forces" in case:
                g.create_dataset("forces", data=case["forces"])
        if split:
            s = f.create_group("splits")
            for name, ids in split.items():
                s.create_dataset(name, data=np.array(ids, dtype="S"))


def close_h5(dataset) -> None:
    handle = getattr(dataset, "_file", None)
    if handle is not None:
        handle.close()
        dataset._file = None


def load_case_ids(path: str | Path, split: str | None = None) -> list[str]:
    with h5py.File(path, "r") as f:
        if split and "splits" in f and split in f["splits"]:
            raw = f["splits"][split][()]
            return [x.decode() if isinstance(x, bytes) else str(x) for x in raw]
        return list(f["cases"].keys())


class FlowSnapshotDataset(Dataset):
    """几何通道 → 单帧 Ux/Uy/p。对应 UNetEx 几何分支。"""

    def __init__(self, h5_path: str | Path, split: str | None = "train", frame: str = "mid"):
        self.h5_path = str(h5_path)
        self.ids = load_case_ids(h5_path, split)
        self.frame = frame
        self._file = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def _h5(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        cid = self.ids[idx]
        g = self._h5()["cases"][cid]
        fields = g["fields"]
        t_index = fields.shape[0] // 2 if self.frame == "mid" else int(self.frame)
        x = torch.from_numpy(np.asarray(g["geom"][()], dtype=np.float32))
        y = torch.from_numpy(np.asarray(fields[t_index], dtype=np.float32))
        mask = torch.from_numpy(np.asarray(g["mask"][()], dtype=np.float32))
        return {"x": x, "y": y, "mask": mask, "case_id": cid, "Re": float(g.attrs["Re"])}


class FlowSequenceDataset(Dataset):
    """几何 + 历史 k 帧 → 未来 t 帧。对应 FNO / 时空 UNet 对照。"""

    def __init__(self, h5_path: str | Path, split: str | None = "train", history_k: int = 4, future_t: int = 1):
        self.h5_path = str(h5_path)
        self.history_k = history_k
        self.future_t = future_t
        self.ids = load_case_ids(h5_path, split)
        self._file = None
        self.samples: list[tuple[str, int]] = []
        with h5py.File(self.h5_path, "r") as f:
            for cid in self.ids:
                n = f["cases"][cid]["fields"].shape[0]
                last = n - history_k - future_t
                if last < 0:
                    continue
                for start in range(last + 1):
                    self.samples.append((cid, start))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def _h5(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        cid, start = self.samples[idx]
        g = self._h5()["cases"][cid]
        fields = g["fields"]
        _, _, height, width = fields.shape
        geom = np.asarray(g["geom"][()], dtype=np.float32)
        hist = np.asarray(fields[start : start + self.history_k], dtype=np.float32).reshape(-1, height, width)
        fut = np.asarray(
            fields[start + self.history_k : start + self.history_k + self.future_t],
            dtype=np.float32,
        )
        y = fut.reshape(-1, height, width)
        x = np.concatenate([geom, hist], axis=0)
        mask = np.asarray(g["mask"][()], dtype=np.float32)
        return {
            "x": torch.from_numpy(x.astype(np.float32)),
            "y": torch.from_numpy(y.astype(np.float32)),
            "mask": torch.from_numpy(mask),
            "case_id": cid,
            "Re": float(g.attrs["Re"]),
        }
