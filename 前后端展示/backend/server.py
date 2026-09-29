#!/usr/bin/env python3
"""FlowProxy 交互台后端。

读取 OpenFOAM 插值后的 dataset.h5、力系数原文，并调用已训练的
UNetEx / FNO 做当场推理。帧序按 VTK 时间序列重排，保证涡街沿物理时间播放。
"""

from __future__ import annotations

import base64
import gzip
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import h5py
import numpy as np
import torch

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT = APP_ROOT.parent
sys.path.insert(0, str(PROJECT / "src"))

from flowproxy.models import build_model  # noqa: E402
from flowproxy.paths import load_config  # noqa: E402

H5_PATH = PROJECT / "data" / "processed" / "dataset.h5"
CFG_PATH = PROJECT / "configs" / "default.yaml"
UNETEX_CKPT = PROJECT / "outputs" / "checkpoints" / "unetex_snapshot" / "best.pt"
FNO_CKPT = PROJECT / "outputs" / "checkpoints" / "fno_sequence" / "best.pt"
COMPARE_JSON = PROJECT / "vis" / "figures" / "compare.json"
GEOMETRY_JSON = PROJECT / "vis" / "figures" / "unetex" / "geometry.json"
FRONTEND = APP_ROOT / "frontend"
HISTORY_K = 4
STRIDE = 4

_LOCK = threading.Lock()
_STATE: dict = {}


def _b64_u8(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.uint8).tobytes()).decode("ascii")


def _quantize(frames: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    """frames: [T,H,W] 或 [H,W]。流体区定标，固体不参与色标。"""
    single = frames.ndim == 2
    data = frames[None] if single else frames
    fluid = mask > 0.5
    vals = data[:, fluid]
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    span = vmax - vmin
    if span < 1e-8:
        scaled = np.zeros(data.shape, dtype=np.uint8)
        vmax = vmin + 1e-8
    else:
        scaled = np.clip((data - vmin) / span * 255.0, 0, 255).astype(np.uint8)
    return (scaled[0] if single else scaled), vmin, vmax


def _vorticity(ux: np.ndarray, uy: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """ux/uy: [T,H,W]，ω = ∂v/∂x − ∂u/∂y。"""
    w = np.zeros_like(ux)
    dvdx = (uy[:, 1:-1, 2:] - uy[:, 1:-1, :-2]) / (2 * dx)
    dudy = (ux[:, 2:, 1:-1] - ux[:, :-2, 1:-1]) / (2 * dy)
    w[:, 1:-1, 1:-1] = dvdx - dudy
    return w


def _mean_abs_div(ux: np.ndarray, uy: np.ndarray, mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    dudx = (ux[:, 1:-1, 2:] - ux[:, 1:-1, :-2]) / (2 * dx)
    dvdy = (uy[:, 2:, 1:-1] - uy[:, :-2, 1:-1]) / (2 * dy)
    div = dudx + dvdy
    fluid = mask[1:-1, 1:-1] > 0.5
    return np.mean(np.abs(div)[:, fluid], axis=1).astype(np.float64)


def _rel_l2(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> list[float]:
    fluid = mask > 0.5
    out = []
    for c in range(3):
        a = pred[c][fluid]
        b = truth[c][fluid]
        out.append(float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-8)))
    return out


def _solid_speed(pred: np.ndarray, mask: np.ndarray) -> float:
    solid = mask < 0.5
    if not np.any(solid):
        return 0.0
    return float(np.mean(np.hypot(pred[0][solid], pred[1][solid])))


def _foam_time_fallback(n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """没有 VTK 序列时不重排，只给一条 0.5–80 的时间轴，力系数游标还能对上。"""
    return np.arange(n_frames), np.linspace(0.5, 80.0, n_frames, dtype=np.float64)


def _physical_order(case_id: str, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """h5 入库时按文件名排序并丢掉前 40 条。这里改回 VTK 序列里的物理时间。"""
    vtk = PROJECT / "data" / "raw" / "cases" / case_id / "VTK"
    series_path = vtk / f"{case_id}.vtm.series"
    files = sorted(p for p in vtk.glob("**/*.vtu") if "internal" in p.name.lower())
    if not series_path.exists() or len(files) < n_frames:
        return _foam_time_fallback(n_frames)
    kept = files[-n_frames:]
    series = json.loads(series_path.read_text(encoding="utf-8"))
    suf_to_t: dict[int, float] = {}
    for item in series["files"]:
        match = re.search(r"_(\d+)\.vtm$", item["name"])
        if match:
            suf_to_t[int(match.group(1))] = float(item["time"])
    times = []
    for path in kept:
        match = re.search(r"_(\d+)$", path.parent.name)
        if not match or int(match.group(1)) not in suf_to_t:
            return _foam_time_fallback(n_frames)
        times.append(suf_to_t[int(match.group(1))])
    times_arr = np.asarray(times, dtype=np.float64)
    order = np.argsort(times_arr, kind="mergesort")
    return order.astype(np.int64), times_arr[order]


def _read_forces(case_id: str) -> dict[str, list[float]]:
    path = PROJECT / "data" / "raw" / "cases" / case_id / "postProcessing" / "forceCoeffs1" / "0" / "coefficient.dat"
    rows: list[tuple[float, float, float]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line or line.startswith("#"):
                continue
            cols = line.split()
            if len(cols) >= 5:
                rows.append((float(cols[0]), float(cols[1]), float(cols[4])))
    if not rows:
        return {"t": [], "cd": [], "cl": []}
    step = max(1, len(rows) // 240)
    picked = rows[::step]
    return {
        "t": [round(r[0], 3) for r in picked],
        "cd": [round(r[1], 5) for r in picked],
        "cl": [round(r[2], 6) for r in picked],
    }


def _load_model(path: Path, cfg: dict, device: torch.device):
    pack = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(pack["model_name"], pack["in_channels"], pack["out_channels"], cfg).to(device)
    model.load_state_dict(pack["model"])
    model.eval()
    return model


def _round_list(values: np.ndarray, ndigits: int = 5) -> list[float]:
    return [round(float(v), ndigits) for v in values]


class Studio:
    def __init__(self) -> None:
        self.cfg = load_config(CFG_PATH)
        domain = self.cfg["domain"]
        grid = self.cfg["grid"]
        self.dx = (domain["x_max"] - domain["x_min"]) / max(int(grid["nx"]) - 1, 1)
        self.dy = (domain["y_max"] - domain["y_min"]) / max(int(grid["ny"]) - 1, 1)
        self.domain = {
            "x_min": float(domain["x_min"]),
            "x_max": float(domain["x_max"]),
            "y_min": float(domain["y_min"]),
            "y_max": float(domain["y_max"]),
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.unetex = _load_model(UNETEX_CKPT, self.cfg, self.device)
        self.fno = _load_model(FNO_CKPT, self.cfg, self.device)
        self.cache: dict[str, dict] = {}
        with h5py.File(H5_PATH, "r") as handle:
            self.case_ids = list(handle["cases"].keys())
            self.catalog = []
            for cid in self.case_ids:
                attrs = handle["cases"][cid].attrs
                self.catalog.append(
                    {
                        "id": cid,
                        "Re": float(attrs["Re"]),
                        "cy": float(attrs["cy"]),
                        "D": float(attrs["D"]),
                    }
                )
        self.catalog.sort(key=lambda row: (row["Re"], row["cy"], row["D"]))

    def overview(self) -> dict:
        compare = json.loads(COMPARE_JSON.read_text(encoding="utf-8"))
        geometry = json.loads(GEOMETRY_JSON.read_text(encoding="utf-8"))
        by_model = {row["model"]: row for row in compare}
        return {
            "title": "二维流场神经代理预测与可视化",
            "codename": "FlowProxy",
            "device": str(self.device),
            "n_cases": len(self.catalog),
            "grid": [64, 64],
            "domain": self.domain,
            "dx": self.dx,
            "dy": self.dy,
            "metrics": {
                "unetex": by_model["unetex"],
                "fno": by_model["fno"],
            },
            "targets": {"speed_rel": 0.05, "pressure_rel": 0.08, "speedup": 20},
            "geometry": geometry,
            "cases": self.catalog,
            "notes": [
                "指标来自全量算例回放，不是留出 Re=200 的隔离测试集。",
                "相对同硬件 icoFoam 的加速比尚未计时。",
                "播放按 VTK 物理时间重排，保证看到的是涡街而不是文件名顺序。",
            ],
        }

    def bundle(self, case_id: str) -> dict:
        if case_id not in self.case_ids:
            raise KeyError(case_id)
        with _LOCK:
            cached = self.cache.get(case_id)
            if cached is not None:
                return cached
            payload = self._build(case_id)
            self.cache[case_id] = payload
            if len(self.cache) > 8:
                self.cache.pop(next(iter(self.cache)))
            return payload

    @torch.no_grad()
    def _build(self, case_id: str) -> dict:
        with h5py.File(H5_PATH, "r") as handle:
            group = handle["cases"][case_id]
            fields = np.asarray(group["fields"][()], dtype=np.float32)
            geom = np.asarray(group["geom"][()], dtype=np.float32)
            mask = np.asarray(group["mask"][()], dtype=np.float32)
            attrs = {k: group.attrs[k] for k in group.attrs}

        order, times = _physical_order(case_id, fields.shape[0])
        fields = fields[order]
        times = times.astype(np.float64)
        pick = np.arange(0, fields.shape[0], STRIDE)
        # 保证最后一帧进播放，涡街能走到时间窗末端
        if pick[-1] != fields.shape[0] - 1:
            pick = np.concatenate([pick, [fields.shape[0] - 1]])
        seq = fields[pick]
        seq_t = times[pick]
        height, width = mask.shape

        ux, uy, p = seq[:, 0], seq[:, 1], seq[:, 2]
        omega = _vorticity(ux, uy, self.dx, self.dy)
        fluid = mask > 0.5
        omega[:, ~fluid] = 0.0

        cfd_ch = {}
        cfd_scale = {}
        for name, arr in (("ux", ux), ("uy", uy), ("p", p), ("omega", omega)):
            q, vmin, vmax = _quantize(arr, mask)
            cfd_ch[name] = _b64_u8(q)
            cfd_scale[name] = [vmin, vmax]

        sdf_q, sdf_min, sdf_max = _quantize(geom[0], np.ones_like(mask))
        div_cfd = _mean_abs_div(ux, uy, mask, self.dx, self.dy)

        # UNetEx：只吃几何，给出一张不随时间走的快照
        unet_pred = self.unetex(torch.from_numpy(geom).unsqueeze(0).to(self.device))[0, :3].detach().cpu().numpy()
        unet_omega = _vorticity(unet_pred[0][None], unet_pred[1][None], self.dx, self.dy)[0]
        unet_pack = {}
        unet_scale = {}
        for name, arr in (
            ("ux", unet_pred[0]),
            ("uy", unet_pred[1]),
            ("p", unet_pred[2]),
            ("omega", unet_omega),
        ):
            q, vmin, vmax = _quantize(arr, mask)
            unet_pack[name] = _b64_u8(q)
            unet_scale[name] = [vmin, vmax]
        mid = int(np.argmin(np.abs(seq_t - np.median(seq_t))))
        unet_rel_mid = _rel_l2(unet_pred, seq[mid, :3], mask)
        unet_rel_frames = [_rel_l2(unet_pred, frame[:3], mask) for frame in seq]
        unet_wall = _solid_speed(unet_pred, mask)

        # FNO：物理时间上的前 4 帧 → 下一帧。开头不足 4 帧不预测。
        fno_pred = np.zeros_like(seq)
        fno_valid = np.zeros(seq.shape[0], dtype=bool)
        for i in range(seq.shape[0]):
            src = int(pick[i])
            if src < HISTORY_K:
                continue
            hist = fields[src - HISTORY_K : src].reshape(-1, height, width)
            x = np.concatenate([geom, hist], axis=0)
            pred = self.fno(torch.from_numpy(x).unsqueeze(0).to(self.device))[0, :3].detach().cpu().numpy()
            fno_pred[i, :3] = pred
            fno_valid[i] = True
        fno_omega = _vorticity(fno_pred[:, 0], fno_pred[:, 1], self.dx, self.dy)
        fno_omega[~fno_valid] = 0.0
        fno_ch = {}
        fno_scale = {}
        for name, arr in (
            ("ux", fno_pred[:, 0]),
            ("uy", fno_pred[:, 1]),
            ("p", fno_pred[:, 2]),
            ("omega", fno_omega),
        ):
            q, vmin, vmax = _quantize(arr[fno_valid] if np.any(fno_valid) else arr, mask)
            # 无效帧仍占位，色标按有效帧
            span = max(vmax - vmin, 1e-8)
            q_all = np.clip((arr - vmin) / span * 255.0, 0, 255).astype(np.uint8)
            fno_ch[name] = _b64_u8(q_all)
            fno_scale[name] = [vmin, vmax]
        fno_rel = []
        div_fno = np.zeros(seq.shape[0], dtype=np.float64)
        wall_fno = []
        if np.any(fno_valid):
            div_all = _mean_abs_div(fno_pred[:, 0], fno_pred[:, 1], mask, self.dx, self.dy)
        else:
            div_all = div_fno
        for i in range(seq.shape[0]):
            if not fno_valid[i]:
                fno_rel.append(None)
                wall_fno.append(None)
                continue
            fno_rel.append([round(v, 5) for v in _rel_l2(fno_pred[i, :3], seq[i, :3], mask)])
            div_fno[i] = div_all[i]
            wall_fno.append(round(_solid_speed(fno_pred[i, :3], mask), 6))
        valid_rel = [row for row in fno_rel if row is not None]
        fno_rel_mean = (
            [round(float(np.mean([row[c] for row in valid_rel])), 5) for c in range(3)] if valid_rel else [None, None, None]
        )

        return {
            "id": case_id,
            "Re": float(attrs["Re"]),
            "cy": float(attrs["cy"]),
            "D": float(attrs["D"]),
            "cx": float(attrs["cx"]),
            "ny": int(height),
            "nx": int(width),
            "times": [round(float(t), 3) for t in seq_t],
            "mask": _b64_u8((mask > 0.5).astype(np.uint8)),
            "sdf": {"data": _b64_u8(sdf_q), "range": [sdf_min, sdf_max]},
            "cfd": {"channels": cfd_ch, "range": cfd_scale, "div": _round_list(div_cfd, 6)},
            "unetex": {
                "channels": unet_pack,
                "range": unet_scale,
                "rel_mid": [round(v, 5) for v in unet_rel_mid],
                "rel_frames": [[round(v, 5) for v in row] for row in unet_rel_frames],
                "wall": round(unet_wall, 6),
                "anchor_time": round(float(seq_t[mid]), 3),
            },
            "fno": {
                "channels": fno_ch,
                "range": fno_scale,
                "valid": [bool(v) for v in fno_valid],
                "rel_frames": fno_rel,
                "rel_mean": fno_rel_mean,
                "div": _round_list(div_fno, 6),
                "wall": wall_fno,
                "history_k": HISTORY_K,
            },
            "forces": _read_forces(case_id),
            "domain": self.domain,
        }


def _overview_public(studio: Studio) -> dict:
    data = studio.overview()
    # 几何误差表前端用来画矩阵，字段保持紧凑
    data["geometry"] = [
        {
            "id": row["case_id"],
            "Re": row["Re"],
            "cy": row["cy"],
            "D": row["D"],
            "ux": round(row["rel_l2_ux"], 4),
            "uy": round(row["rel_l2_uy"], 4),
            "p": round(row["rel_l2_p"], 4),
        }
        for row in data["geometry"]
    ]
    return data


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s\n" % (fmt % args))

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/api/overview":
                self._send_json(_overview_public(_STATE["studio"]))
                return
            if path == "/api/cases":
                self._send_json({"cases": _STATE["studio"].catalog})
                return
            if path == "/api/case":
                qs = parse_qs(parsed.query)
                case_id = (qs.get("id") or [""])[0]
                self._send_json(_STATE["studio"].bundle(case_id))
                return
            self._send_file(path)
        except KeyError:
            self._send_json({"error": "未知算例"}, status=404)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, status=500)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._send_bytes(raw, "application/json; charset=utf-8", status)

    def _send_file(self, path: str) -> None:
        rel = "index.html" if path in {"", "/"} else path.lstrip("/")
        file_path = (FRONTEND / rel).resolve()
        if not str(file_path).startswith(str(FRONTEND.resolve())) or not file_path.is_file():
            self._send_json({"error": "未找到页面"}, status=404)
            return
        kind = {
            ".html": "text/html; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".svg": "image/svg+xml",
        }.get(file_path.suffix, "application/octet-stream")
        self._send_bytes(file_path.read_bytes(), kind, 200)

    def _send_bytes(self, raw: bytes, content_type: str, status: int) -> None:
        accept = self.headers.get("Accept-Encoding", "")
        if "gzip" in accept and len(raw) > 800:
            raw = gzip.compress(raw, compresslevel=5)
            extra = [("Content-Encoding", "gzip")]
        else:
            extra = []
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-cache")
        for key, value in extra:
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(raw)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    print("加载 UNetEx / FNO …", flush=True)
    studio = Studio()
    _STATE["studio"] = studio
    print(f"预热默认算例，设备 {studio.device}", flush=True)
    studio.bundle("cylinder_Re100_cy0_D1")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"FlowProxy 交互台  http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
