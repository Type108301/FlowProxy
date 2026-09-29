from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


CHANNEL_NAMES = ["Ux", "Uy", "p"]


def plot_triple(truth: np.ndarray, pred: np.ndarray, mask: np.ndarray | None, path: str | Path, title: str = "") -> None:
    """真值 | 预测 | 误差 三视图。truth/pred: [3, H, W]"""
    err = np.abs(pred - truth)
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), constrained_layout=True)
    if title:
        fig.suptitle(title)
    for i, name in enumerate(CHANNEL_NAMES[: truth.shape[0]]):
        vmin = np.min(truth[i])
        vmax = np.max(truth[i])
        panels = (truth[i], pred[i], err[i])
        titles = (f"CFD {name}", f"NN {name}", f"|err| {name}")
        for j, (img, ttl) in enumerate(zip(panels, titles)):
            ax = axes[i, j]
            show = img if mask is None else np.where(mask > 0.5, img, np.nan)
            im = ax.imshow(show, origin="lower", cmap="jet", aspect="auto", vmin=None if j == 2 else vmin, vmax=None if j == 2 else vmax)
            ax.set_title(ttl, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_curves(history: list[dict], path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot([h["epoch"] for h in history], [h["train_loss"] for h in history], label="train")
    if "rel_l2_mean" in history[0]:
        ax.plot([h["epoch"] for h in history], [h["rel_l2_mean"] for h in history], label="val relL2")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss / error")
    ax.legend()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def animate_geometry(
    truth: np.ndarray,
    pred: np.ndarray,
    masks: np.ndarray,
    titles: list[str],
    path: str | Path,
    channel: int = 0,
    fps: int = 2,
) -> None:
    """每个几何一帧。truth/pred: [N, 3, H, W]，masks: [N, H, W]。"""
    try:
        from matplotlib.animation import PillowWriter
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("gif 导出需要 pillow") from exc
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    name = CHANNEL_NAMES[channel]
    err = np.abs(pred - truth)
    fluid = masks > 0.5
    truth_c = np.where(fluid[:, None], truth, np.nan)
    pred_c = np.where(fluid[:, None], pred, np.nan)
    err_c = np.where(fluid[:, None], err, np.nan)
    vmin = float(np.nanmin(truth_c[:, channel]))
    vmax = float(np.nanmax(truth_c[:, channel]))
    emax = float(np.nanmax(err_c[:, channel]))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), constrained_layout=True)
    spans = ((vmin, vmax), (vmin, vmax), (0.0, max(emax, 1e-6)))
    labels = (f"CFD {name}", f"UNetEx {name}", f"|err| {name}")
    arrays = (truth_c, pred_c, err_c)
    ims = []
    for ax, data, span, label in zip(axes, arrays, spans, labels):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label, fontsize=10)
        im = ax.imshow(data[0, channel], origin="lower", cmap="jet", aspect="auto", vmin=span[0], vmax=span[1])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ims.append(im)
    writer = PillowWriter(fps=fps)
    with writer.saving(fig, str(path), dpi=100):
        for i, title in enumerate(titles):
            for im, data in zip(ims, arrays):
                im.set_data(data[i, channel])
            fig.suptitle(title, fontsize=11)
            writer.grab_frame()
    plt.close(fig)


def animate_compare(
    truth: np.ndarray,
    pred: np.ndarray,
    path: str | Path,
    mask: np.ndarray | None = None,
    channel: int = 0,
    fps: int = 8,
) -> None:
    """truth/pred: [T, 3, H, W]，写成真值 | 预测 | 误差的 gif。"""
    try:
        from matplotlib.animation import PillowWriter
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("gif 导出需要 pillow") from exc
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    err = np.abs(pred - truth)
    name = CHANNEL_NAMES[channel]
    vmin, vmax = float(np.nanmin(truth[:, channel])), float(np.nanmax(truth[:, channel]))
    emax = float(np.nanmax(err[:, channel]))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), constrained_layout=True)
    panels = []
    for ax, title, span in (
        (axes[0], f"CFD {name}", (vmin, vmax)),
        (axes[1], f"FNO {name}", (vmin, vmax)),
        (axes[2], f"|err| {name}", (0.0, max(emax, 1e-6))),
    ):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10)
        panels.append((ax, span))
    ims = []
    arrays = (truth, pred, err)
    for ax, data, (_, span) in zip(axes, arrays, panels):
        img = data[0, channel]
        if mask is not None:
            img = np.where(mask > 0.5, img, np.nan)
        im = ax.imshow(img, origin="lower", cmap="jet", aspect="auto", vmin=span[0], vmax=span[1])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ims.append(im)
    writer = PillowWriter(fps=fps)
    with writer.saving(fig, str(path), dpi=100):
        for t in range(truth.shape[0]):
            for im, data in zip(ims, arrays):
                img = data[t, channel]
                if mask is not None:
                    img = np.where(mask > 0.5, img, np.nan)
                im.set_data(img)
            axes[0].set_title(f"CFD {name}  t={t}", fontsize=10)
            writer.grab_frame()
    plt.close(fig)


def animate_fields(frames: np.ndarray, path: str | Path, mask: np.ndarray | None = None, channel: int = 0) -> None:
    """frames: [T, 3, H, W] → gif（需要 pillow）。"""
    try:
        from matplotlib.animation import PillowWriter
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("gif 导出需要 pillow") from exc
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    vmin, vmax = np.nanmin(frames[:, channel]), np.nanmax(frames[:, channel])
    img0 = frames[0, channel]
    if mask is not None:
        img0 = np.where(mask > 0.5, img0, np.nan)
    im = ax.imshow(img0, origin="lower", cmap="jet", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)
    writer = PillowWriter(fps=8)
    with writer.saving(fig, str(path), dpi=110):
        for t in range(frames.shape[0]):
            img = frames[t, channel]
            if mask is not None:
                img = np.where(mask > 0.5, img, np.nan)
            im.set_data(img)
            ax.set_title(f"{CHANNEL_NAMES[channel]}  t-index={t}")
            writer.grab_frame()
    plt.close(fig)


@torch.no_grad()
def tensors_to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()
