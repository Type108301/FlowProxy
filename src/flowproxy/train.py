from __future__ import annotations

import json
from pathlib import Path

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import close_h5
from .losses import channel_weights, physics_bundle, report_errors, supervised_loss
from .metrics import relative_l2
from .models import build_model
from .paths import resolve_under_root


def _format_bar(desc: str, done: int, total: int, elapsed: float, loss: float) -> str:
    frac = done / total if total else 1.0
    width = 20
    filled = min(width, int(width * frac))
    bar = "█" * filled + "░" * (width - filled)
    rate = done / elapsed if elapsed > 0 else 0.0
    remain = (total - done) / rate if rate > 0 else 0.0
    return (
        f"{desc}: {frac * 100:5.1f}%|{bar}| {done}/{total} "
        f"[{int(elapsed // 60):02d}:{int(elapsed % 60):02d}<{int(remain // 60):02d}:{int(remain % 60):02d}, "
        f"{rate:.2f}it/s, loss={loss:.4g}]"
    )


def _write_status(path: Path, text: str) -> None:
    path.write_text(text + "\n", encoding="utf-8")


def loader_kwargs(cfg: dict) -> dict:
    workers = int(cfg["train"].get("num_workers", 0))
    kw: dict = {"num_workers": workers}
    # 模型先上了 CUDA 再 fork DataLoader，子进程会继承 CUDA 上下文并空转。
    if workers > 0:
        kw["multiprocessing_context"] = "spawn"
        kw["persistent_workers"] = True
    return kw


def pick_device(name: str) -> torch.device:
    if name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(name if name != "cuda" else "cuda")
    return torch.device("cpu")


def input_output_channels(task: str, cfg: dict) -> tuple[int, int]:
    if task == "snapshot":
        return 4, 3
    k = int(cfg["data"]["history_k"])
    t = int(cfg["data"]["future_t"])
    return 4 + 3 * k, 3 * t


def evaluate(model, loader, device, dx, dy) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    n = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            mask = batch["mask"].to(device)
            pred = model(x)
            err = report_errors(pred, y, mask)
            rel = relative_l2(pred, y, mask)
            err["rel_l2_mean"] = float(rel.mean())
            for k, v in err.items():
                totals[k] = totals.get(k, 0.0) + v
            n += 1
    return {k: v / max(n, 1) for k, v in totals.items()}


def _checkpoint(
    model,
    opt,
    sched,
    scaler,
    model_name: str,
    task: str,
    in_ch: int,
    out_ch: int,
    cfg: dict,
    epoch: int,
    best: float,
    stale: int,
    history: list,
    val_metrics: dict,
) -> dict:
    payload = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "model_name": model_name,
        "task": task,
        "in_channels": in_ch,
        "out_channels": out_ch,
        "config": {k: cfg[k] for k in cfg if not k.startswith("_")},
        "epoch": epoch,
        "best": best,
        "stale": stale,
        "history": history,
        "val": val_metrics,
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    return payload


def train_loop(
    cfg: dict,
    model_name: str,
    task: str,
    train_set,
    val_set,
    tag: str | None = None,
    resume: Path | None = None,
) -> Path:
    device = pick_device(str(cfg["train"]["device"]))
    in_ch, out_ch = input_output_channels(task, cfg)
    model = build_model(model_name, in_ch, out_ch, cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8, min_lr=1e-6)
    use_amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None
    common = loader_kwargs(cfg)
    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        drop_last=False,
        **common,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        **common,
    )
    domain = cfg["domain"]
    dx = (domain["x_max"] - domain["x_min"]) / max(cfg["grid"]["nx"] - 1, 1)
    dy = (domain["y_max"] - domain["y_min"]) / max(cfg["grid"]["ny"] - 1, 1)
    out_dir = resolve_under_root(cfg, "outputs") / "checkpoints" / (tag or f"{model_name}_{task}")
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir.parent.parent / f"{model_name}_status.txt"

    best = float("inf")
    stale = 0
    history = []
    start_epoch = 1
    if resume is not None and Path(resume).is_file():
        saved = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        # 旧验证分数来自另一份数据。续训换了 h5 时不能拿它当最好值，否则新 best.pt 可能一直不写。
        reset_best = bool(cfg["train"].get("reset_best", False))
        if not reset_best:
            if "optimizer" in saved:
                opt.load_state_dict(saved["optimizer"])
            if "scheduler" in saved:
                sched.load_state_dict(saved["scheduler"])
            if scaler is not None and "scaler" in saved:
                scaler.load_state_dict(saved["scaler"])
            if "best" in saved:
                best = float(saved["best"])
            elif "val" in saved:
                best = float(saved["val"].get("rel_l2_mean", saved["val"].get("mse", best)))
            stale = int(saved.get("stale", 0))
            history = list(saved.get("history", []))
            if "epoch" in saved:
                start_epoch = int(saved["epoch"]) + 1
        print(
            f"resumed {resume} at epoch {start_epoch} best {best:.4f}",
            flush=True,
        )
    weights = None
    probe = train_set[0]["y"].unsqueeze(0)
    weights = channel_weights(probe.to(device))
    close_h5(train_set)
    close_h5(val_set)

    epochs = int(cfg["train"]["epochs"])
    patience = int(cfg["train"]["patience"])
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running = 0.0
        total = len(train_loader)
        desc = f"{model_name} ep{epoch}/{epochs}"
        started = time.perf_counter()
        shown = 0.0
        loss_value = 0.0
        for step, batch in enumerate(train_loader, 1):
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            mask = batch["mask"].to(device)
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", enabled=True):
                    pred = model(x)
                    loss = supervised_loss(pred, y, weights)
                    phys = physics_bundle(
                        pred[:, :3],
                        mask,
                        dx,
                        dy,
                        float(cfg["train"]["lambda_div"]),
                        float(cfg["train"]["lambda_wall"]),
                    )
                    loss = loss + phys["phys"]
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                pred = model(x)
                loss = supervised_loss(pred, y, weights)
                phys = physics_bundle(
                    pred[:, :3],
                    mask,
                    dx,
                    dy,
                    float(cfg["train"]["lambda_div"]),
                    float(cfg["train"]["lambda_wall"]),
                )
                loss = loss + phys["phys"]
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            running += float(loss.detach())
            loss_value = float(loss.detach())
            now = time.perf_counter()
            if step == 1 or step == total or now - shown >= 0.5:
                shown = now
                _write_status(
                    status_path,
                    _format_bar(desc, step, total, now - started, loss_value),
                )
        val_metrics = evaluate(model, val_loader, device, dx, dy)
        val_score = val_metrics.get("rel_l2_mean", val_metrics["mse"])
        sched.step(val_score)
        row = {"epoch": epoch, "train_loss": running / max(len(train_loader), 1), **val_metrics}
        history.append(row)
        print(
            f"[{model_name}] epoch {epoch:03d}  train {row['train_loss']:.4e}  "
            f"val relL2 {val_score:.4f}  ux {val_metrics.get('rel_l2_ux', 0):.4f}  "
            f"uy {val_metrics.get('rel_l2_uy', 0):.4f}  p {val_metrics.get('rel_l2_p', 0):.4f}",
            flush=True,
        )
        if val_score < best:
            best = val_score
            stale = 0
        else:
            stale += 1
        snapshot = _checkpoint(
            model, opt, sched, scaler, model_name, task, in_ch, out_ch, cfg,
            epoch, best, stale, history, val_metrics,
        )
        torch.save(snapshot, out_dir / "last.pt")
        if val_score == best:
            torch.save(snapshot, out_dir / "best.pt")
        if stale >= patience:
            print(f"early stopping at epoch {epoch}", flush=True)
            break

    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return out_dir / "best.pt"
