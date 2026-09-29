from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def autodl_root(cfg: dict[str, Any] | None = None) -> Path | None:
    configured = Path((cfg or {}).get("paths", {}).get("autodl_root", "/root/autodl-tmp/flowproxy"))
    if configured.parent.exists():
        return configured
    return None


def work_root(cfg: dict[str, Any] | None = None) -> Path:
    """AutoDL 数据盘优先，否则使用仓库根目录。"""
    remote = autodl_root(cfg)
    if remote is not None and (remote.exists() or Path("/root/autodl-tmp").exists()):
        remote.mkdir(parents=True, exist_ok=True)
        return remote
    return repo_root()


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else repo_root() / "configs" / "default.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = work_root(cfg)
    cfg["_root"] = str(root)
    cfg["_repo"] = str(repo_root())
    cfg["_config_path"] = str(cfg_path)
    return cfg


def resolve_under_root(cfg: dict[str, Any], key: str) -> Path:
    rel = Path(cfg["paths"][key])
    if rel.is_absolute():
        rel.mkdir(parents=True, exist_ok=True)
        return rel
    path = Path(cfg["_root"]) / rel
    path.mkdir(parents=True, exist_ok=True)
    return path
