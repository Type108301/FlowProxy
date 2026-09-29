#!/usr/bin/env python3
"""批量生成并运行 OpenFOAM 算例。在 AutoDL 上用 CPU 跑，不占用 GPU。"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cfd" / "scripts"))

from generate_case import expand_matrix, write_case  # noqa: E402
from flowproxy.paths import load_config, resolve_under_root  # noqa: E402


def chmod_exec(path: Path) -> None:
    path.chmod(path.stat().st_mode | 0o111)


def run_one(case: Path, timeout: int | None) -> int:
    chmod_exec(case / "Allrun")
    chmod_exec(case / "Allclean")
    log = case / "log.Allrun"
    env = os.environ.copy()
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(
            ["bash", str(case / "Allrun")],
            cwd=case,
            stdout=fh,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            env=env,
        )
    return proc.returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--out", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-cases", type=int, default=0)
    p.add_argument("--timeout", type=int, default=0, help="单算例超时秒数，0 表示不限制")
    args = p.parse_args()
    cfg = load_config(args.config)
    out = Path(args.out) if args.out else resolve_under_root(cfg, "data_raw") / "cases"
    jobs = expand_matrix(cfg)
    if args.max_cases:
        jobs = jobs[: args.max_cases]
    print(f"{len(jobs)} cases -> {out}")
    failed = []
    for i, job in enumerate(jobs, 1):
        case = write_case(cfg, out, **job)
        print(f"[{i}/{len(jobs)}] {case.name}", flush=True)
        if args.dry_run:
            continue
        rc = run_one(case, timeout=args.timeout or None)
        if rc != 0:
            failed.append(case.name)
            print(f"  FAILED rc={rc}  see {case / 'log.Allrun'}", flush=True)
        else:
            print("  ok", flush=True)
    if failed:
        print("failed:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
