#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
DATA="${1:-/root/autodl-tmp/flowproxy/data/processed/dataset.h5}"
python scripts/train.py --model unetex --task snapshot --data "$DATA"
python scripts/train.py --model fno --task sequence --data "$DATA"
python scripts/infer.py --ckpt outputs/checkpoints/unetex_snapshot/best.pt --data "$DATA"
python scripts/infer.py --ckpt outputs/checkpoints/fno_sequence/best.pt --data "$DATA"
python scripts/compare_models.py --data "$DATA"
