#!/usr/bin/env bash
# 用圆柱 UNetEx 权重，在圆柱+三角形合并数据上续训。不覆盖 unetex_snapshot/best.pt。
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
LOG="$HERE/outputs/triangle_finetune.log"
DATA="$HERE/data/processed/dataset_triangle.h5"
mkdir -p "$HERE/outputs"
if [ ! -f "$DATA" ]; then
  echo "missing $DATA" >&2
  exit 1
fi
if pgrep -f 'scripts/train.py --model unetex --task snapshot' >/dev/null 2>&1; then
  echo "UNetEx snapshot training is already running" >&2
  pgrep -af 'scripts/train.py --model unetex --task snapshot' >&2
  exit 1
fi
setsid nohup python -u scripts/train.py \
  --model unetex \
  --task snapshot \
  --config "$HERE/configs/finetune_triangle.yaml" \
  --data "$DATA" \
  --tag unetex_snapshot_triangle \
  --resume "$HERE/outputs/checkpoints/unetex_snapshot/best.pt" \
  >>"$LOG" 2>&1 </dev/null &
echo "started pid $!"
echo "log: $LOG"
