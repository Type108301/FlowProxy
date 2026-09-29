#!/usr/bin/env bash
# 脱离 SSH 训练 FNO。断线不会停。进度看 outputs/fno_train.log。
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
LOG="$HERE/outputs/fno_train.log"
mkdir -p "$HERE/outputs"
if pgrep -f 'python -u scripts/train.py --model fno' >/dev/null 2>&1; then
  echo "FNO training is already running:" >&2
  pgrep -af 'python -u scripts/train.py --model fno' >&2
  exit 1
fi
RESUME=()
if [ -f "$HERE/outputs/checkpoints/fno_sequence/last.pt" ]; then
  RESUME=(--resume "$HERE/outputs/checkpoints/fno_sequence/last.pt")
fi
setsid nohup python -u scripts/train.py \
  --model fno \
  --task sequence \
  --data /root/autodl-tmp/flowproxy/data/processed/dataset.h5 \
  "${RESUME[@]}" >>"$LOG" 2>&1 </dev/null &
echo "started pid $!"
echo "progress: bash autodl/watch_fno.sh"
