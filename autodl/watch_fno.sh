#!/usr/bin/env bash
# 只显示正在跑的那一个 epoch。
HERE="$(cd "$(dirname "$0")/.." && pwd)"
STATUS="$HERE/outputs/fno_status.txt"

show_line() {
  python3 - "$STATUS" << 'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8").strip() if path.exists() else ""
match = re.search(
    r"(fno ep\d+/\d+).*?(\d+(?:\.\d+)?)%\|.*?\| (\d+)/(\d+).*?loss=(\S+)",
    text,
)
if match:
    desc, pct_s, done_s, total_s, loss = match.groups()
    pct = float(pct_s)
    width = 12
    filled = min(width, int(width * pct / 100))
    bar = "█" * filled + "░" * (width - filled)
    print(f"{desc} {pct:5.1f}%|{bar}| {done_s}/{total_s} loss={loss.rstrip(']')}")
else:
    print(text or "waiting")
PY
}

if [ "${1:-}" = "--once" ]; then
  show_line
  exit 0
fi

# 不用 watch：它每 0.5 秒清整屏，终端会把这个前台进程杀掉。
trap 'printf "\n"; exit 0' INT TERM
while true; do
  printf '\r\033[2K%s' "$(show_line)"
  sleep 0.5
done
