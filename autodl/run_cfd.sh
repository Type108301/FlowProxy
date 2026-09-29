#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
source "$HERE/cfd/env.sh"
OUT="${1:-/root/autodl-tmp/flowproxy/data/raw/cases}"
python cfd/scripts/run_batch.py --out "$OUT"
python cfd/scripts/qc_forces.py "$OUT" --json-out /root/autodl-tmp/flowproxy/outputs/qc.json
python scripts/build_dataset.py --from-vtk "$OUT" --out /root/autodl-tmp/flowproxy/data/processed/dataset.h5
