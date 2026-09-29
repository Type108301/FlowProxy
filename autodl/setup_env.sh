#!/usr/bin/env bash
set -euo pipefail
# AutoDL 镜像通常已带 conda + PyTorch + CUDA。数据写到数据盘。
ROOT="${FLOWPROXY_ROOT:-/root/autodl-tmp/flowproxy}"
if [ ! -f "$ROOT/pyproject.toml" ]; then
  # 脚本若放在仓库 autodl/ 内，回退到仓库根
  HERE="$(cd "$(dirname "$0")/.." && pwd)"
  ROOT="$HERE"
fi
cd "$ROOT"
echo "project root: $ROOT"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), end=" ")
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print("(CPU only)")
PY
# AutoDL 默认的阿里云 PyPI 镜像经常被 IP 黑名单拒绝（403，versions: none）。
# 环境变量优先于 /etc/pip.conf，构建隔离子进程也会继承。
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"
python -m pip install -U pip
python -m pip install -e ".[vtk,app]"
mkdir -p /root/autodl-tmp/flowproxy/data/raw /root/autodl-tmp/flowproxy/data/processed /root/autodl-tmp/flowproxy/outputs
echo "setup done."
echo "next: bash autodl/install_openfoam.sh   # once per image"
echo "      bash autodl/run_cfd.sh"
echo "      bash autodl/run_train.sh"
