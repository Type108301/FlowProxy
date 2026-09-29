#!/usr/bin/env bash
set -euo pipefail
# 在 AutoDL Ubuntu 上安装 ESI OpenFOAM（含 icoFoam / snappyHexMesh / foamToVTK）。
if command -v icoFoam >/dev/null 2>&1; then
  echo "icoFoam already installed: $(command -v icoFoam)"
  exit 0
fi
export DEBIAN_FRONTEND=noninteractive
if [ "$(id -u)" -ne 0 ]; then
  SUDO="sudo"
else
  SUDO=""
fi
$SUDO apt-get update
$SUDO apt-get install -y wget ca-certificates gnupg
if wget -q -O - https://dl.openfoam.com/add-debian-repo.sh | $SUDO bash; then
  $SUDO apt-get update
  $SUDO apt-get install -y openfoam2406-default || $SUDO apt-get install -y openfoam2312-default
else
  echo "ESI repo failed, trying Foundation OpenFOAM 11"
  curl -s https://dl.openfoam.org/gpg.key | $SUDO gpg --dearmor -o /usr/share/keyrings/openfoam.gpg
  echo "deb [signed-by=/usr/share/keyrings/openfoam.gpg] http://dl.openfoam.org/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    | $SUDO tee /etc/apt/sources.list.d/openfoam.list
  $SUDO apt-get update
  $SUDO apt-get install -y openfoam11
fi
# persist
if ! grep -q "cfd/env.sh" ~/.bashrc 2>/dev/null; then
  echo "source $(cd "$(dirname "$0")/.." && pwd)/cfd/env.sh" >> ~/.bashrc
fi
bash "$(cd "$(dirname "$0")/.." && pwd)/cfd/env.sh"
echo "OpenFOAM install done."
