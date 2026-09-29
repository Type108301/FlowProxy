#!/usr/bin/env bash
# 在 AutoDL / WSL / 本机 Linux 上探测并加载 OpenFOAM。
set +u
# OpenFOAM bashrc 在可选第三方库（system boost / CGAL 等）上会 return 1。
# 调用方若开了 set -e（run_cfd.sh），这个返回值会让脚本在 echo 之前静默退出。
_flowproxy_errexit=0
case $- in
  *e*) _flowproxy_errexit=1 ;;
esac
set +e
_flowproxy_restore_errexit() {
  if [ "${_flowproxy_errexit:-0}" -eq 1 ]; then
    set -e
  fi
  unset _flowproxy_errexit
}
CANDIDATES=(
  /usr/lib/openfoam/openfoam2406/etc/bashrc
  /usr/lib/openfoam/openfoam2312/etc/bashrc
  /usr/lib/openfoam/openfoam2212/etc/bashrc
  /opt/openfoam11/etc/bashrc
  /opt/openfoam10/etc/bashrc
  /opt/openfoam9/etc/bashrc
  /opt/openfoam-dev/etc/bashrc
)
for f in "${CANDIDATES[@]}"; do
  if [ -f "$f" ]; then
    # shellcheck disable=SC1090
    source "$f"
    echo "OpenFOAM loaded: $WM_PROJECT_DIR ($WM_PROJECT_VERSION)"
    _flowproxy_restore_errexit
    set -u
    return 0 2>/dev/null || exit 0
  fi
done
if command -v icoFoam >/dev/null 2>&1; then
  echo "icoFoam already on PATH"
  _flowproxy_restore_errexit
  set -u
  return 0 2>/dev/null || exit 0
fi
echo "ERROR: OpenFOAM not found. See autodl/install_openfoam.sh" >&2
_flowproxy_restore_errexit
set -u
return 1 2>/dev/null || exit 1
