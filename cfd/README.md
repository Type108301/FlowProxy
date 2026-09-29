# 二维圆柱绕流 OpenFOAM 真值生成

算例由 `cfd/scripts/generate_case.py` 写出完整 `icoFoam` 目录，不手改网格。批跑入口是 `cfd/scripts/run_batch.py`，它会把 `Allrun` 的完整路径传给 bash。

## 物理设置

- 控制方程：二维不可压 NS，`icoFoam`（OpenFOAM 2406）
- 已完成矩阵：Re = 80 / 100 / 150 / 200，`cy` = -0.4 / 0 / 0.4，`D` = 0.9 / 1.0 / 1.1，共 36 组
- 计算域：入口 8D、出口 20D、上下 8D（`x ∈ [-8, 20]`，`y ∈ [-8, 8]`）
- 网格：背景 `blockMesh`（140×80）+ `snappyHexMesh`（`searchableCylinder`，`snap_level=2`）
- 时间：`endTime=80`，`deltaT=0.02`，`writeInterval=0.25`，组 h5 时丢掉启动段 `skip_time=40`
- 输出：`Ux, Uy, p` 经 `foamToVTK` 导出，再插值到 64×64。每例保留 281 帧

`checkMesh` 可能报告非对齐的二维 snappy 边并写 “Failed 1 mesh checks”。这没有挡住求解器，36 组都算到了 t=80。

## 生成过程

2026-09-23 在这台 AutoDL 上出完 36 组，用的命令是：

```bash
cd /root/autodl-tmp/flowproxy
bash autodl/run_cfd.sh
```

它依次做三件事。

1. `source cfd/env.sh`，加载 `/usr/lib/openfoam/openfoam2406`。这一步关掉了 `set -e`，避免 OpenFOAM 的 bashrc 在检测系统库时把脚本直接退出。
2. `python cfd/scripts/run_batch.py` 按 `configs/default.yaml` 展开矩阵，逐个写算例并跑 `Allrun`。CPU 求解，不占 GPU。日志在每个算例的 `log.Allrun`。
3. 全部返回 0 之后，`qc_forces.py` 读 `postProcessing` 里 t≥40 的力系数，写出 `outputs/qc.json`。然后 `scripts/build_dataset.py --from-vtk` 把 VTK 插值成 `dataset.h5`。h5 用写模式打开，所以只有这一步成功结束才会替换旧文件。

单个算例由 `generate_case.py` 写成目录 `data/raw/cases/cylinder_Re{Re}_cy{cy}_D{D}/`，里面是 `0.orig/`、`constant/`、`system/`、`Allrun`。`Allrun` 必须用带路径的 `bash .../Allrun` 调用。它自己做：

```text
清掉旧的 0/、polyMesh、VTK、postProcessing
cp -r 0.orig 0
blockMesh
snappyHexMesh -overwrite
把 0.orig/U 和 0.orig/p 拷回 0/     # snappy 会改 0/，边界条件要恢复
checkMesh | tee log.checkMesh
icoFoam | tee log.icoFoam           # endTime 80，deltaT 0.02
foamToVTK -useTimeName | tee log.foamToVTK
```

`0/U`、`0/p` 的文件头是闭合的 `FoamFile` 字典。横幅注释如果用 `\` 续行且不闭合 `/*`，`icoFoam` 会报 “First token could not be read or is not FoamFile”。

组 h5 时，每个 VTK 内部点被插值到 64×64，乘上流体掩膜，再丢掉 `t < skip_time`（40）的帧。留下 281 帧，通道是 `Ux, Uy, p`。几何通道另算，不从 VTK 来：障碍物 SDF、流体掩膜、壁面距离、`Re/200`。

求解没有随机数。同一套字典重跑，场是一样的。`run_batch.py` 每次都会清网格再算，没有“跳过已完成算例”的开关。

## 当前结果

2026-09-23 全部算完。`outputs/qc.json` 里 36/36 `ok=true`。居中、D=1 的平均阻力系数大约是：

| Re | Cd_mean（cy=0, D=1） |
| --- | --- |
| 80 | 1.28 |
| 100 | 1.20 |
| 150 | 1.13 |
| 200 | 1.14 |

纵向偏移后的 Cd 更高，Re=100 偏移算例大约在 1.26–1.30。质检脚本里的 St 仍是空的，正式报告要另算斯特劳哈尔数。文献对照（Schäfer & Turek, 1996；Re=100 量级）：Cd 约 1.3，St 约 0.16。这只是粗门禁。

求解是确定性的。`bash autodl/run_cfd.sh` 每次都会清掉旧网格、VTK 和时间目录再跑。`dataset.h5` 要等全部算例成功后才整文件重写。

## 重跑

```bash
cd /root/autodl-tmp/flowproxy
source cfd/env.sh
python cfd/scripts/generate_case.py --re 100 --out data/raw/cases
bash data/raw/cases/cylinder_Re100_cy0_D1/Allrun
python cfd/scripts/qc_forces.py data/raw/cases --json-out outputs/qc.json
```

批量：

```bash
bash autodl/run_cfd.sh
```

方柱：把 `configs/default.yaml` 里 `physics.shape` 改成 `square`，再生成算例。现有 36 组圆柱结果会被覆盖。
