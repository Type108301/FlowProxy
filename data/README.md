# 规则网格数据包

- `raw/cases/`：OpenFOAM 算例与 VTK。体积大约 31 GB，不进 git
- `processed/dataset.h5`：36 组圆柱，插值后约 403 MB，2026-09-23 写出。在 `tyx` 分支
- `processed/dataset_triangle.h5`：上述 36 组圆柱，加上 36 组等边三角形，约 809 MB。在 `tyx` 分支

`main` 不含这些 h5。`tyx` 分支里只有这两份文件，没有代码，用 Git LFS 存放。`processed/triangle_cases/*.npz` 已并进 `dataset_triangle.h5`，不另存一份。

h5 结构是 `cases/<case_id>/`：

| 数据集 | 形状 | 含义 |
| --- | --- | --- |
| `fields` | `(281, 3, 64, 64)` float32 | 时间 × `Ux, Uy, p` |
| `times` | `(281,)` | 对应物理时间 |
| `geom` | `(4, 64, 64)` | SDF、掩膜、壁面距离、Re 图 |
| `mask` | `(64, 64)` | 流体掩膜 |
| `forces` | `(400, 3)` | 力系数时间序列 |

这批正式数据由下面这条命令生成。它会重跑 OpenFOAM 并覆盖 `dataset.h5`：

```bash
cd /root/autodl-tmp/flowproxy
bash autodl/run_cfd.sh
```

字段从 OpenFOAM 来的过程：`Allrun` 里的 `foamToVTK` 写出体网格上的 `U` 和 `p`；`build_dataset.py` 把这些点插值到 64×64，乘流体掩膜，丢掉 t<40，再和另外算好的几何通道一起写入 h5。几何通道不来自 VTK。

构建（会覆盖已有 h5）：

```bash
python scripts/build_dataset.py --from-vtk data/raw/cases \
    --out data/processed/dataset.h5
```

本机没有 OpenFOAM、只想看训练闭环时：

```bash
python scripts/build_dataset.py --synthetic
```

合成场不能当正式标签。

## 训练时怎么读

快照任务读一帧。时序任务读 `history_k + future_t` 帧（默认 4+1），窗口起点仍扫过全部合法时间，不会丢掉后半段序列。数值与先把 281 帧解压出来再切片相同。DataLoader 用 `spawn`，打开 h5 前设置 `HDF5_USE_FILE_LOCKING=FALSE`。
