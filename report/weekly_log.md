# 周工作日志（队员 C 维护，每周日更新三行即可）

数字截至 2026-09-23。FNO 仍在训练，下面的 relL2 都是验证集，不是最终测试集。

## 第 1 周 2026-09-15 — 09-21  通闭环

- CFD：环境与 `icoFoam` 算例模板打通。正式 36 组在第 2 周才全部算完
- 模型：本机冒烟可以过拟合合成涡街并出对比图
- 系统/材料：仓库、配置和 AutoDL 脚本就位
- 周会三样东西：CFD 云图 / 过拟合对比图 / 样本清单 → 正式云图与样本清单顺延到 09-23

## 第 2 周 2026-09-22 — 09-28  堆数据 + 基线

- 入库样本数：36 个圆柱算例，命令是 `bash autodl/run_cfd.sh`。内部是 `run_batch.py` 逐个 `Allrun`（blockMesh、snappyHexMesh、icoFoam 到 t=80、foamToVTK），再质检，再把 VTK 插值到 64×64 并丢掉 t<40，得到每例 281 帧。`dataset.h5` 约 403 MB。质检 36/36 通过。居中 D=1 的 Cd 均值：Re80 1.28，Re100 1.20，Re150 1.13，Re200 1.14。St 还没算
- UNetEx：`train.py --model unetex --task snapshot`，几何四通道预测每个算例的中间一帧，200 epoch、batch 8、AMP。h5 没有 splits，36 个算例同时用于训练和验证。验证 rel L2 均值 0.374（ux 0.054，uy 0.725，p 0.344）。权重 `outputs/checkpoints/unetex_snapshot/best.pt`，曲线 `history.json`
- FNO 验证集 rel L2：epoch 4 最好，均值 0.175（ux 0.027，uy 0.289，p 0.209）。09-23 晚从 `last.pt` 续到 epoch 6，后台训练中，目标 200 epoch / 早停 40
- 训练注意：FNO 用 `bash autodl/run_fno_bg.sh` 脱离 SSH，用 `bash autodl/watch_fno.sh` 看进度。不要同时开第二份 `train.py`，也不要重跑 `bash autodl/run_cfd.sh`

## 第 3 周 2026-09-29 — 10-05  变完整

- 时间窗：
- 对比表：
- 演示链路：

## 第 4 周 2026-10-06 — 10-15  交作品

- 技术报告 / 视频 / 网盘：
- 格式自检：
