# FlowProxy · 二维流场神经代理预测与可视化

AIC 算法主题赛（AI+力学）开放式赛题工程仓库。

用 **OpenFOAM** 生成二维圆柱绕流真值，训练 **UNetEx**（几何→场）和 **FNO**（历史→未来）做对照，在有限时间窗内预测 `Ux, Uy, p`，并用三视图 / 动画 / 可选 Streamlit 做演示。

对应组会方案：《组会汇报-AIC-AI+流体力学项目方案》。正式材料不得出现学校、LOGO、导师信息。

工作目录在 AutoDL 数据盘：`/root/autodl-tmp/flowproxy`。当前实例是 RTX 3090，conda Python 3.12，PyTorch 2.8 + CUDA。

## 仓库分支

`main` 只放可复用代码：几何、数据集构建、UNetEx / FNO、OpenFOAM 出数和演示。训练标签不进 `main`，每个人的 label 分开放，避免和组员混在同一次提交里。

个人标签仍放在代码原来读取的路径上，只是活在各自的分支里：

| 分支 | 内容 |
| --- | --- |
| `main` | 可复用代码，不含 h5 |
| `tyx` | `main` 的代码，另加圆柱 `data/processed/dataset.h5`，以及圆柱加三角形 `data/processed/dataset_triangle.h5` |

这两份 h5 走 Git LFS。OpenFOAM 原算例在 `data/raw/cases/`，体积大约 31 GB，不放进仓库。

| 项 | 现状（2026-09-23） |
| --- | --- |
| 赛道 | 开放式 · 流体力学 |
| 求解器 | OpenFOAM 2406，`icoFoam`。Fluent 不作为数据源 |
| 算例 | 36 组圆柱绕流已跑完：Re ∈ {80,100,150,200} × cy ∈ {-0.4,0,0.4} × D ∈ {0.9,1.0,1.1} |
| 数据包 | `data/processed/dataset.h5`（约 403 MB）。每例 `fields` 为 `(281, 3, 64, 64)` |
| 模型 | UNetEx 快照已训满 200 epoch。FNO 时序正在后台训练 |
| 分辨率 | 64×64。128×128 尚未做 |

## 目录

```text
flowproxy/
  cfd/                 OpenFOAM 算例生成、批跑、Cd/Cl 质检
  data/                raw VTK → processed dataset.h5
  src/flowproxy/       几何 SDF、数据集、UNetEx/FNO、训练与插值
  models/              权重放置约定（gitignore，实际在 outputs/checkpoints）
  vis/                 Streamlit 交互
  scripts/             冒烟 / 组数据 / 训练 / 推理 / 对比表
  autodl/              云端装环境、出数、训练、后台看进度
  report/              周日志、引用、成效表模板
  configs/default.yaml 物理、网格、训练超参
```

代码来源（必须引用，见 `report/citations.md`）：

- UNetEx 改编自 [DeepCFD](https://github.com/mdribeiro/DeepCFD)（Ribeiro et al., arXiv:2004.08826, MIT）
- FNO 按 Li et al., ICLR 2021 独立精简实现，思想对齐 [neuraloperator](https://github.com/neuraloperator/neuraloperator)
- CFD 模板对齐本仓库 `openFOAM/` 中的 `icoFoam` 字典风格

## 本机立刻做（不需要 OpenFOAM / GPU）

```bash
cd /root/autodl-tmp/flowproxy
python -m pip install -e .
python scripts/smoke_test.py --epochs 5
```

会写出合成卡门涡街 h5、过拟合 UNetEx 与 FNO，并在 `outputs/smoke/` 保存真值|预测|误差图。这是“网络能跑、对比图能出”的闭环；**合成场不能当正式训练标签**。

可选界面：

```bash
python scripts/build_dataset.py --synthetic
streamlit run vis/streamlit_app.py
```

## 已跑完的两段过程

OpenFOAM 出数（2026-09-23）用的就是：

```bash
cd /root/autodl-tmp/flowproxy
bash autodl/run_cfd.sh
```

它加载 OpenFOAM 2406，按配置写出 36 个算例。每个算例走 `blockMesh` → `snappyHexMesh` → `icoFoam`（到 t=80）→ `foamToVTK`。全部成功后做 Cd/Cl 质检，再把 VTK 插值成 64×64，丢掉 t<40 的帧，写成 `dataset.h5`。这条命令会删掉已有网格并重跑全部算例，最后覆盖 `dataset.h5`。结果是确定性的。没有改物理参数时不要重跑。逐步说明在 [`cfd/README.md`](cfd/README.md)。

UNetEx 训练（同日 17:23 结束）：

```bash
python scripts/train.py --model unetex --task snapshot \
  --data /root/autodl-tmp/flowproxy/data/processed/dataset.h5
```

输入是几何四通道，标签是每个算例的中间一帧 `Ux, Uy, p`。200 epoch，batch 8，AdamW，混合精度，带散度和壁面惩罚。权重和曲线在 `outputs/checkpoints/unetex_snapshot/`。验证均值 relL2 从 1.197 降到 0.374；当时 h5 没有划分，这 36 个算例同时出现在训练和验证里。过程见 [`models/README.md`](models/README.md)。

## AutoDL 出数 + 训练

流程与路径见 [`autodl/README.md`](autodl/README.md)。当前机器上环境、36 组 CFD 和 `dataset.h5` 已经就绪。按顺序是：

```bash
cd /root/autodl-tmp/flowproxy
bash autodl/setup_env.sh
bash autodl/run_cfd.sh             # 36 组 icoFoam，已跑完；再跑会覆盖 dataset.h5
python scripts/train.py --model unetex --task snapshot \
  --data /root/autodl-tmp/flowproxy/data/processed/dataset.h5   # UNetEx 已训完
bash autodl/run_fno_bg.sh          # 已有 last.pt 时从断点续训，且不绑 SSH
bash autodl/watch_fno.sh           # 每个 epoch 一行，当前进度条在最后一行往前走
```

`bash autodl/run_train.sh` 会在前台依次重训 UNetEx 和 FNO。FNO 已经在后台跑时不要再启动它，两份进程会抢同一块 GPU。

输入通道与 DeepCFD 对齐：障碍物 SDF、流体掩膜、壁面距离、来流 Re 图；时序模型再叠 k 帧历史场。FNO 每个样本是长度为 `history_k=4`、预测 `future_t=1` 的滑动窗口，窗口起点覆盖全部时间帧，数值与先读完整序列再切片相同。测试集按**算例**划分，默认留 Re=200 与一种未见几何。

## 指标（方案 2.6，提交前填 `report/metrics_template.md`）

下面是**验证集**中间结果，不是最终测试集，也还没有相对 `icoFoam` 的加速比。

- UNetEx 快照，200 epoch 结束：val relL2 均值 0.374（ux 0.054，uy 0.725，p 0.344）
- FNO 时序，截至 epoch 5：最好 val relL2 均值 0.175（epoch 4；ux 0.027，uy 0.289，p 0.209）。训练仍在继续，目标 200 epoch，早停耐心 40
- 提交目标仍是：测试集速度相对 L2 < 5%，压力 < 8%；至少 2 个涡脱落周期；相对同硬件 OpenFOAM 推理加速 ≥20×；必须有 UNetEx vs FNO 表

## 三人分工（方案 5.4）

| 角色 | 主责 | 本仓库入口 |
| --- | --- | --- |
| A · CFD | 算例、批跑、质检、文献 Cd/St | `cfd/` `autodl/run_cfd.sh` |
| B · 模型 | UNetEx/FNO、误差、对比 | `scripts/train.py` `autodl/run_fno_bg.sh` |
| C · 系统与材料 | 数据管道、可视化、报告、网盘 | `scripts/build_dataset.py` `vis/` `report/` |
