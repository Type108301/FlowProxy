# 模型权重

## UNetEx 训练过程

2026-09-23 下午用正式 `dataset.h5` 训完，命令是：

```bash
cd /root/autodl-tmp/flowproxy
python scripts/train.py --model unetex --task snapshot \
  --data /root/autodl-tmp/flowproxy/data/processed/dataset.h5
```

`run_train.sh` 里的第一段就是这条。它在前台跑，当时没有脱离 SSH。17:23 结束，没有 `--resume`。训练读的 `dataset.h5` 来自 `bash autodl/run_cfd.sh`。

数据是 `FlowSnapshotDataset`：每个算例只取时间序列正中间那一帧当标签，输入是 4 通道几何图（障碍物 SDF、流体掩膜、壁面距离、`Re/200`），输出是该帧的 `Ux, Uy, p`。当前 h5 **没有写入** `splits`，所以训练和验证都读到了全部 36 个算例。验证 relL2 不是留出集上的数字。按算例留出 Re=200 和 `cylinder_cy0.4_D1.0` 要在重新 `build_dataset` 时把划分写进 h5 之后才生效。

网络是 DeepCFD 的 UNetEx：三个输出场各有一个 decoder，共用 encoder。`configs/default.yaml` 里 `filters=[8,16,32,32]`，卷积核 5，每个 block 2 层，不用 batch norm，也不用 weight norm。优化器 AdamW（lr `1e-3`，weight decay `5e-3`），`ReduceLROnPlateau`（factor 0.5，patience 8）。batch size 8，所以一个 epoch 大约 5 个 step。混合精度打开。损失是按通道 RMS 加权的均方误差，再加上散度残差（权重 0.05）和物体内速度惩罚（权重 0.02）。早停耐心 40，这次一直到第 200 个 epoch 仍有改进，没有提前停。

验证 relL2 均值从 epoch 1 的 1.197 降到 epoch 200 的 0.374。分通道：ux 0.054，uy 0.725，p 0.344。压力和横向速度还明显高于方案里的测试集目标。曲线在 `outputs/checkpoints/unetex_snapshot/history.json`，权重是同目录的 `best.pt`（约 2.3 MB）。这次的 `best.pt` 没有优化器状态；FNO 后来改成每个 epoch 另存带优化器的 `last.pt`，UNetEx 那次没有这份文件。

不要再跑 `bash autodl/run_train.sh`。它会从头重训 UNetEx，然后接着在前台训 FNO，和正在跑的后台 FNO 抢 GPU。

网络定义在 `src/flowproxy/models/`：

- `unetex.py`：DeepCFD 的 UNetEx，几何 → 一场
- `fno.py`：二维傅里叶神经算子，历史场 → 下一场。谱权重存成实数参数，AMP 里在 float32 做 FFT
- `unet.py`：普通 U-Net 对照，还没正式训练

训练写出的文件在 `outputs/checkpoints/<tag>/`：

```text
outputs/checkpoints/unetex_snapshot/best.pt    # 已完成，200 epoch，2026-09-23
outputs/checkpoints/fno_sequence/best.pt       # 验证 relL2 变好才更新
outputs/checkpoints/fno_sequence/last.pt       # 每个 epoch 都写，用来 --resume
```

`last.pt` 含模型、优化器、调度器、GradScaler、epoch、best、history。续训：

```bash
python scripts/train.py --model fno --task sequence \
  --data data/processed/dataset.h5 \
  --resume outputs/checkpoints/fno_sequence/last.pt
```

后台方式见 `autodl/README.md`。提交网盘时再把最终权重复制到本目录，例如 `unetex_snapshot.pt`、`fno_sequence.pt`。不要提交中间 checkpoint 和 toy 过拟合权重。
