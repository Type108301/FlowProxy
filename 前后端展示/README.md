# 交互展示

按组会方案里的可视化要求做的前后端：换雷诺数、纵向位置和直径，播放卡门涡街，并排看 CFD 真值、代理预测和误差，同时读阻力/升力。

脚本在数据盘仓库里，不要在 `~` 下直接敲相对路径。从任意目录：

```bash
bash /root/autodl-tmp/flowproxy/前后端展示/run.sh
```

或先进入仓库再跑：

```bash
cd /root/autodl-tmp/flowproxy
bash 前后端展示/run.sh
```

浏览器打开 `http://127.0.0.1:8765`。AutoDL 需映射 8765 端口。

- 产品亮点：非定常时间窗、UNetEx / FNO 分工、OpenFOAM 真值、物理残差、可操作的三视图。
- 交互风洞台：当场调用 `outputs/checkpoints` 里的权重。UNetEx 只吃几何，给出一张不随时间走的快照。FNO 用物理时间上的前 4 帧预测当前帧。
- 双模型分工：全量回放误差，以及 36 个几何上 UNetEx 流向误差矩阵。点一格会切到对应算例。

播放顺序按各算例 `VTK/*.vtm.series` 的物理时间，而不是数据包里的文件名顺序。页面上的汇总误差来自 `vis/figures/compare.json`，单帧误差是当前算例现算的，两套数字不要混读。
