# AutoDL 使用说明（数据生成 + GPU 训练）

本项目本地只负责写代码和冒烟测试。OpenFOAM 批量出数和 PyTorch 训练放到 AutoDL 租的实例上。

推荐：先租 **CPU 够用的 GPU 机**（例如 RTX 4090 / 3090，系统盘 30 GB + 数据盘 ≥50 GB）。CFD 走 CPU，训练走 GPU，不必分开两台。

工作目录一律放到数据盘，避免实例关机丢系统盘：

```text
/root/autodl-tmp/flowproxy
```

## 1. 上传代码

在 AutoDL Jupyter / 终端：

```bash
# 若用 git
cd /root/autodl-tmp
git clone <你的仓库> flowproxy
cd flowproxy

# 或用网盘/scp 把整个 project/ 传到 /root/autodl-tmp/flowproxy
```

学术镜像（PyPI / GitHub 加速）按 AutoDL 控制台提示开启即可。

## 2. 环境

```bash
bash autodl/setup_env.sh
# 另开终端装 OpenFOAM（只需一次，写在系统盘；重装镜像后要再跑）
bash autodl/install_openfoam.sh
```

`setup_env.sh` 会：

- 使用镜像自带的 PyTorch + CUDA
- `pip install -e .` 安装 `flowproxy`
- 把数据/输出目录指到 `/root/autodl-tmp/flowproxy`

## 3. 生成 CFD 数据（CPU）

先跑通 1 个 Re=100 圆柱，确认涡街和 Cd：

```bash
source cfd/env.sh
python cfd/scripts/generate_case.py --re 100 --out /root/autodl-tmp/flowproxy/data/raw/cases
bash /root/autodl-tmp/flowproxy/data/raw/cases/cylinder_Re100_cy0_D1/Allrun
python cfd/scripts/qc_forces.py /root/autodl-tmp/flowproxy/data/raw/cases
```

确认无误后批量出数。2026-09-23 已用这条跑完 36 组，并写出 `dataset.h5`：

```bash
cd /root/autodl-tmp/flowproxy
bash autodl/run_cfd.sh
```

`run_cfd.sh` 会 `source cfd/env.sh`，再跑 `run_batch.py`、`qc_forces.py` 和 `build_dataset.py`。它会删掉已有网格并重跑全部算例，成功后覆盖 `dataset.h5`。没有改物理参数时不要重跑。要缩小矩阵，先改 `configs/default.yaml` 的 `Re_list` / `cy_offsets` / `D_scales`。

单算例粗估：2 万级网格、`endTime=80`，约十几分钟到一小时，视 CPU 而定。可先把 `end_time` 降到 50、`delta_t` 提到 0.025 做第 1 周闭环。

## 4. 训练（GPU）

```bash
python scripts/train.py --model unetex --task snapshot \
    --data /root/autodl-tmp/flowproxy/data/processed/dataset.h5
python scripts/train.py --model fno --task sequence \
    --data /root/autodl-tmp/flowproxy/data/processed/dataset.h5
python scripts/infer.py --ckpt outputs/checkpoints/unetex_snapshot/best.pt
python scripts/compare_models.py
```

日志和权重在 `outputs/`。关机前把 `data/processed`、`outputs/checkpoints` 打包下载或拷到网盘。

## 5. 费用注意

- OpenFOAM 不吃显卡，训练时才满载 GPU。出数阶段可租更便宜的 GPU 或闲时实例。
- `data/raw` 的 VTK 很大，质检通过后可只保留 `dataset.h5`。
- 实例到期系统盘清空；**只认数据盘路径**。
