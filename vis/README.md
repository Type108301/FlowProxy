# 可视化

- `scripts/infer.py`：用权重画真值 | 预测 | 误差三视图 PNG。序列模型加 `--gif` 时，按真实历史逐帧导出动画。
- `scripts/geometry_gallery.py`：UNetEx 按几何扫描中间帧，写出 `vis/figures/unetex/geometry.gif`。
- `scripts/compare_by_re.py`：居中圆柱（cy=0，D=1），Re 取 80、100、150、200，同一中间时刻。UNetEx 只吃几何，FNO 吃几何加前 4 帧。图写到 `vis/figures/compare/`。
- `vis/streamlit_app.py`：选算例看云图。需要 `pip install -e ".[app]"`，然后 `streamlit run vis/streamlit_app.py`。
- 训练进度不在这里看。FNO 用 `bash autodl/watch_fno.sh`。
- 演示视频仍建议 ParaView 打开 OpenFOAM VTK，与预测图并排。VTK 在 `bash autodl/run_cfd.sh` 写出的 `data/raw/cases/*/VTK`。

```bash
python scripts/infer.py --ckpt outputs/checkpoints/unetex_snapshot/best.pt
python scripts/infer.py --ckpt outputs/checkpoints/fno_sequence/best.pt --gif
python scripts/geometry_gallery.py
python scripts/compare_by_re.py
```
