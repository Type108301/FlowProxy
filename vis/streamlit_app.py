"""简易交互可视化：选算例、看真值/预测/误差。

  streamlit run vis/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flowproxy.dataset import FlowSnapshotDataset, load_case_ids  # noqa: E402
from flowproxy.models import build_model  # noqa: E402
from flowproxy.paths import load_config, resolve_under_root  # noqa: E402


st.set_page_config(page_title="FlowProxy", layout="wide")
st.title("二维流场神经代理 · 真值 / 预测 / 误差")

cfg = load_config()
h5 = st.text_input("数据集 h5", str(resolve_under_root(cfg, "data_processed") / "dataset.h5"))
ckpt = st.text_input("权重", str(resolve_under_root(cfg, "outputs") / "checkpoints/unetex_snapshot/best.pt"))
if not Path(h5).exists():
    st.warning("找不到数据集。先跑 scripts/smoke_test.py 或 build_dataset.py。")
    st.stop()
ids = load_case_ids(h5, None)
cid = st.selectbox("算例", ids)
ds = FlowSnapshotDataset(h5, None)
idx = ds.ids.index(cid)
sample = ds[idx]
pred = None
if Path(ckpt).exists():
    pack = torch.load(ckpt, map_location="cpu")
    model = build_model(pack["model_name"], pack["in_channels"], pack["out_channels"], cfg)
    model.load_state_dict(pack["model"])
    model.eval()
    with torch.no_grad():
        pred = model(sample["x"].unsqueeze(0))[0, :3].numpy()
else:
    st.info("未找到权重，只显示 CFD 真值。")

truth = sample["y"][:3].numpy()
mask = sample["mask"].numpy()
names = ["Ux", "Uy", "p"]
ch = st.selectbox("场量", names)
i = names.index(ch)
cols = st.columns(3)
panels = [("CFD", truth[i])]
if pred is not None:
    panels.append(("NN", pred[i]))
    panels.append(("|err|", np.abs(pred[i] - truth[i])))
for col, (title, img) in zip(cols, panels):
    show = np.where(mask > 0.5, img, np.nan)
    col.subheader(title)
    col.image(np.flipud(show), clamp=True, use_container_width=True)
st.caption(f"Re={sample['Re']}  · 仅供演示，正式动画请用 ParaView / scripts/infer.py --gif")
