const RES = ["80", "100", "150", "200"];
const CYS = [
  { v: -0.4, label: "−0.4" },
  { v: 0, label: "0" },
  { v: 0.4, label: "+0.4" },
];
const DS = [
  { v: 0.9, label: "0.9" },
  { v: 1, label: "1.0" },
  { v: 1.1, label: "1.1" },
];
const FIELDS = [
  { id: "ux", label: "Ux" },
  { id: "uy", label: "Uy" },
  { id: "p", label: "p" },
  { id: "omega", label: "涡量" },
];
const MODES = [
  { id: "truth", label: "真值回放" },
  { id: "unetex", label: "UNetEx 快照" },
  { id: "fno", label: "FNO 时序" },
];

const state = {
  page: "highlights",
  overview: null,
  bundle: null,
  re: 100,
  cy: 0,
  d: 1,
  field: "ux",
  mode: "fno",
  stream: true,
  frame: 0,
  playing: true,
  heroFrame: 0,
  heatMetric: "uy",
  compareView: "pred",
  compareField: "uy",
};

const decoded = new Map();

function b64(s) {
  const bin = atob(s);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) out[i] = bin.charCodeAt(i);
  return out;
}

function caseId(re, cy, d) {
  const cyLabel = cy === 0 ? "0" : cy.toFixed(1);
  const dLabel = d === 1 ? "1" : d.toFixed(1);
  return `cylinder_Re${re}_cy${cyLabel}_D${dLabel}`;
}

function pct(x) {
  if (x == null || Number.isNaN(x)) return "—";
  return `${(x * 100).toFixed(2)}%`;
}

function jet(t) {
  const u = Math.min(1, Math.max(0, t));
  const r = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * u - 3)));
  const g = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * u - 2)));
  const b = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * u - 1)));
  return [r * 255, g * 255, b * 255];
}

function ember(t) {
  const u = Math.min(1, Math.max(0, t));
  return [
    Math.min(255, u * 2.4 * 255),
    Math.min(255, Math.max(0, u - 0.28) * 1.7 * 255),
    Math.min(255, Math.max(0, u - 0.72) * 2.4 * 180),
  ];
}

function coolwarm(value, vmin, vmax) {
  const neg = Math.min(vmin, 0);
  const pos = Math.max(vmax, 0);
  let t = 0.5;
  if (value >= 0) t = 0.5 + 0.5 * (pos === 0 ? 0 : Math.min(1, value / pos));
  else t = 0.5 - 0.5 * (neg === 0 ? 0 : Math.min(1, value / neg));
  const r = t < 0.5 ? 40 + t * 2 * 200 : 255;
  const g = t < 0.5 ? 90 + t * 2 * 150 : 255 - (t - 0.5) * 2 * 170;
  const b = t < 0.5 ? 220 : 220 - (t - 0.5) * 2 * 180;
  return [r, g, b];
}

function sample(u8, t, row, col, h, w) {
  return u8[t * h * w + row * w + col];
}

function dequant(byte, range) {
  return range[0] + (byte / 255) * (range[1] - range[0]);
}

function ensureDecoded(bundle) {
  if (decoded.has(bundle.id)) return decoded.get(bundle.id);
  const pack = {
    mask: b64(bundle.mask),
    cfd: {},
    fno: {},
    unetex: {},
    sdf: b64(bundle.sdf.data),
  };
  for (const key of ["ux", "uy", "p", "omega"]) {
    pack.cfd[key] = b64(bundle.cfd.channels[key]);
    pack.fno[key] = b64(bundle.fno.channels[key]);
    pack.unetex[key] = b64(bundle.unetex.channels[key]);
  }
  decoded.set(bundle.id, pack);
  if (decoded.size > 3) decoded.delete(decoded.keys().next().value);
  return pack;
}

function paintField(canvas, u8, t, range, mask, h, w, opts) {
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(w, h);
  const data = img.data;
  const diverging = opts.diverging;
  const err = opts.err;
  for (let row = 0; row < h; row += 1) {
    for (let col = 0; col < w; col += 1) {
      const src = (h - 1 - row) * w + col;
      const dst = (row * w + col) * 4;
      if (mask[src] < 1 && !opts.ignoreMask) {
        data[dst] = 12;
        data[dst + 1] = 16;
        data[dst + 2] = 14;
        data[dst + 3] = 255;
        continue;
      }
      const byte = u8[t * h * w + src];
      const value = dequant(byte, range);
      let rgb;
      if (err) rgb = ember(byte / 255);
      else if (diverging) rgb = coolwarm(value, range[0], range[1]);
      else rgb = jet(byte / 255);
      data[dst] = rgb[0];
      data[dst + 1] = rgb[1];
      data[dst + 2] = rgb[2];
      data[dst + 3] = 255;
    }
  }
  const off = document.createElement("canvas");
  off.width = w;
  off.height = h;
  off.getContext("2d").putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
  if (opts.stream && opts.ux && opts.uy) {
    drawStreamlines(ctx, opts.ux, opts.uy, opts.uxRange, opts.uyRange, mask, t, h, w, canvas.width, canvas.height);
  }
}

function drawStreamlines(ctx, ux, uy, xr, yr, mask, t, h, w, cw, ch) {
  ctx.save();
  ctx.strokeStyle = "rgba(255,255,255,0.55)";
  ctx.lineWidth = 1.1;
  const seeds = [];
  for (let row = 6; row < h; row += 8) {
    for (let col = 4; col < w; col += 10) {
      if (mask[row * w + col] > 0) seeds.push([col, row]);
    }
  }
  const speedAt = (col, row) => {
    const c = Math.max(0, Math.min(w - 1, col));
    const r = Math.max(0, Math.min(h - 1, row));
    const c0 = Math.floor(c);
    const r0 = Math.floor(r);
    const c1 = Math.min(w - 1, c0 + 1);
    const r1 = Math.min(h - 1, r0 + 1);
    const tx = c - c0;
    const ty = r - r0;
    const grab = (arr, range, rr, cc) => dequant(sample(arr, t, rr, cc, h, w), range);
    const u00 = grab(ux, xr, r0, c0);
    const u10 = grab(ux, xr, r0, c1);
    const u01 = grab(ux, xr, r1, c0);
    const u11 = grab(ux, xr, r1, c1);
    const v00 = grab(uy, yr, r0, c0);
    const v10 = grab(uy, yr, r0, c1);
    const v01 = grab(uy, yr, r1, c0);
    const v11 = grab(uy, yr, r1, c1);
    const u = (u00 * (1 - tx) + u10 * tx) * (1 - ty) + (u01 * (1 - tx) + u11 * tx) * ty;
    const v = (v00 * (1 - tx) + v10 * tx) * (1 - ty) + (v01 * (1 - tx) + v11 * tx) * ty;
    return [u, v];
  };
  const toPx = (col, row) => [(col / (w - 1)) * cw, (1 - row / (h - 1)) * ch];
  for (const [sc, sr] of seeds) {
    let col = sc;
    let row = sr;
    ctx.beginPath();
    const p0 = toPx(col, row);
    ctx.moveTo(p0[0], p0[1]);
    for (let k = 0; k < 26; k += 1) {
      const [u, v] = speedAt(col, row);
      const mag = Math.hypot(u, v) + 1e-6;
      col += (u / mag) * 1.15;
      row += (v / mag) * 1.15;
      const ri = Math.round(row);
      const ci = Math.round(col);
      if (ri < 1 || ci < 1 || ri >= h - 1 || ci >= w - 1 || mask[ri * w + ci] < 1) break;
      const p = toPx(col, row);
      ctx.lineTo(p[0], p[1]);
    }
    ctx.stroke();
  }
  ctx.restore();
}

function errorFrame(truthU8, predU8, tPred, rangeT, rangeP, mask, h, w, tTruth) {
  const out = new Uint8Array(h * w);
  let max = 1e-8;
  const vals = new Float32Array(h * w);
  for (let i = 0; i < h * w; i += 1) {
    if (mask[i] < 1) continue;
    const tv = dequant(truthU8[tTruth * h * w + i], rangeT);
    const pv = dequant(predU8[tPred * h * w + i], rangeP);
    const e = Math.abs(tv - pv);
    vals[i] = e;
    if (e > max) max = e;
  }
  for (let i = 0; i < h * w; i += 1) out[i] = mask[i] < 1 ? 0 : Math.round((vals[i] / max) * 255);
  return { u8: out, max };
}

function tone(value, good, warn) {
  if (value == null) return "";
  if (value <= good) return "good";
  if (value <= warn) return "warn";
  return "bad";
}

function renderHero() {
  const bundle = state.bundle;
  if (!bundle || state.page !== "highlights") return;
  const pack = ensureDecoded(bundle);
  const canvas = document.getElementById("heroCanvas");
  const h = bundle.ny;
  const w = bundle.nx;
  paintField(canvas, pack.cfd.ux, state.heroFrame, bundle.cfd.range.ux, pack.mask, h, w, {
    stream: true,
    ux: pack.cfd.ux,
    uy: pack.cfd.uy,
    uxRange: bundle.cfd.range.ux,
    uyRange: bundle.cfd.range.uy,
  });
  const t = bundle.times[state.heroFrame];
  document.getElementById("heroCap").textContent =
    `Re ${bundle.Re} · cy ${bundle.cy} · D ${bundle.D} · t = ${t.toFixed(2)} · Ux`;
}

function renderStudio() {
  const bundle = state.bundle;
  if (!bundle) return;
  const pack = ensureDecoded(bundle);
  const h = bundle.ny;
  const w = bundle.nx;
  const t = state.frame;
  const field = state.field;
  const mode = state.mode;
  const tri = document.getElementById("triptych");
  tri.classList.toggle("single", mode === "truth");
  document.getElementById("panePred").style.display = mode === "truth" ? "none" : "";
  document.getElementById("paneErr").style.display = mode === "truth" ? "none" : "";

  const cfdRange = bundle.cfd.range[field];
  paintField(document.getElementById("cfdCanvas"), pack.cfd[field], t, cfdRange, pack.mask, h, w, {
    diverging: field === "omega",
    stream: state.stream && (field === "ux" || field === "uy" || field === "omega"),
    ux: pack.cfd.ux,
    uy: pack.cfd.uy,
    uxRange: bundle.cfd.range.ux,
    uyRange: bundle.cfd.range.uy,
  });

  let rel = null;
  let div = bundle.cfd.div[t];
  let wall = 0;
  const time = bundle.times[t];
  document.getElementById("capTruth").textContent = field.toUpperCase();

  if (mode !== "truth") {
    const model = mode === "unetex" ? bundle.unetex : bundle.fno;
    const predPack = mode === "unetex" ? pack.unetex : pack.fno;
    const predT = mode === "unetex" ? 0 : t;
    const valid = mode === "fno" ? bundle.fno.valid[t] : true;
    const predRange = model.range[field];
    if (valid) {
      paintField(document.getElementById("predCanvas"), predPack[field], predT, predRange, pack.mask, h, w, {
        diverging: field === "omega",
        stream: state.stream && mode === "fno" && (field === "ux" || field === "uy" || field === "omega"),
        ux: predPack.ux,
        uy: predPack.uy,
        uxRange: model.range.ux,
        uyRange: model.range.uy,
      });
      const err = errorFrame(pack.cfd[field], predPack[field], predT, cfdRange, predRange, pack.mask, h, w, t);
      const errCanvas = document.getElementById("errCanvas");
      paintField(errCanvas, err.u8, 0, [0, err.max], pack.mask, h, w, { err: true });
      document.getElementById("capErr").textContent = `max ${err.max.toFixed(3)}`;
    } else {
      const ctx = document.getElementById("predCanvas").getContext("2d");
      ctx.clearRect(0, 0, 840, 480);
      ctx.fillStyle = "#93a399";
      ctx.font = "16px sans-serif";
      ctx.fillText("前 4 帧用作历史，从下一帧开始预测", 24, 40);
      document.getElementById("errCanvas").getContext("2d").clearRect(0, 0, 840, 480);
      document.getElementById("capErr").textContent = "—";
    }
    if (mode === "unetex") {
      rel = bundle.unetex.rel_frames[t];
      wall = bundle.unetex.wall;
      document.getElementById("capPred").textContent = `快照 @ t ${bundle.unetex.anchor_time}`;
    } else {
      rel = bundle.fno.rel_frames[t];
      div = valid ? bundle.fno.div[t] : null;
      wall = valid ? bundle.fno.wall[t] : null;
      document.getElementById("capPred").textContent = valid ? "历史 4 帧 → 当前" : "历史不足";
    }
  }

  document.getElementById("scaleMin").textContent = cfdRange[0].toFixed(3);
  document.getElementById("scaleMax").textContent = cfdRange[1].toFixed(3);
  document.getElementById("colorBar").classList.toggle("err", false);
  document.getElementById("timeRead").textContent = `t = ${time.toFixed(2)}`;
  const scrub = document.getElementById("scrub");
  scrub.max = String(bundle.times.length - 1);
  if (document.activeElement !== scrub) scrub.value = String(t);

  const lines = {
    truth: "正在回放 OpenFOAM 真值。打开右侧对照，才能看见网络跟不跟得上涡街。",
    unetex: "UNetEx 只看几何，预测停在时间窗中部的一张快照。播放时真值继续走，误差会随涡对相位涨落。",
    fno: "FNO 用物理时间上的前 4 帧预测当前帧。三块画面一起动，误差应贴着涡核而不是整片发亮。",
  };
  document.getElementById("modeLine").textContent = lines[mode];
  document.getElementById("pillModel").textContent = mode === "truth" ? "CFD" : mode === "unetex" ? "UNetEx" : "FNO";
  document.getElementById("caseTitle").textContent = `Re ${bundle.Re} · cy ${bundle.cy} · D ${bundle.D}`;
  document.getElementById("caseSub").textContent = bundle.id;

  const kv = document.getElementById("readKv");
  const relCells = rel
    ? [
        ["Ux 相对 L2", rel[0], tone(rel[0], 0.05, 0.12)],
        ["Uy 相对 L2", rel[1], tone(rel[1], 0.08, 0.2)],
        ["p 相对 L2", rel[2], tone(rel[2], 0.08, 0.2)],
      ]
    : [
        ["Ux 相对 L2", null, ""],
        ["Uy 相对 L2", null, ""],
        ["p 相对 L2", null, ""],
      ];
  const mean = mode === "fno" ? bundle.fno.rel_mean : mode === "unetex" ? null : null;
  const rows = [
    ["物理时间", time.toFixed(2), ""],
    ...relCells.map(([name, value, cls]) => [name, value == null ? "—" : pct(value), cls]),
    ["|∇·u|", div == null ? "—" : div.toExponential(2), ""],
    ["物体内速度", wall == null ? "—" : wall.toExponential(2), ""],
  ];
  if (mode === "fno" && mean[0] != null) {
    rows.push(["本例 FNO 均值", `${pct(mean[0])} / ${pct(mean[1])} / ${pct(mean[2])}`, ""]);
  }
  if (mode === "unetex") {
    const mid = bundle.unetex.rel_mid;
    rows.push(["对齐中帧", `${pct(mid[0])} / ${pct(mid[1])} / ${pct(mid[2])}`, ""]);
  }
  kv.innerHTML = rows
    .map(([name, value, cls]) => `<i>${name}</i><b class="${cls}">${value}</b>`)
    .join("");
  drawForces(bundle, time);
}

function drawForces(bundle, time) {
  const canvas = document.getElementById("forceCanvas");
  const ctx = canvas.getContext("2d");
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  const forces = bundle.forces;
  if (!forces.t.length) {
    ctx.fillStyle = "#93a399";
    ctx.fillText("这个算例没有力系数文件", 16, 28);
    document.getElementById("stRead").textContent = "";
    return;
  }
  const drawSeries = (values, top, band, color) => {
    let min = Infinity;
    let max = -Infinity;
    for (const v of values) {
      min = Math.min(min, v);
      max = Math.max(max, v);
    }
    const pad = (max - min) * 0.12 || 0.01;
    min -= pad;
    max += pad;
    const t0 = forces.t[0];
    const t1 = forces.t[forces.t.length - 1];
    ctx.beginPath();
    values.forEach((v, i) => {
      const x = ((forces.t[i] - t0) / (t1 - t0)) * (width - 24) + 12;
      const y = top + ((max - v) / (max - min)) * band;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.4;
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.font = "12px sans-serif";
    ctx.fillText(top < height / 2 ? "Cd" : "Cl", 14, top + 14);
  };
  drawSeries(forces.cd, 8, height / 2 - 18, "#dff25a");
  drawSeries(forces.cl, height / 2 + 4, height / 2 - 16, "#7ddec8");
  const t0 = forces.t[0];
  const t1 = forces.t[forces.t.length - 1];
  const x = ((time - t0) / (t1 - t0)) * (width - 24) + 12;
  ctx.strokeStyle = "rgba(255,255,255,0.8)";
  ctx.beginPath();
  ctx.moveTo(x, 4);
  ctx.lineTo(x, height - 4);
  ctx.stroke();
  document.getElementById("stRead").textContent = strouhalText(bundle);
}

function strouhalText(bundle) {
  const { t, cl, cd } = bundle.forces;
  const late = [];
  for (let i = 0; i < t.length; i += 1) if (t[i] >= 40) late.push([t[i], cl[i], cd[i]]);
  if (late.length < 30) return "";
  const mean = late.reduce((s, p) => s + p[1], 0) / late.length;
  const crossings = [];
  for (let i = 1; i < late.length; i += 1) {
    if (late[i - 1][1] < mean && late[i][1] >= mean) crossings.push(late[i][0]);
  }
  if (crossings.length < 3) return "升力起伏较弱，未估出周期";
  let acc = 0;
  for (let i = 1; i < crossings.length; i += 1) acc += crossings[i] - crossings[i - 1];
  const period = acc / (crossings.length - 1);
  const st = bundle.D / period;
  const cdMean = late.reduce((s, p) => s + p[2], 0) / late.length;
  return `t>40  Cd≈${cdMean.toFixed(3)}  St≈${st.toFixed(3)}`;
}

const HEATS = [
  { id: "ux", label: "Ux" },
  { id: "uy", label: "Uy" },
  { id: "p", label: "p" },
];
const COMPARE_VIEWS = [
  { id: "pred", label: "预测场" },
  { id: "err", label: "误差场" },
];
const HEAT_SCALE = { ux: 0.1, uy: 1.15, p: 0.7 };

function selectedId() {
  return caseId(state.re, state.cy, state.d);
}

function renderMeters() {
  const box = document.getElementById("meterBox");
  const ov = state.overview;
  if (!ov) return;
  const rows = [
    ["FNO Ux", ov.metrics.fno.rel_l2_ux, "fno"],
    ["FNO Uy", ov.metrics.fno.rel_l2_uy, "fno"],
    ["FNO p", ov.metrics.fno.rel_l2_p, "fno"],
    ["UNetEx Ux", ov.metrics.unetex.rel_l2_ux, "ux"],
    ["UNetEx Uy", ov.metrics.unetex.rel_l2_uy, "ux"],
    ["UNetEx p", ov.metrics.unetex.rel_l2_p, "ux"],
  ];
  const scale = 0.8;
  box.innerHTML = rows
    .map(([name, value, cls]) => {
      const width = Math.max(1.2, Math.min(100, (value / scale) * 100));
      return `<div class="meter"><span>${name}</span><div class="meter-track"><span class="meter-fill ${cls}" style="width:${width}%"></span></div><b>${pct(value)}</b></div>`;
    })
    .join("") + `<div class="target-key">色条按 0–80% 拉伸，好让 1% 和 70% 同时看得见。方案基线：速度 5%，压力 8%。</div>`;
}

function heatColor(value, metric) {
  const t = Math.min(1, value / (HEAT_SCALE[metric] || 0.1));
  const r = Math.round(40 + t * 210);
  const g = Math.round(180 - t * 120);
  const b = Math.round(120 - t * 80);
  return `rgb(${r},${g},${b})`;
}

function renderMatrix() {
  const host = document.getElementById("matrix");
  const ov = state.overview;
  if (!ov) return;
  const metric = state.heatMetric;
  const names = { ux: "流向 Ux", uy: "横向 Uy", p: "压力 p" };
  document.getElementById("matrixTitle").textContent = `UNetEx 换几何：${names[metric]} 相对误差（%）`;
  const headers = [];
  for (const cy of CYS) {
    for (const d of DS) headers.push(`cy ${cy.label}<br>D ${d.label}`);
  }
  const current = selectedId();
  const head = `<div class="mrow"><span></span>${headers.map((h) => `<span class="mhead">${h}</span>`).join("")}</div>`;
  const body = RES.map((re) => {
    const cells = [];
    for (const cy of CYS) {
      for (const d of DS) {
        const id = caseId(Number(re), cy.v, d.v);
        const row = ov.geometry.find((g) => g.id === id);
        const value = row ? row[metric] : 0;
        const on = id === current ? " on" : "";
        cells.push(
          `<button class="cell${on}" style="background:${heatColor(value, metric)}" data-case="${id}">${row ? (value * 100).toFixed(1) : "—"}</button>`,
        );
      }
    }
    return `<div class="mrow"><span>Re ${re}</span>${cells.join("")}</div>`;
  }).join("");
  host.innerHTML = head + body;
  const tip = document.getElementById("heatTip");
  host.querySelectorAll(".cell").forEach((btn) => {
    btn.addEventListener("click", () => pickCompareCase(btn.dataset.case));
    btn.addEventListener("mouseenter", (event) => {
      const row = ov.geometry.find((g) => g.id === btn.dataset.case);
      if (!row) return;
      tip.classList.remove("hidden");
      tip.innerHTML = `<div>${row.id}</div><b>Ux ${pct(row.ux)}</b><br><b>Uy ${pct(row.uy)}</b><br><b>p ${pct(row.p)}</b>`;
      placeTip(event, tip);
    });
    btn.addEventListener("mousemove", (event) => placeTip(event, tip));
    btn.addEventListener("mouseleave", () => tip.classList.add("hidden"));
  });
}

function placeTip(event, tip) {
  const box = document.querySelector(".matrix-wrap").getBoundingClientRect();
  tip.style.left = `${event.clientX - box.left + 14}px`;
  tip.style.top = `${event.clientY - box.top + 14}px`;
}

function pickCompareCase(id) {
  const row = state.overview.cases.find((c) => c.id === id);
  if (!row) return;
  state.re = row.Re;
  state.cy = row.cy;
  state.d = row.D;
  syncSegs();
  renderMatrix();
  loadCase();
}

function paintComparePane(canvas, u8, t, range, mask, h, w, opts, emptyText) {
  if (emptyText) {
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#93a399";
    ctx.font = "16px sans-serif";
    ctx.fillText(emptyText, 24, 40);
    return;
  }
  paintField(canvas, u8, t, range, mask, h, w, opts);
}

function renderCompare() {
  const bundle = state.bundle;
  if (!bundle) return;
  const pack = ensureDecoded(bundle);
  const h = bundle.ny;
  const w = bundle.nx;
  const t = state.frame;
  const field = state.compareField;
  const time = bundle.times[t];
  const cfdRange = bundle.cfd.range[field];
  const viewErr = state.compareView === "err";
  const diverging = !viewErr && field === "omega";
  document.getElementById("cmpColorBar").classList.toggle("err", viewErr);

  paintComparePane(document.getElementById("cmpCfd"), pack.cfd[field], t, cfdRange, pack.mask, h, w, {
    diverging,
    stream: !viewErr && (field === "ux" || field === "uy" || field === "omega"),
    ux: pack.cfd.ux,
    uy: pack.cfd.uy,
    uxRange: bundle.cfd.range.ux,
    uyRange: bundle.cfd.range.uy,
  });

  const unetErr = errorFrame(pack.cfd[field], pack.unetex[field], 0, cfdRange, bundle.unetex.range[field], pack.mask, h, w, t);
  if (viewErr) {
    paintComparePane(document.getElementById("cmpUnet"), unetErr.u8, 0, [0, unetErr.max], pack.mask, h, w, { err: true });
  } else {
    paintComparePane(document.getElementById("cmpUnet"), pack.unetex[field], 0, bundle.unetex.range[field], pack.mask, h, w, {
      diverging,
    });
  }

  const fnoValid = bundle.fno.valid[t];
  if (!fnoValid) {
    paintComparePane(document.getElementById("cmpFno"), null, 0, [0, 1], pack.mask, h, w, {}, "前 4 帧用作历史，下一帧开始预测");
  } else if (viewErr) {
    const fnoErr = errorFrame(pack.cfd[field], pack.fno[field], t, cfdRange, bundle.fno.range[field], pack.mask, h, w, t);
    paintComparePane(document.getElementById("cmpFno"), fnoErr.u8, 0, [0, fnoErr.max], pack.mask, h, w, { err: true });
  } else {
    paintComparePane(document.getElementById("cmpFno"), pack.fno[field], t, bundle.fno.range[field], pack.mask, h, w, {
      diverging,
      stream: field === "ux" || field === "uy" || field === "omega",
      ux: pack.fno.ux,
      uy: pack.fno.uy,
      uxRange: bundle.fno.range.ux,
      uyRange: bundle.fno.range.uy,
    });
  }

  const scale = viewErr ? [0, Math.max(unetErr.max, 1e-6)] : cfdRange;
  document.getElementById("cmpScaleMin").textContent = scale[0].toFixed(3);
  document.getElementById("cmpScaleMax").textContent = scale[1].toFixed(3);
  document.getElementById("cmpCapCfd").textContent = field.toUpperCase();
  document.getElementById("cmpCapUnet").textContent = viewErr ? `max ${unetErr.max.toFixed(3)}` : `锁在 t ${bundle.unetex.anchor_time}`;
  document.getElementById("cmpCapFno").textContent = fnoValid ? (viewErr ? "当前帧误差" : "历史 4 帧 → 当前") : "历史不足";
  document.getElementById("cmpTimeRead").textContent = `t = ${time.toFixed(2)}`;
  document.getElementById("cmpModeLine").textContent = viewErr
    ? "误差场：UNetEx 整片会随涡街相位发亮，FNO 应当只剩涡核附近的细线。"
    : "预测场：中间那张快照不随时间走，右边 FNO 跟着 CFD 一起动。";
  document.getElementById("cmpCaseTitle").textContent = `Re ${bundle.Re} · cy ${bundle.cy} · D ${bundle.D}`;
  document.getElementById("cmpCaseSub").textContent = bundle.id;
  document.getElementById("cmpPillField").textContent = field.toUpperCase();

  const unetRel = bundle.unetex.rel_frames[t];
  const fnoRel = bundle.fno.rel_frames[t];
  const mean = bundle.fno.rel_mean;
  const rows = [
    ["物理时间", time.toFixed(2), ""],
    ["UNetEx Ux", pct(unetRel[0]), tone(unetRel[0], 0.05, 0.12)],
    ["UNetEx Uy", pct(unetRel[1]), tone(unetRel[1], 0.08, 0.2)],
    ["UNetEx p", pct(unetRel[2]), tone(unetRel[2], 0.08, 0.2)],
    ["FNO Ux", fnoRel ? pct(fnoRel[0]) : "—", fnoRel ? tone(fnoRel[0], 0.05, 0.12) : ""],
    ["FNO Uy", fnoRel ? pct(fnoRel[1]) : "—", fnoRel ? tone(fnoRel[1], 0.08, 0.2) : ""],
    ["FNO p", fnoRel ? pct(fnoRel[2]) : "—", fnoRel ? tone(fnoRel[2], 0.08, 0.2) : ""],
    ["本例 FNO 均值", mean[0] == null ? "—" : `${pct(mean[0])} / ${pct(mean[1])} / ${pct(mean[2])}`, ""],
    ["对齐中帧 UNetEx", `${pct(bundle.unetex.rel_mid[0])} / ${pct(bundle.unetex.rel_mid[1])} / ${pct(bundle.unetex.rel_mid[2])}`, ""],
  ];
  document.getElementById("cmpKv").innerHTML = rows
    .map(([name, value, cls]) => `<i>${name}</i><b class="${cls}">${value}</b>`)
    .join("");

  const scrub = document.getElementById("cmpScrub");
  scrub.max = String(bundle.times.length - 1);
  if (document.activeElement !== scrub) scrub.value = String(t);
  document.getElementById("cmpPlayBtn").textContent = state.playing ? "暂停" : "播放";
}

function renderHeroStats() {
  const ov = state.overview;
  const host = document.getElementById("heroStats");
  if (!ov) return;
  const fno = ov.metrics.fno;
  const ux = ov.metrics.unetex;
  host.innerHTML = `
    <div class="stat"><b><em>${ov.n_cases}</em> 组</b><span>圆柱绕流，Re、纵向位置、直径三个旋钮，全部来自 icoFoam。</span></div>
    <div class="stat"><b><em>${pct(fno.rel_l2_ux)}</em></b><span>FNO 流向速度相对 L2，全量时间窗回放。压力 ${pct(fno.rel_l2_p)}。</span></div>
    <div class="stat"><b><em>${pct(ux.rel_l2_uy)}</em></b><span>UNetEx 的横向速度误差。几何快照抓不住涡相位，所以时序必须另给模型。</span></div>
    <div class="stat"><b>4 帧 → 1 帧</b><span>FNO 的输入是几何加上刚刚发生的历史，输出下一时刻的 Ux、Uy、p。</span></div>
  `;
  document.getElementById("deviceMeta").textContent = `${ov.device} · ${ov.n_cases} cases · 64×64`;
  document.getElementById("overviewNotes").textContent = ov.notes.join(" ");
}

function showPage(page) {
  state.page = page;
  document.querySelectorAll("nav button").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.page === page);
  });
  document.querySelectorAll(".view").forEach((view) => {
    view.classList.toggle("active", view.id === `page-${page}`);
  });
  if (page === "highlights") renderHero();
  if (page === "studio") renderStudio();
  if (page === "compare") {
    renderMatrix();
    renderCompare();
  }
}

function syncSegs() {
  const mark = (root, current) => {
    root.querySelectorAll("button").forEach((btn) => {
      btn.classList.toggle("on", btn.dataset.value === String(current));
    });
  };
  mark(document.getElementById("segRe"), state.re);
  mark(document.getElementById("segCy"), state.cy);
  mark(document.getElementById("segD"), state.d);
  mark(document.getElementById("segField"), state.field);
  mark(document.getElementById("segMode"), state.mode);
}

function fillSeg(root, items, getValue, onPick) {
  root.innerHTML = items
    .map((item) => `<button type="button" data-value="${getValue(item)}">${item.label || item}</button>`)
    .join("");
  root.querySelectorAll("button").forEach((btn) => {
    btn.addEventListener("click", () => onPick(btn.dataset.value));
  });
}

async function loadCase() {
  const id = caseId(state.re, state.cy, state.d);
  document.getElementById("loading").classList.remove("hidden");
  try {
    const res = await fetch(`/api/case?id=${encodeURIComponent(id)}`);
    if (!res.ok) throw new Error(await res.text());
    state.bundle = await res.json();
    state.frame = 0;
    state.heroFrame = 0;
    renderHero();
    renderStudio();
    renderCompare();
    renderMatrix();
  } catch (err) {
    document.getElementById("modeLine").textContent = `这个算例没有返回：${err.message}`;
  } finally {
    document.getElementById("loading").classList.add("hidden");
  }
}

function bootControls() {
  fillSeg(document.getElementById("segRe"), RES.map((r) => ({ label: r, value: r })), (x) => x.value, (v) => {
    state.re = Number(v);
    syncSegs();
    loadCase();
  });
  fillSeg(document.getElementById("segCy"), CYS, (x) => x.v, (v) => {
    state.cy = Number(v);
    syncSegs();
    loadCase();
  });
  fillSeg(document.getElementById("segD"), DS, (x) => x.v, (v) => {
    state.d = Number(v);
    syncSegs();
    loadCase();
  });
  fillSeg(document.getElementById("segField"), FIELDS, (x) => x.id, (v) => {
    state.field = v;
    syncSegs();
    renderStudio();
  });
  fillSeg(document.getElementById("segMode"), MODES, (x) => x.id, (v) => {
    state.mode = v;
    syncSegs();
    renderStudio();
  });
  fillSeg(document.getElementById("segHeat"), HEATS, (x) => x.id, (v) => {
    state.heatMetric = v;
    markCompareSegs();
    renderMatrix();
  });
  fillSeg(document.getElementById("segCompareView"), COMPARE_VIEWS, (x) => x.id, (v) => {
    state.compareView = v;
    markCompareSegs();
    renderCompare();
  });
  fillSeg(document.getElementById("segCompareField"), FIELDS, (x) => x.id, (v) => {
    state.compareField = v;
    markCompareSegs();
    renderCompare();
  });
  syncSegs();
  markCompareSegs();
  document.getElementById("streamToggle").addEventListener("click", (event) => {
    state.stream = !state.stream;
    event.currentTarget.classList.toggle("on", state.stream);
    renderStudio();
    renderHero();
  });
  document.getElementById("playBtn").addEventListener("click", () => setPlaying(!state.playing));
  document.getElementById("cmpPlayBtn").addEventListener("click", () => setPlaying(!state.playing));
  document.getElementById("scrub").addEventListener("input", (event) => {
    state.frame = Number(event.target.value);
    renderStudio();
    renderCompare();
  });
  document.getElementById("cmpScrub").addEventListener("input", (event) => {
    state.frame = Number(event.target.value);
    renderStudio();
    renderCompare();
  });
  document.querySelectorAll("nav button").forEach((btn) => {
    btn.addEventListener("click", () => showPage(btn.dataset.page));
  });
  document.querySelectorAll("[data-jump]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const jump = btn.dataset.jump;
      if (jump === "truth" || jump === "unetex" || jump === "fno") state.mode = jump;
      showPage("studio");
      syncSegs();
      renderStudio();
    });
  });
}

let last = performance.now();
let acc = 0;
function loop(now) {
  const dt = now - last;
  last = now;
  if (state.playing && state.bundle) {
    acc += dt;
    const step = 85;
    while (acc >= step) {
      acc -= step;
      const n = state.bundle.times.length;
      state.frame = (state.frame + 1) % n;
      state.heroFrame = (state.heroFrame + 1) % n;
    }
    if (state.page === "studio") renderStudio();
    if (state.page === "highlights") renderHero();
    if (state.page === "compare") renderCompare();
  }
  requestAnimationFrame(loop);
}

function setPlaying(on) {
  state.playing = on;
  document.getElementById("playBtn").textContent = on ? "暂停" : "播放";
  document.getElementById("cmpPlayBtn").textContent = on ? "暂停" : "播放";
}

function markCompareSegs() {
  const mark = (root, current) => {
    if (!root) return;
    root.querySelectorAll("button").forEach((btn) => {
      btn.classList.toggle("on", btn.dataset.value === String(current));
    });
  };
  mark(document.getElementById("segHeat"), state.heatMetric);
  mark(document.getElementById("segCompareView"), state.compareView);
  mark(document.getElementById("segCompareField"), state.compareField);
}

async function main() {
  bootControls();
  const ov = await fetch("/api/overview");
  state.overview = await ov.json();
  renderHeroStats();
  renderMeters();
  renderMatrix();
  await loadCase();
  requestAnimationFrame(loop);
}

main();
