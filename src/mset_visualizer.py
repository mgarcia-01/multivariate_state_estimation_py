"""
mset_visualizer.py
==================

Interactive matplotlib visualization for the MSET / SPRT pipeline implemented
in `mset_rul.py` (Cheng & Pecht, "Multivariate State Estimation Technique for
Remaining Useful Life Prediction of Electronic Products").

One figure, two linked charts:

    +--------------------------------------------------+----------------+
    |                                                  |  time slider   |
    |     3D chart - monitored input states,           |  layer toggles |
    |     MSET estimates and detected anomalies        |  SPRT channel  |
    |     (rotate / zoom with the mouse,               |  selector      |
    |      click any point for details)                |  info panel    |
    |                                                  |                |
    +--------------------------------------------------+----------------+
    |     SPRT chart - log-likelihood-ratio            |                |
    |     trajectories, decision boundaries,           |                |
    |     fault alarms (click to jump in time)         |                |
    +--------------------------------------------------+-----------------+

Interactivity (pure matplotlib, no extra dependencies):

    * native 3D rotation / zoom on the top chart;
    * a time Slider that scrubs through the observation history - the current
      state is highlighted in 3D and a synchronized cursor moves on the SPRT
      chart;
    * CheckButtons to toggle layers (training cloud, MSET estimates, residual
      lines, anomaly markers);
    * RadioButtons to switch the SPRT panel between monitored parameters
      (p1, p2, p3, ... or the residual norm ||R||);
    * pick events: click a 3D point or anywhere on the SPRT panel to move the
      cursor to that sample and print its details in the info panel.

Usage
-----
    from mset_rul import MSET, inverse_distance_kernel
    from mset_visualizer import MSETAnomalyVisualizer

    model = MSET().fit(training_states)
    viz = MSETAnomalyVisualizer(
        model=model,
        observations=monitored_states,     # list of [p1, p2, p3] states
        times=days,                        # optional, defaults to 1..N
        param_names=("Temp [degC]", "RH [%]", "Vib [g]"),
    )
    viz.show()

or simply run this file for a self-contained demo:

    python3 mset_visualizer.py

Only the first three monitored parameters are drawn on the 3D axes (a 3D chart
has three axes); the SPRT panel supports every parameter.
"""

from __future__ import annotations

import json
import math
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider, CheckButtons, RadioButtons
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3D projection)

from mset_rul import (
    MSET,
    euclidean_norm,
)


# --------------------------------------------------------------------------- #
# SPRT with full trajectory recording (for plotting)
# --------------------------------------------------------------------------- #

class SPRTTrace:
    """
    Two-sided Wald SPRT identical in logic to `mset_rul.SPRT`, but recording
    the complete log-likelihood-ratio trajectories and every decision so that
    they can be plotted.

        H0: r ~ N(mu0, sigma^2)                  (healthy)
        H1+: r ~ N(mu0 + M*sigma, sigma^2)       (positive drift)
        H1-: r ~ N(mu0 - M*sigma, sigma^2)       (negative drift)

    Recorded per sample i:
        llr_pos[i], llr_neg[i]  - the two running LLRs *after* sample i
        verdicts[i]             - 'fault' / 'healthy' / 'continue'
        fault_indices           - samples at which a fault alarm fired
    """

    def __init__(self,
                 mean: float,
                 sigma: float,
                 alpha: float = 0.01,
                 beta: float = 0.01,
                 disturbance: float = 3.0) -> None:
        self.mu0 = mean
        self.sigma = max(sigma, 1e-12)
        self.shift = disturbance * self.sigma
        self.upper = math.log((1.0 - beta) / alpha)
        self.lower = math.log(beta / (1.0 - alpha))

        self.llr_pos: List[float] = []
        self.llr_neg: List[float] = []
        self.verdicts: List[str] = []
        self.fault_indices: List[int] = []
        self.healthy_indices: List[int] = []

    def run(self, series: Sequence[float]) -> "SPRTTrace":
        pos = neg = 0.0
        gain = self.shift / self.sigma ** 2
        for i, value in enumerate(series):
            deviation = value - self.mu0
            pos += gain * (deviation - self.shift / 2.0)
            neg += gain * (-deviation - self.shift / 2.0)

            verdict = "continue"
            if pos >= self.upper or neg >= self.upper:
                verdict = "fault"
                self.fault_indices.append(i)
            elif pos <= self.lower and neg <= self.lower:
                verdict = "healthy"
                self.healthy_indices.append(i)

            # record post-update values, then apply the reset rules
            self.llr_pos.append(pos)
            self.llr_neg.append(neg)
            self.verdicts.append(verdict)

            if verdict != "continue":
                pos = neg = 0.0
            else:
                pos = max(pos, self.lower)
                neg = max(neg, self.lower)
        return self


# --------------------------------------------------------------------------- #
# HTML/JS export template (mirrors the browser output of mset_visualizer.c,
# so the Python and C versions produce the same interaction model).
# The `DATA` object referenced throughout is written immediately before this
# template in the same <script> tag by MSETAnomalyVisualizer.to_html().
# --------------------------------------------------------------------------- #

_HTML_TOP = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MSET input &amp; anomalies (3D) + SPRT fault detection</title>
<style>
  body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;
       margin:18px;color:#222;background:#fff;}
  h1{font-size:16px;font-weight:600;margin:0 0 4px 0;}
  .legend{font-size:12px;color:#555;margin-bottom:10px;}
  .hint{font-size:11px;color:#888;margin-top:4px;}
  .container{display:flex;gap:18px;align-items:flex-start;flex-wrap:wrap;}
  .left{flex:0 0 auto;}
  .right{flex:0 0 260px;display:flex;flex-direction:column;gap:12px;}
  canvas{border:1px solid #ddd;border-radius:6px;background:#fff;
         display:block;cursor:grab;}
  canvas:active{cursor:grabbing;}
  #chartSprt{cursor:pointer;margin-top:12px;}
  .box{border:1px solid #ccc;border-radius:6px;padding:10px 12px;font-size:13px;}
  .box-title{font-weight:600;font-size:12px;color:#444;margin-bottom:6px;
             text-transform:uppercase;letter-spacing:.02em;}
  .box label{display:block;padding:2px 0;cursor:pointer;font-size:13px;}
  #infoPanel{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;
             white-space:pre-wrap;background:#f7f7f7;border:1px solid #ccc;
             border-radius:6px;padding:10px;min-height:230px;line-height:1.45;}
  #sprtTitle{font-size:13px;font-weight:600;margin:14px 0 4px 2px;}
  input[type=range]{width:100%;}
  .swatch{display:inline-block;width:10px;height:10px;border-radius:2px;
          margin-right:5px;vertical-align:middle;}
</style>
</head>
<body>
<h1>Input states, MSET estimates and detected anomalies</h1>
<div class="legend">
  <span class="swatch" style="background:#9ecae1"></span>healthy training data &nbsp;
  <span class="swatch" style="background:#31688e"></span>input (colour = ||residual||) &nbsp;
  <span class="swatch" style="background:#7f7f7f"></span>MSET estimate (triangle) &nbsp;
  <span class="swatch" style="background:#d62728"></span>anomaly (SPRT fault, X) &nbsp;
  <span class="swatch" style="background:#ff7f0e"></span>current time (ring)
</div>
<div class="container">
  <div class="left">
    <canvas id="chart3d" width="900" height="540"></canvas>
    <div class="hint">rotate: drag &middot; zoom: scroll &middot; click a point to inspect</div>
    <div id="sprtTitle"></div>
    <canvas id="chartSprt" width="900" height="220"></canvas>
    <div style="margin-top:10px;">
      <label for="timeSlider" style="font-size:12px;color:#444;">time index</label>
      <input type="range" id="timeSlider" min="0" max="1" step="1" value="0">
    </div>
  </div>
  <div class="right">
    <div class="box">
      <div class="box-title">layers</div>
      <label><input type="checkbox" id="chk_training" checked> training data</label>
      <label><input type="checkbox" id="chk_estimates" checked> MSET estimates</label>
      <label><input type="checkbox" id="chk_residuals" checked> residual lines</label>
      <label><input type="checkbox" id="chk_anomalies" checked> anomalies</label>
      <label><input type="checkbox" id="chk_trajectory" checked> trajectory</label>
    </div>
    <div class="box">
      <div class="box-title">SPRT channel</div>
      <div id="channelRadios"></div>
    </div>
    <div class="box">
      <div class="box-title">sample info</div>
      <div id="infoPanel"></div>
    </div>
  </div>
</div>
"""

# The renderer: vanilla Canvas 2D, no external dependencies. A raw string, so
# the one literal `\n` (the JS join separator near the bottom) passes through
# to the output unchanged instead of becoming a real newline in this source.
_JS_TEMPLATE = r"""(function(){
  const state = {
    rotX: -0.32, rotY: 0.55, zoom: 1.0,
    cursor: DATA.k - 1,
    channel: DATA.channels.length - 1,
    layers: {training:true, estimates:true, residuals:true, anomalies:true, trajectory:true},
    dragging:false, moved:false, lastX:0, lastY:0, downX:0, downY:0
  };

  const c3d = document.getElementById('chart3d');
  const ctx3d = c3d.getContext('2d');
  const cSprt = document.getElementById('chartSprt');
  const ctxSprt = cSprt.getContext('2d');
  const slider = document.getElementById('timeSlider');
  const info = document.getElementById('infoPanel');
  const sprtTitle = document.getElementById('sprtTitle');

  slider.min = 0; slider.max = DATA.k - 1; slider.value = state.cursor;

  /* ---- build the SPRT channel radio buttons (count is data-dependent) --- */
  const radioHost = document.getElementById('channelRadios');
  DATA.channels.forEach((name, i) => {
    const label = document.createElement('label');
    const input = document.createElement('input');
    input.type = 'radio'; input.name = 'sprtChannel'; input.id = 'chan_' + i;
    if (i === state.channel) input.checked = true;
    input.addEventListener('change', () => { state.channel = i; redrawSPRT(); });
    label.appendChild(input);
    label.appendChild(document.createTextNode(' ' + name));
    radioHost.appendChild(label);
  });

  /* ---- bounds + normalisation into a [-1,1]^3 cube ---- */
  const allX = DATA.objX.concat(DATA.trainX, DATA.estX);
  const allY = DATA.objY.concat(DATA.trainY, DATA.estY);
  const allZ = DATA.objZ.concat(DATA.trainZ, DATA.estZ);
  const lo = [Math.min(...allX), Math.min(...allY), Math.min(...allZ)];
  const hi = [Math.max(...allX), Math.max(...allY), Math.max(...allZ)];
  function norm3(x, y, z) {
    return [
      2*(x-lo[0])/((hi[0]-lo[0])||1) - 1,
      2*(y-lo[1])/((hi[1]-lo[1])||1) - 1,
      2*(z-lo[2])/((hi[2]-lo[2])||1) - 1
    ];
  }

  function project(p) {
    const cy = Math.cos(state.rotY), sy = Math.sin(state.rotY);
    const cx = Math.cos(state.rotX), sx = Math.sin(state.rotX);
    let x = p[0]*cy - p[2]*sy;
    let z = p[0]*sy + p[2]*cy;
    let y = p[1]*cx - z*sx;
    z    = p[1]*sx + z*cx;
    const scale = 150 * state.zoom;
    const persp = 1 / (1 + (z + 2) * 0.12);
    return {
      x: c3d.width/2 + x*scale*persp,
      y: c3d.height/2 + 30 - y*scale*persp,
      depth: z
    };
  }

  const VIRIDIS = [[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
  function colorRamp(v, vmin, vmax) {
    let t = (v - vmin) / ((vmax - vmin) || 1);
    t = Math.max(0, Math.min(1, t));
    const seg = t * (VIRIDIS.length - 1);
    const i = Math.min(VIRIDIS.length - 2, Math.floor(seg));
    const f = seg - i, a = VIRIDIS[i], b = VIRIDIS[i+1];
    const r = Math.round(a[0]+(b[0]-a[0])*f);
    const g = Math.round(a[1]+(b[1]-a[1])*f);
    const bl= Math.round(a[2]+(b[2]-a[2])*f);
    return `rgb(${r},${g},${bl})`;
  }

  function drawAxes() {
    const O = project(norm3(lo[0], lo[1], lo[2]));
    const X = project(norm3(hi[0], lo[1], lo[2]));
    const Y = project(norm3(lo[0], hi[1], lo[2]));
    const Z = project(norm3(lo[0], lo[1], hi[2]));
    ctx3d.strokeStyle = '#aaa'; ctx3d.lineWidth = 1;
    ctx3d.font = '11px sans-serif'; ctx3d.fillStyle = '#444';
    [[O,X,DATA.paramNames[0]],[O,Y,DATA.paramNames[1]],[O,Z,DATA.paramNames[2]]].forEach(([a,b,label]) => {
      ctx3d.beginPath(); ctx3d.moveTo(a.x,a.y); ctx3d.lineTo(b.x,b.y); ctx3d.stroke();
      ctx3d.fillText(label, b.x+4, b.y+4);
    });
  }

  function redraw3D() {
    ctx3d.clearRect(0, 0, c3d.width, c3d.height);
    drawAxes();

    const items = [];

    if (state.layers.training) {
      for (let i = 0; i < DATA.trainX.length; ++i) {
        const P = project(norm3(DATA.trainX[i], DATA.trainY[i], DATA.trainZ[i]));
        items.push({type:'dot', p:P, r:2.4, color:'rgba(158,202,225,0.55)', depth:P.depth});
      }
    }

    if (state.layers.trajectory) {
      ctx3d.strokeStyle = 'rgba(120,120,120,0.55)'; ctx3d.lineWidth = 1;
      ctx3d.beginPath();
      for (let i = 0; i < DATA.k; ++i) {
        const P = project(norm3(DATA.objX[i], DATA.objY[i], DATA.objZ[i]));
        if (i === 0) ctx3d.moveTo(P.x, P.y); else ctx3d.lineTo(P.x, P.y);
      }
      ctx3d.stroke();
    }

    if (state.layers.residuals) {
      for (let i = 0; i < DATA.k; ++i) {
        const A = project(norm3(DATA.objX[i], DATA.objY[i], DATA.objZ[i]));
        const B = project(norm3(DATA.estX[i], DATA.estY[i], DATA.estZ[i]));
        items.push({type:'line', a:A, b:B, color:'rgba(127,127,127,0.55)', depth:(A.depth+B.depth)/2});
      }
    }

    if (state.layers.estimates) {
      for (let i = 0; i < DATA.k; ++i) {
        const P = project(norm3(DATA.estX[i], DATA.estY[i], DATA.estZ[i]));
        items.push({type:'tri', p:P, color:'#7f7f7f', depth:P.depth});
      }
    }

    const vmin = Math.min(...DATA.residualNorm), vmax = Math.max(...DATA.residualNorm);
    for (let i = 0; i < DATA.k; ++i) {
      if (DATA.anomalyMask[i]) continue;
      const P = project(norm3(DATA.objX[i], DATA.objY[i], DATA.objZ[i]));
      items.push({type:'dot', p:P, r:5, color:colorRamp(DATA.residualNorm[i], vmin, vmax), depth:P.depth});
    }

    if (state.layers.anomalies) {
      for (let i = 0; i < DATA.k; ++i) {
        if (!DATA.anomalyMask[i]) continue;
        const P = project(norm3(DATA.objX[i], DATA.objY[i], DATA.objZ[i]));
        items.push({type:'x', p:P, color:'#d62728', depth:P.depth});
      }
    }

    items.sort((a, b) => a.depth - b.depth);
    items.forEach(it => {
      if (it.type === 'dot') {
        ctx3d.beginPath(); ctx3d.arc(it.p.x, it.p.y, it.r, 0, 2*Math.PI);
        ctx3d.fillStyle = it.color; ctx3d.fill();
      } else if (it.type === 'tri') {
        ctx3d.beginPath();
        ctx3d.moveTo(it.p.x, it.p.y-5); ctx3d.lineTo(it.p.x-5, it.p.y+4); ctx3d.lineTo(it.p.x+5, it.p.y+4);
        ctx3d.closePath(); ctx3d.fillStyle = it.color; ctx3d.fill();
      } else if (it.type === 'line') {
        ctx3d.strokeStyle = it.color; ctx3d.lineWidth = 1;
        ctx3d.beginPath(); ctx3d.moveTo(it.a.x, it.a.y); ctx3d.lineTo(it.b.x, it.b.y); ctx3d.stroke();
      } else if (it.type === 'x') {
        ctx3d.strokeStyle = it.color; ctx3d.lineWidth = 2.4;
        ctx3d.beginPath();
        ctx3d.moveTo(it.p.x-6, it.p.y-6); ctx3d.lineTo(it.p.x+6, it.p.y+6);
        ctx3d.moveTo(it.p.x+6, it.p.y-6); ctx3d.lineTo(it.p.x-6, it.p.y+6);
        ctx3d.stroke();
      }
    });

    const ci = state.cursor;
    const CP = project(norm3(DATA.objX[ci], DATA.objY[ci], DATA.objZ[ci]));
    ctx3d.beginPath(); ctx3d.arc(CP.x, CP.y, 11, 0, 2*Math.PI);
    ctx3d.strokeStyle = '#ff7f0e'; ctx3d.lineWidth = 2.5; ctx3d.stroke();

    updateInfo();
  }

  function pickIndex(mx, my) {
    let best = -1, bestD = 1e18;
    for (let i = 0; i < DATA.k; ++i) {
      const P = project(norm3(DATA.objX[i], DATA.objY[i], DATA.objZ[i]));
      const d = Math.hypot(P.x - mx, P.y - my);
      if (d < bestD) { bestD = d; best = i; }
    }
    return bestD < 16 ? best : -1;
  }

  function setCursor(i) {
    state.cursor = Math.max(0, Math.min(DATA.k - 1, i));
    slider.value = state.cursor;
    redraw3D();
    redrawSPRT();
  }

  c3d.addEventListener('mousedown', e => {
    const rect = c3d.getBoundingClientRect();
    state.dragging = true; state.moved = false;
    state.downX = e.clientX - rect.left; state.downY = e.clientY - rect.top;
    state.lastX = state.downX; state.lastY = state.downY;
  });
  window.addEventListener('mousemove', e => {
    if (!state.dragging) return;
    const rect = c3d.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    if (Math.hypot(mx - state.downX, my - state.downY) > 3) state.moved = true;
    const dx = mx - state.lastX, dy = my - state.lastY;
    state.rotY += dx * 0.008; state.rotX += dy * 0.008;
    state.lastX = mx; state.lastY = my;
    redraw3D();
  });
  window.addEventListener('mouseup', e => {
    if (state.dragging && !state.moved) {
      const rect = c3d.getBoundingClientRect();
      const idx = pickIndex(e.clientX - rect.left, e.clientY - rect.top);
      if (idx >= 0) setCursor(idx);
    }
    state.dragging = false;
  });
  c3d.addEventListener('wheel', e => {
    e.preventDefault();
    state.zoom *= (e.deltaY < 0 ? 1.08 : 0.93);
    state.zoom = Math.max(0.3, Math.min(3.5, state.zoom));
    redraw3D();
  }, {passive:false});

  const PAD_L = 46, PAD_R = 12, PAD_T = 24, PAD_B = 28;
  function redrawSPRT() {
    ctxSprt.clearRect(0, 0, cSprt.width, cSprt.height);
    const ch = DATA.sprt[state.channel];
    const W = cSprt.width, H = cSprt.height;
    const tmin = DATA.times[0], tmax = DATA.times[DATA.k - 1];
    const allVals = ch.llrPos.concat(ch.llrNeg, [ch.upper, ch.lower, 0]);
    const vmin = Math.min(...allVals), vmax = Math.max(...allVals);
    const X = t => PAD_L + (t - tmin) / ((tmax - tmin) || 1) * (W - PAD_L - PAD_R);
    const Y = v => H - PAD_B - (v - vmin) / ((vmax - vmin) || 1) * (H - PAD_T - PAD_B);

    if (state.layers.anomalies) {
      ctxSprt.fillStyle = 'rgba(214,39,40,0.09)';
      for (let i = 0; i < DATA.k; ++i) {
        if (!DATA.anomalyMask[i]) continue;
        const x = X(DATA.times[i]);
        ctxSprt.fillRect(x - 3, PAD_T, 6, H - PAD_T - PAD_B);
      }
    }

    ctxSprt.strokeStyle = '#333'; ctxSprt.lineWidth = 1;
    ctxSprt.strokeRect(PAD_L, PAD_T, W - PAD_L - PAD_R, H - PAD_T - PAD_B);
    ctxSprt.fillStyle = '#333'; ctxSprt.font = '10px monospace';
    ctxSprt.fillText(vmax.toFixed(1), 4, Y(vmax) + 3);
    ctxSprt.fillText(vmin.toFixed(1), 4, Y(vmin) + 3);
    ctxSprt.fillText(String(Math.round(tmin)), PAD_L - 6, H - 8);
    ctxSprt.fillText(String(Math.round(tmax)), W - PAD_R - 22, H - 8);
    ctxSprt.fillText(DATA.timeLabel, (W + PAD_L - PAD_R) / 2 - 18, H - 6);

    ctxSprt.setLineDash([5, 4]);
    ctxSprt.strokeStyle = '#d62728';
    ctxSprt.beginPath(); ctxSprt.moveTo(PAD_L, Y(ch.upper)); ctxSprt.lineTo(W - PAD_R, Y(ch.upper)); ctxSprt.stroke();
    ctxSprt.strokeStyle = '#666';
    ctxSprt.beginPath(); ctxSprt.moveTo(PAD_L, Y(ch.lower)); ctxSprt.lineTo(W - PAD_R, Y(ch.lower)); ctxSprt.stroke();
    ctxSprt.setLineDash([]);

    function poly(vals, color) {
      ctxSprt.strokeStyle = color; ctxSprt.lineWidth = 1.6;
      ctxSprt.beginPath();
      for (let i = 0; i < DATA.k; ++i) {
        const x = X(DATA.times[i]), y = Y(vals[i]);
        if (i === 0) ctxSprt.moveTo(x, y); else ctxSprt.lineTo(x, y);
      }
      ctxSprt.stroke();
    }
    poly(ch.llrPos, '#1f77b4');
    poly(ch.llrNeg, '#2ca02c');

    ctxSprt.strokeStyle = '#d62728'; ctxSprt.lineWidth = 2.2;
    ch.faultIndices.forEach(i => {
      const x = X(DATA.times[i]), y = Y(Math.max(ch.llrPos[i], ch.llrNeg[i]));
      ctxSprt.beginPath();
      ctxSprt.moveTo(x-5,y-5); ctxSprt.lineTo(x+5,y+5);
      ctxSprt.moveTo(x+5,y-5); ctxSprt.lineTo(x-5,y+5);
      ctxSprt.stroke();
    });

    const cx = X(DATA.times[state.cursor]);
    ctxSprt.strokeStyle = '#ff7f0e'; ctxSprt.lineWidth = 2;
    ctxSprt.beginPath(); ctxSprt.moveTo(cx, PAD_T); ctxSprt.lineTo(cx, H - PAD_B); ctxSprt.stroke();

    sprtTitle.textContent = 'SPRT fault detection - channel: ' + DATA.channels[state.channel] +
      '   (mu0=' + ch.mu0.toFixed(4) + ', sigma=' + ch.sigma.toFixed(4) +
      ', alarms=' + ch.faultIndices.length + ')';
  }

  cSprt.addEventListener('click', e => {
    const rect = cSprt.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const tmin = DATA.times[0], tmax = DATA.times[DATA.k - 1];
    const t = tmin + (mx - PAD_L) / (cSprt.width - PAD_L - PAD_R) * (tmax - tmin);
    let best = 0, bestD = Infinity;
    for (let i = 0; i < DATA.k; ++i) {
      const d = Math.abs(DATA.times[i] - t);
      if (d < bestD) { bestD = d; best = i; }
    }
    setCursor(best);
  });

  slider.addEventListener('input', () => setCursor(parseInt(slider.value, 10)));

  ['training','estimates','residuals','anomalies','trajectory'].forEach(name => {
    document.getElementById('chk_' + name).addEventListener('change', e => {
      state.layers[name] = e.target.checked;
      redraw3D(); redrawSPRT();
    });
  });

  function updateInfo() {
    const i = state.cursor;
    const ch = DATA.sprt[state.channel];
    const names = [DATA.paramNames[0], DATA.paramNames[1], DATA.paramNames[2]];
    const obs = [DATA.objX[i], DATA.objY[i], DATA.objZ[i]];
    const est = [DATA.estX[i], DATA.estY[i], DATA.estZ[i]];
    const lines = [];
    lines.push('sample  #' + (i+1) + '/' + DATA.k);
    lines.push('time    ' + DATA.times[i]);
    lines.push('status  ' + (DATA.anomalyMask[i] ? 'ANOMALY' : 'healthy'));
    lines.push('||R||   ' + DATA.residualNorm[i].toFixed(4));
    lines.push('');
    for (let j = 0; j < 3; ++j) {
      lines.push(names[j].slice(0,12).padEnd(12) + ' obs ' + obs[j].toFixed(3));
      lines.push(''.padEnd(12) + ' est ' + est[j].toFixed(3));
    }
    lines.push('');
    lines.push('SPRT [' + DATA.channels[state.channel].slice(0,14) + ']');
    lines.push('  LLR+  ' + ch.llrPos[i].toFixed(2));
    lines.push('  LLR-  ' + ch.llrNeg[i].toFixed(2));
    info.textContent = lines.join('\n');
  }

  redraw3D();
  redrawSPRT();
})();
"""


# --------------------------------------------------------------------------- #
# The visualizer
# --------------------------------------------------------------------------- #

class MSETAnomalyVisualizer:
    """
    Interactive two-panel matplotlib figure:

      top    - 3D scatter of the input states with MSET estimates, residual
               lines and SPRT-flagged anomalies, all in the same graph;
      bottom - SPRT module: LLR trajectories against the Wald decision
               boundaries with fault alarms, linked to the 3D chart through a
               shared time cursor.
    """

    HEALTHY_CMAP = "viridis"
    ANOMALY_COLOR = "#d62728"      # red
    ESTIMATE_COLOR = "#7f7f7f"     # grey
    TRAINING_COLOR = "#9ecae1"     # light blue
    CURSOR_COLOR = "#ff7f0e"       # orange

    def __init__(self,
                 model: MSET,
                 observations: Sequence[Sequence[float]],
                 times: Optional[Sequence[float]] = None,
                 param_names: Optional[Sequence[str]] = None,
                 alpha: float = 0.01,
                 beta: float = 0.01,
                 disturbance: float = 3.0,
                 time_label: str = "Time (days)") -> None:
        if model.n_params < 3:
            raise ValueError("the 3D chart needs at least three monitored parameters")
        if len(observations) < 2:
            raise ValueError("need at least two observations")

        self.model = model
        self.obs = [list(o) for o in observations]
        self.k = len(self.obs)
        self.times = list(times) if times is not None else list(range(1, self.k + 1))
        if len(self.times) != self.k:
            raise ValueError("times and observations must have the same length")
        self.time_label = time_label

        n = model.n_params
        self.param_names = (list(param_names) if param_names
                            else [f"p{j + 1}" for j in range(n)])

        # ---- MSET quantities -------------------------------------------- #
        self.estimates = [model.estimate(o) for o in self.obs]
        self.residuals = model.residuals(self.obs)                 # scaled units
        self.residual_norms = [euclidean_norm(r) for r in self.residuals]

        # ---- SPRT traces: one per parameter, plus the residual norm ----- #
        stats = model.healthy_statistics()
        healthy_norms = [euclidean_norm(r) for r in model.healthy_residuals()]
        norm_mu = sum(healthy_norms) / len(healthy_norms)
        norm_var = (sum((v - norm_mu) ** 2 for v in healthy_norms)
                    / max(1, len(healthy_norms) - 1))
        norm_sd = max(math.sqrt(norm_var), 1e-9)

        self.sprt_channels: List[str] = self.param_names[:n] + ["||R|| norm"]
        self.sprt_traces: List[SPRTTrace] = []
        for j in range(n):
            mu, sd = stats[j]
            trace = SPRTTrace(mu, sd, alpha, beta, disturbance)
            trace.run([r[j] for r in self.residuals])
            self.sprt_traces.append(trace)
        norm_trace = SPRTTrace(norm_mu, norm_sd, alpha, beta, disturbance)
        norm_trace.run(self.residual_norms)
        self.sprt_traces.append(norm_trace)

        # A sample is an anomaly if ANY channel raised a fault alarm there.
        self.anomaly_mask = [any(t.verdicts[i] == "fault" for t in self.sprt_traces)
                             for i in range(self.k)]
        self.anomaly_indices = [i for i, a in enumerate(self.anomaly_mask) if a]

        # ---- state ------------------------------------------------------- #
        self.current_index = self.k - 1
        self.current_channel = len(self.sprt_channels) - 1   # start on ||R||

        self._build_figure()

    # ------------------------------------------------------------------ #
    # Figure construction
    # ------------------------------------------------------------------ #
    #
    # Every axes below is placed with an explicit [left, bottom, width, height]
    # rectangle in figure-fraction coordinates rather than GridSpec ratios.
    # GridSpec sizes columns/rows *proportionally*, so it can't guarantee a
    # fixed gap in front of a 3D axes' auto-expanding bounding box (its axis
    # labels and tick text render outside the nominal box and were colliding
    # with the colorbar) or between an axes' auto-placed title and its
    # neighbour above. Explicit rectangles let every gap be sized once and
    # verified by rendering, independent of how any one panel's content
    # happens to lay itself out.
    #
    # Layout map (figure-fraction coordinates):
    #
    #   LEFT_L .. LEFT_R   : 3D legend strip, ax3d, SPRT legend strip,
    #                        ax_sprt, slider - all share this x-range.
    #   CBAR_L .. CBAR_R   : colorbar, in the gap between the left column
    #                        and the control column.
    #   RIGHT_L .. RIGHT_R : layers / channel / info control column.
    #
    LEFT_L, LEFT_R = 0.055, 0.595
    CBAR_L, CBAR_R = 0.625, 0.645
    RIGHT_L, RIGHT_R = 0.715, 0.965
    LEFT_W = LEFT_R - LEFT_L

    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(15, 12))
        try:
            self.fig.canvas.manager.set_window_title(
                "MSET input & anomalies (3D) + SPRT fault detection")
        except AttributeError:
            pass  # some embedded/headless backends have no window manager

        L, R = self.LEFT_L, self.LEFT_R
        W = self.LEFT_W

        # -- left column, top to bottom -----------------------------------
        self.ax_legend3d = self.fig.add_axes((L, 0.930, W, 0.035))
        self.ax3d = self.fig.add_axes((L, 0.405, W, 0.478), projection="3d")
        self.ax_legend_sprt = self.fig.add_axes((L, 0.330, W, 0.032))
        self.ax_sprt = self.fig.add_axes((L, 0.148, W, 0.150))
        self.ax_slider = self.fig.add_axes((L + 0.05, 0.078, W - 0.05, 0.022))

        # -- colorbar, in the reserved gap between the columns -------------
        self.ax_cbar = self.fig.add_axes((self.CBAR_L, 0.44, self.CBAR_R - self.CBAR_L, 0.40))

        # -- right-hand control column --------------------------------------
        RL, RR = self.RIGHT_L, self.RIGHT_R
        RW = RR - RL
        self.ax_checks = self.fig.add_axes((RL, 0.780, RW, 0.150))
        self.ax_radio = self.fig.add_axes((RL, 0.500, RW, 0.250))
        self.ax_info = self.fig.add_axes((RL, 0.060, RW, 0.415))

        self.fig.text(0.055, 0.035,
                      "drag to rotate the 3D plot  \u00b7  scroll to zoom  \u00b7  "
                      "click a point to inspect",
                      fontsize=9, color="0.4")

        self._draw_3d()
        self._draw_sprt()
        self._draw_widgets()
        self._connect_events()
        self._sync(self.current_index)

    # ------------------------------------------------------------------ #
    # Top panel - 3D input + anomalies
    # ------------------------------------------------------------------ #
    def _draw_3d(self) -> None:
        ax = self.ax3d
        ax.set_box_aspect((1, 1, 0.85), zoom=1.12)   # fill the panel, with room to spare
        ax.set_title("Input states, MSET estimates and detected anomalies",
                     fontsize=11, pad=8)
        ax.set_xlabel(self.param_names[0], labelpad=6)
        ax.set_ylabel(self.param_names[1], labelpad=6)
        ax.set_zlabel(self.param_names[2], labelpad=6)

        xs = [o[0] for o in self.obs]
        ys = [o[1] for o in self.obs]
        zs = [o[2] for o in self.obs]

        # Healthy training cloud (unscaled memory + remaining training data)
        train = [self.model._unscale(s)
                 for s in (self.model.D_states + self.model.L_states)]
        self.scatter_training = ax.scatter(
            [t[0] for t in train], [t[1] for t in train], [t[2] for t in train],
            s=14, c=self.TRAINING_COLOR, alpha=0.35, depthshade=False,
            label="healthy training data", marker="o")

        # Trajectory of the monitored unit through state space
        self.line_trajectory, = ax.plot(xs, ys, zs, lw=0.8, color="0.55",
                                        alpha=0.7, zorder=1)

        # Healthy observations, colored by residual norm
        healthy_idx = [i for i in range(self.k) if not self.anomaly_mask[i]]
        self.scatter_obs = ax.scatter(
            [xs[i] for i in healthy_idx],
            [ys[i] for i in healthy_idx],
            [zs[i] for i in healthy_idx],
            c=[self.residual_norms[i] for i in healthy_idx],
            cmap=self.HEALTHY_CMAP, s=42, depthshade=False, picker=5,
            label="input (color = ||residual||)")
        self._healthy_idx = healthy_idx

        # Anomalies - SPRT fault alarms
        self.scatter_anom = ax.scatter(
            [xs[i] for i in self.anomaly_indices],
            [ys[i] for i in self.anomaly_indices],
            [zs[i] for i in self.anomaly_indices],
            s=130, c=self.ANOMALY_COLOR, marker="X", depthshade=False,
            picker=5, edgecolors="k", linewidths=0.6,
            label=f"anomaly (SPRT fault, {len(self.anomaly_indices)})")

        # MSET estimates + residual lines obs -> est
        ex = [e[0] for e in self.estimates]
        ey = [e[1] for e in self.estimates]
        ez = [e[2] for e in self.estimates]
        self.scatter_est = ax.scatter(ex, ey, ez, s=22, marker="^",
                                      c=self.ESTIMATE_COLOR, alpha=0.75,
                                      depthshade=False, label="MSET estimate")
        self.residual_lines = []
        for o, e in zip(self.obs, self.estimates):
            ln, = ax.plot([o[0], e[0]], [o[1], e[1]], [o[2], e[2]],
                          color=self.ESTIMATE_COLOR, lw=0.7, alpha=0.5)
            self.residual_lines.append(ln)

        # Current-time highlight (driven by the slider)
        self.cursor3d = ax.scatter([xs[-1]], [ys[-1]], [zs[-1]],
                                   s=340, facecolors="none",
                                   edgecolors=self.CURSOR_COLOR,
                                   linewidths=2.2, depthshade=False,
                                   label="current time")

        cbar = self.fig.colorbar(self.scatter_obs, cax=self.ax_cbar)
        cbar.set_label("residual Euclidean norm  ||R(t)||", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        self._draw_legend_strip(
            self.ax_legend3d,
            [
                Line2D([], [], marker="o", linestyle="none", markersize=6,
                      markerfacecolor=self.TRAINING_COLOR, markeredgecolor="none",
                      alpha=0.7, label="healthy training data"),
                Line2D([], [], marker="o", linestyle="none", markersize=7,
                      markerfacecolor="#31688e", markeredgecolor="none",
                      label="input (colour = ||residual||)"),
                Line2D([], [], marker="^", linestyle="none", markersize=7,
                      markerfacecolor=self.ESTIMATE_COLOR, markeredgecolor="none",
                      alpha=0.85, label="MSET estimate"),
                Line2D([], [], marker="X", linestyle="none", markersize=9,
                      markerfacecolor=self.ANOMALY_COLOR, markeredgecolor="k",
                      label=f"anomaly (SPRT fault, {len(self.anomaly_indices)})"),
                Line2D([], [], marker="o", linestyle="none", markersize=10,
                      markerfacecolor="none", markeredgecolor=self.CURSOR_COLOR,
                      markeredgewidth=2, label="current time"),
            ],
            ncol=5,
        )

    def _draw_legend_strip(self, ax, handles, ncol: int) -> None:
        """
        Render a legend into its own dedicated, data-free axes strip instead
        of overlaying it on a data axes with ax.legend(). A legend box's size
        and exact placement aren't fully predictable in advance (especially
        against a 3D axes' auto-expanding bounding box), so anchoring it
        in-plot risks covering data points as the view rotates or the window
        resizes. A separate strip guarantees the legend can never overlap the
        chart beneath it.
        """
        ax.axis("off")
        ax.legend(handles=handles, loc="center", ncol=ncol, fontsize=7.5,
                  frameon=False, borderaxespad=0.0, handletextpad=0.5,
                  columnspacing=1.4, handlelength=1.6)

    # ------------------------------------------------------------------ #
    # Bottom panel - SPRT module
    # ------------------------------------------------------------------ #
    def _draw_sprt(self) -> None:
        ax = self.ax_sprt
        ax.set_xlabel(self.time_label, fontsize=9)
        ax.set_ylabel("cumulative log-likelihood ratio", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.25)

        trace = self.sprt_traces[self.current_channel]

        self.line_pos, = ax.plot(self.times, trace.llr_pos, lw=1.6,
                                 color="#1f77b4", label="LLR  H1+ (positive shift)")
        self.line_neg, = ax.plot(self.times, trace.llr_neg, lw=1.6,
                                 color="#2ca02c", label="LLR  H1- (negative shift)")

        self.hline_upper = ax.axhline(trace.upper, color=self.ANOMALY_COLOR,
                                      ls="--", lw=1.2,
                                      label="fault boundary  ln((1-b)/a)")
        self.hline_lower = ax.axhline(trace.lower, color="0.35", ls=":", lw=1.2,
                                      label="healthy boundary  ln(b/(1-a))")
        ax.axhline(0.0, color="0.8", lw=0.8)

        fx = [self.times[i] for i in trace.fault_indices]
        fy = [max(trace.llr_pos[i], trace.llr_neg[i]) for i in trace.fault_indices]
        self.fault_markers, = ax.plot(fx, fy, "X", ms=10,
                                      color=self.ANOMALY_COLOR, mec="k",
                                      mew=0.6, ls="none", label="fault alarm")

        # Shade anomalous samples (any channel) so both panels tell one story
        self.anom_spans = []
        for i in self.anomaly_indices:
            half = ((self.times[1] - self.times[0]) / 2.0) if self.k > 1 else 0.5
            span = ax.axvspan(self.times[i] - half, self.times[i] + half,
                              color=self.ANOMALY_COLOR, alpha=0.07)
            self.anom_spans.append(span)

        self.cursor_line = ax.axvline(self.times[self.current_index],
                                      color=self.CURSOR_COLOR, lw=1.8, alpha=0.9)

        self._set_sprt_title()
        self._draw_legend_strip(
            self.ax_legend_sprt,
            [
                Line2D([], [], color="#1f77b4", lw=1.6, label="LLR H1+ (positive shift)"),
                Line2D([], [], color="#2ca02c", lw=1.6, label="LLR H1- (negative shift)"),
                Line2D([], [], color=self.ANOMALY_COLOR, ls="--", lw=1.2,
                      label="fault boundary  ln((1-b)/a)"),
                Line2D([], [], color="0.35", ls=":", lw=1.2,
                      label="healthy boundary  ln(b/(1-a))"),
                Line2D([], [], marker="X", linestyle="none", markersize=9,
                      markerfacecolor=self.ANOMALY_COLOR, markeredgecolor="k",
                      label="fault alarm"),
            ],
            ncol=5,
        )
        ax.margins(x=0.01)

    def _set_sprt_title(self) -> None:
        trace = self.sprt_traces[self.current_channel]
        name = self.sprt_channels[self.current_channel]
        self.ax_sprt.set_title(
            f"SPRT fault detection - channel: {name}   "
            f"(mu0={trace.mu0:+.4f}, sigma={trace.sigma:.4f}, "
            f"alarms={len(trace.fault_indices)})",
            fontsize=10)

    def _refresh_sprt(self) -> None:
        """Redraw the SPRT lines for the currently selected channel."""
        trace = self.sprt_traces[self.current_channel]
        self.line_pos.set_data(self.times, trace.llr_pos)
        self.line_neg.set_data(self.times, trace.llr_neg)
        self.hline_upper.set_ydata([trace.upper, trace.upper])
        self.hline_lower.set_ydata([trace.lower, trace.lower])
        fx = [self.times[i] for i in trace.fault_indices]
        fy = [max(trace.llr_pos[i], trace.llr_neg[i]) for i in trace.fault_indices]
        self.fault_markers.set_data(fx, fy)
        self._set_sprt_title()
        self.ax_sprt.relim()
        self.ax_sprt.autoscale_view()
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------ #
    # Widgets
    # ------------------------------------------------------------------ #
    def _draw_widgets(self) -> None:
        # Time slider
        self.slider = Slider(self.ax_slider, "time index",
                             valmin=0, valmax=self.k - 1,
                             valinit=self.current_index, valstep=1,
                             color=self.CURSOR_COLOR)
        self.slider.label.set_fontsize(9)
        self.slider.on_changed(self._on_slider)

        # Layer toggles
        self.ax_checks.set_title("layers", fontsize=9)
        self.check_labels = ["training data", "MSET estimates",
                             "residual lines", "anomalies", "trajectory"]
        self.checks = CheckButtons(self.ax_checks, self.check_labels,
                                   [True] * len(self.check_labels))
        for lbl in self.checks.labels:
            lbl.set_fontsize(8)
        self.checks.on_clicked(self._on_check)

        # SPRT channel selector
        self.ax_radio.set_title("SPRT channel", fontsize=9)
        self.radio = RadioButtons(self.ax_radio, self.sprt_channels,
                                  active=self.current_channel)
        for lbl in self.radio.labels:
            lbl.set_fontsize(8)
        self.radio.on_clicked(self._on_radio)

        # Info panel
        self.ax_info.set_axis_off()
        self.info_text = self.ax_info.text(
            0.02, 0.98, "", transform=self.ax_info.transAxes,
            fontsize=8, family="monospace", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.45", fc="#f7f7f7", ec="0.7"))

    # ------------------------------------------------------------------ #
    # Events / synchronization
    # ------------------------------------------------------------------ #
    def _connect_events(self) -> None:
        self.fig.canvas.mpl_connect("pick_event", self._on_pick)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    def _on_slider(self, value: float) -> None:
        self._sync(int(round(value)))

    def _on_check(self, label: str) -> None:
        states = dict(zip(self.check_labels, self.checks.get_status()))
        self.scatter_training.set_visible(states["training data"])
        self.scatter_est.set_visible(states["MSET estimates"])
        for ln in self.residual_lines:
            ln.set_visible(states["residual lines"])
        self.scatter_anom.set_visible(states["anomalies"])
        for span in self.anom_spans:
            span.set_visible(states["anomalies"])
        self.fault_markers.set_visible(states["anomalies"])
        self.line_trajectory.set_visible(states["trajectory"])
        self.fig.canvas.draw_idle()

    def _on_radio(self, label: str) -> None:
        self.current_channel = self.sprt_channels.index(label)
        self._refresh_sprt()

    def _on_pick(self, event) -> None:
        """Click a 3D point -> jump the cursor to that sample."""
        if event.artist is self.scatter_obs and len(event.ind):
            self._sync(self._healthy_idx[event.ind[0]], move_slider=True)
        elif event.artist is self.scatter_anom and len(event.ind):
            self._sync(self.anomaly_indices[event.ind[0]], move_slider=True)

    def _on_click(self, event) -> None:
        """Click on the SPRT chart -> jump the cursor to that time."""
        if event.inaxes is not self.ax_sprt or event.xdata is None:
            return
        idx = min(range(self.k), key=lambda i: abs(self.times[i] - event.xdata))
        self._sync(idx, move_slider=True)

    def _sync(self, index: int, move_slider: bool = False) -> None:
        """Move the shared time cursor: 3D highlight + SPRT line + info text."""
        index = max(0, min(self.k - 1, index))
        self.current_index = index

        o = self.obs[index]
        self.cursor3d._offsets3d = ([o[0]], [o[1]], [o[2]])
        self.cursor_line.set_xdata([self.times[index], self.times[index]])

        if move_slider:
            self.slider.eventson = False
            self.slider.set_val(index)
            self.slider.eventson = True

        trace = self.sprt_traces[self.current_channel]
        est = self.estimates[index]
        res = self.residuals[index]
        lines = [
            f"sample  #{index + 1}/{self.k}",
            f"time    {self.times[index]:g}",
            f"status  {'ANOMALY' if self.anomaly_mask[index] else 'healthy'}",
            f"||R||   {self.residual_norms[index]:.4f}",
            "",
        ]
        for j in range(min(3, self.model.n_params)):
            lines.append(f"{self.param_names[j][:10]:<10}"
                         f" obs {o[j]:9.3f}")
            lines.append(f"{'':<10} est {est[j]:9.3f}  r {res[j]:+.3f}")
        lines += [
            "",
            f"SPRT [{self.sprt_channels[self.current_channel][:10]}]",
            f"  LLR+  {trace.llr_pos[index]:+8.2f}",
            f"  LLR-  {trace.llr_neg[index]:+8.2f}",
            f"  verdict: {trace.verdicts[index]}",
        ]
        self.info_text.set_text("\n".join(lines))
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------ #
    def show(self) -> None:
        plt.show()

    def save(self, path: str, dpi: int = 140) -> None:
        self.fig.savefig(path, dpi=dpi, bbox_inches="tight")

    def to_html(self,
                path: str = "mset_visualization.html",
                title: Optional[str] = None,
                time_label: Optional[str] = None) -> str:
        """
        Export this visualization as a single self-contained, interactive
        HTML file (Canvas 2D + vanilla JS, no external dependencies, no
        network access) - the browser-native equivalent of this matplotlib
        figure, for sharing or viewing outside a Python/matplotlib session.

        Reuses the same JS renderer as mset_visualizer.c, so the Python and
        C versions produce the same rotate/zoom/pick/slider/checkbox/radio
        interactions and visual layout. All data already computed in
        __init__ (estimates, residuals, SPRT traces, anomaly mask) is reused
        directly - nothing is recomputed.

        Returns the path written to.
        """
        n = self.model.n_params
        train = [self.model._unscale(s)
                for s in (self.model.D_states + self.model.L_states)]

        data = {
            "title": title or "MSET input & anomalies (3D) + SPRT fault detection",
            "timeLabel": time_label or self.time_label,
            "paramNames": list(self.param_names[:n]),
            "k": self.k,
            "times": self.times,
            "objX": [o[0] for o in self.obs],
            "objY": [o[1] for o in self.obs],
            "objZ": [o[2] for o in self.obs],
            "estX": [e[0] for e in self.estimates],
            "estY": [e[1] for e in self.estimates],
            "estZ": [e[2] for e in self.estimates],
            "residualNorm": self.residual_norms,
            "anomalyMask": self.anomaly_mask,
            "trainX": [t[0] for t in train],
            "trainY": [t[1] for t in train],
            "trainZ": [t[2] for t in train],
            "channels": self.sprt_channels,
            "sprt": [
                {
                    "mu0": tr.mu0, "sigma": tr.sigma,
                    "upper": tr.upper, "lower": tr.lower,
                    "llrPos": tr.llr_pos, "llrNeg": tr.llr_neg,
                    "faultIndices": tr.fault_indices,
                }
                for tr in self.sprt_traces
            ],
        }

        html = (_HTML_TOP
               + "<script>\nconst DATA = " + json.dumps(data) + ";\n"
               + _JS_TEMPLATE
               + "</script>\n</body>\n</html>\n")

        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        return path


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #

def demo(show: bool = True, save_path: Optional[str] = None,
        html_path: Optional[str] = None) -> MSETAnomalyVisualizer:
    """
    Self-contained demonstration reusing the synthetic-data generator of
    mset_rul.py: a healthy training history, then a monitored unit that
    develops a drift which the SPRT flags as anomalous.
    """
    import random
    from mset_rul import _generate_unit, inverse_distance_kernel

    rng = random.Random(20070101)
    training = _generate_unit(days=60, drift_rate=0.0, rng=rng)
    model = MSET(kernel=inverse_distance_kernel(scale=0.30)).fit(training,
                                                                 memory_size=14)

    # Monitored unit: healthy for ~2 weeks, then degrading
    healthy_part = _generate_unit(days=14, drift_rate=0.0005, rng=rng)
    drifting_part = _generate_unit(days=22, drift_rate=0.030, rng=rng)
    monitored = healthy_part + drifting_part
    times = list(range(1, len(monitored) + 1))

    viz = MSETAnomalyVisualizer(
        model=model,
        observations=monitored,
        times=times,
        param_names=("Temperature [degC]", "Rel. humidity [%]", "Vibration [g]"),
        alpha=0.01, beta=0.01, disturbance=3.0,
    )
    if save_path:
        viz.save(save_path)
    if html_path:
        viz.to_html(html_path)
    if show:
        viz.show()
    return viz


if __name__ == "__main__":
    demo(html_path="assets/anomaly_detection.html")
