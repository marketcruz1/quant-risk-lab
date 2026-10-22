# =============================================================================
# models/visuals.py: diseño de la app (colores, estilo de las pestañas y la
# animación de Monte Carlo). Aquí no hay ningún modelo de riesgo: solo aspecto.
# La animación se dibuja en el navegador con JavaScript (un <canvas>), así que
# no hace falta ninguna librería nueva.
# =============================================================================

import json

import numpy as np

# --- Paleta de colores (la misma que .streamlit/config.toml) ------------------
BG = "#0B1220"        # fondo, azul noche
PANEL = "#111A2E"     # tarjetas y barra de pestañas
BORDER = "#1E2A44"    # bordes suaves
TEXT = "#E6EDF3"      # texto principal
MUTED = "#94A3B8"     # texto secundario
ACCENT = "#2DD4BF"    # turquesa (color principal)
BLUE = "#60A5FA"      # series principales de los gráficos
RED = "#F87171"       # pérdidas, alertas, VaR
GREY = "#64748B"      # series secundarias
GREY_LIGHT = "#475569"

GITHUB_URL = "https://github.com/marketcruz1/quant-risk-lab"

# --- CSS: pestañas grandes tipo "píldora", métricas como tarjetas -------------
CSS = f"""
<style>
/* menos espacio vacío arriba */
[data-testid="stMainBlockContainer"] {{ padding-top: 3.6rem; }}
/* barra de pestañas principal (estilo "píldora") */
.stTabs [role="tablist"] {{
    gap: 6px; background: {PANEL}; padding: 6px; border-radius: 14px;
    border: 1px solid {BORDER}; flex-wrap: wrap; width: fit-content; max-width: 100%;
}}
.stTabs [role="tab"] {{
    height: auto; padding: 9px 20px; border-radius: 10px; background: transparent;
    color: {MUTED}; font-weight: 600; transition: background .15s ease, color .15s ease;
}}
.stTabs [role="tab"] p {{ font-size: 1.02rem; }}
.stTabs [role="tab"]:hover {{ color: {TEXT}; background: rgba(148,163,184,.08); }}
.stTabs [role="tab"][aria-selected="true"] {{
    color: {ACCENT}; background: rgba(45,212,191,.12);
    box-shadow: inset 0 0 0 1px rgba(45,212,191,.35);
}}
.stTabs .react-aria-SelectionIndicator {{ display: none; }}
/* sub-pestañas (dentro de una pestaña): sin fondo y algo más pequeñas */
.stTabs .stTabs [role="tablist"] {{ background: transparent; border: none; padding: 0; }}
.stTabs .stTabs [role="tab"] {{ padding: 6px 14px; box-shadow: inset 0 0 0 1px {BORDER}; }}
.stTabs .stTabs [role="tab"] p {{ font-size: .92rem; }}
/* métricas como tarjetas */
[data-testid="stMetric"] {{
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px; padding: 14px 16px;
}}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; }}
/* desplegables */
[data-testid="stExpander"] details {{ border-color: {BORDER}; border-radius: 12px; }}
/* cabecera de la página */
.qrl-brand {{ display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; margin: -8px 0 6px; }}
.qrl-brand .logo {{ font-size: 1.55rem; font-weight: 700; color: {TEXT}; letter-spacing: -.01em; }}
.qrl-brand .logo span {{ color: {ACCENT}; }}
.qrl-brand .tag {{ color: {MUTED}; font-size: .95rem; }}
/* tarjetas de la portada */
.qrl-chip {{
    display: inline-block; padding: 3px 10px; margin: 0 6px 6px 0; border-radius: 999px;
    font-size: .8rem; color: {ACCENT}; background: rgba(45,212,191,.10);
    border: 1px solid rgba(45,212,191,.25);
}}
.qrl-kicker {{ color: {ACCENT}; font-size: .78rem; font-weight: 700; letter-spacing: .12em; }}
.qrl-muted {{ color: {MUTED}; }}
</style>
"""

BRAND = """
<div class="qrl-brand">
  <div class="logo">Quant Risk Lab<span>.</span></div>
  <div class="tag">Market risk and trading surveillance models on public data</div>
</div>
"""


def chips(items):
    """Etiquetas pequeñas (HTML) con las técnicas de cada modelo."""
    return "".join(f'<span class="qrl-chip">{i}</span>' for i in items)


# =============================================================================
# Animación de Monte Carlo
# Las trayectorias se dibujan poco a poco, como raíces que crecen desde un
# punto. Al terminar aparece la distribución del resultado final y la línea
# del VaR. Después se borra y empieza otra vez con trayectorias nuevas.
# =============================================================================
_ENGINE = r"""
<!doctype html><html><head><meta charset="utf-8">
<style>
  html, body { margin: 0; padding: 0; background: transparent; overflow: hidden; }
  body { font-family: "Source Sans Pro", "Source Sans 3", system-ui, -apple-system, "Segoe UI", sans-serif; }
  #wrap { position: relative; width: 100%; height: __HEIGHT__px; border-radius: 16px;
          background: radial-gradient(ellipse at 70% 40%, #12213A 0%, #0B1220 70%);
          border: 1px solid #1E2A44; overflow: hidden; }
  canvas { position: absolute; inset: 0; width: 100%; height: 100%; }
  #text { position: absolute; left: 32px; top: 28px; max-width: 420px; pointer-events: none; }
  .kicker { color: #2DD4BF; font-size: 12px; font-weight: 700; letter-spacing: .14em; }
  h1 { color: #E6EDF3; font-size: 40px; line-height: 1.08; margin: 10px 0 12px; font-weight: 700; }
  h1 span { color: #2DD4BF; }
  p { color: #94A3B8; font-size: 16px; line-height: 1.5; margin: 0; }
  #legend { position: absolute; right: 18px; top: 10px; color: #94A3B8; font-size: 12px; }
  @media (max-width: 760px) { #text { display: none; } }
</style></head><body>
<div id="wrap"><canvas id="c"></canvas>__TEXT__<div id="legend"></div></div>
<script>
// Parámetros que llegan desde Python
const CFG = __CONFIG__;
const canvas = document.getElementById("c");
const ctx = canvas.getContext("2d");
const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
let W = 0, H = 0, paths = [], order = [], t0 = null, setIdx = 0;

// Generador aleatorio con semilla (para que sea reproducible)
let seed = 7;
function rnd() { seed = (seed * 1664525 + 1013904223) % 4294967296; return seed / 4294967296; }
function gauss() { let u = 0; while (u === 0) u = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rnd()); }

// Trayectorias: o vienen ya calculadas desde Python, o se simulan aquí (GBM)
function newPaths() {
  if (CFG.sets) { const s = CFG.sets[setIdx % CFG.sets.length]; setIdx++; return s; }
  const out = [], dt = 1 / CFG.steps;
  for (let i = 0; i < CFG.n; i++) {
    const p = [CFG.s0]; let s = CFG.s0;
    for (let k = 0; k < CFG.steps; k++) {
      s *= Math.exp((CFG.mu - 0.5 * CFG.sigma ** 2) * dt + CFG.sigma * Math.sqrt(dt) * gauss());
      p.push(s);
    }
    out.push(p);
  }
  return out;
}

function quantile(sorted, q) { return sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor(q * sorted.length)))]; }

let yMin, yMax, finals, varLevel, esLevel;
function prepare() {
  paths = newPaths();
  finals = paths.map(p => p[p.length - 1]).slice().sort((a, b) => a - b);
  const all = paths.flat().slice().sort((a, b) => a - b);
  yMin = quantile(all, 0.002); yMax = quantile(all, 0.998);
  const pad = (yMax - yMin) * 0.08; yMin -= pad; yMax += pad;
  varLevel = quantile(finals, 1 - CFG.conf);
  const tail = finals.filter(v => v <= varLevel);
  esLevel = tail.reduce((a, b) => a + b, 0) / tail.length;
  // color de cada trayectoria según su valor final (rojo abajo, turquesa arriba)
  const rank = new Map(); finals.forEach((v, i) => rank.set(v, i / (finals.length - 1)));
  order = paths.map(p => rank.get(p[p.length - 1]));
  t0 = null;
}

function resize() {
  const dpr = window.devicePixelRatio || 1;
  const r = canvas.getBoundingClientRect();
  W = r.width; H = r.height;
  if (W < 10) return;
  canvas.width = W * dpr; canvas.height = H * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function mix(a, b, t) { return a.map((v, i) => Math.round(v + (b[i] - v) * t)); }
const RED = [248, 113, 113], MID = [100, 116, 139], TEAL = [45, 212, 191];
function color(r, alpha) {
  const c = r < 0.5 ? mix(RED, MID, r / 0.5) : mix(MID, TEAL, (r - 0.5) / 0.5);
  return `rgba(${c[0]},${c[1]},${c[2]},${alpha})`;
}

// zona del gráfico: las trayectorias a la izquierda, la distribución a la derecha
function box() {
  const left = W * (W < 760 ? Math.min(CFG.left, 0.06) : CFG.left), right = W - Math.max(90, W * 0.13), top = 26, bottom = H - 34;
  return { left, right, top, bottom, histW: W - right - 14 };
}
function X(b, k) { return b.left + (b.right - b.left) * k / CFG.steps; }
function Y(b, v) { return b.bottom - (b.bottom - b.top) * (v - yMin) / (yMax - yMin); }

function draw(progress, histP, varP, fade) {
  ctx.clearRect(0, 0, W, H);
  const b = box();
  ctx.globalAlpha = fade;
  // línea del valor inicial
  ctx.strokeStyle = "rgba(148,163,184,0.25)"; ctx.setLineDash([3, 5]); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(b.left, Y(b, CFG.s0)); ctx.lineTo(b.right, Y(b, CFG.s0)); ctx.stroke();
  ctx.setLineDash([]);
  // trayectorias hasta el paso actual (crecen desde el origen)
  const kMax = progress * CFG.steps, kInt = Math.floor(kMax), frac = kMax - kInt;
  ctx.lineWidth = 1.1;
  for (let i = 0; i < paths.length; i++) {
    const p = paths[i];
    ctx.strokeStyle = color(order[i], 0.32);
    ctx.beginPath(); ctx.moveTo(X(b, 0), Y(b, p[0]));
    for (let k = 1; k <= kInt; k++) ctx.lineTo(X(b, k), Y(b, p[k]));
    let tipX = X(b, kInt), tipY = Y(b, p[kInt]);
    if (kInt < CFG.steps) {
      tipX = X(b, kInt + frac); tipY = Y(b, p[kInt] + (p[kInt + 1] - p[kInt]) * frac);
      ctx.lineTo(tipX, tipY);
    }
    ctx.stroke();
    // punta brillante mientras crece
    if (progress < 1) {
      ctx.fillStyle = color(order[i], 0.9);
      ctx.beginPath(); ctx.arc(tipX, tipY, 1.6, 0, 2 * Math.PI); ctx.fill();
    }
  }
  // origen
  ctx.fillStyle = "#E6EDF3"; ctx.shadowColor = "#2DD4BF"; ctx.shadowBlur = 12;
  ctx.beginPath(); ctx.arc(X(b, 0), Y(b, CFG.s0), 3.5, 0, 2 * Math.PI); ctx.fill();
  ctx.shadowBlur = 0;
  // ejes mínimos
  ctx.fillStyle = "#64748B"; ctx.font = "12px sans-serif"; ctx.textAlign = "left";
  ctx.fillText(CFG.xStart, b.left - 4, H - 12);
  ctx.textAlign = "right"; ctx.fillText(CFG.xEnd, b.right, H - 12);
  // distribución del valor final (histograma horizontal que crece)
  if (histP > 0) {
    const bins = 28, counts = new Array(bins).fill(0);
    finals.forEach(v => { const j = Math.floor((v - yMin) / (yMax - yMin) * bins); if (j >= 0 && j < bins) counts[j]++; });
    const cMax = Math.max(...counts), hb = (b.bottom - b.top) / bins;
    for (let j = 0; j < bins; j++) {
      const v0 = yMin + (yMax - yMin) * j / bins;
      const len = counts[j] / cMax * b.histW * histP;
      ctx.fillStyle = v0 + (yMax - yMin) / bins <= varLevel ? "rgba(248,113,113,0.85)" : "rgba(96,165,250,0.55)";
      ctx.fillRect(b.right + 10, Y(b, v0) - hb + 1, len, hb - 2);
    }
  }
  // línea del VaR (de izquierda a derecha) y del ES
  if (varP > 0) {
    const yV = Y(b, varLevel), xEnd = b.left + (b.right + 10 + b.histW - b.left) * varP;
    ctx.strokeStyle = "#F87171"; ctx.lineWidth = 1.6; ctx.setLineDash([6, 4]);
    ctx.beginPath(); ctx.moveTo(b.left, yV); ctx.lineTo(xEnd, yV); ctx.stroke(); ctx.setLineDash([]);
    if (varP >= 1) {
      ctx.fillStyle = "#F87171"; ctx.font = "600 13px sans-serif"; ctx.textAlign = "right";
      ctx.fillText(CFG.label(varLevel, esLevel), b.right - 4, yV - 8);
    }
  }
  ctx.globalAlpha = 1;
}

// Etiqueta del VaR: % de pérdida (portada) o importe (cartera)
CFG.label = function (v, es) {
  const pct = CFG.conf * 100;
  const conf = Number.isInteger(pct) ? pct.toFixed(0) : pct.toFixed(1);
  if (CFG.money) {
    const f = x => CFG.ccy + " " + Math.round(CFG.s0 - x).toLocaleString("en-US");
    return `VaR ${conf}%: ${f(v)}  ·  ES: ${f(es)}`;
  }
  return `VaR ${conf}%: ${((1 - v / CFG.s0) * 100).toFixed(1)}% loss  ·  ES: ${((1 - es / CFG.s0) * 100).toFixed(1)}%`;
};

// Tiempos de cada fase (milisegundos)
const GROW = CFG.grow, HIST = 900, LINE = 700, HOLD = 3200, FADE = 700;
function frame(ts) {
  if (W < 10) { resize(); requestAnimationFrame(frame); return; }
  if (t0 === null) t0 = ts;
  const t = ts - t0;
  const ease = x => 1 - Math.pow(1 - Math.min(1, Math.max(0, x)), 2);
  const g = Math.min(1, t / GROW);
  const h = ease((t - GROW) / HIST);
  const l = ease((t - GROW - HIST) / LINE);
  const endT = GROW + HIST + LINE + HOLD;
  const f = t > endT ? Math.max(0, 1 - (t - endT) / FADE) : 1;
  draw(g, h, l, f);
  if (t > endT + FADE) prepare();
  requestAnimationFrame(frame);
}

document.getElementById("legend").textContent = CFG.legend;
window.addEventListener("resize", resize);
if (window.ResizeObserver) new ResizeObserver(resize).observe(canvas);
resize(); prepare();
if (reduce) { // sin animación: se dibuja directamente el resultado final
  const still = () => { resize(); if (W >= 10) draw(1, 1, 1, 1); };
  still(); window.addEventListener("resize", still);
} else {
  requestAnimationFrame(frame);
}
</script></body></html>
"""

_HERO_TEXT = """
<div id="text">
  <div class="kicker">MARKET RISK · SURVEILLANCE · QUANT</div>
  <h1>Quant Risk<br>Lab<span>.</span></h1>
  <p>Risk models running live on public market data: Value at Risk, Expected Shortfall,
     stress testing, yield curve risk and unusual-volume surveillance.</p>
</div>
"""


def _html(config, height, text=""):
    return (_ENGINE.replace("__HEIGHT__", str(height))
            .replace("__TEXT__", text)
            .replace("__CONFIG__", json.dumps(config)))


def hero_html(height=440):
    """Portada: Monte Carlo ilustrativo de un activo (GBM), calculado en el navegador."""
    config = {"n": 140, "steps": 252, "s0": 100, "mu": 0.07, "sigma": 0.20, "conf": 0.99,
              "left": 0.42, "grow": 4200, "money": False, "xStart": "Today",
              "xEnd": "1 year",
              "legend": "Illustrative: 140 simulated one-year price paths "
                        "(geometric Brownian motion, 7% drift, 20% volatility)"}
    return _html(config, height, _HERO_TEXT)


def portfolio_mc_html(daily_pnl, value, ccy, confidence, horizon=10, n_paths=150,
                      n_sets=3, height=360, seed=42):
    """Animación con los datos reales de la cartera: trayectorias de 10 días
    construidas remuestreando (bootstrap) las pérdidas y ganancias diarias de
    la ventana de estimación. Python calcula las trayectorias; el navegador
    solo las dibuja."""
    rng = np.random.default_rng(seed)
    pnl = np.asarray(daily_pnl, dtype=float)
    sets = []
    for _ in range(n_sets):
        draws = rng.choice(pnl, size=(n_paths, horizon), replace=True)
        paths = value + np.concatenate([np.zeros((n_paths, 1)), draws.cumsum(axis=1)], axis=1)
        sets.append(np.round(paths, 0).tolist())
    config = {"sets": sets, "steps": horizon, "s0": value, "conf": confidence,
              "left": 0.04, "grow": 3000, "money": True, "ccy": ccy,
              "xStart": "Today", "xEnd": f"Day {horizon}",
              "legend": f"{n_paths} simulated {horizon}-day paths, resampling the "
                        "portfolio's historical daily P&L"}
    return _html(config, height)
