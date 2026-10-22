# =============================================================================
# models/rates.py
#
# Modelo: Market Risk Engine - riesgo de tipos de interés (curva del euro)
# Versión pública para la demo (datos públicos del Banco Central Europeo).
#
# QUÉ HACE:
#   1) Descarga la curva cupón cero AAA de la zona euro del BCE (9 plazos,
#      de 1 a 30 años) y la limpia: festivos y plazos que faltan un día se
#      rellenan interpolando con los otros plazos del MISMO día.
#   2) Valora una cartera ficticia de dos bonos del Estado (5 y 10 años) con
#      esa curva y calcula DV01, key-rate DV01, duración modificada y el
#      efecto de la convexidad (revaluación completa vs DV01).
#   3) PCA de los cambios diarios de la curva: nivel, pendiente y curvatura,
#      y cuánto explica cada componente.
#   4) VaR de tipos de la cartera de bonos con tres métodos: covarianza
#      completa de los 9 plazos, solo 3 componentes principales, y simulación
#      histórica con revaluación completa.
#
# SIMPLIFICACIONES (decirlas siempre):
#   - Bonos "genéricos": vencimiento residual fijo (4.6 y 9.3 años), cupón
#     anual, sin riesgo de crédito ni diferencial frente a la curva AAA.
#   - Interpolación lineal de los tipos cero entre plazos.
# =============================================================================

import io
import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import requests   # viene instalado con yfinance

# --- Parámetros ---------------------------------------------------------------
TENORS = [1, 2, 3, 5, 7, 10, 15, 20, 30]          # plazos de la curva (años)
START_DATE = "2006-01-01"
ECB_URL = ("https://data-api.ecb.europa.eu/service/data/YC/"
           "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_{tenor}Y?format=csvdata&startPeriod={start}")
FFILL_LIMIT = 3          # festivos: copiar el último valor como mucho 3 días
N_COMPONENTS = 3         # componentes principales que se usan en el VaR PCA

# Cartera ficticia de bonos: (nombre, nominal EUR, cupón anual, vencimiento en años)
BONDS = [
    ("EUR govt 5Y 2.25%", 300_000, 0.0225, 4.6),
    ("EUR govt 10Y 2.75%", 300_000, 0.0275, 9.3),
]

# Escenarios de tipos: movimiento en puntos básicos para cada plazo
RATE_SCENARIOS = {
    "Parallel +200bp": {t: 200 for t in TENORS},
    "Parallel +100bp": {t: 100 for t in TENORS},
    "Parallel -100bp": {t: -100 for t in TENORS},
    "Bear steepener (+25bp 1Y to +125bp 30Y)": {t: 25 + (t - 1) / 29 * 100 for t in TENORS},
    "Bull flattener (-25bp 1Y to -125bp 30Y)": {t: -25 - (t - 1) / 29 * 100 for t in TENORS},
}

_NORMAL = NormalDist()


# =============================================================================
# 1) DATOS: curva del BCE
# =============================================================================
def _parse_ecb_csv(text):
    """Serie (fecha -> valor) de una respuesta CSV del BCE."""
    raw = pd.read_csv(io.StringIO(text))
    cols = {c.upper(): c for c in raw.columns}
    time_col = cols.get("TIME_PERIOD")
    value_col = cols.get("OBS_VALUE")
    if time_col is None or value_col is None or raw.empty:
        raise ValueError("Unexpected ECB response")
    out = pd.Series(pd.to_numeric(raw[value_col], errors="coerce").to_numpy(),
                    index=pd.to_datetime(raw[time_col]))
    return out.groupby(level=0).last().dropna()


def download_curve():
    """Curva cupón cero AAA del euro (decimales), una columna por plazo.
    Si el BCE falla devuelve un DataFrame vacío."""
    series = {}
    try:
        for t in TENORS:
            url = ECB_URL.format(tenor=t, start=START_DATE)
            resp = requests.get(url, timeout=60, headers={"Accept": "text/csv"})
            resp.raise_for_status()
            series[t] = _parse_ecb_csv(resp.text)
    except Exception:
        return pd.DataFrame()
    curve = pd.DataFrame(series)[TENORS] / 100      # % -> decimales
    return curve.sort_index()


def clean_curve(raw):
    """Limpieza de la curva. Devuelve (curva limpia, registro de incidencias).
    - Días sin ningún dato: se eliminan.
    - Un plazo que falta un día: se interpola con los demás plazos de ese día
      (mejor que copiar el valor de ayer: la curva se ha movido hoy).
    - Si faltan casi todos los plazos: se copia el día anterior (máx. 3 días)."""
    log = []
    curve = raw.dropna(how="all").copy()
    tenors = np.array(TENORS, dtype=float)
    partial = curve.isna().any(axis=1) & (curve.notna().sum(axis=1) >= 2)
    for date in curve.index[partial]:
        row = curve.loc[date]
        have = row.notna().to_numpy()
        curve.loc[date, ~have] = np.interp(tenors[~have], tenors[have], row[have].to_numpy())
        missing = ", ".join(f"{t:g}Y" for t in tenors[~have])
        log.append([date.strftime("%Y-%m-%d"), missing, "Missing tenor",
                    "Linear interpolation from the other tenors"])
    n_before = int(curve.isna().sum().sum())
    curve = curve.ffill(limit=FFILL_LIMIT).dropna()
    if n_before:
        log.append(["Several dates", "All tenors", "Missing day",
                    f"{n_before} values carried forward (max {FFILL_LIMIT} days)"])
    log = pd.DataFrame(log, columns=["Date", "Tenor", "Issue", "Treatment"])
    return curve, log


# =============================================================================
# 2) BONOS: precio, DV01, key-rate DV01, duración y convexidad
# =============================================================================
def _interp_weights(times):
    """Matriz W con tipo(times) = W @ tipos de los plazos (interpolación lineal,
    plana fuera del rango 1-30 años). Así un mismo W sirve para miles de curvas."""
    eye = np.eye(len(TENORS))
    t = np.clip(times, TENORS[0], TENORS[-1])
    return np.column_stack([np.interp(t, TENORS, eye[j]) for j in range(len(TENORS))])


class Bond:
    """Bono bullet con cupón anual, valorado con la curva cupón cero
    (capitalización continua): precio = suma de flujo * exp(-z(t) * t)."""

    def __init__(self, name, notional, coupon, maturity):
        self.name = name
        n = int(math.ceil(maturity - 1e-9))
        self.times = np.array([maturity - k for k in range(n)][::-1])   # fechas de pago
        self.cashflows = np.full(n, coupon * notional)
        self.cashflows[-1] += notional
        self.weights = _interp_weights(self.times)

    def price(self, curves):
        """Precio para una curva (vector) o muchas a la vez (matriz días x plazos)."""
        z = np.atleast_2d(curves) @ self.weights.T
        pv = (np.exp(-z * self.times) * self.cashflows).sum(axis=1)
        return pv if np.ndim(curves) == 2 else float(pv[0])

    def key_rate_dv01(self, curve):
        """Cuánto GANA el bono si baja 1pb cada plazo por separado."""
        bumped = np.asarray(curve) + np.eye(len(TENORS)) * 1e-4
        return self.price(curve) - self.price(bumped)


def bond_book():
    return [Bond(*b) for b in BONDS]


def book_price(bonds, curves):
    return sum(b.price(curves) for b in bonds)


# =============================================================================
# 3) PCA de los cambios diarios de la curva
# =============================================================================
def curve_pca(changes_bp):
    """Componentes principales de la matriz de covarianzas de los cambios
    diarios (en puntos básicos). Devuelve (autovalores, autovectores) ordenados
    de mayor a menor, con signos fijados para que se lean igual siempre:
    PC1 (nivel) positiva, PC2 (pendiente) sube con el plazo,
    PC3 (curvatura) positiva en la zona media de la curva."""
    cov = np.cov(changes_bp, rowvar=False)
    eigval, eigvec = np.linalg.eigh(cov)            # eigh: matriz simétrica
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]
    if eigvec[:, 0].sum() < 0:
        eigvec[:, 0] *= -1
    if eigvec[-1, 1] < eigvec[0, 1]:
        eigvec[:, 1] *= -1
    belly = TENORS.index(5)
    if eigvec[belly, 2] < 0.5 * (eigvec[0, 2] + eigvec[-1, 2]):
        eigvec[:, 2] *= -1
    return eigval, eigvec, cov


# =============================================================================
# 4) TODO EL MODELO EN UNA LLAMADA
# =============================================================================
def run(curve, confidence, window_days):
    """Curva limpia -> bonos, PCA, VaR de tipos y escenarios.
    window_days = días de cambios usados en el PCA y en el VaR."""
    bonds = bond_book()
    today = curve.iloc[-1].to_numpy()
    changes = curve.diff().dropna().iloc[-window_days:]
    changes_bp = changes.to_numpy() * 1e4

    # --- Bonos: precio y sensibilidades ---------------------------------------
    rows, krd = [], {}
    for (name, notional, coupon, maturity), b in zip(BONDS, bonds):
        p = b.price(today)
        k = b.key_rate_dv01(today)
        krd[name] = k
        dv01 = k.sum()
        up = b.price(today + 0.02) - p                       # +200pb revaluando
        rows.append([name, notional, maturity, p, dv01, dv01 / p * 1e4,
                     up, -dv01 * 200, up + dv01 * 200])
    bond_table = pd.DataFrame(rows, columns=[
        "Bond", "Notional", "Maturity (years)", "Price (EUR)", "DV01 (EUR/bp)",
        "Modified duration", "P&L +200bp, full revaluation", "P&L +200bp, DV01 only",
        "Convexity effect"])
    krd = pd.DataFrame(krd, index=[f"{t}Y" for t in TENORS])
    krd["Book"] = krd.sum(axis=1)
    d = krd["Book"].to_numpy()                             # EUR por pb que BAJA

    # --- PCA -------------------------------------------------------------------
    eigval, eigvec, cov = curve_pca(changes_bp)
    explained = eigval / eigval.sum()
    loadings = pd.DataFrame(eigvec[:, :N_COMPONENTS], index=[f"{t}Y" for t in TENORS],
                            columns=["PC1 (level)", "PC2 (slope)", "PC3 (curvature)"])
    pca_table = pd.DataFrame({
        "Component": [f"PC{i + 1}" for i in range(len(TENORS))],
        "Daily volatility (bp)": np.sqrt(eigval),
        "Variance explained": explained,
        "Cumulative": np.cumsum(explained),
    })
    # exposición de la cartera a cada componente: EUR por 1 desviación típica
    pc_exposure = pd.DataFrame({
        "Component": loadings.columns,
        "P&L for a +1 std move (EUR)": [-(d @ eigvec[:, i]) * math.sqrt(eigval[i])
                                        for i in range(N_COMPONENTS)],
    })

    # --- VaR de tipos de la cartera de bonos ----------------------------------
    z = _NORMAL.inv_cdf(confidence)
    sigma_full = math.sqrt(d @ cov @ d)
    v3, l3 = eigvec[:, :N_COMPONENTS], eigval[:N_COMPONENTS]
    sigma_pca = math.sqrt(((d @ v3) ** 2 * l3).sum())
    base = book_price(bonds, today)
    hist_pnl = book_price(bonds, today + changes.to_numpy()) - base   # revaluación completa
    hist_losses = -hist_pnl
    hist_var = np.quantile(hist_losses, confidence)
    var_table = pd.DataFrame([
        ["Parametric, full covariance (9 tenors)", z * sigma_full,
         sigma_full * _NORMAL.pdf(z) / (1 - confidence)],
        [f"Parametric, {N_COMPONENTS} principal components", z * sigma_pca,
         sigma_pca * _NORMAL.pdf(z) / (1 - confidence)],
        ["Historical simulation, full revaluation", hist_var,
         hist_losses[hist_losses >= hist_var].mean()],
    ], columns=["Method", "VaR 1d", "ES 1d"])

    # --- Escenarios de tipos ---------------------------------------------------
    sc_rows = []
    for label, shock in RATE_SCENARIOS.items():
        move_bp = np.array([shock[t] for t in TENORS])
        full = book_price(bonds, today + move_bp / 1e4) - base
        linear = -(d @ move_bp)
        sc_rows.append([label, full, linear, full - linear, full / base])
    scenarios = pd.DataFrame(sc_rows, columns=[
        "Scenario", "P&L full revaluation", "P&L DV01 only", "Convexity effect",
        "% of book"])

    return {
        "as_of": curve.index[-1].strftime("%Y-%m-%d"),
        "window_start": changes.index[0].strftime("%Y-%m-%d"),
        "curve_today": curve.iloc[-1],
        "curve_history": curve,
        "book_value": base,
        "book_dv01": d.sum(),
        "bond_table": bond_table,
        "key_rate_dv01": krd,
        "pca_table": pca_table,
        "loadings": loadings,
        "pc_exposure": pc_exposure,
        "var_table": var_table,
        "pca_share_of_var": sigma_pca / sigma_full,
        "scenarios": scenarios,
    }
