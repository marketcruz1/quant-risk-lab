# =============================================================================
# models/var_es.py
#
# Modelo: Market Risk Engine - Value at Risk (VaR) y Expected Shortfall (ES)
# Versión pública para la demo (solo datos públicos de Yahoo Finance).
#
# QUÉ HACE:
#   1) Descarga precios diarios de Yahoo Finance de una cartera ficticia y los
#      pasa a la moneda base de la cartera (EUR o USD).
#   2) Calcula el VaR y el ES a 1 día con cuatro métodos, usando una ventana
#      de los últimos 500 días:
#        - Simulación histórica
#        - Paramétrico normal con covarianza de pesos iguales
#        - Paramétrico normal con covarianza EWMA (lambda = 0.94, RiskMetrics)
#        - Monte Carlo t de Student con Cholesky (grados de libertad estimados
#          a partir de la curtosis de la cartera)
#   3) Backtesting fuera de muestra: cada día calcula el VaR SOLO con el pasado
#      y lo compara con la pérdida real del día siguiente. Cuenta excepciones
#      y aplica los tests de Kupiec y Christoffersen y el semáforo de Basilea.
#
# DIFERENCIAS con el Market Risk Engine completo (decirlas siempre):
#   - Solo acciones, ETF y divisas; sin bonos, curva de tipos ni opciones.
#   - Pesos constantes (se rebalancea cada día) y P&L lineal.
#   - Sin los controles de calidad de datos del motor completo: solo se
#     rellenan festivos (máx. 3 días) y se quitan fechas sin datos.
#
# No necesita scipy: las funciones de la normal salen de statistics.NormalDist
# y los p-valores de la chi-cuadrado con 1 y 2 grados de libertad tienen
# fórmula cerrada.
# =============================================================================

import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import yfinance as yf

# --- Parámetros del modelo ---------------------------------------------------
LOOKBACK_DAYS = 500      # ventana de estimación (unos dos años de mercado)
BACKTEST_DAYS = 500      # días de backtesting fuera de muestra
EWMA_LAMBDA = 0.94       # factor de decaimiento diario de RiskMetrics
MC_SIMULATIONS = 20_000  # escenarios para el VaR de hoy
MC_SIMULATIONS_BT = 5_000  # escenarios por día en el backtesting (más rápido)
MC_SEED = 42             # semilla fija: mismos resultados en cada ejecución
FFILL_LIMIT = 3          # festivos: copiar el último precio como mucho 3 días
HISTORY = "6y"           # historia descargada: ventana + backtesting + margen

METHODS = [
    "Historical simulation",
    "Parametric normal (equal weights)",
    "Parametric normal (EWMA, lambda 0.94)",
    "Monte Carlo Student t (Cholesky)",
]

# Carteras ficticias. Cada posición: (nombre, ticker de Yahoo, moneda, peso).
# Los pesos suman 1. Las divisas en Yahoo: EURUSD=X = dólares por 1 euro.
PORTFOLIOS = {
    "European and US equities (EUR)": {
        "currency": "EUR",
        "value": 1_000_000,
        "positions": [
            ("Banco Santander", "SAN.MC", "EUR", 0.15),
            ("BBVA", "BBVA.MC", "EUR", 0.15),
            ("Inditex", "ITX.MC", "EUR", 0.15),
            ("SAP", "SAP.DE", "EUR", 0.15),
            ("ASML", "ASML.AS", "EUR", 0.15),
            ("SPDR S&P 500 ETF", "SPY", "USD", 0.15),
            ("HSBC", "HSBA.L", "GBp", 0.10),   # cotiza en peniques
        ],
    },
    "US 60/40 stocks and bonds (USD)": {
        "currency": "USD",
        "value": 1_000_000,
        "positions": [
            ("SPDR S&P 500 ETF", "SPY", "USD", 0.60),
            ("iShares 7-10Y Treasury ETF", "IEF", "USD", 0.40),
        ],
    },
    "Diversified multi-asset ETFs (USD)": {
        "currency": "USD",
        "value": 1_000_000,
        "positions": [
            ("US equities (SPY)", "SPY", "USD", 0.30),
            ("Developed ex-US equities (EFA)", "EFA", "USD", 0.20),
            ("Emerging markets (EEM)", "EEM", "USD", 0.10),
            ("US Treasuries 7-10Y (IEF)", "IEF", "USD", 0.25),
            ("Gold (GLD)", "GLD", "USD", 0.15),
        ],
    },
}

# Tipo de cambio necesario para pasar cada moneda a la moneda base
FX_TICKERS = {("USD", "EUR"): "EURUSD=X", ("GBP", "EUR"): "EURGBP=X"}

_NORMAL = NormalDist()


# =============================================================================
# 1) DATOS
# =============================================================================
def download_prices(portfolio_name):
    """Descarga los precios de cierre de Yahoo y los pasa a la moneda base.
    Devuelve un DataFrame: filas = fechas, columnas = nombres de las posiciones.
    Si Yahoo falla devuelve un DataFrame vacío."""
    pf = PORTFOLIOS[portfolio_name]
    base = pf["currency"]
    tickers = [p[1] for p in pf["positions"]]
    fx_needed = {}
    for _, _, ccy, _ in pf["positions"]:
        ccy = "GBP" if ccy == "GBp" else ccy
        if ccy != base:
            fx_needed[ccy] = FX_TICKERS[(ccy, base)]

    raw = yf.download(tickers + list(fx_needed.values()), period=HISTORY,
                      auto_adjust=True, progress=False)
    if raw is None or raw.empty or "Close" not in raw:
        return pd.DataFrame()
    close = raw["Close"]
    if isinstance(close, pd.Series):               # por si solo hay un ticker
        close = close.to_frame(tickers[0])
    missing = [t for t in tickers + list(fx_needed.values())
               if t not in close or close[t].dropna().empty]
    if missing:
        return pd.DataFrame()

    close.index = pd.to_datetime(close.index)
    if close.index.tz is not None:                 # quitar zona horaria
        close.index = close.index.tz_convert(None)
    close.index = close.index.normalize()
    close = close.where(close > 0)                 # precios <= 0 = dato malo
    close = close.ffill(limit=FFILL_LIMIT)         # festivos de cada bolsa

    prices = pd.DataFrame(index=close.index)
    for name, ticker, ccy, _ in pf["positions"]:
        p = close[ticker]
        if ccy == "GBp":                           # peniques -> libras
            p, ccy = p / 100.0, "GBP"
        if ccy != base:                            # p. ej. USD -> EUR
            p = p / close[fx_needed[ccy]]
        prices[name] = p
    return prices.dropna()


# =============================================================================
# 2) VaR y ES de una ventana de rendimientos
#    Todos devuelven (VaR, ES) como FRACCIÓN del valor de la cartera,
#    en positivo (pérdida). Media de los rendimientos = 0 (hipótesis usual
#    a 1 día) en los métodos paramétricos y Monte Carlo.
# =============================================================================
def var_es_historical(window, w, conf):
    """Simulación histórica: reordenar las pérdidas reales de la ventana."""
    losses = -(window @ w)
    var = np.quantile(losses, conf)
    es = losses[losses >= var].mean()
    return var, es


def _normal_var_es(sigma, conf):
    """VaR y ES de una normal de media 0 y desviación sigma."""
    z = _NORMAL.inv_cdf(conf)
    var = z * sigma
    es = sigma * _NORMAL.pdf(z) / (1 - conf)
    return var, es


def var_es_parametric(window, w, conf):
    """Normal con covarianza muestral (todos los días pesan igual)."""
    cov = np.cov(window, rowvar=False)
    sigma = math.sqrt(float(w @ np.atleast_2d(cov) @ w))
    return _normal_var_es(sigma, conf)


def ewma_cov(window, lam=EWMA_LAMBDA):
    """Covarianza EWMA: el día de ayer pesa (1 - lambda), el anterior
    (1 - lambda) * lambda, etc. Los días recientes pesan más."""
    n = len(window)
    weights = lam ** np.arange(n - 1, -1, -1)      # el último día pesa más
    weights = weights / weights.sum()
    return (window * weights[:, None]).T @ window


def var_es_ewma(window, w, conf):
    """Normal con covarianza EWMA: reacciona antes a los cambios de volatilidad."""
    sigma = math.sqrt(float(w @ ewma_cov(window) @ w))
    return _normal_var_es(sigma, conf)


def student_t_dof(pnl):
    """Grados de libertad de la t por el método de los momentos:
    curtosis en exceso de una t = 6 / (nu - 4)  ->  nu = 4 + 6 / exceso.
    Se limita a [3, 30] (30 ya es casi una normal)."""
    x = pnl - pnl.mean()
    excess = (x ** 4).mean() / (x ** 2).mean() ** 2 - 3
    if excess <= 0.2:
        return 30.0
    return float(np.clip(4 + 6 / excess, 3.0, 30.0))


def var_es_monte_carlo(window, w, conf, n_sims=MC_SIMULATIONS, rng=None):
    """Monte Carlo con t de Student multivariante.
    1) Cholesky de la covarianza: cov = L L'.
    2) Normales independientes z -> correladas L z.
    3) Se divide por sqrt(chi2 / nu) para tener colas gruesas (t de Student)
       y se reescala para que la covarianza siga siendo la de la ventana."""
    rng = rng if rng is not None else np.random.default_rng(MC_SEED)
    cov = np.atleast_2d(np.cov(window, rowvar=False))
    cov = cov + np.eye(len(w)) * 1e-12             # estabilidad numérica
    L = np.linalg.cholesky(cov)
    nu = student_t_dof(window @ w)
    z = rng.standard_normal((n_sims, len(w)))
    chi2 = rng.chisquare(nu, size=(n_sims, 1))
    sims = (z @ L.T) / np.sqrt(chi2 / nu) * math.sqrt((nu - 2) / nu)
    losses = -(sims @ w)
    var = np.quantile(losses, conf)
    es = losses[losses >= var].mean()
    return var, es, nu


def var_es(method, window, w, conf, rng=None, n_sims=MC_SIMULATIONS):
    """Elige el método por su nombre (los nombres de METHODS)."""
    if method == METHODS[0]:
        return var_es_historical(window, w, conf)
    if method == METHODS[1]:
        return var_es_parametric(window, w, conf)
    if method == METHODS[2]:
        return var_es_ewma(window, w, conf)
    var, es, _ = var_es_monte_carlo(window, w, conf, n_sims, rng)
    return var, es


# =============================================================================
# 3) BACKTESTING y tests estadísticos
# =============================================================================
def _xlogy(x, y):
    """x * log(y), con 0 * log(0) = 0 (para cuando no hay excepciones)."""
    return 0.0 if x == 0 else x * math.log(y)


def kupiec_test(n_obs, n_exc, p):
    """Kupiec POF: ¿el número de excepciones es compatible con p?
    LR ~ chi-cuadrado(1). p-valor = erfc(sqrt(LR / 2))."""
    phat = n_exc / n_obs
    lr = -2 * (_xlogy(n_obs - n_exc, 1 - p) + _xlogy(n_exc, p)
               - _xlogy(n_obs - n_exc, 1 - phat) - _xlogy(n_exc, phat))
    lr = max(lr, 0.0)
    return lr, math.erfc(math.sqrt(lr / 2))


def christoffersen_test(exceptions):
    """Christoffersen: ¿las excepciones llegan agrupadas (un día malo
    hace más probable otro día malo)? LR de independencia ~ chi-cuadrado(1)."""
    e = np.asarray(exceptions, dtype=int)
    prev, curr = e[:-1], e[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    log_l0 = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    log_l1 = (_xlogy(n00, 1 - pi01) + _xlogy(n01, pi01)
              + _xlogy(n10, 1 - pi11) + _xlogy(n11, pi11))
    lr = max(-2 * (log_l0 - log_l1), 0.0)
    return lr, math.erfc(math.sqrt(lr / 2))


def basel_zone(n_exc_250):
    """Semáforo de Basilea para VaR 99% y 250 días."""
    if n_exc_250 <= 4:
        return "Green"
    if n_exc_250 <= 9:
        return "Yellow"
    return "Red"


def backtest(returns, w, conf, method, value):
    """VaR de cada día calculado SOLO con los LOOKBACK_DAYS anteriores,
    comparado con la pérdida real de ese día."""
    rng = np.random.default_rng(MC_SEED)
    arr = returns.to_numpy()
    start = max(LOOKBACK_DAYS, len(arr) - BACKTEST_DAYS)
    rows = []
    for t in range(start, len(arr)):
        window = arr[t - LOOKBACK_DAYS: t]
        var, _ = var_es(method, window, w, conf, rng, MC_SIMULATIONS_BT)
        pnl = float(arr[t] @ w)
        rows.append((returns.index[t], pnl * value, var * value))
    bt = pd.DataFrame(rows, columns=["date", "pnl", "var"]).set_index("date")
    bt["loss"] = -bt["pnl"]
    bt["exception"] = bt["loss"] > bt["var"]
    return bt


# =============================================================================
# 4) TODO EL MODELO EN UNA LLAMADA
# =============================================================================
def run(prices, portfolio_name, confidence, method):
    """Calcula VaR/ES de hoy con los cuatro métodos, el backtesting del método
    elegido y los tests. 'prices' es la salida de download_prices()."""
    pf = PORTFOLIOS[portfolio_name]
    value = pf["value"]
    w = np.array([p[3] for p in pf["positions"]])
    returns = prices.pct_change().dropna()
    if len(returns) < LOOKBACK_DAYS + 250:
        raise ValueError("Not enough price history")

    # --- VaR y ES de hoy con todos los métodos --------------------------------
    window = returns.to_numpy()[-LOOKBACK_DAYS:]
    rows = []
    for m in METHODS:
        var, es = var_es(m, window, w, confidence)
        rows.append([m, var * value, es * value,
                     var * value * math.sqrt(10), es * value * math.sqrt(10)])
    table = pd.DataFrame(rows, columns=["Method", "VaR 1d", "ES 1d",
                                        "VaR 10d (√10)", "ES 10d (√10)"])
    nu = student_t_dof(window @ w)

    # --- Backtesting del método elegido ---------------------------------------
    bt = backtest(returns, w, confidence, method, value)
    p = 1 - confidence
    n_obs, n_exc = len(bt), int(bt["exception"].sum())
    kup_lr, kup_p = kupiec_test(n_obs, n_exc, p)
    chr_lr, chr_p = christoffersen_test(bt["exception"])
    cc_lr = kup_lr + chr_lr
    n_exc_250 = int(bt["exception"].iloc[-250:].sum())
    tests = {
        "observations": n_obs,
        "exceptions": n_exc,
        "expected": n_obs * p,
        "kupiec_lr": kup_lr, "kupiec_p": kup_p,
        "christoffersen_lr": chr_lr, "christoffersen_p": chr_p,
        "cc_lr": cc_lr, "cc_p": math.exp(-cc_lr / 2),   # chi-cuadrado(2)
        "exceptions_250": n_exc_250,
        "basel_zone": basel_zone(n_exc_250) if abs(confidence - 0.99) < 1e-9 else None,
    }

    return {
        "as_of": returns.index[-1].strftime("%Y-%m-%d"),
        "currency": pf["currency"],
        "value": value,
        "weights": pd.DataFrame(
            [(n, t, c.replace("GBp", "GBP"), wt) for n, t, c, wt in pf["positions"]],
            columns=["Position", "Ticker", "Currency", "Weight"]),
        "table": table,
        "selected": table.set_index("Method").loc[method],
        "window_pnl": window @ w * value,
        "student_t_dof": nu,
        "backtest": bt,
        "tests": tests,
    }
