# =============================================================================
# models/var_es.py
#
# Modelo: Market Risk Engine - Value at Risk (VaR) y Expected Shortfall (ES)
# Versión pública para la demo (solo datos públicos de Yahoo Finance).
#
# QUÉ HACE:
#   1) Descarga precios diarios de Yahoo Finance de una cartera ficticia y los
#      pasa a la moneda base de la cartera (EUR o USD).
#   2) Calcula el VaR y el ES a 1 día con cinco métodos, usando una ventana
#      de los últimos 500 días:
#        - Simulación histórica
#        - Paramétrico normal con covarianza de pesos iguales
#        - Paramétrico normal con covarianza EWMA (lambda = 0.94, RiskMetrics)
#        - Monte Carlo t de Student con Cholesky (grados de libertad estimados
#          a partir de la curtosis de la cartera)
#        - Simulación histórica filtrada con GARCH(1,1): la volatilidad de
#          mañana sale del GARCH y la forma de la cola, de los residuos reales
#   3) Volatilidad y autocorrelación: compara la volatilidad EWMA y GARCH(1,1)
#      y aplica el test de Ljung-Box a los rendimientos, a sus cuadrados y a
#      los residuos del GARCH (¿queda agrupamiento de volatilidad sin explicar?).
#   4) Backtesting fuera de muestra: cada día calcula el VaR SOLO con el pasado
#      y lo compara con la pérdida real del día siguiente. Cuenta excepciones
#      y aplica los tests de Kupiec y Christoffersen y el semáforo de Basilea.
#   5) Stressed VaR: el VaR histórico de la cartera de HOY calculado en la peor
#      ventana de 250 días desde 2006 (y en 2008 y 2020), con el capital
#      simplificado de Basilea 2.5.
#   6) Escenarios de estrés: crisis históricas (Lehman, euro 2011, Brexit,
#      COVID, 2022) y escenarios hipotéticos aplicados a la cartera de hoy.
#   7) Calidad de datos: precios <= 0, festivos, precios congelados y
#      "bad ticks" (saltos de un día que se deshacen al día siguiente).
#
# DIFERENCIAS con el Market Risk Engine completo (decirlas siempre):
#   - Solo acciones, ETF y divisas; sin bonos, curva de tipos ni opciones.
#   - Pesos constantes (se rebalancea cada día) y P&L lineal.
#   - Controles de calidad de datos más sencillos que en el motor completo.
#
# No necesita scipy: las funciones de la normal salen de statistics.NormalDist,
# los p-valores de la chi-cuadrado con grados de libertad pares tienen fórmula
# cerrada, y el GARCH se estima por máxima verosimilitud con una búsqueda en
# rejilla (gruesa y luego fina) en vez de un optimizador.
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
HISTORY_START = "2006-01-01"  # historia descargada: incluye 2008 para el stressed VaR
SVAR_WINDOW = 250        # ventana del stressed VaR (12 meses, Basilea 2.5)
BASEL_MULTIPLIER = 3.0   # multiplicador mínimo de Basilea para VaR y sVaR
SPIKE_Z = 10.0           # bad tick: salto de más de 10 desviaciones robustas
SPIKE_WINDOW = 60        # ...medidas con los 60 días ANTERIORES (volatilidad local)...
SPIKE_REVERSAL = 0.7     # ...que se deshace al menos un 70% al día siguiente...
SPIKE_CALM_Z = 3.0       # ...y con días tranquilos antes y después (aislado)
STALE_DAYS = 5           # 5 cierres idénticos seguidos = precio congelado
GARCH_REFIT_DAYS = 20    # backtesting: reestimar el GARCH cada 20 días
LJUNG_BOX_LAGS = 10      # retardos del test de Ljung-Box (número par)
TRADING_DAYS = 252       # para anualizar la volatilidad

METHODS = [
    "Historical simulation",
    "Parametric normal (equal weights)",
    "Parametric normal (EWMA, lambda 0.94)",
    "Monte Carlo Student t (Cholesky)",
    "Filtered historical simulation (GARCH(1,1))",
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

# Tipo de activo de cada ticker (para los escenarios hipotéticos)
ASSET_CLASS = {
    "SAN.MC": "Banks", "BBVA.MC": "Banks", "HSBA.L": "Banks",
    "ITX.MC": "Equity", "SAP.DE": "Equity", "ASML.AS": "Equity", "SPY": "Equity",
    "EFA": "Equity", "EEM": "Equity",
    "IEF": "Bonds", "GLD": "Gold",
}
BOND_ETF_DURATION = 7.5  # duración aproximada de IEF (bonos EE. UU. 7-10 años)

# Fechas fijas de estrés: stressed VaR (ventanas) y escenarios históricos (inicio, fin)
STRESS_WINDOWS = {
    "Global financial crisis (Jul 2008 - Jun 2009)": ("2008-07-01", "2009-06-30"),
    "COVID (2020)": ("2020-01-02", "2020-12-31"),
}
HISTORICAL_SCENARIOS = {
    "Lehman collapse (Sep-Oct 2008)": ("2008-09-12", "2008-10-10"),
    "Euro sovereign crisis (Jul-Sep 2011)": ("2011-07-01", "2011-09-22"),
    "Brexit vote (Jun 2016)": ("2016-06-23", "2016-06-27"),
    "COVID crash (Feb-Mar 2020)": ("2020-02-19", "2020-03-23"),
    "Rate shock (Jan-Oct 2022)": ("2021-12-31", "2022-10-21"),
}

# Tipo de cambio necesario para pasar cada moneda a la moneda base
FX_TICKERS = {("USD", "EUR"): "EURUSD=X", ("GBP", "EUR"): "EURGBP=X"}

_NORMAL = NormalDist()


# =============================================================================
# 1) DATOS
# =============================================================================
def quality_controls(close):
    """Controles de calidad sobre los precios de Yahoo. Devuelve
    (precios corregidos, registro de incidencias)."""
    log = []
    close = close.copy()
    for col in close.columns:
        s = close[col]
        # 1) precios <= 0: imposibles, se eliminan
        bad = s <= 0
        if bad.any():
            log.append([col, "Non-positive price", int(bad.sum()), "Removed"])
            close.loc[bad, col] = np.nan
        # 2) precio congelado: el mismo cierre muchos días seguidos (solo aviso)
        x = close[col].dropna()
        same = x.eq(x.shift())
        runs = same.groupby((~same).cumsum()).sum()
        n_stale = int((runs >= STALE_DAYS - 1).sum())
        if n_stale:
            log.append([col, f"Stale price ({STALE_DAYS}+ identical closes)", n_stale,
                        "Flagged only"])
        # 3) bad tick: un salto enorme que se deshace al día siguiente y que
        #    está AISLADO: el día anterior al salto y el día posterior a la
        #    vuelta son normales. Un error de datos es así; un día de crisis
        #    real (marzo 2020: -10%, +9%, -11%...) llega rodeado de otros días
        #    muy volátiles, así que no se confunde con un error y no se borra.
        r = np.log(x).diff()
        mad = 1.4826 * r.abs().rolling(SPIKE_WINDOW, min_periods=20).median().shift(1)
        z = r / mad
        nxt = r.shift(-1)
        isolated = (z.shift(1).abs() < SPIKE_CALM_Z) & (z.shift(-2).abs() < SPIKE_CALM_Z)
        tick = (z.abs() > SPIKE_Z) & (np.sign(nxt) == -np.sign(r)) & \
               (nxt.abs() > SPIKE_REVERSAL * r.abs()) & isolated
        if tick.any():
            dates = ", ".join(d.strftime("%Y-%m-%d") for d in tick[tick].index[:5])
            log.append([col, "Bad tick (spike that reverses)", int(tick.sum()),
                        f"Replaced by previous close ({dates})"])
            close.loc[tick[tick].index, col] = np.nan
        # 4) festivos y días sin dato: copiar el último precio (máx. 3 días)
        gaps = int(close[col].loc[close[col].first_valid_index():].isna().sum())
        if gaps:
            log.append([col, "Missing day (holiday or gap)", gaps,
                        f"Previous close carried forward (max {FFILL_LIMIT} days)"])
    close = close.ffill(limit=FFILL_LIMIT)
    return close, pd.DataFrame(log, columns=["Series", "Check", "Cases", "Treatment"])


def download_prices(portfolio_name):
    """Descarga los precios de cierre de Yahoo y los pasa a la moneda base.
    Devuelve (precios, registro de calidad de datos). Precios: filas = fechas,
    columnas = nombres de las posiciones. Si Yahoo falla, precios vacíos."""
    empty = pd.DataFrame(), pd.DataFrame()
    pf = PORTFOLIOS[portfolio_name]
    base = pf["currency"]
    tickers = [p[1] for p in pf["positions"]]
    fx_needed = {}
    for _, _, ccy, _ in pf["positions"]:
        ccy = "GBP" if ccy == "GBp" else ccy
        if ccy != base:
            fx_needed[ccy] = FX_TICKERS[(ccy, base)]

    raw = yf.download(tickers + list(fx_needed.values()), start=HISTORY_START,
                      auto_adjust=True, progress=False)
    if raw is None or raw.empty or "Close" not in raw:
        return empty
    close = raw["Close"]
    if isinstance(close, pd.Series):               # por si solo hay un ticker
        close = close.to_frame(tickers[0])
    missing = [t for t in tickers + list(fx_needed.values())
               if t not in close or close[t].dropna().empty]
    if missing:
        return empty

    close.index = pd.to_datetime(close.index)
    if close.index.tz is not None:                 # quitar zona horaria
        close.index = close.index.tz_convert(None)
    close.index = close.index.normalize()
    close, dq_log = quality_controls(close)

    prices = pd.DataFrame(index=close.index)
    for name, ticker, ccy, _ in pf["positions"]:
        p = close[ticker]
        if ccy == "GBp":                           # peniques -> libras
            p, ccy = p / 100.0, "GBP"
        if ccy != base:                            # p. ej. USD -> EUR
            p = p / close[fx_needed[ccy]]
        prices[name] = p
    return prices.dropna(), dq_log


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


# --- GARCH(1,1) ---------------------------------------------------------------
#   varianza de mañana = omega + alpha * (rendimiento de hoy)^2 + beta * varianza de hoy
# "Variance targeting": omega = (1 - alpha - beta) * varianza de la ventana, así
# la volatilidad de largo plazo es la de la muestra y solo hay que estimar
# alpha y beta.
def garch_variances(r, alpha, beta, omega):
    """Varianzas condicionales del GARCH. alpha y beta pueden ser arrays
    (una rejilla de valores): se calculan todos a la vez.
    Devuelve (varianzas de cada día, varianza prevista para mañana)."""
    alpha, beta, omega = np.asarray(alpha), np.asarray(beta), np.asarray(omega)
    var_t = np.full(alpha.shape, r.var())          # arranque: varianza muestral
    out = np.empty((len(r),) + alpha.shape)
    for t in range(len(r)):
        out[t] = var_t
        var_t = omega + alpha * r[t] ** 2 + beta * var_t
    return out, var_t


def _garch_loglik(r, alpha, beta):
    """Log-verosimilitud normal (cuasi-máxima verosimilitud) de cada pareja."""
    omega = (1 - alpha - beta) * r.var()
    variances, _ = garch_variances(r, alpha, beta, omega)
    return -0.5 * (np.log(variances) + r[:, None] ** 2 / variances).sum(axis=0)


def fit_garch(r):
    """Estima alpha y beta maximizando la verosimilitud en una rejilla:
    primero gruesa (pasos de 0.01) y después fina (0.001) alrededor del mejor."""
    r = np.asarray(r, dtype=float) - np.mean(r)
    best = None
    for step, a_rng, b_rng in [(0.01, (0.01, 0.30), (0.50, 0.99)), (0.001, None, None)]:
        if a_rng is None:                          # rejilla fina
            a_rng = (max(best[0] - 0.01, 0.001), best[0] + 0.01)
            b_rng = (max(best[1] - 0.01, 0.0), min(best[1] + 0.01, 0.998))
        a, b = np.meshgrid(np.arange(a_rng[0], a_rng[1] + step / 2, step),
                           np.arange(b_rng[0], b_rng[1] + step / 2, step))
        a, b = a.ravel(), b.ravel()
        ok = a + b < 0.999                         # estacionariedad
        a, b = a[ok], b[ok]
        ll = _garch_loglik(r, a, b)
        i = int(np.argmax(ll))
        best = (float(a[i]), float(b[i]))
    alpha, beta = round(best[0], 3), round(best[1], 3)
    omega = (1 - alpha - beta) * r.var()
    return {"alpha": alpha, "beta": beta, "omega": omega}


def garch_filter(r, params):
    """Aplica unos parámetros ya estimados a una serie: devuelve la volatilidad
    de cada día, la prevista para mañana y los residuos estandarizados."""
    r = np.asarray(r, dtype=float) - np.mean(r)
    variances, var_next = garch_variances(r, params["alpha"], params["beta"],
                                          params["omega"])
    sigma = np.sqrt(variances)
    return sigma, math.sqrt(float(var_next)), r / sigma


def var_es_garch_fhs(window, w, conf, params=None):
    """Simulación histórica filtrada (Barone-Adesi): residuos estandarizados
    reales de la ventana, escalados con la volatilidad GARCH de mañana.
    Recoge el agrupamiento de volatilidad (GARCH) y las colas gruesas
    (residuos reales, sin suponer una normal)."""
    pnl = window @ w
    params = params if params is not None else fit_garch(pnl)
    _, sigma_next, z = garch_filter(pnl, params)
    losses = -z * sigma_next
    var = np.quantile(losses, conf)
    es = losses[losses >= var].mean()
    return var, es


def ewma_vol_series(r, lam=EWMA_LAMBDA):
    """Volatilidad EWMA de la cartera día a día (para el gráfico)."""
    r = np.asarray(r, dtype=float)
    var_t, out = r.var(), np.empty(len(r))
    for t in range(len(r)):
        out[t] = var_t
        var_t = lam * var_t + (1 - lam) * r[t] ** 2
    return np.sqrt(out), math.sqrt(var_t)


def var_es(method, window, w, conf, rng=None, n_sims=MC_SIMULATIONS, garch_params=None):
    """Elige el método por su nombre (los nombres de METHODS)."""
    if method == METHODS[0]:
        return var_es_historical(window, w, conf)
    if method == METHODS[1]:
        return var_es_parametric(window, w, conf)
    if method == METHODS[2]:
        return var_es_ewma(window, w, conf)
    if method == METHODS[4]:
        return var_es_garch_fhs(window, w, conf, garch_params)
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


def chi2_sf_even(x, dof):
    """P(chi-cuadrado(dof) > x) para dof par: fórmula cerrada
    exp(-x/2) * suma de (x/2)^k / k!  para k = 0 ... dof/2 - 1."""
    h = x / 2
    term, total = 1.0, 1.0
    for k in range(1, dof // 2):
        term *= h / k
        total += term
    return min(1.0, math.exp(-h) * total)


def ljung_box(x, lags=LJUNG_BOX_LAGS):
    """Ljung-Box: ¿hay autocorrelación en los primeros 'lags' retardos?
    Q = n (n + 2) * suma de rho_k^2 / (n - k)  ~  chi-cuadrado(lags)."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    denom = (x ** 2).sum()
    q = 0.0
    for k in range(1, lags + 1):
        rho = (x[k:] * x[:-k]).sum() / denom
        q += rho ** 2 / (n - k)
    q *= n * (n + 2)
    return q, chi2_sf_even(q, lags)


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
    rows, params = [], None
    for t in range(start, len(arr)):
        window = arr[t - LOOKBACK_DAYS: t]
        # GARCH: reestimar cada GARCH_REFIT_DAYS (más rápido); entre medias se
        # usan los mismos parámetros, pero la volatilidad se actualiza cada día
        if method == METHODS[4] and (t - start) % GARCH_REFIT_DAYS == 0:
            params = fit_garch(window @ w)
        var, _ = var_es(method, window, w, conf, rng, MC_SIMULATIONS_BT, params)
        pnl = float(arr[t] @ w)
        rows.append((returns.index[t], pnl * value, var * value))
    bt = pd.DataFrame(rows, columns=["date", "pnl", "var"]).set_index("date")
    bt["loss"] = -bt["pnl"]
    bt["exception"] = bt["loss"] > bt["var"]
    return bt


# =============================================================================
# 4) STRESSED VaR y ESCENARIOS DE ESTRÉS (cartera de hoy, historia desde 2006)
# =============================================================================
def stressed_var(pnl_hist, confidence):
    """VaR histórico de cada ventana de 250 días y la peor de todas.
    pnl_hist: rendimientos diarios de la cartera de HOY (pesos actuales)."""
    rolling_var = (-pnl_hist).rolling(SVAR_WINDOW).quantile(confidence).dropna()
    end = rolling_var.idxmax()
    start = pnl_hist.index[pnl_hist.index.get_loc(end) - SVAR_WINDOW + 1]

    def row(label, sample):
        losses = -sample.to_numpy()
        var = np.quantile(losses, confidence)
        es = losses[losses >= var].mean()
        return [label, sample.index[0].strftime("%Y-%m-%d"),
                sample.index[-1].strftime("%Y-%m-%d"), len(sample), var, es,
                var * math.sqrt(10)]

    rows = [row(f"Current ({LOOKBACK_DAYS} days)", pnl_hist.iloc[-LOOKBACK_DAYS:]),
            row(f"Current ({SVAR_WINDOW} days)", pnl_hist.iloc[-SVAR_WINDOW:]),
            row(f"Stressed: worst {SVAR_WINDOW}-day window", pnl_hist.loc[start:end])]
    for label, (a, b) in STRESS_WINDOWS.items():
        sample = pnl_hist.loc[a:b]
        if len(sample) > 100:                      # solo si hay historia suficiente
            rows.append(row(f"Stressed: {label}", sample))
    table = pd.DataFrame(rows, columns=["Window", "From", "To", "Days", "VaR 1d", "ES 1d",
                                        "VaR 10d (√10)"])
    return table, rolling_var, (start, end)


def historical_scenarios(prices):
    """Movimiento acumulado de cada posición entre dos fechas de una crisis
    pasada, aplicado a la cartera de hoy (en moneda base, con su FX)."""
    out = {}
    for label, (a, b) in HISTORICAL_SCENARIOS.items():
        if prices.index[0] > pd.Timestamp(a):
            continue                               # no hay historia tan antigua
        p0 = prices.asof(pd.Timestamp(a))
        p1 = prices.asof(pd.Timestamp(b))
        out[f"Historical: {label}"] = (p1 / p0 - 1).to_numpy()
    return out


def hypothetical_scenarios(pf):
    """Escenarios diseñados a mano (movimiento relativo de cada posición)."""
    base = pf["currency"]
    cls = [ASSET_CLASS.get(t, "Equity") for _, t, _, _ in pf["positions"]]
    foreign = [c.replace("GBp", "GBP") != base for _, _, c, _ in pf["positions"]]
    rate_move = -BOND_ETF_DURATION * 0.02          # +200bp con la duración

    def scen(equity=0.0, banks=None, bonds=0.0, gold=0.0, fx=0.0):
        banks = equity if banks is None else banks
        local = np.array([{"Equity": equity, "Banks": banks, "Bonds": bonds,
                           "Gold": gold}[c] for c in cls])
        fx_move = np.where(foreign, fx, 0.0)
        return (1 + local) * (1 + fx_move) - 1     # local y divisa a la vez

    out = {
        "Equities -20%": scen(equity=-0.20),
        "Banks -30%": scen(banks=-0.30),
        "Flight to quality: equities -20%, bonds +5%, gold +10%":
            scen(equity=-0.20, bonds=0.05, gold=0.10),
        f"US rates +200bp (bond ETF duration {BOND_ETF_DURATION:g})": scen(bonds=rate_move),
        "Stagflation: equities -15%, US rates +150bp": scen(
            equity=-0.15, bonds=-BOND_ETF_DURATION * 0.015),
    }
    if any(foreign):
        # el euro sube un 10%: un dólar o una libra valen 1/1.10 - 1 = -9.1% en euros
        out["EUR +10% vs USD and GBP"] = scen(fx=1 / 1.10 - 1)
    # quitar escenarios que no afectan a esta cartera o que repiten otro
    # (p. ej. "flight to quality" en una cartera sin bonos ni oro)
    unique = {}
    for k, v in out.items():
        if np.abs(v).max() > 0 and not any(np.allclose(v, u) for u in unique.values()):
            unique[k] = v
    return unique


def stress_tests(prices, pf):
    """P&L de cada escenario por posición y total, con la cartera de hoy."""
    names = [p[0] for p in pf["positions"]]
    pos_value = np.array([p[3] for p in pf["positions"]]) * pf["value"]
    scenarios = {**historical_scenarios(prices),
                 **hypothetical_scenarios(pf)}
    rows = []
    for label, move in scenarios.items():
        pnl = move * pos_value
        rows.append([label, *pnl, pnl.sum(), pnl.sum() / pf["value"]])
    return pd.DataFrame(rows, columns=["Scenario", *names, "Total P&L", "% of portfolio"])


# =============================================================================
# 5) TODO EL MODELO EN UNA LLAMADA
# =============================================================================
def run(prices, portfolio_name, confidence, method):
    """Calcula VaR/ES de hoy con los cinco métodos, la volatilidad (EWMA y
    GARCH), los tests de Ljung-Box, el backtesting del método
    elegido y los tests. 'prices' es la salida de download_prices()."""
    pf = PORTFOLIOS[portfolio_name]
    value = pf["value"]
    w = np.array([p[3] for p in pf["positions"]])
    returns = prices.pct_change().dropna()
    if len(returns) < LOOKBACK_DAYS + 250:
        raise ValueError("Not enough price history")

    # --- VaR y ES de hoy con todos los métodos --------------------------------
    window = returns.to_numpy()[-LOOKBACK_DAYS:]
    pnl = window @ w                               # rendimientos de la cartera
    garch = fit_garch(pnl)
    rows = []
    for m in METHODS:
        var, es = var_es(m, window, w, confidence, garch_params=garch)
        rows.append([m, var * value, es * value,
                     var * value * math.sqrt(10), es * value * math.sqrt(10)])
    table = pd.DataFrame(rows, columns=["Method", "VaR 1d", "ES 1d",
                                        "VaR 10d (√10)", "ES 10d (√10)"])
    nu = student_t_dof(pnl)

    # --- Volatilidad: EWMA vs GARCH(1,1) --------------------------------------
    sigma_garch, sigma_next, z = garch_filter(pnl, garch)
    sigma_ewma, sigma_ewma_next = ewma_vol_series(pnl - pnl.mean())
    ann = math.sqrt(TRADING_DAYS)                  # diaria -> anual
    vol_series = pd.DataFrame({
        "EWMA (lambda 0.94)": sigma_ewma * ann,
        "GARCH(1,1)": sigma_garch * ann,
    }, index=returns.index[-LOOKBACK_DAYS:])
    persistence = garch["alpha"] + garch["beta"]
    garch_info = {
        **garch,
        "persistence": persistence,
        # días que tarda en reducirse a la mitad un shock de volatilidad
        "half_life": math.log(0.5) / math.log(persistence),
        "long_run_vol": math.sqrt(garch["omega"] / (1 - persistence)) * ann,
        "vol_tomorrow": sigma_next * ann,
        "vol_ewma_tomorrow": sigma_ewma_next * ann,
    }

    # --- Ljung-Box -------------------------------------------------------------
    lb_rows = []
    for label, series in [("Portfolio returns", pnl),
                          ("Squared returns (volatility clustering)", (pnl - pnl.mean()) ** 2),
                          ("GARCH standardised residuals", z),
                          ("Squared GARCH standardised residuals", z ** 2)]:
        q, p_val = ljung_box(series)
        lb_rows.append([label, q, p_val])
    ljung = pd.DataFrame(lb_rows, columns=["Series", f"Q({LJUNG_BOX_LAGS})", "p-value"])

    # --- Stressed VaR y escenarios de estrés -----------------------------------
    pnl_hist = returns @ w                         # cartera de hoy sobre toda la historia
    svar_table, rolling_var, worst = stressed_var(pnl_hist, confidence)
    svar_table[["VaR 1d", "ES 1d", "VaR 10d (√10)"]] *= value
    var10 = svar_table.loc[0, "VaR 10d (√10)"]
    svar10 = svar_table.loc[2, "VaR 10d (√10)"]
    capital = {
        "var10": var10, "svar10": svar10,
        "capital": BASEL_MULTIPLIER * (var10 + svar10),
    }
    stress = stress_tests(prices, pf)

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
        "garch": garch_info,
        "vol_series": vol_series,
        "ljung_box": ljung,
        "backtest": bt,
        "history_start": returns.index[0].strftime("%Y-%m-%d"),
        "svar_table": svar_table,
        "rolling_var": rolling_var * value,
        "worst_window": (worst[0].strftime("%Y-%m-%d"), worst[1].strftime("%Y-%m-%d")),
        "capital": capital,
        "stress": stress,
        "tests": tests,
    }
