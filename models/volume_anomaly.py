# =============================================================================
# models/volume_anomaly.py
#
# Modelo: Market Surveillance - Unusual Trading Volume Detection
# Versión pública para la demo (sin Interactive Brokers ni Databento).
#
# QUÉ HACE:
#   1) Descarga velas de 5 minutos de Yahoo Finance (datos públicos, gratis).
#   2) Compara el volumen de cada vela con el volumen NORMAL A ESA MISMA HORA
#      en las 20 sesiones anteriores (solo pasado: sin mirar al futuro).
#        RVOL = volumen / mediana de la línea base
#        z    = z-score robusto del log-volumen (mediana y MAD)
#      ALERTA si RVOL >= 3 y z >= 3 (los mismos umbrales del modelo original).
#   3) Estima si el volumen fue comprador o vendedor con Bulk Volume
#      Classification: fracción compradora = Phi(cambio de precio / sigma).
#   4) Agrupa las alertas seguidas en EVENTOS (lo que sería un aviso).
#
# DIFERENCIAS con el modelo de investigación (decirlas siempre):
#   - Velas de 5 min en vez de operaciones a 1 min (Yahoo no da operaciones
#     y solo guarda 60 días de velas de 5 min).
#   - La dirección es una ESTIMACIÓN a partir de velas, no el lado agresor real.
#   - Yahoo no da volumen en divisas (FX): ahí el modelo no puede funcionar.
# =============================================================================

import math

import numpy as np
import pandas as pd
import yfinance as yf

# --- Parámetros del modelo ---------------------------------------------------
BASELINE_SESSIONS = 20   # sesiones anteriores para el volumen "normal"
MARGIN_BARS = 1          # misma hora ± 1 vela (± 5 minutos)
RVOL_MIN = 3.0           # el volumen debe ser al menos 3 veces la mediana
Z_MIN = 3.0              # y estar a 3 desviaciones robustas por encima
EVENT_GAP_MIN = 15       # alertas separadas por menos de 15 min = un evento
SIGMA_BARS = 100         # velas previas para medir la volatilidad (dirección)
DIRECTION_CUTOFF = 0.65  # >= 65 % comprador o vendedor para darle dirección


def download_bars(ticker, period="60d", interval="5m"):
    """Descarga velas de Yahoo Finance y las devuelve en hora UTC.
    Columnas: Open, High, Low, Close, Volume. Índice: fecha y hora UTC."""
    bars = yf.Ticker(ticker).history(period=period, interval=interval,
                                     auto_adjust=False, prepost=False)
    if bars.empty:
        return bars
    bars = bars[["Open", "High", "Low", "Close", "Volume"]].copy()
    bars.index = bars.index.tz_convert("UTC")      # todo en UTC
    return bars


def add_direction(bars):
    """Bulk Volume Classification: fracción compradora estimada de cada vela.
    Si el precio sube mucho respecto a su volatilidad normal, casi todo el
    volumen se cuenta como compra; si baja mucho, como venta."""
    change = bars["Close"].diff()                          # cambio de precio
    # volatilidad de las velas ANTERIORES (shift(1): sin mirar al futuro)
    sigma = change.rolling(SIGMA_BARS, min_periods=20).std().shift(1)
    ratio = (change / sigma).replace([np.inf, -np.inf], np.nan)
    # Phi(x) = función de distribución de la normal, escrita con math.erf
    bars["buy_fraction"] = ratio.apply(
        lambda x: np.nan if pd.isna(x) else 0.5 * (1 + math.erf(x / math.sqrt(2))))
    return bars


def detect(bars):
    """Calcula RVOL, z y ALERTA de cada vela frente a las 20 sesiones previas."""
    bars = bars.copy()
    bars["day"] = bars.index.strftime("%Y-%m-%d")
    bars["time"] = bars.index.strftime("%H:%M")

    # Tabla de volúmenes: filas = hora del día, columnas = días
    table = bars.pivot_table(index="time", columns="day", values="Volume",
                             aggfunc="sum")
    times = list(table.index)
    days = list(table.columns)

    rvol_col, z_col = [], []
    for timestamp, row in bars.iterrows():
        rvol, z = np.nan, np.nan
        d = days.index(row["day"])
        previous_days = days[max(0, d - BASELINE_SESSIONS): d]   # solo pasado
        if len(previous_days) >= 10:                # pedir al menos 10 sesiones
            i = times.index(row["time"])
            window = table.iloc[max(0, i - MARGIN_BARS): i + MARGIN_BARS + 1]
            sample = window[previous_days].to_numpy().ravel()
            sample = sample[~np.isnan(sample)]
            if len(sample) > 0:
                median = np.median(sample)
                log_s = np.log1p(sample)
                mad = 1.4826 * np.median(np.abs(log_s - np.median(log_s)))
                if median > 0:
                    rvol = row["Volume"] / median
                if mad > 0:
                    z = (np.log1p(row["Volume"]) - np.median(log_s)) / mad
        rvol_col.append(rvol)
        z_col.append(z)

    bars["rvol"] = rvol_col
    bars["z"] = z_col
    bars["alert"] = (bars["rvol"] >= RVOL_MIN) & (bars["z"] >= Z_MIN)
    return bars


def direction_label(buy_fraction):
    """Texto de la dirección estimada (en inglés, porque se ve en la app)."""
    if pd.isna(buy_fraction):
        return "n/a"
    if buy_fraction >= DIRECTION_CUTOFF:
        return "Buy-side"
    if buy_fraction <= 1 - DIRECTION_CUTOFF:
        return "Sell-side"
    return "Mixed"


def events(bars):
    """Agrupa las velas con alerta en eventos y devuelve una tabla
    (más reciente primero) con lo que se enseñará en la app."""
    alerts = bars[bars["alert"]]
    rows, current = [], None
    for timestamp, row in alerts.iterrows():
        new_event = (current is None or
                     timestamp - current["last"] >= pd.Timedelta(minutes=EVENT_GAP_MIN))
        if new_event:
            if current is not None:
                rows.append(current)
            current = {"start": timestamp, "last": timestamp, "volume": 0,
                       "buy_volume": 0.0, "max_rvol": 0.0, "max_z": 0.0}
        current["last"] = timestamp
        current["volume"] += row["Volume"]
        if not pd.isna(row["buy_fraction"]):
            current["buy_volume"] += row["Volume"] * row["buy_fraction"]
        current["max_rvol"] = max(current["max_rvol"], row["rvol"])
        current["max_z"] = max(current["max_z"], row["z"])
    if current is not None:
        rows.append(current)

    columns = ["Start (UTC)", "Bars", "Volume", "Max RVOL", "Max z",
               "Buy share", "Direction (est.)"]
    if not rows:
        return pd.DataFrame(columns=columns)

    result = []
    for e in rows:
        share = e["buy_volume"] / e["volume"] if e["volume"] > 0 else np.nan
        n_bars = int(((alerts.index >= e["start"]) & (alerts.index <= e["last"])).sum())
        result.append([e["start"].strftime("%Y-%m-%d %H:%M"), n_bars,
                       int(e["volume"]), round(e["max_rvol"], 1),
                       round(e["max_z"], 1), round(share, 2),
                       direction_label(share)])
    return pd.DataFrame(result, columns=columns).iloc[::-1].reset_index(drop=True)


def run(ticker):
    """Todo el modelo en una llamada: descarga, dirección, detección y eventos."""
    bars = download_bars(ticker)
    if bars.empty:
        return bars, pd.DataFrame()
    bars = add_direction(bars)
    bars = detect(bars)
    return bars, events(bars)