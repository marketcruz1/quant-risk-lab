# =============================================================================
# app.py: Quant Risk Lab
# Plataforma web con un modelo por pestaña. Todo lo que se VE va en inglés;
# los comentarios del código, en español.
# Para arrancarla en local:  py -m streamlit run app.py
# =============================================================================

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from models.volume_anomaly import run
from models import var_es

st.set_page_config(page_title="Quant Risk Lab", layout="wide")

st.title("Quant Risk Lab")
st.write("Interactive platform for market risk and trading surveillance models")

# Una pestaña por modelo
tab_volume, tab_var = st.tabs(["Unusual Volume Detection", "VaR & Expected Shortfall"])

# Activos que hemos probado con Yahoo Finance y que traen volumen.
# (Las divisas no aparecen: Yahoo no da volumen en FX.)
INSTRUMENTS = {
    "WTI Crude Oil futures (CL=F)": "CL=F",
    "Gold futures (GC=F)": "GC=F",
    "E-mini S&P 500 futures (ES=F)": "ES=F",
    "10-Year T-Note futures (ZN=F)": "ZN=F",
    "Apple Inc. (AAPL)": "AAPL",
}


@st.cache_data(ttl=3600, show_spinner=False)
def load(ticker):
    """Ejecuta el modelo y guarda el resultado 1 hora (ttl=3600 segundos),
    para no descargar de Yahoo cada vez que alguien toca un botón.
    Si Yahoo no devuelve datos, lanza un error: así ese fallo NO se guarda
    y se vuelve a intentar en la siguiente visita."""
    bars, events = run(ticker)
    if bars.empty:
        raise ValueError("No data from Yahoo Finance")
    return bars, events


with tab_volume:
    st.header("Market Surveillance: Unusual Trading Volume Detection")
    st.caption("Research prototype. It flags statistically unusual volume; "
               "it does not identify insider trading and is not a trading signal.")

    # --- Controles -----------------------------------------------------------
    col_a, col_b = st.columns([2, 1])
    label = col_a.selectbox("Instrument", list(INSTRUMENTS))
    days_shown = col_b.slider("Days shown in the chart", 1, 15, 5)
    ticker = INSTRUMENTS[label]

    # --- Ejecutar el modelo (con aviso claro si Yahoo falla) ------------------
    try:
        with st.spinner("Downloading public data and running the model..."):
            bars, events = load(ticker)
    except Exception:
        bars, events = pd.DataFrame(), pd.DataFrame()

    if bars.empty:
        st.warning("Market data is temporarily unavailable from Yahoo Finance. "
                   "Please try again in a few minutes.")
    else:
        # --- Cifras resumen ---------------------------------------------------
        n_days = bars["day"].nunique()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sessions analysed", n_days)
        m2.metric("Unusual-volume events", len(events))
        m3.metric("Events per day", f"{len(events) / n_days:.1f}")
        m4.metric("Latest event (UTC)",
                  events["Start (UTC)"].iloc[0] if len(events) else "None")

        # --- Gráfico: precio arriba, volumen abajo, alertas en rojo -----------
        last_days = sorted(bars["day"].unique())[-days_shown:]
        view = bars[bars["day"].isin(last_days)]
        alerts = view[view["alert"]]

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.4, 0.6], vertical_spacing=0.04)
        fig.add_trace(go.Scatter(x=view.index, y=view["Close"], name="Price",
                                 line=dict(color="#4C78A8", width=1.5)),
                      row=1, col=1)
        fig.add_trace(go.Bar(x=view.index, y=view["Volume"], name="Volume (5-min)",
                             marker_color="#9AA5B1"),
                      row=2, col=1)
        fig.add_trace(go.Bar(x=alerts.index, y=alerts["Volume"], name="Unusual volume",
                             marker_color="#D62728"),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=alerts.index, y=alerts["Close"], mode="markers",
                                 name="Alert", marker=dict(color="#D62728", size=8),
                                 showlegend=False),
                      row=1, col=1)
        fig.update_layout(height=550, barmode="overlay", bargap=0,
                          margin=dict(l=10, r=10, t=30, b=10),
                          legend=dict(orientation="h", y=1.08))
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_xaxes(title_text="Time (UTC)", row=2, col=1)
        st.plotly_chart(fig, width="stretch")

        # --- Tabla de los últimos eventos ------------------------------------
        st.subheader("Latest detected events")
        if len(events):
            st.dataframe(events.head(20), hide_index=True, width="stretch")
        else:
            st.info("No unusual-volume events in the period analysed.")

    # --- Explicación, límites y fuentes de datos ------------------------------
    with st.expander("How the model works"):
        st.markdown("""
- **Baseline:** each 5-minute bar is compared with the volume at the same time of day
  (±5 min) over the previous 20 sessions. Only past data is used (no look-ahead).
- **Alert:** relative volume (volume / baseline median) ≥ 3 **and** robust z-score of
  log-volume (median and MAD) ≥ 3.
- **Direction (estimate):** Bulk Volume Classification, buy share = Φ(ΔP / σ).
  Labelled Buy-side or Sell-side when the share is ≥ 65% on one side, otherwise Mixed.
- **Events:** consecutive alerts less than 15 minutes apart are grouped into one event.
""")

    with st.expander("Limitations"):
        st.markdown("""
- It detects **statistically unusual activity**; it does not prove informed or insider trading.
- **No real-time news feed:** without a minute-by-minute professional feed (e.g. Refinitiv or
  Bloomberg), the model cannot check automatically whether a public headline explains an alert.
  This is the main limitation. A test with second-level X/Twitter headlines did not solve it:
  during periods of heavy news flow, almost every alert had relevant headlines just before it.
- In the research backtest it flagged the reference cases before the public announcement,
  but it also produced around 2-3 alerts per day, and no volume metric separated those cases
  from reactions to public news. Alerts need human review against the news flow.
- This demo uses 5-minute bars, so trade direction is estimated, not the real aggressor side.
- Yahoo Finance data can be delayed or incomplete, and it has no volume for FX.
""")

    with st.expander("Data sources"):
        st.markdown("""
**Used in the research backtest**
- Databento: tick-level trades for ICE Brent futures (with aggressor side) and
  1-minute bars for CME WTI (CL) and E-mini S&P 500 (ES)
- X/Twitter headlines via twitterapi.io: news timestamped to the second, used to test a news filter
- GDELT and Alpaca News (Benzinga): tested as news filters and discarded
  (too coarse or no commodity coverage)
- Manual calendar of scheduled events (EIA, Fed, CPI, payrolls, OPEC+)

**Used in this live demo**
- Yahoo Finance (yfinance): public 5-minute bars, last 60 days, refreshed hourly
""")


# =============================================================================
# PESTAÑA 2: VaR y Expected Shortfall (models/var_es.py)
# =============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_var_prices(portfolio):
    """Descarga los precios de la cartera y los guarda 1 hora.
    Si Yahoo no devuelve datos, lanza un error para que el fallo NO se guarde."""
    prices = var_es.download_prices(portfolio)
    if prices.empty:
        raise ValueError("No data from Yahoo Finance")
    return prices


@st.cache_data(ttl=3600, show_spinner=False)
def run_var(portfolio, confidence, method):
    """Ejecuta el modelo (VaR/ES de hoy y backtesting) y guarda el resultado."""
    return var_es.run(load_var_prices(portfolio), portfolio, confidence, method)


def money(x, ccy):
    """Formato de importes para la app, p. ej. EUR 25,431."""
    return f"{ccy} {x:,.0f}"


def test_result(p_value):
    """Lectura del test al 5% (texto en inglés, porque se ve en la app)."""
    return "Rejected at 5%" if p_value < 0.05 else "Not rejected"


with tab_var:
    st.header("Market Risk: Value at Risk and Expected Shortfall")
    st.caption("Version 1, more features in progress. Fictitious portfolios on public "
               "Yahoo Finance data. Educational model, not investment advice.")

    # --- Controles -----------------------------------------------------------
    col_a, col_b, col_c = st.columns([2, 1, 2])
    portfolio = col_a.selectbox("Portfolio", list(var_es.PORTFOLIOS))
    confidence = col_b.selectbox("Confidence level", [0.99, 0.975, 0.95],
                                 format_func=lambda c: f"{c:.1%}")
    method = col_c.selectbox("Method", var_es.METHODS)

    # --- Ejecutar el modelo (con aviso claro si Yahoo falla) ------------------
    result = None
    try:
        with st.spinner("Downloading public data and running the model..."):
            result = run_var(portfolio, confidence, method)
    except Exception:
        result = None

    if result is None:
        st.warning("Market data is temporarily unavailable from Yahoo Finance. "
                   "Please try again in a few minutes.")
    else:
        ccy = result["currency"]
        tests = result["tests"]
        selected = result["selected"]

        # --- Cifras resumen ---------------------------------------------------
        st.write(f"Portfolio value: **{money(result['value'], ccy)}**, "
                 f"valuation date **{result['as_of']}**, 1-day horizon, "
                 f"estimation window {var_es.LOOKBACK_DAYS} trading days.")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"VaR {confidence:.1%}, 1 day", money(selected["VaR 1d"], ccy))
        m2.metric(f"ES {confidence:.1%}, 1 day", money(selected["ES 1d"], ccy))
        m3.metric(f"Backtest exceptions (expected {tests['expected']:.1f})",
                  f"{tests['exceptions']} / {tests['observations']} days")
        m4.metric("Basel traffic light (last 250 days)",
                  f"{tests['basel_zone']} ({tests['exceptions_250']} exc.)"
                  if tests["basel_zone"] else "Only defined for 99%")

        # --- Gráfico: pérdidas diarias vs VaR, excepciones en rojo ------------
        st.subheader("Backtest: daily loss vs VaR forecast")
        bt = result["backtest"]
        exc = bt[bt["exception"]]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=bt.index, y=bt["loss"], name="Daily loss (gain < 0)",
                             marker_color="#9AA5B1"))
        fig.add_trace(go.Scatter(x=bt.index, y=bt["var"], name=f"VaR {confidence:.1%}",
                                 line=dict(color="#4C78A8", width=2)))
        fig.add_trace(go.Scatter(x=exc.index, y=exc["loss"], mode="markers",
                                 name="Exception (loss > VaR)",
                                 marker=dict(color="#D62728", size=9)))
        fig.update_layout(height=450, bargap=0, margin=dict(l=10, r=10, t=30, b=10),
                          legend=dict(orientation="h", y=1.08))
        fig.update_yaxes(title_text=f"Loss ({ccy})")
        st.plotly_chart(fig, width="stretch")
        st.caption("Each day's VaR is estimated only with the previous "
                   f"{var_es.LOOKBACK_DAYS} days (out of sample) and compared with "
                   "that day's actual portfolio loss.")

        # --- Tests de Kupiec y Christoffersen ---------------------------------
        st.subheader("Backtesting tests")
        tests_table = pd.DataFrame([
            ["Kupiec POF (number of exceptions)", tests["kupiec_lr"], tests["kupiec_p"]],
            ["Christoffersen independence (clustering)", tests["christoffersen_lr"],
             tests["christoffersen_p"]],
            ["Christoffersen conditional coverage (both)", tests["cc_lr"], tests["cc_p"]],
        ], columns=["Test", "LR statistic", "p-value"])
        tests_table["Result"] = tests_table["p-value"].apply(test_result)
        st.dataframe(tests_table.round(3), hide_index=True, width="stretch")

        # --- Comparación de métodos e histograma ------------------------------
        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("All methods, today")
            st.dataframe(result["table"].round(0), hide_index=True, width="stretch")
            st.caption("Monte Carlo Student t degrees of freedom (fitted): "
                       f"{result['student_t_dof']:.1f}. "
                       "10-day figures use the square-root-of-time rule.")
        with col_r:
            st.subheader("P&L distribution, estimation window")
            hist = go.Figure()
            hist.add_trace(go.Histogram(x=result["window_pnl"], nbinsx=60,
                                        marker_color="#9AA5B1", name="Daily P&L"))
            hist.add_vline(x=-selected["VaR 1d"], line_color="#4C78A8", line_width=2,
                           annotation_text="VaR", annotation_position="top left")
            hist.add_vline(x=-selected["ES 1d"], line_color="#D62728", line_width=2,
                           annotation_text="ES", annotation_position="bottom left")
            hist.update_layout(height=350, showlegend=False,
                               margin=dict(l=10, r=10, t=30, b=10))
            hist.update_xaxes(title_text=f"Daily P&L ({ccy})")
            st.plotly_chart(hist, width="stretch")

        with st.expander("Portfolio composition"):
            st.dataframe(result["weights"], hide_index=True, width="stretch",
                         column_config={"Weight": st.column_config.NumberColumn(
                             format="percent")})

    # --- Explicación, límites y fuentes de datos ------------------------------
    with st.expander("How the model works"):
        st.markdown(f"""
- **Portfolio P&L:** fixed weights (rebalanced daily) times daily returns in the base
  currency. Foreign assets are converted with the daily FX rate, so FX risk is included.
- **Historical simulation:** VaR is the loss quantile of the last {var_es.LOOKBACK_DAYS}
  daily P&Ls; ES is the average loss beyond VaR.
- **Parametric normal:** zero-mean normal with the sample covariance matrix
  (equal weights) or an EWMA covariance (λ = 0.94, RiskMetrics), which reacts faster
  to volatility changes. ES = σ·φ(z)/(1 − α).
- **Monte Carlo Student t:** correlated scenarios from the Cholesky factor of the
  covariance matrix, with fat tails from a multivariate Student t whose degrees of
  freedom are fitted to the portfolio kurtosis.
- **Backtesting:** {var_es.BACKTEST_DAYS} days out of sample. Kupiec POF tests whether
  the number of exceptions matches the confidence level; Christoffersen tests whether
  exceptions cluster in time; conditional coverage combines both.
  Basel traffic light: green 0-4, yellow 5-9, red 10+ exceptions in 250 days at 99%.
""")

    with st.expander("Limitations"):
        st.markdown("""
- Equities, ETFs and FX only, with linear P&L: no bonds priced off a yield curve,
  no options and no non-linear risk in this demo.
- Fixed portfolio weights; real portfolios change over time.
- Normal and Student t models assume zero mean returns and a stable covariance
  over the window. Historical simulation assumes the next day looks like the last two years.
- 10-day figures are scaled with √10, which ignores autocorrelation and volatility clustering.
- No stressed VaR or stress scenarios yet, and only basic data cleaning (holidays filled
  for up to 3 days, non-positive prices removed). Yahoo Finance data can contain errors.
- With 500 observations the tests have limited power: "not rejected" does not prove the
  model is right.
""")

    with st.expander("Data sources"):
        st.markdown("""
**Used in this live demo**
- Yahoo Finance (yfinance): public daily adjusted closing prices and FX rates
  (EURUSD=X, EURGBP=X), last 6 years, refreshed hourly

**Used in the full research version (offline)**
- Yahoo Finance prices and ECB euro area AAA yield curve, with data quality controls,
  bonds, an equity index option, stressed VaR and stress tests
""")
