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

st.set_page_config(page_title="Quant Risk Lab", layout="wide")

st.title("Quant Risk Lab")
st.write("Interactive platform for market risk and trading surveillance models")

# Una pestaña por modelo (de momento solo hay uno)
tab_volume, = st.tabs(["Unusual Volume Detection"])

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