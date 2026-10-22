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
from models import rates, var_es, visuals

st.set_page_config(page_title="Quant Risk Lab", layout="wide")

# Estilo (colores, pestañas, tarjetas) y cabecera
st.markdown(visuals.CSS, unsafe_allow_html=True)
st.markdown(visuals.BRAND, unsafe_allow_html=True)

# Portada y una pestaña por modelo
tab_home, tab_volume, tab_var = st.tabs(
    ["Home", "Unusual Volume Detection", "VaR & Expected Shortfall"])

# =============================================================================
# PORTADA: presentación de la app con la animación de Monte Carlo
# =============================================================================
with tab_home:
    st.iframe(visuals.hero_html(440), height=446)
    st.caption("The animation is illustrative: simulated paths, not a forecast. "
               "The models in the other tabs use real public market data.")

    st.write("")
    col_1, col_2 = st.columns(2)
    with col_1, st.container(border=True):
        st.markdown('<div class="qrl-kicker">MODEL 1 · TRADING SURVEILLANCE</div>',
                    unsafe_allow_html=True)
        st.subheader("Unusual Volume Detection")
        st.write("Flags statistically unusual 5-minute trading volume in futures and "
                 "equities, compared with the same time of day in previous sessions, "
                 "and estimates whether the flow was buy-side or sell-side.")
        st.markdown(visuals.chips(["Intraday seasonality", "Robust z-score (MAD)",
                                   "Bulk Volume Classification", "Event clustering"]),
                    unsafe_allow_html=True)
    with col_2, st.container(border=True):
        st.markdown('<div class="qrl-kicker">MODEL 2 · MARKET RISK</div>',
                    unsafe_allow_html=True)
        st.subheader("VaR & Expected Shortfall")
        st.write("A market risk engine for three fictitious portfolios: five VaR and ES "
                 "methods, out-of-sample backtesting, GARCH volatility, stressed VaR, "
                 "stress scenarios and euro yield curve risk.")
        st.markdown(visuals.chips(["Historical & parametric VaR", "Monte Carlo (Cholesky)",
                                   "Filtered HS with GARCH(1,1)", "Kupiec & Christoffersen",
                                   "Stressed VaR", "PCA of the yield curve"]),
                    unsafe_allow_html=True)

    st.write("")
    a, b, c = st.columns(3)
    a.markdown("**Public data only**  \n"
               '<span class="qrl-muted">Yahoo Finance and the European Central Bank, '
               "refreshed automatically.</span>", unsafe_allow_html=True)
    b.markdown("**Honest about limits**  \n"
               '<span class="qrl-muted">Every tab explains how the model works and '
               "where it falls short.</span>", unsafe_allow_html=True)
    c.markdown("**Open source**  \n"
               f'<span class="qrl-muted">Python code on <a href="{visuals.GITHUB_URL}" '
               'target="_blank">GitHub</a>. Not investment advice.</span>',
               unsafe_allow_html=True)
    st.caption("Open a model from the tab bar above.")

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
                                 line=dict(color=visuals.BLUE, width=1.5)),
                      row=1, col=1)
        fig.add_trace(go.Bar(x=view.index, y=view["Volume"], name="Volume (5-min)",
                             marker_color=visuals.GREY),
                      row=2, col=1)
        fig.add_trace(go.Bar(x=alerts.index, y=alerts["Volume"], name="Unusual volume",
                             marker_color=visuals.RED),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=alerts.index, y=alerts["Close"], mode="markers",
                                 name="Alert", marker=dict(color=visuals.RED, size=8),
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
# PESTAÑA 2: Market Risk Engine (models/var_es.py y models/rates.py)
# Dentro hay sub-pestañas: VaR y backtesting, volatilidad, estrés, curva de
# tipos y calidad de datos.
# =============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_var_prices(portfolio):
    """Descarga los precios de la cartera (desde 2006) y los guarda 1 hora.
    Si Yahoo no devuelve datos, lanza un error para que el fallo NO se guarde."""
    prices, dq_log = var_es.download_prices(portfolio)
    if prices.empty:
        raise ValueError("No data from Yahoo Finance")
    return prices, dq_log


@st.cache_data(ttl=3600, show_spinner=False)
def run_var(portfolio, confidence, method):
    """Ejecuta el modelo de la cartera y guarda el resultado."""
    prices, dq_log = load_var_prices(portfolio)
    result = var_es.run(prices, portfolio, confidence, method)
    result["dq_log"] = dq_log
    return result


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def load_curve():
    """Curva del BCE (se publica una vez al día): se guarda 12 horas."""
    raw = rates.download_curve()
    if raw.empty:
        raise ValueError("No data from the ECB")
    return rates.clean_curve(raw)


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def run_rates(confidence, window_days):
    curve, curve_log = load_curve()
    result = rates.run(curve, confidence, window_days)
    result["curve_log"] = curve_log
    return result


def money(x, ccy):
    """Formato de importes para la app, p. ej. EUR 25,431."""
    return f"{ccy} {x:,.0f}"


def test_result(p_value):
    """Lectura del test al 5% (texto en inglés, porque se ve en la app)."""
    return "Rejected at 5%" if p_value < 0.05 else "Not rejected"


def autocorrelation_result(p_value):
    """Lectura del Ljung-Box al 5% (texto en inglés)."""
    return "Autocorrelation at 5%" if p_value < 0.05 else "No significant autocorrelation"


def round_money(df, pct_col):
    """Importes sin decimales, dejando la columna de porcentaje como está."""
    out = df.copy()
    cols = out.select_dtypes("number").columns.drop(pct_col)
    out[cols] = out[cols].round(0)
    return out


def chart_layout(fig, height=380, legend=True):
    """Mismo estilo en todos los gráficos de la pestaña."""
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10),
                      showlegend=legend, legend=dict(orientation="h", y=1.1))
    return fig


RATES_WINDOWS = {"Last 2 years (500 days)": 500, "Last 5 years (1,250 days)": 1250,
                 "Since 2006": 10_000}

with tab_var:
    st.header("Market Risk Engine: VaR, Expected Shortfall and Stress Testing")
    st.caption("Fictitious portfolios on public data (Yahoo Finance and the European "
               "Central Bank). Educational model, not investment advice.")

    # --- Controles comunes -----------------------------------------------------
    col_a, col_b, col_c = st.columns([2, 1, 2])
    portfolio = col_a.selectbox("Portfolio", list(var_es.PORTFOLIOS))
    confidence = col_b.selectbox("Confidence level", [0.99, 0.975, 0.95],
                                 format_func=lambda c: f"{c:.1%}")
    method = col_c.selectbox("Method", var_es.METHODS)

    # --- Ejecutar el modelo (con aviso claro si Yahoo falla) ------------------
    try:
        with st.spinner("Downloading public data and running the model..."):
            result = run_var(portfolio, confidence, method)
    except Exception:
        result = None

    sub_var, sub_vol, sub_stress, sub_rates, sub_dq = st.tabs([
        "VaR & backtesting", "Volatility (GARCH)", "Stressed VaR & stress tests",
        "Yield curve & DV01 (PCA)", "Data quality"])

    yahoo_warning = ("Market data is temporarily unavailable from Yahoo Finance. "
                     "Please try again in a few minutes.")

    # ======================= Sub-pestaña: VaR y backtesting ====================
    with sub_var:
        if result is None:
            st.warning(yahoo_warning)
        else:
            ccy = result["currency"]
            tests = result["tests"]
            selected = result["selected"]

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

            # --- Animación: trayectorias de 10 días con los datos de la cartera -
            st.subheader("10-day simulated paths for this portfolio")
            st.iframe(visuals.portfolio_mc_html(result["window_pnl"], result["value"],
                                                      ccy, confidence), height=366)
            st.caption("Illustration: each path adds 10 daily P&Ls drawn at random from the "
                       f"last {var_es.LOOKBACK_DAYS} days (bootstrap, independent days). "
                       "The red line is the 10-day VaR of these paths. The figures above "
                       "and in the table come from the selected method, not from this "
                       "animation.")

            # --- Gráfico: pérdidas diarias vs VaR, excepciones en rojo --------
            st.subheader("Backtest: daily loss vs VaR forecast")
            bt = result["backtest"]
            exc = bt[bt["exception"]]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=bt.index, y=bt["loss"], name="Daily loss (gain < 0)",
                                 marker_color=visuals.GREY))
            fig.add_trace(go.Scatter(x=bt.index, y=bt["var"], name=f"VaR {confidence:.1%}",
                                     line=dict(color=visuals.BLUE, width=2)))
            fig.add_trace(go.Scatter(x=exc.index, y=exc["loss"], mode="markers",
                                     name="Exception (loss > VaR)",
                                     marker=dict(color=visuals.RED, size=9)))
            chart_layout(fig, 450).update_layout(bargap=0)
            fig.update_yaxes(title_text=f"Loss ({ccy})")
            st.plotly_chart(fig, width="stretch")
            st.caption("Each day's VaR is estimated only with the previous "
                       f"{var_es.LOOKBACK_DAYS} days (out of sample) and compared with "
                       "that day's actual portfolio loss.")

            # --- Tests de Kupiec y Christoffersen -----------------------------
            st.subheader("Backtesting tests")
            tests_table = pd.DataFrame([
                ["Kupiec POF (number of exceptions)", tests["kupiec_lr"], tests["kupiec_p"]],
                ["Christoffersen independence (clustering)", tests["christoffersen_lr"],
                 tests["christoffersen_p"]],
                ["Christoffersen conditional coverage (both)", tests["cc_lr"], tests["cc_p"]],
            ], columns=["Test", "LR statistic", "p-value"])
            tests_table["Result"] = tests_table["p-value"].apply(test_result)
            st.dataframe(tests_table.round(3), hide_index=True, width="stretch")

            # --- Comparación de métodos e histograma --------------------------
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
                                            marker_color=visuals.GREY, name="Daily P&L"))
                hist.add_vline(x=-selected["VaR 1d"], line_color=visuals.BLUE, line_width=2,
                               annotation_text="VaR", annotation_position="top left")
                hist.add_vline(x=-selected["ES 1d"], line_color=visuals.RED, line_width=2,
                               annotation_text="ES", annotation_position="bottom left")
                chart_layout(hist, 350, legend=False)
                hist.update_xaxes(title_text=f"Daily P&L ({ccy})")
                st.plotly_chart(hist, width="stretch")

            with st.expander("Portfolio composition"):
                st.dataframe(result["weights"], hide_index=True, width="stretch",
                             column_config={"Weight": st.column_config.NumberColumn(
                                 format="percent")})

    # ======================= Sub-pestaña: volatilidad ==========================
    with sub_vol:
        if result is None:
            st.warning(yahoo_warning)
        else:
            g = result["garch"]
            v1, v2, v3, v4 = st.columns(4)
            v1.metric("GARCH(1,1) α / β", f"{g['alpha']:.3f} / {g['beta']:.3f}")
            v2.metric("Persistence α + β", f"{g['persistence']:.3f}",
                      f"shock half-life {g['half_life']:.0f} days", delta_color="off",
                      delta_arrow="off")
            v3.metric("Volatility forecast, 1 day ahead (annualised)",
                      f"{g['vol_tomorrow']:.1%}",
                      f"EWMA {g['vol_ewma_tomorrow']:.1%}", delta_color="off",
                      delta_arrow="off")
            v4.metric("GARCH long-run volatility (annualised)", f"{g['long_run_vol']:.1%}")

            st.subheader("Conditional volatility: EWMA vs GARCH(1,1)")
            vol = result["vol_series"]
            vfig = go.Figure()
            vfig.add_trace(go.Scatter(x=vol.index, y=vol["EWMA (lambda 0.94)"],
                                      name="EWMA (λ = 0.94)",
                                      line=dict(color=visuals.GREY, width=1.5)))
            vfig.add_trace(go.Scatter(x=vol.index, y=vol["GARCH(1,1)"], name="GARCH(1,1)",
                                      line=dict(color=visuals.BLUE, width=2)))
            chart_layout(vfig, 380)
            vfig.update_yaxes(title_text="Annualised volatility", tickformat=".0%")
            st.plotly_chart(vfig, width="stretch")
            st.caption("Daily conditional volatility of the portfolio over the estimation "
                       "window, annualised with √252.")

            st.subheader("Ljung-Box autocorrelation tests")
            lb = result["ljung_box"].copy()
            lb["Result"] = lb["p-value"].apply(autocorrelation_result)
            st.dataframe(lb.round(3), hide_index=True, width="stretch")
            st.caption(f"Ljung-Box test with {var_es.LJUNG_BOX_LAGS} lags on the last "
                       f"{var_es.LOOKBACK_DAYS} days. Autocorrelation in squared returns "
                       "means volatility clustering; if it disappears in the squared GARCH "
                       "residuals, the GARCH model has captured it.")

    # ======================= Sub-pestaña: stressed VaR y estrés ================
    with sub_stress:
        if result is None:
            st.warning(yahoo_warning)
        else:
            ccy = result["currency"]
            sv = result["svar_table"]
            cap = result["capital"]
            w_start, w_end = result["worst_window"]
            s1, s2, s3, s4 = st.columns(4)
            s1.metric(f"Current historical VaR {confidence:.1%}, 1 day",
                      money(sv.loc[0, "VaR 1d"], ccy))
            s2.metric(f"Stressed VaR {confidence:.1%}, 1 day", money(sv.loc[2, "VaR 1d"], ccy))
            s3.metric("Worst 250-day window", f"{w_start[:7]} to {w_end[:7]}")
            s4.metric("Basel 2.5 capital, simplified", money(cap["capital"], ccy))

            st.subheader("Historical VaR of every 250-day window since "
                         f"{result['history_start'][:4]}")
            rv = result["rolling_var"]
            sfig = go.Figure()
            sfig.add_vrect(x0=w_start, x1=w_end, fillcolor=visuals.RED, opacity=0.12,
                           line_width=0, annotation_text="Stressed window",
                           annotation_position="top left")
            sfig.add_trace(go.Scatter(x=rv.index, y=rv.values, name="250-day historical VaR",
                                      line=dict(color=visuals.BLUE, width=1.5)))
            chart_layout(sfig, 380, legend=False)
            sfig.update_yaxes(title_text=f"VaR {confidence:.1%}, 1 day ({ccy})")
            st.plotly_chart(sfig, width="stretch")
            st.caption("Today's portfolio (current weights) revalued over the whole history. "
                       "Each point is the historical VaR of the 250 days ending on that date.")

            st.dataframe(sv.round(0), hide_index=True, width="stretch")
            st.caption(f"Basel 2.5 capital (simplified): {var_es.BASEL_MULTIPLIER:g} × "
                       f"10-day VaR ({money(cap['var10'], ccy)}) + "
                       f"{var_es.BASEL_MULTIPLIER:g} × 10-day stressed VaR "
                       f"({money(cap['svar10'], ccy)}). The regulation uses 99% and "
                       "60-day averages; this uses today's values at the selected level.")

            st.subheader("Stress scenarios on today's portfolio")
            stress = result["stress"]
            bfig = go.Figure(go.Bar(
                x=stress["Total P&L"], y=stress["Scenario"], orientation="h",
                marker_color=[visuals.RED if v < 0 else visuals.BLUE
                              for v in stress["Total P&L"]]))
            chart_layout(bfig, 60 + 38 * len(stress), legend=False)
            bfig.update_yaxes(autorange="reversed")
            bfig.update_xaxes(title_text=f"P&L ({ccy})")
            st.plotly_chart(bfig, width="stretch")
            st.dataframe(round_money(stress, "% of portfolio"),
                         hide_index=True, width="stretch",
                         column_config={"% of portfolio": st.column_config.NumberColumn(
                             format="percent")})
            st.caption("Historical scenarios apply each position's actual move between the "
                       "two dates (including FX) to today's positions. Hypothetical scenarios "
                       f"are designed shocks; the bond ETF uses a duration of "
                       f"{var_es.BOND_ETF_DURATION:g} for rate moves.")

    # ======================= Sub-pestaña: curva de tipos =======================
    with sub_rates:
        st.caption("A separate fictitious book of two euro area government bonds "
                   "(EUR 300,000 nominal each), priced off the ECB AAA zero-coupon curve. "
                   "The confidence level above also applies here.")
        window_label = st.selectbox("Window for the PCA and the rates VaR", list(RATES_WINDOWS))
        try:
            with st.spinner("Downloading the ECB yield curve..."):
                rr = run_rates(confidence, RATES_WINDOWS[window_label])
        except Exception:
            rr = None

        if rr is None:
            st.warning("Yield curve data is temporarily unavailable from the European Central "
                       "Bank. Please try again in a few minutes.")
        else:
            rv_table = rr["var_table"]
            pca_tab = rr["pca_table"]
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Bond book value", money(rr["book_value"], "EUR"))
            r2.metric("DV01 (gain if all rates fall 1bp)", money(rr["book_dv01"], "EUR"))
            r3.metric(f"Rates VaR {confidence:.1%}, 1 day (historical)",
                      money(rv_table.loc[2, "VaR 1d"], "EUR"))
            r4.metric("Variance explained by PC1 (level)",
                      f"{pca_tab.loc[0, 'Variance explained']:.1%}")
            st.write(f"Curve date **{rr['as_of']}**, PCA and VaR on daily changes since "
                     f"**{rr['window_start']}**.")

            col_l, col_r = st.columns(2)
            with col_l:
                st.subheader("EUR AAA zero-coupon curve")
                hist_curve = rr["curve_history"]
                cfig = go.Figure()
                for label, years_ago, color in [("Today", 0, visuals.BLUE),
                                                ("1 year ago", 1, visuals.GREY),
                                                ("5 years ago", 5, visuals.GREY_LIGHT)]:
                    when = hist_curve.index[-1] - pd.DateOffset(years=years_ago)
                    if when >= hist_curve.index[0]:
                        cfig.add_trace(go.Scatter(x=rates.TENORS, y=hist_curve.asof(when),
                                                  name=label, mode="lines+markers",
                                                  line=dict(color=color, width=2)))
                chart_layout(cfig, 360)
                cfig.update_xaxes(title_text="Maturity (years)")
                cfig.update_yaxes(title_text="Zero rate", tickformat=".1%")
                st.plotly_chart(cfig, width="stretch")
            with col_r:
                st.subheader("Key-rate DV01 of the book")
                krd = rr["key_rate_dv01"]
                kfig = go.Figure(go.Bar(x=krd.index, y=krd["Book"], marker_color=visuals.BLUE))
                chart_layout(kfig, 360, legend=False)
                kfig.update_yaxes(title_text="EUR per bp (rate down)")
                st.plotly_chart(kfig, width="stretch")

            st.subheader("Bonds: price, DV01, duration and convexity")
            st.dataframe(rr["bond_table"].round(2), hide_index=True, width="stretch")

            st.subheader("PCA of daily curve changes")
            col_l, col_r = st.columns([3, 2])
            with col_l:
                load = rr["loadings"]
                pfig = go.Figure()
                for col, color in zip(load.columns, [visuals.BLUE, visuals.RED, visuals.GREY]):
                    pfig.add_trace(go.Scatter(x=rates.TENORS, y=load[col], name=col,
                                              mode="lines+markers",
                                              line=dict(color=color, width=2)))
                chart_layout(pfig, 360)
                pfig.update_xaxes(title_text="Maturity (years)")
                pfig.update_yaxes(title_text="Loading")
                st.plotly_chart(pfig, width="stretch")
            with col_r:
                st.dataframe(pca_tab.head(5).round(3), hide_index=True, width="stretch",
                             column_config={
                                 "Variance explained": st.column_config.NumberColumn(
                                     format="percent"),
                                 "Cumulative": st.column_config.NumberColumn(
                                     format="percent")})
                st.dataframe(rr["pc_exposure"].round(0), hide_index=True, width="stretch")

            st.subheader("Rates VaR of the bond book")
            st.dataframe(rv_table.round(0), hide_index=True, width="stretch")
            st.caption(f"The first {rates.N_COMPONENTS} principal components give "
                       f"{rr['pca_share_of_var']:.1%} of the full-covariance volatility of "
                       "the book.")

            st.subheader("Rate scenarios: full revaluation vs DV01")
            st.dataframe(round_money(rr["scenarios"], "% of book"),
                         hide_index=True, width="stretch",
                         column_config={"% of book": st.column_config.NumberColumn(
                             format="percent")})

    # ======================= Sub-pestaña: calidad de datos =====================
    with sub_dq:
        st.subheader("Equity, ETF and FX prices (Yahoo Finance)")
        if result is None:
            st.warning(yahoo_warning)
        elif result["dq_log"].empty:
            st.info("No data quality issues found in this portfolio's price history.")
        else:
            st.dataframe(result["dq_log"], hide_index=True, width="stretch")
        st.subheader("EUR yield curve (ECB)")
        if rr is None:
            st.warning("Yield curve data is temporarily unavailable from the European "
                       "Central Bank.")
        elif rr["curve_log"].empty:
            st.info("No data quality issues found in the yield curve.")
        else:
            log = rr["curve_log"]
            st.write(f"{len(log)} issues logged. Most recent:")
            st.dataframe(log.iloc[::-1].head(20), hide_index=True, width="stretch")

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
- **GARCH(1,1):** σ²ₜ₊₁ = ω + α·rₜ² + β·σ²ₜ on the portfolio returns, estimated by
  Gaussian (quasi) maximum likelihood with variance targeting (ω fixed so that the
  long-run variance equals the sample variance).
- **Filtered historical simulation (GARCH):** the real returns of the window are
  divided by their GARCH volatility (standardised residuals) and rescaled with the
  volatility forecast for tomorrow. It captures volatility clustering and fat tails
  without assuming a normal distribution.
- **Ljung-Box:** Q = n(n+2)·Σ ρₖ²/(n−k) over {var_es.LJUNG_BOX_LAGS} lags, chi-square
  distributed. Applied to returns (serial correlation), squared returns (volatility
  clustering) and GARCH residuals (what the model leaves unexplained).
- **Backtesting:** {var_es.BACKTEST_DAYS} days out of sample. Kupiec POF tests whether
  the number of exceptions matches the confidence level; Christoffersen tests whether
  exceptions cluster in time; conditional coverage combines both.
  Basel traffic light: green 0-4, yellow 5-9, red 10+ exceptions in 250 days at 99%.
- **Stressed VaR:** today's portfolio is revalued over the full history since 2006 and
  the historical VaR is computed for every 250-day window; the worst window is the
  stressed period (Basel 2.5). The 2008-09 and 2020 windows are also shown.
- **Stress tests:** historical scenarios (Lehman, euro crisis 2011, Brexit, COVID,
  2022 rate shock) and hypothetical shocks applied instantly to today's positions.
- **Yield curve:** the bonds are priced off the ECB AAA zero curve (continuous
  compounding, linear interpolation). Key-rate DV01 bumps each tenor by 1bp; the
  scenarios compare full revaluation with the DV01 approximation (the difference is
  convexity). The PCA is the eigen-decomposition of the covariance matrix of daily
  curve changes: level, slope and curvature. The rates VaR uses the full 9-tenor
  covariance, only the first {rates.N_COMPONENTS} components, or historical curve moves
  with full revaluation.
- **Data quality:** non-positive prices removed, stale prices flagged, isolated
  spikes that reverse the next day (bad ticks) replaced, holidays filled for up to
  {var_es.FFILL_LIMIT} days; missing curve tenors interpolated from the other tenors
  of the same day.
""")

    with st.expander("Limitations"):
        st.markdown("""
- The equity portfolios hold equities, ETFs and FX with linear P&L; the bond book is
  shown separately and is not aggregated with them. There are no options.
- Fixed portfolio weights; real portfolios change over time.
- Normal and Student t models assume zero mean returns and a stable covariance
  over the window. Historical simulation assumes the next day looks like the last two years.
- 10-day figures are scaled with √10, which ignores autocorrelation and volatility clustering.
- GARCH(1,1) is fitted to the portfolio returns (univariate), not to each asset with
  dynamic correlations. The likelihood is maximised on a grid (steps of 0.001) rather
  than with a numerical optimiser, and in the backtest the parameters are re-estimated
  every 20 days (the volatility itself is updated daily).
- The stressed window is searched only from 2006, and only over the dates when every
  position has a price. The Basel 2.5 capital is a simplified version (no 60-day
  averages, no qualitative add-ons).
- Hypothetical scenarios are hand-designed and the bond ETF is approximated with a
  fixed duration, ignoring convexity.
- The bonds are generic: fixed residual maturities, annual coupons, no credit spread
  over the AAA curve.
- With 500 observations the tests have limited power: "not rejected" does not prove the
  model is right. Public data can contain errors that the checks do not catch.
""")

    with st.expander("Data sources"):
        st.markdown("""
**Used in this live demo**
- Yahoo Finance (yfinance): public daily adjusted closing prices and FX rates
  (EURUSD=X, EURGBP=X) since 2006, refreshed hourly
- European Central Bank Data Portal: euro area AAA government zero-coupon yield curve
  (dataset YC), 9 tenors from 1 to 30 years, since 2006, refreshed twice a day
""")
