# Quant Risk Lab

Interactive platform for market risk and trading surveillance models, built with Streamlit.

## Models

### Market Surveillance: Unusual Trading Volume Detection
Flags statistically unusual trading volume and estimates whether it was buyer- or seller-initiated.

- **Baseline:** each 5-minute bar is compared with the volume at the same time of day over the previous 20 sessions (no look-ahead).
- **Alert:** relative volume ≥ 3 and robust z-score of log-volume (median/MAD) ≥ 3.
- **Direction (estimate):** Bulk Volume Classification, buy share = Φ(ΔP / σ).
- **Events:** alerts less than 15 minutes apart are grouped into one event.

The research version was backtested on tick data (Databento, ICE Brent and CME futures) against
documented cases of unusual pre-announcement volume. This public demo runs on free Yahoo Finance data.

**Limitations:** it detects unusual activity, not insider trading, and it is not a trading signal.
Without a real-time professional news feed, alerts need human review against the news flow.

### Market Risk Engine: VaR, Expected Shortfall and Stress Testing
1-day VaR and ES for three fictitious portfolios (European and US equities in EUR with FX risk,
a US 60/40 stock and bond portfolio, and a diversified multi-asset ETF portfolio), plus a
fictitious book of two euro area government bonds.

- **Methods:** historical simulation, parametric normal (equal weights and EWMA λ = 0.94),
  Monte Carlo Student t with correlated scenarios from a Cholesky factorisation, and
  filtered historical simulation with GARCH(1,1) volatility.
- **Volatility and autocorrelation:** EWMA vs GARCH(1,1) conditional volatility
  (quasi-maximum likelihood with variance targeting) and Ljung-Box tests on returns,
  squared returns and GARCH residuals.
- **Backtesting:** 500 days out of sample, Kupiec POF, Christoffersen independence and
  conditional coverage, Basel traffic light at 99%.
- **Stressed VaR and stress testing:** worst 250-day window since 2006, 2008-09 and 2020
  windows, simplified Basel 2.5 capital; historical (Lehman, euro crisis, Brexit, COVID,
  2022) and hypothetical scenarios on today's portfolio.
- **Rates:** ECB AAA zero-coupon curve, bond pricing, DV01, key-rate DV01, modified
  duration and convexity; PCA of daily curve changes (level, slope, curvature); rates VaR
  with full covariance, 3 principal components and historical full revaluation.
- **Data quality:** non-positive prices, stale prices, bad ticks, holidays and missing
  curve tenors, with an issue log.

**Limitations:** linear P&L for the equity portfolios; fixed weights; generic bonds with no
credit spread; 10-day figures by the square-root-of-time rule; univariate GARCH.

## Run locally

    pip install -r requirements.txt
    streamlit run app.py

## Project structure

    app.py                     Streamlit app (home page and one tab per model)
    .streamlit/config.toml     Colour theme
    models/visuals.py          Page styling and the animated Monte Carlo (HTML canvas)
    models/volume_anomaly.py   Unusual volume detection model
    models/var_es.py           Market risk engine: VaR, ES, GARCH, backtesting, stress
    models/rates.py            Yield curve, DV01 and PCA (ECB data)
    requirements.txt           Python dependencies
