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

## Run locally

    pip install -r requirements.txt
    streamlit run app.py

## Project structure

    app.py                     Streamlit app (one tab per model)
    models/volume_anomaly.py   Unusual volume detection model
    requirements.txt           Python dependencies