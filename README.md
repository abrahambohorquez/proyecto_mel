# Injury Risk Predictor — Sports Medicine Module

Predictive web tool based on a logistic regression model trained on 800 professional athletes. Estimates the probability of injury during the upcoming season and prescribes a prioritized prevention plan.

**Performance:** AUC = 0.997 · Pseudo-R² Nagelkerke = 0.913 · 10-fold cross-validation.

---

## Installation

### Requirements
- Python 3.10 or higher
- pip

### Steps

1. Open a terminal in this folder.
2. Install dependencies:

```
pip install -r requirements.txt
```

3. Launch the app:

```
streamlit run predictor_lesiones.py
```

4. Your browser will open automatically at `http://localhost:8501`.

---

## How to use

1. Enter the player's baseline values on the left panel (12 variables across 4 domains: physical, mental, clinical, training load).
2. The right panel shows in real time:
   - **Forecast:** estimated probability and risk band.
   - **Head-to-head:** player vs. study cohort, metric by metric.
   - **Key drivers:** top risk and protective factors with log-odds contribution.
   - **Decomposition:** per-variable contribution chart.
   - **Action plan:** prioritized interventions ranked by impact.

---

## File contents

- `predictor_lesiones.py` — Streamlit application (single file).
- `requirements.txt` — Python dependencies pinned to working versions.
- `README.md` — this file.

---

## Project

Final Project — Linear Statistical Models (IIND-4100) — Universidad de los Andes — 2026.

**Authors:** Samuel Arbeláez · Abraham Bohorquez.
**Advisor:** Prof. Alejandra Tabares Pozos.

---

## Notes

- This tool is informational and does not replace professional medical evaluation.
- Streamlit is pinned to `1.39.0` because newer versions have a dependency conflict with `starlette` on certain Windows setups.
- If the default port `8501` is busy, run with `streamlit run predictor_lesiones.py --server.port 8502`.
