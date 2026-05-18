import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ============ CONFIG ============
st.set_page_config(
    page_title="Injury Risk Predictor",
    page_icon="●",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============ ESTILOS ============
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --bg:           #F7F7F5;
    --paper:        #FFFFFF;
    --ink:          #0F1115;
    --ink-soft:     #3A3F47;
    --ink-mute:     #6B7280;
    --line:         #E5E7EB;
    --line-soft:    #F0F1F3;

    --brand:        #0F4D2E;
    --brand-soft:   #1B6B43;
    --brand-tint:   #EAF1ED;

    --low:          #137548;
    --low-tint:     #E6F2EC;
    --mid:          #A36912;
    --mid-tint:     #FAF1DD;
    --high:         #9A2C18;
    --high-tint:    #F6E0D9;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
html { color-scheme: light !important; }
body { color: var(--ink); }

.stApp { background: var(--bg); }
.main, .main > div, [data-testid="stAppViewContainer"] { background: transparent !important; }
#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1320px;
}

/* ===== TOPBAR ===== */
.topbar {
    background: var(--paper);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 1.2rem 1.6rem;
    margin-bottom: 1.1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 2rem;
    flex-wrap: wrap;
}
.topbar-brand { display: flex; align-items: center; gap: 0.85rem; }
.topbar-mark {
    width: 36px; height: 36px;
    background: var(--brand);
    color: white;
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: -0.02em;
}
.topbar-title {
    font-family: 'Inter', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--ink) !important;
    margin: 0;
    letter-spacing: -0.01em;
    line-height: 1.2;
}
.topbar-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--ink-mute) !important;
    margin: 0.1rem 0 0;
    letter-spacing: 0.05em;
}
.topbar-stats {
    display: flex;
    gap: 1.8rem;
    align-items: center;
}
.topbar-stat {
    text-align: right;
}
.topbar-stat .v {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    color: var(--ink) !important;
    font-size: 0.95rem;
    line-height: 1;
}
.topbar-stat .l {
    font-size: 0.65rem;
    color: var(--ink-mute) !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
    margin-top: 0.25rem;
}

/* ===== INTRO ===== */
.intro {
    background: var(--paper);
    border: 1px solid var(--line);
    border-left: 3px solid var(--brand);
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.1rem;
}
.intro h1 {
    font-family: 'Inter', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--ink) !important;
    margin: 0;
    letter-spacing: -0.015em;
    line-height: 1.25;
}
.intro p {
    font-size: 0.92rem;
    color: var(--ink-soft) !important;
    margin: 0.45rem 0 0;
    line-height: 1.55;
    max-width: 780px;
}

/* ===== PANEL ===== */
.panel {
    background: var(--paper);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 1.4rem 1.5rem;
    margin-bottom: 1.1rem;
}
.panel-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 1.1rem;
    padding-bottom: 0.9rem;
    border-bottom: 1px solid var(--line-soft);
}
.panel-head .ph-lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 700;
    color: var(--brand) !important;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}
.panel-head .ph-title {
    font-family: 'Inter', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--ink) !important;
    margin: 0;
    letter-spacing: -0.01em;
    flex: 1;
    text-align: left;
}
.panel-head .ph-hint {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--ink-mute) !important;
    letter-spacing: 0.04em;
}

/* ===== INPUT GROUP ===== */
.group {
    margin: 1.3rem 0 0.5rem;
    padding: 0.6rem 0.9rem;
    background: var(--line-soft);
    border-radius: 6px;
}
.group-title {
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    color: var(--ink) !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 0;
}
.group-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--ink-mute) !important;
    margin: 0.15rem 0 0;
}

/* ===== SCORE ===== */
.score-card {
    background: var(--paper);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 1.4rem 1.5rem 1rem;
    margin-bottom: 1.1rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.score-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.score-card.low::before  { background: var(--low); }
.score-card.mid::before  { background: var(--mid); }
.score-card.high::before { background: var(--high); }

.score-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--ink-mute) !important;
    margin: 0;
}

.risk-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.32rem 0.8rem;
    border-radius: 100px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}
.risk-pill .dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.risk-pill.low  { background: var(--low-tint);  color: var(--low); }
.risk-pill.mid  { background: var(--mid-tint);  color: var(--mid); }
.risk-pill.high { background: var(--high-tint); color: var(--high); }

.score-msg {
    color: var(--ink-soft) !important;
    margin: 0.7rem auto 0;
    font-size: 0.88rem;
    line-height: 1.5;
    max-width: 360px;
}

/* ===== MATCHSTATS ===== */
.matchstats {
    display: grid;
    grid-template-columns: 1fr;
    gap: 0;
    margin-top: 0.4rem;
}
.matchstats-head {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    font-weight: 700;
    color: var(--ink-mute) !important;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 0.55rem 0.4rem;
    border-bottom: 1px solid var(--line);
}
.matchstats-head .mh-you   { text-align: left;   color: var(--brand) !important; }
.matchstats-head .mh-lbl   { text-align: center; }
.matchstats-head .mh-them  { text-align: right; }

.ms-row {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    align-items: center;
    padding: 0.55rem 0.4rem;
    border-bottom: 1px solid var(--line-soft);
}
.ms-row:last-child { border-bottom: none; }
.ms-you {
    text-align: left;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.92rem;
    font-weight: 700;
    color: var(--ink) !important;
}
.ms-lbl {
    text-align: center;
    font-size: 0.82rem;
    color: var(--ink-soft) !important;
    font-weight: 500;
}
.ms-them {
    text-align: right;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.88rem;
    color: var(--ink-mute) !important;
}
.ms-good .ms-you { color: var(--low) !important; }
.ms-bad  .ms-you { color: var(--high) !important; }

/* ===== DRIVERS ===== */
.driver {
    background: var(--paper);
    padding: 0.95rem 1.15rem;
    border: 1px solid var(--line);
    border-left: 3px solid currentColor;
    border-radius: 6px;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.driver.risk  { color: var(--high); }
.driver.guard { color: var(--low); }
.driver-body { flex: 1; }
.driver-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: currentColor !important;
    margin: 0;
}
.driver-title {
    font-family: 'Inter', sans-serif;
    font-size: 0.98rem;
    font-weight: 700;
    color: var(--ink) !important;
    margin: 0.22rem 0;
    letter-spacing: -0.005em;
}
.driver-desc {
    font-size: 0.83rem;
    color: var(--ink-soft) !important;
    line-height: 1.45;
    margin: 0;
}
.driver-stat {
    flex-shrink: 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.35rem;
    font-weight: 700;
    color: currentColor !important;
    text-align: right;
}
.driver-stat .l {
    display: block;
    font-size: 0.52rem;
    color: var(--ink-mute) !important;
    font-weight: 600;
    letter-spacing: 0.12em;
    margin-top: 0.1rem;
    text-transform: uppercase;
}

/* ===== ADVICE ===== */
.advice {
    background: var(--paper);
    padding: 0.9rem 1.1rem;
    border-radius: 6px;
    margin-bottom: 0.55rem;
    border: 1px solid var(--line);
    border-left: 3px solid currentColor;
}
.advice.danger  { color: var(--high); }
.advice.warning { color: var(--mid); }
.advice.success { color: var(--low); }
.advice-head {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.3rem;
}
.advice-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 0.14rem 0.5rem;
    border-radius: 3px;
    color: currentColor !important;
}
.advice.danger  .advice-tag { background: var(--high-tint); }
.advice.warning .advice-tag { background: var(--mid-tint); }
.advice.success .advice-tag { background: var(--low-tint); }
.advice h4 {
    font-family: 'Inter', sans-serif;
    font-size: 0.92rem;
    font-weight: 700;
    color: var(--ink) !important;
    margin: 0;
    letter-spacing: -0.005em;
}
.advice p {
    margin: 0.3rem 0 0;
    color: var(--ink-soft) !important;
    font-size: 0.83rem;
    line-height: 1.5;
}

/* ===== NOTE ===== */
.note {
    background: var(--brand-tint);
    border-radius: 6px;
    padding: 0.7rem 0.95rem;
    margin-bottom: 0.85rem;
    font-size: 0.8rem;
    color: var(--ink-soft) !important;
    border-left: 3px solid var(--brand);
    line-height: 1.5;
}
.note .dot-green { color: var(--low) !important; font-weight: 700; }
.note .dot-red   { color: var(--high) !important; font-weight: 700; }

/* ===== FOOTER ===== */
.footer {
    background: var(--paper);
    border: 1px solid var(--line);
    border-top: 3px solid var(--brand);
    border-radius: 8px;
    padding: 1.3rem 1.5rem;
    margin-top: 1.8rem;
    text-align: center;
}
.footer-title {
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--ink) !important;
    margin: 0;
}
.footer-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--ink-mute) !important;
    margin: 0.5rem 0 0;
    letter-spacing: 0.08em;
}
.footer-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    color: var(--ink-mute) !important;
    margin: 0.3rem 0 0;
}

/* ===== STREAMLIT OVERRIDES ===== */
[data-testid="stExpander"] details {
    background: var(--paper) !important;
    border-radius: 6px !important;
    border: 1px solid var(--line) !important;
    box-shadow: none !important;
}
[data-testid="stExpander"] summary {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    color: var(--ink) !important;
    padding: 0.85rem 1.2rem !important;
    font-size: 0.92rem !important;
    background: var(--paper) !important;
}
[data-testid="stExpander"] summary:hover { background: var(--bg) !important; }
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] {
    padding: 0.5rem 1.2rem 1rem !important;
    background: var(--paper) !important;
    color: var(--ink-soft) !important;
    font-size: 0.88rem;
    line-height: 1.65;
}
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] h3 {
    color: var(--ink) !important;
    font-weight: 700 !important;
    margin-top: 1.2rem !important;
    margin-bottom: 0.4rem !important;
    font-size: 0.95rem !important;
}
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] strong { color: var(--ink) !important; }

.stSlider label, .stRadio label, .stNumberInput label {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    color: var(--ink) !important;
    font-size: 0.85rem !important;
}
[data-testid="stSlider"] > div > div { background: transparent !important; }
[data-baseweb="slider"] [role="slider"] {
    background: var(--paper) !important;
    border: 2px solid var(--brand) !important;
    box-shadow: 0 1px 3px rgba(15,77,46,0.18) !important;
    width: 16px !important; height: 16px !important;
}
[data-baseweb="slider"] > div > div > div { background: var(--brand) !important; }

.stNumberInput input, [data-baseweb="input"] input {
    background-color: var(--paper) !important;
    color: var(--ink) !important;
    border: 1px solid var(--line) !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
}
.stNumberInput button {
    background-color: var(--bg) !important;
    color: var(--ink) !important;
    border: 1px solid var(--line) !important;
}
.stNumberInput button:hover { background-color: var(--brand-tint) !important; }

.stRadio > div { background: transparent !important; }
.stRadio label, .stRadio label *, .stRadio div[role="radiogroup"] * {
    color: var(--ink) !important;
    font-weight: 500 !important;
}

div[data-testid="stVerticalBlock"] > div { gap: 0.4rem; }
</style>
""", unsafe_allow_html=True)

# ============ MODELO ============
COEFS = {
    'const': 0.9176,
    'Training_Hours_Per_Week': 0.4559, 'Previous_Injury_Count': 0.9926,
    'Knee_Strength_Score': -0.9198, 'Hamstring_Flexibility': -1.1525,
    'Reaction_Time_ms': 1.3704, 'Balance_Test_Score': -1.4269,
    'Sprint_Speed_10m_s': -1.3407, 'Agility_Score': -0.7132,
    'Sleep_Hours_Per_Night': -1.1503, 'Stress_Level_Score': 1.3720,
    'Nutrition_Quality_Score': -1.2051, 'Warmup_Routine_Adherence': -1.5408,
    'Stress_x_Sleep': -0.8034
}
MEANS = {
    'Training_Hours_Per_Week': 9.95, 'Previous_Injury_Count': 1.51,
    'Knee_Strength_Score': 74.94, 'Hamstring_Flexibility': 79.03,
    'Reaction_Time_ms': 249.14, 'Balance_Test_Score': 83.69,
    'Sprint_Speed_10m_s': 5.95, 'Agility_Score': 78.39,
    'Sleep_Hours_Per_Night': 7.40, 'Stress_Level_Score': 53.99,
    'Nutrition_Quality_Score': 74.36
}
STDS = {
    'Training_Hours_Per_Week': 2.78, 'Previous_Injury_Count': 1.30,
    'Knee_Strength_Score': 6.50, 'Hamstring_Flexibility': 6.90,
    'Reaction_Time_ms': 21.00, 'Balance_Test_Score': 7.40,
    'Sprint_Speed_10m_s': 0.36, 'Agility_Score': 9.40,
    'Sleep_Hours_Per_Night': 0.96, 'Stress_Level_Score': 11.50,
    'Nutrition_Quality_Score': 11.00
}

def standardize(value, var):
    return (value - MEANS[var]) / STDS[var]

def predict_probability(inputs):
    z = {var: standardize(val, var) for var, val in inputs.items() if var in MEANS}
    z['Warmup_Routine_Adherence'] = inputs['Warmup_Routine_Adherence']
    eta = COEFS['const']
    for var in z:
        if var in COEFS:
            eta += COEFS[var] * z[var]
    eta += COEFS['Stress_x_Sleep'] * z['Stress_Level_Score'] * z['Sleep_Hours_Per_Night']
    return 1 / (1 + np.exp(-eta)), z

# ============ TOPBAR ============
st.markdown(
    '<div class="topbar">'
    '<div class="topbar-brand">'
    '<div class="topbar-mark">IR</div>'
    '<div>'
    '<p class="topbar-title">Injury Risk Predictor</p>'
    '<p class="topbar-sub">Sports Medicine Module · v1.0</p>'
    '</div>'
    '</div>'
    '<div class="topbar-stats">'
    '<div class="topbar-stat"><div class="v">800</div><div class="l">Cohort</div></div>'
    '<div class="topbar-stat"><div class="v">0.997</div><div class="l">AUC</div></div>'
    '<div class="topbar-stat"><div class="v">13</div><div class="l">Predictors</div></div>'
    '<div class="topbar-stat"><div class="v">10×</div><div class="l">CV folds</div></div>'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)

# ============ INTRO ============
st.markdown(
    '<div class="intro">'
    '<h1>Estimate a player\'s injury probability for the upcoming season</h1>'
    '<p>Enter the athlete\'s baseline values below. The model returns a calibrated probability, '
    'a head-to-head comparison against the study cohort, the key risk and protective drivers, '
    'and a prioritized intervention plan.</p>'
    '</div>',
    unsafe_allow_html=True
)

# ============ EXPANDERS ============
with st.expander("How this tool works", expanded=False):
    st.markdown("""
This tool applies a **logistic regression model** trained on 800 professional players. It combines 13 variables — physical, mental, clinical, and habit-based — to deliver a personalized probability of injury during the upcoming season.

**Pipeline:**

1. Each input is standardized against the study cohort mean.
2. The model assigns a statistically significant weight (Odds Ratio) to each variable.
3. The linear combination passes through a sigmoid, yielding a probability between 0 and 1.
4. The system flags where the player deviates most from the optimum and generates prioritized prescriptions.

**Validity:** AUC = 0.997 · Pseudo-R² Nagelkerke = 0.913 · 10-fold cross-validation confirms stability.

**Disclaimer:** Informational tool. Does not replace professional medical evaluation.
""")

with st.expander("How to measure each variable", expanded=False):
    st.markdown("""
### Physical capacity

**Knee strength (0–100)** — Isokinetic quadriceps test. Without equipment: single-leg squats before form breaks. 0–5 → 60 · 6–10 → 70 · 11–15 → 80 · 16+ → 90.

**Hamstring flexibility (0–100)** — Sit-and-reach test. −10 cm → 60 · 0 cm → 75 · +5 cm → 85 · +15 cm → 95.

**Reaction time (180–320 ms)** — humanbenchmark.com/tests/reactiontime. Average 5 trials. Adult: 250 ms · Elite: 200–220 ms.

**Balance score (0–100)** — Y-Balance test or single-leg stand with eyes closed: <10 s → 65 · 10–30 s → 75 · 30–60 s → 85 · 60+ s → 95.

**Sprint speed 10 m (m/s)** — Time a 10 m sprint: m/s = 10 / seconds. Elite athletes: 6.5+ m/s.

**Agility (0–100)** — T-test: >12 s → 65 · 11–12 s → 75 · 10–11 s → 85 · <10 s → 95.

### Mental & habits

**Stress level (0–100)** — PSS-10 × 2.5 or subjective scale. Low <35 · Medium 35–65 · High >65.

**Sleep hours / night** — Average of past 7 nights. Recommended 7.5–9 h.

**Nutrition quality (0–100)** — 50 poor · 70 average · 80 good · 90+ excellent.

**Warmup adherence** — Mobility + activation + sport-specific drills before every session.

### Clinical history
**Previous injuries** — Total muscular, joint, or ligament injuries in career.

### Training load
**Hours per week** — Amateur 5–10 h · Semi-pro 10–15 h · Pro 15–20 h.
""")

# ============ LAYOUT ============
col_input, col_result = st.columns([1, 1.1], gap="large")

# ============ COLUMNA IZQUIERDA — INPUTS ============
with col_input:
    st.markdown(
        '<div class="panel">'
        '<div class="panel-head">'
        '<span class="ph-lbl">01 / Inputs</span>'
        '<h3 class="ph-title">Player profile</h3>'
        '<span class="ph-hint">12 variables</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="group">'
        '<p class="group-title">Physical capacity</p>'
        '<p class="group-sub">Neuromuscular and conditioning measures</p>'
        '</div>',
        unsafe_allow_html=True
    )
    c1, c2 = st.columns(2)
    with c1:
        knee = st.slider("Knee strength", 50, 100, 75,
                         help="Isokinetic quadriceps test. Without gear: count single-leg squats before form breaks.")
        balance = st.slider("Balance", 50, 100, 84,
                            help="Y-Balance or single-leg stand with eyes closed.")
        agility = st.slider("Agility", 50, 100, 78,
                            help="T-test or Illinois Agility.")
    with c2:
        hamstring = st.slider("Hamstring flexibility", 50, 100, 79,
                              help="Sit-and-reach test. Higher = more flexible.")
        sprint = st.slider("Sprint 10 m (m/s)", 4.5, 7.0, 5.95, 0.05,
                           help="m/s = 10 / seconds in a 10 m sprint.")
        reaction = st.slider("Reaction time (ms)", 180, 320, 249,
                             help="humanbenchmark.com/tests/reactiontime. Lower = faster.")

    st.markdown(
        '<div class="group">'
        '<p class="group-title">Mental state and habits</p>'
        '<p class="group-sub">Psychological load and recovery routines</p>'
        '</div>',
        unsafe_allow_html=True
    )
    c3, c4 = st.columns(2)
    with c3:
        stress = st.slider("Stress level", 20, 100, 54,
                           help="Subjective scale or PSS-10 × 2.5.")
        nutrition = st.slider("Nutrition quality", 50, 100, 74,
                              help="50 poor · 70 average · 80 good · 90+ excellent.")
    with c4:
        sleep = st.slider("Sleep per night (h)", 4.0, 11.0, 7.4, 0.1)
        warmup = st.radio("Always warms up before training",
                          options=[1, 0],
                          format_func=lambda x: "Yes" if x == 1 else "No",
                          horizontal=True)

    st.markdown(
        '<div class="group">'
        '<p class="group-title">Clinical history and training load</p>'
        '<p class="group-sub">Background and weekly exposure</p>'
        '</div>',
        unsafe_allow_html=True
    )
    c5, c6 = st.columns(2)
    with c5:
        prev_injury = st.number_input("Previous injuries (career)", 0, 15, 1)
    with c6:
        training_hours = st.slider("Training hours / week", 4.0, 20.0, 10.0, 0.5)

# ============ COMPUTE ============
inputs = {
    'Training_Hours_Per_Week': training_hours, 'Previous_Injury_Count': prev_injury,
    'Knee_Strength_Score': knee, 'Hamstring_Flexibility': hamstring,
    'Reaction_Time_ms': reaction, 'Balance_Test_Score': balance,
    'Sprint_Speed_10m_s': sprint, 'Agility_Score': agility,
    'Sleep_Hours_Per_Night': sleep, 'Stress_Level_Score': stress,
    'Nutrition_Quality_Score': nutrition, 'Warmup_Routine_Adherence': warmup
}
prob, z_inputs = predict_probability(inputs)
prob_pct = prob * 100

# ============ COLUMNA DERECHA — RESULTADOS ============
with col_result:
    # --- 02 / Forecast ---
    if prob < 0.30:
        risk_color = "#137548"; risk_label = "LOW"; risk_class = "low"
        risk_msg = "The player's profile reflects low risk. Maintain current routines and monitor monthly."
    elif prob < 0.60:
        risk_color = "#A36912"; risk_label = "MODERATE"; risk_class = "mid"
        risk_msg = "Modifiable factors are present. Apply the prescribed actions below and reassess in 4 weeks."
    else:
        risk_color = "#9A2C18"; risk_label = "HIGH"; risk_class = "high"
        risk_msg = "Composite risk is elevated. Preventive intervention is strongly recommended."

    st.markdown(
        '<div class="panel">'
        '<div class="panel-head">'
        '<span class="ph-lbl">02 / Forecast</span>'
        '<h3 class="ph-title">Injury probability</h3>'
        '<span class="ph-hint">next season</span>'
        '</div>'
        f'<div class="score-card {risk_class}">'
        '<p class="score-eyebrow">Estimated probability</p>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_pct,
        number={'suffix': "%", 'font': {'size': 64, 'color': risk_color, 'family': 'Inter'}},
        gauge={
            'axis': {'range': [0, 100], 'tickfont': {'size': 10, 'color': '#6B7280', 'family': 'JetBrains Mono'},
                     'tickwidth': 1, 'tickcolor': '#E5E7EB'},
            'bar': {'color': risk_color, 'thickness': 0.82},
            'bgcolor': "#F7F7F5",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 30], 'color': '#E6F2EC'},
                {'range': [30, 60], 'color': '#FAF1DD'},
                {'range': [60, 100], 'color': '#F6E0D9'}
            ],
        }
    ))
    fig.update_layout(height=220, margin=dict(t=0, b=0, l=20, r=20),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font={'family': 'Inter'})
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown(
        '<div style="text-align:center; margin-top:-0.6rem;">'
        f'<div class="risk-pill {risk_class}"><div class="dot"></div>{risk_label} RISK</div>'
        f'<p class="score-msg">{risk_msg}</p>'
        '</div>',
        unsafe_allow_html=True
    )

    # --- 03 / Head-to-head ---
    avg_inputs = {
        'Knee_Strength_Score': 74.94, 'Hamstring_Flexibility': 79.03,
        'Reaction_Time_ms': 249.14, 'Balance_Test_Score': 83.69,
        'Sprint_Speed_10m_s': 5.95, 'Agility_Score': 78.39,
        'Sleep_Hours_Per_Night': 7.40, 'Stress_Level_Score': 53.99,
        'Nutrition_Quality_Score': 74.36, 'Warmup_Routine_Adherence': 1
    }

    def stat_class(user, cohort, higher_better):
        diff = user - cohort
        if higher_better:
            return 'ms-good' if diff > 0 else ('ms-bad' if diff < 0 else '')
        return 'ms-good' if diff < 0 else ('ms-bad' if diff > 0 else '')

    match_stats = [
        ('Stress level',       stress,         avg_inputs['Stress_Level_Score'],          f'{stress}',        f'{avg_inputs["Stress_Level_Score"]:.0f}',         False),
        ('Sleep (h)',          sleep,          avg_inputs['Sleep_Hours_Per_Night'],       f'{sleep:.1f}',     f'{avg_inputs["Sleep_Hours_Per_Night"]:.1f}',      True),
        ('Balance',            balance,        avg_inputs['Balance_Test_Score'],          f'{balance}',       f'{avg_inputs["Balance_Test_Score"]:.0f}',         True),
        ('Reaction (ms)',      reaction,       avg_inputs['Reaction_Time_ms'],            f'{reaction}',      f'{avg_inputs["Reaction_Time_ms"]:.0f}',           False),
        ('Sprint (m/s)',       sprint,         avg_inputs['Sprint_Speed_10m_s'],          f'{sprint:.2f}',    f'{avg_inputs["Sprint_Speed_10m_s"]:.2f}',         True),
        ('Knee strength',      knee,           avg_inputs['Knee_Strength_Score'],         f'{knee}',          f'{avg_inputs["Knee_Strength_Score"]:.0f}',        True),
        ('Hamstring flex.',    hamstring,      avg_inputs['Hamstring_Flexibility'],       f'{hamstring}',     f'{avg_inputs["Hamstring_Flexibility"]:.0f}',      True),
        ('Agility',            agility,        avg_inputs['Agility_Score'],               f'{agility}',       f'{avg_inputs["Agility_Score"]:.0f}',              True),
        ('Nutrition',          nutrition,      avg_inputs['Nutrition_Quality_Score'],     f'{nutrition}',     f'{avg_inputs["Nutrition_Quality_Score"]:.0f}',    True),
        ('Previous injuries',  prev_injury,    MEANS['Previous_Injury_Count'],            f'{prev_injury}',   f'{MEANS["Previous_Injury_Count"]:.1f}',           False),
        ('Warmup',             1 if warmup==1 else 0, 0.61,                                'Yes' if warmup==1 else 'No', '61% yes',                              True),
    ]

    ms_rows = ""
    for label, u_num, c_num, u_str, c_str, higher_better in match_stats:
        klass = stat_class(u_num, c_num, higher_better)
        ms_rows += f'<div class="ms-row {klass}"><div class="ms-you">{u_str}</div><div class="ms-lbl">{label}</div><div class="ms-them">{c_str}</div></div>'

    st.markdown(
        '<div class="panel">'
        '<div class="panel-head">'
        '<span class="ph-lbl">03 / Comparison</span>'
        '<h3 class="ph-title">Head-to-head vs. cohort</h3>'
        '<span class="ph-hint">study average</span>'
        '</div>'
        '<div class="matchstats">'
        '<div class="matchstats-head">'
        '<span class="mh-you">Player</span>'
        '<span class="mh-lbl">Metric</span>'
        '<span class="mh-them">Cohort avg.</span>'
        '</div>'
        f'{ms_rows}'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    # --- 04 / Drivers ---
    var_meta = {
        'Training_Hours_Per_Week':  ('Training load',        training_hours, MEANS['Training_Hours_Per_Week'],     'higher_risk'),
        'Previous_Injury_Count':    ('Previous injuries',    prev_injury,    MEANS['Previous_Injury_Count'],       'higher_risk'),
        'Knee_Strength_Score':      ('Knee strength',        knee,           MEANS['Knee_Strength_Score'],         'higher_better'),
        'Hamstring_Flexibility':    ('Hamstring flexibility', hamstring,     MEANS['Hamstring_Flexibility'],       'higher_better'),
        'Reaction_Time_ms':         ('Reaction time',        reaction,       MEANS['Reaction_Time_ms'],            'lower_better'),
        'Balance_Test_Score':       ('Balance',              balance,        MEANS['Balance_Test_Score'],          'higher_better'),
        'Sprint_Speed_10m_s':       ('Sprint speed',         sprint,         MEANS['Sprint_Speed_10m_s'],          'higher_better'),
        'Agility_Score':            ('Agility',              agility,        MEANS['Agility_Score'],               'higher_better'),
        'Sleep_Hours_Per_Night':    ('Sleep hours',          sleep,          MEANS['Sleep_Hours_Per_Night'],       'higher_better'),
        'Stress_Level_Score':       ('Stress level',         stress,         MEANS['Stress_Level_Score'],          'lower_better'),
        'Nutrition_Quality_Score':  ('Nutrition quality',    nutrition,      MEANS['Nutrition_Quality_Score'],     'higher_better'),
        'Warmup_Routine_Adherence': ('Warmup adherence',     warmup,         0.61,                                 'binary'),
    }

    contribs = []
    for var, (name, user_val, avg_val, kind) in var_meta.items():
        contrib = COEFS[var] * z_inputs[var]
        contribs.append({'var': var, 'name': name, 'user': user_val, 'avg': avg_val, 'kind': kind, 'contrib': contrib})

    def format_diff(c):
        if c['kind'] == 'binary':
            return ("Warmup is consistent — a strong protective behavior."
                    if warmup == 1 else
                    "Warmup is inconsistent — one of the strongest risk amplifiers.")
        diff = c['user'] - c['avg']
        diff_pct = (diff / c['avg']) * 100 if c['avg'] != 0 else 0
        if c['kind'] == 'higher_better':
            d = "above" if diff > 0 else "below"
            lbl = "protective" if diff > 0 else "adds risk"
            return f"{abs(diff_pct):.0f}% {d} cohort mean — {lbl}."
        if c['kind'] == 'lower_better':
            d = "below" if diff < 0 else "above"
            lbl = "protective" if diff < 0 else "adds risk"
            return f"{abs(diff_pct):.0f}% {d} cohort mean — {lbl}."
        if c['kind'] == 'higher_risk':
            d = "below" if diff < 0 else "above"
            lbl = "protective" if diff < 0 else "adds risk"
            return f"Player sits {d} cohort mean — {lbl}."
        return ""

    UMBRAL = 0.15
    risks   = sorted([c for c in contribs if c['contrib'] >  UMBRAL], key=lambda x: -x['contrib'])
    protecs = sorted([c for c in contribs if c['contrib'] < -UMBRAL], key=lambda x:  x['contrib'])

    drivers_block = (
        '<div class="panel">'
        '<div class="panel-head">'
        '<span class="ph-lbl">04 / Drivers</span>'
        '<h3 class="ph-title">Key risk and protective factors</h3>'
        '<span class="ph-hint">log-odds contribution</span>'
        '</div>'
    )
    if risks:
        rt = risks[0]
        drivers_block += (
            '<div class="driver risk">'
            '<div class="driver-body">'
            '<p class="driver-tag">Top risk driver</p>'
            f'<p class="driver-title">{rt["name"]}</p>'
            f'<p class="driver-desc">{format_diff(rt)}</p>'
            '</div>'
            f'<div class="driver-stat">+{rt["contrib"]:.2f}<span class="l">log-odds</span></div>'
            '</div>'
        )
    else:
        drivers_block += (
            '<div class="driver guard">'
            '<div class="driver-body">'
            '<p class="driver-tag">Risk drivers</p>'
            '<p class="driver-title">No significant risk drivers</p>'
            '<p class="driver-desc">No variable is contributing material risk to the current profile.</p>'
            '</div>'
            '</div>'
        )
    if protecs:
        pt = protecs[0]
        drivers_block += (
            '<div class="driver guard">'
            '<div class="driver-body">'
            '<p class="driver-tag">Top protective factor</p>'
            f'<p class="driver-title">{pt["name"]}</p>'
            f'<p class="driver-desc">{format_diff(pt)}</p>'
            '</div>'
            f'<div class="driver-stat">{pt["contrib"]:.2f}<span class="l">log-odds</span></div>'
            '</div>'
        )
    drivers_block += '</div>'
    st.markdown(drivers_block, unsafe_allow_html=True)

    # --- 05 / Decomposition ---
    st.markdown(
        '<div class="panel">'
        '<div class="panel-head">'
        '<span class="ph-lbl">05 / Decomposition</span>'
        '<h3 class="ph-title">Contribution per variable</h3>'
        '<span class="ph-hint">log-odds scale</span>'
        '</div>'
        '<div class="note">'
        'Each bar shows how much a single variable adds <span class="dot-red">●</span> or removes <span class="dot-green">●</span> from the player\'s risk, controlling for all other inputs.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    contribs_chart = sorted(contribs, key=lambda x: x['contrib'])
    labels = [c['name'] for c in contribs_chart]
    values = [round(c['contrib'], 3) for c in contribs_chart]
    colors = ['#9A2C18' if v > 0 else ('#137548' if v < 0 else '#A09C8F') for v in values]

    fig_contrib = go.Figure()
    fig_contrib.add_trace(go.Bar(
        y=labels, x=values, orientation='h',
        marker=dict(color=colors, line=dict(color='#FFFFFF', width=2)),
        text=[f'{v:+.2f}' for v in values],
        textposition='outside',
        textfont=dict(family='JetBrains Mono', size=11, color='#3A3F47'),
        hovertemplate='<b>%{y}</b><br>Contribution: %{x:+.3f}<extra></extra>'
    ))
    max_abs = max(abs(v) for v in values) if values else 1
    fig_contrib.update_layout(
        height=400,
        margin=dict(t=10, b=40, l=20, r=70),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='#F0F1F3',
                   zeroline=True, zerolinecolor='#0F1115', zerolinewidth=1.2,
                   tickfont=dict(size=10, color='#6B7280', family='JetBrains Mono'),
                   range=[-max_abs * 1.25, max_abs * 1.25]),
        yaxis=dict(showgrid=False,
                   tickfont=dict(family='Inter', size=11.5, color='#0F1115')),
        font=dict(family='Inter'),
        annotations=[
            dict(x=-max_abs * 1.1, y=-0.6, xref='x', yref='paper',
                 text='Protective', showarrow=False,
                 font=dict(size=10, color='#137548', family='JetBrains Mono')),
            dict(x=max_abs * 1.1, y=-0.6, xref='x', yref='paper',
                 text='Risk-additive', showarrow=False,
                 font=dict(size=10, color='#9A2C18', family='JetBrains Mono'))
        ]
    )
    st.plotly_chart(fig_contrib, use_container_width=True, config={'displayModeBar': False})

    # --- 06 / Action plan ---
    advices = []
    if z_inputs['Stress_Level_Score'] > 0.5 and z_inputs['Sleep_Hours_Per_Night'] < -0.5:
        advices.append(("danger", "CRITICAL", "Compounded stress × sleep risk",
                        "The combination is the model's most dangerous pattern. Sleep buffers stress; with both compromised, risk multiplies non-linearly. Address sleep duration first."))
    if z_inputs['Stress_Level_Score'] > 0.5:
        advices.append(("danger", "HIGH", "Reduce psychological stress",
                        "Stress sits above the cohort baseline. Consider mindfulness, structured breathing, or sport-psychology support. Normalizing stress reduces injury probability by up to 4 pp."))
    if z_inputs['Sleep_Hours_Per_Night'] < -0.5:
        advices.append(("danger", "HIGH", "Extend sleep duration",
                        f"Player sleeps below cohort baseline ({sleep:.1f} h vs 7.4 h). Sleep is one of the strongest protective factors. Target 7.5–9 h with consistent timing."))
    if warmup == 0:
        advices.append(("danger", "HIGH", "Establish warmup adherence",
                        "Warmup is one of the strongest protective behaviors (−4.84 pp). In the cohort, non-warmers injured at 66%; consistent warmers at 39%. Highest marginal return of any intervention."))
    if z_inputs['Balance_Test_Score'] < -0.5:
        advices.append(("warning", "MODIFIABLE", "Improve postural control",
                        "Balance is the strongest protective predictor. Add proprioception 3×/week: BOSU, unstable surfaces, single-leg deadlifts, eyes-closed holds."))
    if z_inputs['Reaction_Time_ms'] > 0.5:
        advices.append(("warning", "MODIFIABLE", "Sharpen reaction time",
                        "Reaction time is the strongest risk-additive predictor. Train visual and auditory reaction: reaction lights, color-coded cone drills, signal-triggered cuts."))
    if z_inputs['Knee_Strength_Score'] < -0.5:
        advices.append(("warning", "MODIFIABLE", "Build knee strength",
                        "Front squats, Romanian deadlift, unilateral press, single-leg squats. 2–3 sessions per week."))
    if z_inputs['Hamstring_Flexibility'] < -0.5:
        advices.append(("warning", "MODIFIABLE", "Improve hamstring flexibility",
                        "Daily mobility (15 min) + Nordic Curls 2×/week. Reduces probability by 3.5 pp."))
    if z_inputs['Nutrition_Quality_Score'] < -0.5:
        advices.append(("warning", "MODIFIABLE", "Upgrade nutritional quality",
                        "Aim for 5+ servings of produce, protein 1.6–2.2 g/kg, hydration, fewer processed foods. Estimated reduction: 4 pp."))
    if z_inputs['Sprint_Speed_10m_s'] < -0.5:
        advices.append(("warning", "MODIFIABLE", "Develop sprint output",
                        "Plyometrics, progressive short sprints (5×10 m, 5×20 m), explosive lower-body strength."))
    if z_inputs['Agility_Score'] < -0.5:
        advices.append(("warning", "MODIFIABLE", "Sharpen agility",
                        "Change-of-direction drills 2×/week: T-test, zigzag cones, agility ladder, accel/decel patterns."))
    if prev_injury >= 3:
        advices.append(("danger", "HIGH", "Heavy clinical history — intensify monitoring",
                        f"Player reports {prev_injury} previous injuries. Each one multiplies the odds by 2.53. Build a targeted protocol for previously affected areas."))
    if not advices:
        advices.append(("success", "OPTIMAL", "Profile is well-conditioned",
                        "All indicators sit at or above the cohort baseline. Maintain current training, recovery, and nutrition routines."))

    actions_block = (
        '<div class="panel">'
        '<div class="panel-head">'
        '<span class="ph-lbl">06 / Action plan</span>'
        '<h3 class="ph-title">Prioritized interventions</h3>'
        '<span class="ph-hint">ranked by impact</span>'
        '</div>'
    )
    for tipo, priority, titulo, texto in advices:
        actions_block += (
            f'<div class="advice {tipo}">'
            '<div class="advice-head">'
            f'<span class="advice-tag">{priority}</span>'
            f'<h4>{titulo}</h4>'
            '</div>'
            f'<p>{texto}</p>'
            '</div>'
        )
    actions_block += '</div>'
    st.markdown(actions_block, unsafe_allow_html=True)

# ============ FOOTER ============
st.markdown(
    '<div class="footer">'
    '<p class="footer-title">Injury Risk Predictor — Sports Medicine Module</p>'
    '<p class="footer-meta">LOGISTIC REGRESSION · AUC 0.997 · NAGELKERKE R² 0.913 · 10-FOLD CV</p>'
    '<p class="footer-sub">Final Project — Linear Statistical Models (IIND-4100) — Universidad de los Andes — 2026</p>'
    '</div>',
    unsafe_allow_html=True
)
