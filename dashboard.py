# app.py
import io
import json
import os
import base64
from glob import glob
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ------------------------
# App config
# ------------------------
st.set_page_config(
    page_title="CardioPredict – Cardiovascular Risk Preview",
    page_icon="❤️",
    layout="centered",
)

# ------------------------
# Utilities
# ------------------------
FEATURES = [
    "age", "sex_female", "sex_male",
    "sbp", "dbp", "bmi",
    "total_chol", "hdl", "ldl", "trig",
    "fasting_glucose", "hba1c",
    "smoker", "diabetes", "bp_meds", "family_history"
]

def load_model():
    p = "models/cvd_model.pkl"
    if os.path.exists(p):
        import pickle
        with open(p, "rb") as f:
            return pickle.load(f)
    return None

MODEL = load_model()

def demo_risk_score(row: pd.Series) -> tuple[float, dict]:
    w = {
        "age": 0.04,
        "sbp": 0.02,
        "bmi": 0.03,
        "total_chol": 0.015,
        "hdl": -0.03,
        "ldl": 0.012,
        "fasting_glucose": 0.01,
        "smoker": 0.5,
        "diabetes": 0.6,
        "bp_meds": 0.25,
        "family_history": 0.25,
    }
    intercept = -8.0
    score = intercept
    contribs = {}
    for k, wk in w.items():
        val = float(row.get(k, 0) or 0)
        s = wk * val
        score += s
        contribs[k] = s
    prob = 1 / (1 + np.exp(-score))
    return float(prob * 100), contribs

def explain_top_factors(contribs: dict, top_k: int = 5):
    items = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
    nice = {
        "sbp": "Systolic blood pressure",
        "dbp": "Diastolic blood pressure",
        "bmi": "Body mass index",
        "total_chol": "Total cholesterol",
        "hdl": "HDL cholesterol",
        "ldl": "LDL cholesterol",
        "trig": "Triglycerides",
        "fasting_glucose": "Fasting glucose",
        "hba1c": "HbA1c",
        "smoker": "Smoking status",
        "diabetes": "Diabetes",
        "bp_meds": "On blood pressure medication",
        "family_history": "Family history of early heart disease",
        "age": "Age",
    }
    return [(nice.get(k, k), v) for k, v in items]

def parse_structured(file) -> dict:
    name = file.name.lower()
    try:
        if name.endswith(".json"):
            data = json.loads(file.read().decode("utf-8"))
            return {k: data.get(k) for k in FEATURES if k in data}
        if name.endswith(".csv"):
            df = pd.read_csv(file)
            if set(["name", "value"]).issubset(df.columns):
                d = {r["name"]: r["value"] for _, r in df.iterrows()}
                return {k: d.get(k) for k in FEATURES if k in d}
    except Exception:
        pass
    return {}

def coerce_float(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None

def compute_bmi(height_cm, weight_kg):
    if height_cm and weight_kg and height_cm > 0:
        return round(weight_kg / (height_cm/100)**2, 1)
    return None

# ---------- Plotly visual helpers you provided (wired in) ----------
def risk_category(score: float):
    if score < 33:
        return "Low", "#22c55e"    # green
    elif score < 66:
        return "Moderate", "#f59e0b"  # amber
    return "High", "#ef4444"       # red

def indicator_gauge(score: float):
    """
    Plotly 'Indicator' gauge with 0-100 scale and three color zones:
    0-33 Low (green), 33-66 Moderate (yellow), 66-100 High (red)
    """
    cat, color = risk_category(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(score),
        number={'suffix': " /100", 'font': {'size': 22}},
        title={'text': f"<b>Cardiovascular Risk</b><br><span style='color:{color}'>{cat}</span>", 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': '#111827'},
            'steps': [
                {'range': [0, 33],  'color': '#22c55e'},
                {'range': [33, 66], 'color': '#f59e0b'},
                {'range': [66, 100],'color': '#ef4444'},
            ],
            'threshold': {
                'line': {'color': '#111827', 'width': 3},
                'thickness': 0.8,
                'value': float(score)
            }
        }
    ))
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor='black')
    return fig

def progress_line(months, values, y_title="LDL (mg/dL)"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months, y=values, mode="lines+markers",
        line=dict(width=3), marker=dict(size=8)
    ))
    fig.update_layout(
        yaxis_title=y_title,
        xaxis_title=None,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="black", plot_bgcolor="black",
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#e5e7eb")
    )
    return fig

def clinical_color(metric: str, value: float):
    """
    Return a color based on common clinical cut-points.
    """
    if metric == "LDL (mg/dL)":
        if value < 100: return "#22c55e"
        if value < 130: return "#f59e0b"
        if value < 160: return "#f97316"
        return "#ef4444"

    if metric == "HDL (mg/dL)":
        if value < 40: return "#ef4444"
        if value < 60: return "#f59e0b"
        return "#22c55e"

    if metric == "Systolic BP (mmHg)":
        if value < 120: return "#22c55e"
        if value < 140: return "#f59e0b"
        if value < 160: return "#f97316"
        return "#ef4444"

    if metric == "BMI":
        if value < 25: return "#22c55e"
        if value < 30: return "#f59e0b"
        if value < 35: return "#f97316"
        return "#ef4444"

    return "#64748B"  # default gray

def risk_indicator_bars(values_dict):
    names = list(values_dict.keys())
    vals = [values_dict[k] for k in names]
    colors = [clinical_color(n, v) for n, v in zip(names, vals)]

    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker=dict(color=colors, line=dict(color="black", width=1))
    ))
    fig.update_layout(
        xaxis_title=None, yaxis_title=None,
        xaxis=dict(showgrid=True, gridcolor="#e5e7eb"),
        yaxis=dict(showgrid=False),
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="black", plot_bgcolor="black"
    )
    return fig

def chip(text, color):
    st.markdown(
        f"""
        <div style="display:inline-block;padding:4px 10px;border-radius:999px;
        background:{color};color:white;font-size:12px;margin-left:8px;">{text}</div>
        """, unsafe_allow_html=True
    )

# ---------- Global background helpers ----------
def find_background_image() -> str | None:
    preferred = ["background.jpg", "hero.jpg", "background.png", "hero.png"]
    for p in preferred:
        if os.path.exists(p):
            return p
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        matches = glob(ext)
        if matches:
            return matches[0]
    return None

def apply_global_background(img_path: str):
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    mime = "image/jpeg" if img_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:{mime};base64,{b64}") no-repeat center center fixed;
            background-size: cover;
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.35);
            z-index: 0;
        }}
        .block-container {{
            position: relative;
            z-index: 1;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ------------------------
# Session init
# ------------------------
if "consented" not in st.session_state:
    st.session_state.consented = False
if "inputs" not in st.session_state:
    st.session_state.inputs = {k: None for k in FEATURES}

# ------------------------
# Apply background to ALL pages
# ------------------------
_bg = find_background_image()
if _bg:
    apply_global_background(_bg)
else:
    st.warning("Place a background image (e.g., 'background.jpg') next to app.py for a full-page background.")

# ------------------------
# Sidebar navigation
# ------------------------
page = st.sidebar.radio("Navigate", ["Home", "Upload report", "Manual entry", "Risk results", "Privacy & about"])

# ------------------------
# Pages
# ------------------------
if page == "Home":
    st.markdown(
        """
        <div style="
            text-align:center;
            margin: 1rem auto 0.5rem auto;
            background: rgba(0,0,0,0.45);
            color: white;
            padding: 12px 20px;
            border-radius: 12px;
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: 0.5px;
            display: inline-block;">
            CardioPredict
        </div>
        """,
        unsafe_allow_html=True
    )
    st.subheader("A simple, educational preview of cardiovascular risk")
    st.write(
        "Upload your lab report or enter your numbers to see a model-based risk estimate "
        "and what factors may be contributing. **This is not medical advice.**"
    )
    st.divider()
    st.checkbox("I understand this is for education only (not a medical diagnosis).", key="consented")

    with st.expander("What data do we use?"):
        st.markdown(
            "- Age, sex\n"
            "- Height, weight (to calculate BMI)\n"
            "- Blood pressure (systolic/diastolic)\n"
            "- Cholesterol panel (total, HDL, LDL, triglycerides)\n"
            "- Fasting glucose or HbA1c\n"
            "- Lifestyle and history (smoking, diabetes, BP meds, family history)\n"
        )

elif page == "Upload report":
    st.title("Upload your report")
    if not st.session_state.consented:
        st.warning("Please confirm consent on the Home page first.")
    file = st.file_uploader(
        "Upload a medical report (JSON/CSV preferred; PDF/images optional)",
        type=["json","csv","pdf","png","jpg","jpeg"]
    )
    if file is not None:
        if file.name.lower().endswith((".json",".csv")):
            parsed = parse_structured(file)
            if parsed:
                st.success("Structured data found. Pre-filling your values below.")
                st.session_state.inputs.update({k: coerce_float(v) for k, v in parsed.items()})
            else:
                st.info("Could not parse values from this file. You can still enter them manually.")
        else:
            st.info("PDF/images are stored in memory only during this session. Parsing is optional and disabled in this demo.")
    st.caption("We do not store your data by default. Close the tab to clear the session, or use the button on the Privacy page.")

elif page == "Manual entry":
    st.title("Enter or review your numbers")
    if not st.session_state.consented:
        st.warning("Please confirm consent on the Home page first.")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=int(st.session_state.inputs.get("age") or 50))
        sex = st.selectbox("Sex (for risk equations)", ["Female","Male"])
        height = st.number_input("Height (cm)", min_value=120, max_value=230, value=170)
    with col2:
        weight = st.number_input("Weight (kg)", min_value=35.0, max_value=250.0, value=75.0, step=0.1)
        sbp = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=240, value=130)
        dbp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=140, value=80)

    bmi = compute_bmi(height, weight)
    st.write(f"**BMI:** {bmi if bmi else '—'}")

    col3, col4 = st.columns(2)
    with col3:
        total_chol = st.number_input("Total cholesterol (mg/dL)", min_value=80, max_value=400, value=200)
        hdl = st.number_input("HDL (mg/dL)", min_value=15, max_value=120, value=50)
        trig = st.number_input("Triglycerides (mg/dL)", min_value=40, max_value=800, value=150)
    with col4:
        ldl = st.number_input("LDL (mg/dL)", min_value=40, max_value=300, value=120)
        fasting_glucose = st.number_input("Fasting glucose (mg/dL)", min_value=60, max_value=300, value=95)
        hba1c = st.number_input("HbA1c (%)", min_value=4.0, max_value=14.0, value=5.4, step=0.1)

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        smoker = st.selectbox("Do you smoke?", ["No","Yes"])
    with col6:
        diabetes = st.selectbox("Diabetes diagnosed?", ["No","Yes"])
    with col7:
        bp_meds = st.selectbox("On BP medication?", ["No","Yes"])
    with col8:
        fam = st.selectbox("Family history (early CVD <55M/<65F)?", ["No","Yes"])

    st.session_state.inputs.update({
        "age": age,
        "sex_female": 1 if sex == "Female" else 0,
        "sex_male": 1 if sex == "Male" else 0,
        "sbp": sbp, "dbp": dbp, "bmi": bmi,
        "total_chol": total_chol, "hdl": hdl, "ldl": ldl, "trig": trig,
        "fasting_glucose": fasting_glucose, "hba1c": hba1c,
        "smoker": 1 if smoker == "Yes" else 0,
        "diabetes": 1 if diabetes == "Yes" else 0,
        "bp_meds": 1 if bp_meds == "Yes" else 0,
        "family_history": 1 if fam == "Yes" else 0,
    })
    st.success("Saved! Go to ‘Risk results’ to see your estimate.")

elif page == "Risk results":
    st.title("Your risk preview")
    if not st.session_state.consented:
        st.warning("Please confirm consent on the Home page first.")

    row = pd.Series(st.session_state.inputs)
    required = ["age","sbp","bmi","total_chol","hdl","ldl","dbp"]
    missing = [k for k in required if row.get(k) in (None, "", np.nan)]
    if missing:
        st.info(f"Please provide these before estimating risk: {', '.join(missing)}")
    else:
        # 1) Compute model/heuristic risk %
        if MODEL is not None:
            try:
                X = pd.DataFrame([row.reindex(FEATURES).fillna(0.0).astype(float)])
                proba = MODEL.predict_proba(X)[:,1][0] * 100.0
                risk_pct = float(proba)
                contribs = {"model_prediction": risk_pct}
            except Exception:
                risk_pct, contribs = demo_risk_score(row)
        else:
            risk_pct, contribs = demo_risk_score(row)

        # 2) Map variables for your visuals
        sys = float(row["sbp"])
        dia = float(row["dbp"])
        ldl = float(row["ldl"])
        hdl = float(row["hdl"])
        bmi = float(row["bmi"] or 0)
        age = int(row["age"])
        smoker_str = "Yes" if int(row["smoker"] or 0) == 1 else "No"

        # 3) Your composite score (0-100) for the gauge
        score = (
            0.28 * (sys/180*100) +
            0.22 * (ldl/190*100) +
            0.16 * (bmi/35*100) +
            0.14 * ((1 - min(hdl/70, 1)) * 100) +
            0.10 * (70 if smoker_str == "No" else 100) +
            0.10 * (age/80*100)
        )
        score = max(0, min(100, float(score)))
        cat_label, cat_color = risk_category(score)

        # 4) Header row (title + category)
        left_col, right_col = st.columns([1, 2])
        with left_col:
            st.markdown("### 🫀 **CardioPredict**")
        with right_col:
            st.markdown(
                f"<div style='text-align:right;'>Overall Risk: "
                f"<span style='font-weight:700;color:{cat_color};'>{cat_label}</span></div>",
                unsafe_allow_html=True
            )
        st.markdown("---")

        # 5) Top row: Gauge | Key Indicators
        c1, c2 = st.columns([1.1, 1.2])
        with c1:
            st.subheader("Cardiovascular Risk")
            st.plotly_chart(indicator_gauge(score), use_container_width=True)

        with c2:
            st.subheader("Key Health Indicators")
            st.markdown(
                """
                <style>
                .keytbl td {padding:10px 8px;border-bottom:1px solid #e5e7eb;}
                .keytbl tr:last-child td {border-bottom:none;}
                </style>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <table class="keytbl" style="width:100%;font-size:16px;background:rgba(0,0,0,0.35);border-radius:12px;">
                  <tr><td><b>Blood Pressure</b></td><td style="text-align:right;"><b>{int(sys)}/{int(dia)} mmHg</b></td></tr>
                  <tr><td><b>LDL / HDL</b></td><td style="text-align:right;"><b>{int(ldl)} / {int(hdl)} mg/dL</b></td></tr>
                  <tr><td><b>BMI</b></td><td style="text-align:right;"><b>{bmi:.1f}</b></td></tr>
                  <tr><td><b>Age</b></td><td style="text-align:right;"><b>{age}</b></td></tr>
                  <tr><td><b>Smoking</b></td><td style="text-align:right;"><b>{smoker_str}</b></td></tr>
                </table>
                """,
                unsafe_allow_html=True
            )

        # 6) Middle: Progress line | Reference table
        m1, m2 = st.columns([1.2, 1.1])
        with m1:
            st.subheader("Progress Tracker")
            months = ["Jan", "Mar", "May", "Jul", "Sep"]
            trend = np.clip(
                np.linspace(ldl + 15, ldl - 15, len(months)) + np.random.randn(len(months)) * 2,
                60, 220
            )
            st.plotly_chart(progress_line(months, trend, y_title="LDL (mg/dL)"), use_container_width=True)

        with m2:
            st.subheader("Risk Categories (Reference)")
            df = pd.DataFrame({
                "Risk Level": ["Low Risk", "Moderate Risk", "High Risk"],
                "Description": [
                    "Minimal likelihood of CVD",
                    "Elevated but manageable risk",
                    "Significant risk of heart disease"
                ],
                "Recommended Action": [
                    "Maintain healthy lifestyle",
                    "Lifestyle changes + regular monitoring",
                    "Immediate intervention, consult a doctor"
                ]
            })
            st.dataframe(df, use_container_width=True, hide_index=True)

        # 7) Bottom: Indicators bar | Guidance + export
        b1, b2 = st.columns([1.1, 1.0])
        with b1:
            st.subheader("Risk Indicators (Current Values)")
            indicators_dict = {
                "LDL (mg/dL)": float(ldl),
                "HDL (mg/dL)": float(hdl),
                "Systolic BP (mmHg)": float(sys),
                "BMI": float(bmi),
            }
            st.plotly_chart(risk_indicator_bars(indicators_dict), use_container_width=True)

        with b2:
            st.subheader("General educational guidance")
            st.markdown(
                "- Discuss these results with your healthcare professional.\n"
                "- Consider lifestyle supports: diet quality, physical activity, sleep.\n"
                "- If BP, lipids, or glucose are high, clinicians may adjust medications.\n"
            )
            st.download_button(
                "Download summary (JSON)",
                data=json.dumps(
                    {"timestamp": datetime.utcnow().isoformat()+"Z",
                     "inputs": st.session_state.inputs,
                     "risk_pct": risk_pct,
                     "composite_score": score,
                     "category": cat_label},
                    indent=2
                ).encode("utf-8"),
                file_name="cardio_predict_summary.json",
                mime="application/json"
            )

elif page == "Privacy & about":
    st.title("Privacy & about")
    st.markdown(
        "We process your data **in your browser session** by default. "
        "Close the tab to clear it, or click the button below."
    )
    if st.button("Delete all session data now"):
        st.session_state.clear()
        st.success("Session cleared.")

    st.divider()
    st.caption("This tool is for education only and is **not** a medical diagnosis.")
