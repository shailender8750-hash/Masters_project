import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="CardioPredict", page_icon="❤️", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
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
            'bar': {'color': '#111827'},  # needle/bar color
            'steps': [
                {'range': [0, 33],  'color': '#22c55e'},  # Low
                {'range': [33, 66], 'color': '#f59e0b'},  # Moderate
                {'range': [66, 100],'color': '#ef4444'},  # High
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
    Colors: green, yellow, orange, red
    """
    # LDL (mg/dL): <100 green, 100-129 yellow, 130-159 orange, >=160 red
    if metric == "LDL (mg/dL)":
        if value < 100: return "#22c55e"
        if value < 130: return "#f59e0b"
        if value < 160: return "#f97316"
        return "#ef4444"

    # HDL (mg/dL) - higher is better: <40 red, 40-59 yellow, >=60 green
    if metric == "HDL (mg/dL)":
        if value < 40: return "#ef4444"
        if value < 60: return "#f59e0b"
        return "#22c55e"

    # Systolic BP (mmHg): <120 green, 120-139 yellow, 140-159 orange, >=160 red
    if metric == "Systolic BP (mmHg)":
        if value < 120: return "#22c55e"
        if value < 140: return "#f59e0b"
        if value < 160: return "#f97316"
        return "#ef4444"

    # BMI: <25 green, 25-29.9 yellow, 30-34.9 orange, >=35 red
    if metric == "BMI":
        if value < 25: return "#22c55e"
        if value < 30: return "#f59e0b"
        if value < 35: return "#f97316"
        return "#ef4444"

    return "#64748B"  # default gray

def risk_indicator_bars(values_dict):
    """
    values_dict: dict like
    {
      "LDL (mg/dL)": 160,
      "HDL (mg/dL)": 50,
      "Systolic BP (mmHg)": 150,
      "BMI": 29.0
    }
    """
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

# -----------------------------
# Sidebar (interactive inputs)
# -----------------------------
st.sidebar.header("Patient Inputs")
age = st.sidebar.number_input("Age", 18, 100, 54)
sys = st.sidebar.number_input("Systolic BP (mmHg)", 90, 220, 150)
dia = st.sidebar.number_input("Diastolic BP (mmHg)", 60, 140, 95)
ldl = st.sidebar.number_input("LDL (mg/dL)", 40, 300, 160)
hdl = st.sidebar.number_input("HDL (mg/dL)", 20, 120, 50)
bmi = st.sidebar.number_input("BMI", 14.0, 50.0, 29.0, step=0.1)
smoker = st.sidebar.selectbox("Smoking", ["No", "Yes"])

# -----------------------------
# Simple illustrative risk score (Option A)
# -----------------------------
score = (
    0.28 * (sys/180*100) +
    0.22 * (ldl/190*100) +
    0.16 * (bmi/35*100) +
    0.14 * ((1 - min(hdl/70, 1)) * 100) +
    0.10 * (70 if smoker == "No" else 100) +
    0.10 * (age/80*100)
)
score = max(0, min(100, score))
cat_label, cat_color = risk_category(score)

# -----------------------------
# Header
# -----------------------------
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

# -----------------------------
# Top row: Gauge | Key Indicators
# -----------------------------
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
        <table class="keytbl" style="width:100%;font-size:16px;">
          <tr><td><b>Blood Pressure</b></td><td style="text-align:right;"><b>{int(sys)}/{int(dia)} mmHg</b></td></tr>
          <tr><td><b>LDL / HDL</b></td><td style="text-align:right;"><b>{int(ldl)} / {int(hdl)} mg/dL</b></td></tr>
          <tr><td><b>BMI</b></td><td style="text-align:right;"><b>{bmi:.1f}</b></td></tr>
          <tr><td><b>Age</b></td><td style="text-align:right;"><b>{age}</b></td></tr>
          <tr><td><b>Smoking</b></td><td style="text-align:right;"><b>{smoker}</b></td></tr>
        </table>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# Middle: Progress (Line) | Risk Table
# -----------------------------
m1, m2 = st.columns([1.2, 1.1])
with m1:
    st.subheader("Progress Tracker")
    months = ["Jan", "Mar", "May", "Jul", "Sep"]
    # Dummy LDL trajectory trending down with light noise
    trend = np.clip(np.linspace(ldl + 15, ldl - 15, len(months)) + np.random.randn(len(months)) * 2, 60, 220)
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

# -----------------------------
# Bottom: Risk Indicators (Bar) | Recommendations
# -----------------------------
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
    st.subheader("Recommendations")
    st.markdown("- Eat a heart-healthy diet (more fiber, less saturated fat)")
    st.markdown("- Increase daily activity (≥150 min/week moderate exercise)")
    st.markdown("- Target LDL reduction by ~20 points")
    st.markdown("- Discuss BP control plan with a clinician")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("This tool is for education and awareness only. It does not provide medical diagnosis or treatment advice.")
