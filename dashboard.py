import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib
from pathlib import Path
from typing import Dict, Tuple
import base64

import plotly.graph_objects as go
import plotly.express as px

# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(
    page_title="CardioPredict",
    page_icon="❤️",
    layout="wide"
)

# =========================
# BACKGROUND & GLOBAL STYLES
# =========================
def add_bg_image(image_path: Path):
    if not image_path.exists():
        return
    with open(image_path, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}
        /* Card styling */
        .card {{
            background-color: rgba(255, 255, 255, 0.9);
            padding: 1.5rem;
            border-radius: 1rem;
            box-shadow: 0 8px 18px rgba(0, 0, 0, 0.12);
            margin-bottom: 1.5rem;
        }}
        .card-title {{
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }}
        .card-subtitle {{
            font-size: 0.95rem;
            color: #555;
            margin-bottom: 1rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

base_path = Path(__file__).resolve().parent
add_bg_image(base_path / "background.jpg")

# =========================
# LOAD ARTIFACTS
# =========================
@st.cache_resource
def load_artifacts(
    model_filename: str = "log_reg_model.pkl",
    scaler_filename: str = "scaler.pkl",
    feature_cols_filename: str = "feature_columns.pkl",
):
    """
    Load:
    - Trained logistic regression model
    - Fitted scaler
    - Feature column list (order the model expects)
    from the same folder as this script.
    """
    def _load_pkl(fname: str):
        path = base_path / fname
        if not path.exists():
            st.error(f"File not found: {path}")
            return None
        # Try joblib first, then pickle
        try:
            return joblib.load(path)
        except Exception:
            with open(path, "rb") as f:
                return pickle.load(f)

    model = _load_pkl(model_filename)
    scaler = _load_pkl(scaler_filename)
    feature_cols = _load_pkl(feature_cols_filename)

    if model is None or scaler is None or feature_cols is None:
        st.error("One or more artifacts (model, scaler, feature_columns) could not be loaded.")
        return None, None, None

    if not isinstance(feature_cols, (list, tuple, np.ndarray)):
        st.error("feature_columns.pkl did not contain a list-like object.")
        return None, None, None

    feature_cols = list(feature_cols)
    return model, scaler, feature_cols


model, scaler, feature_columns = load_artifacts()

# =========================
# CLINICAL STATUS HELPERS
# =========================
def assess_bmi(bmi: float) -> str:
    if np.isnan(bmi):
        return "Unknown"
    if bmi < 18.5:
        return "Borderline (Underweight)"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Borderline (Overweight)"
    else:
        return "Risky (Obese)"


def assess_bp(bp: float) -> str:
    if np.isnan(bp):
        return "Unknown"
    if bp < 120:
        return "Normal"
    elif bp < 130:
        return "Borderline (Elevated)"
    else:
        return "Risky (Hypertension)"


def assess_heart_rate(hr: float) -> str:
    if np.isnan(hr):
        return "Unknown"
    if 60 <= hr <= 100:
        return "Normal"
    elif (50 <= hr < 60) or (100 < hr <= 110):
        return "Borderline"
    else:
        return "Risky"


def assess_chol_total(val: float) -> str:
    if np.isnan(val):
        return "Unknown"
    if val < 200:
        return "Normal"
    elif val < 240:
        return "Borderline"
    else:
        return "Risky"


def assess_chol_ldl(val: float) -> str:
    if np.isnan(val):
        return "Unknown"
    if val < 100:
        return "Normal"
    elif val < 130:
        return "Borderline (Near optimal)"
    elif val < 160:
        return "Borderline (High)"
    else:
        return "Risky"


def assess_chol_hdl(val: float) -> str:
    if np.isnan(val):
        return "Unknown"
    if val < 40:
        return "Risky (Low HDL)"
    elif val < 60:
        return "Normal"
    else:
        return "Borderline (High / Protective)"


def assess_trig(val: float) -> str:
    if np.isnan(val):
        return "Unknown"
    if val < 150:
        return "Normal"
    elif val < 200:
        return "Borderline"
    else:
        return "Risky"


def map_status_to_color(status: str) -> str:
    if status.startswith("Normal"):
        return "green"
    if status.startswith("Borderline"):
        return "orange"
    if status.startswith("Risky"):
        return "red"
    return "gray"


def build_risk_table(inputs: Dict[str, float]) -> pd.DataFrame:
    """
    Build a table with Normal / Borderline / Risky labels per feature.
    """
    rows = []

    age = inputs.get("AGE", np.nan)
    if np.isnan(age):
        age_status = "Unknown"
    elif age < 40:
        age_status = "Normal"
    elif age < 60:
        age_status = "Borderline"
    else:
        age_status = "Risky"
    rows.append({"Feature": "Age (years)", "Value": age, "Status": age_status})

    bp = inputs.get("bp", np.nan)
    rows.append({"Feature": "Blood Pressure (systolic, mmHg)", "Value": bp, "Status": assess_bp(bp)})

    hr = inputs.get("heart_rate", np.nan)
    rows.append({"Feature": "Heart Rate (bpm)", "Value": hr, "Status": assess_heart_rate(hr)})

    bmi = inputs.get("bmi", np.nan)
    rows.append({"Feature": "BMI (kg/m²)", "Value": bmi, "Status": assess_bmi(bmi)})

    chol_total = inputs.get("chol_total", np.nan)
    rows.append({"Feature": "Total Cholesterol (mg/dL)", "Value": chol_total, "Status": assess_chol_total(chol_total)})

    chol_ldl = inputs.get("chol_ldl", np.nan)
    rows.append({"Feature": "LDL Cholesterol (mg/dL)", "Value": chol_ldl, "Status": assess_chol_ldl(chol_ldl)})

    chol_hdl = inputs.get("chol_hdl", np.nan)
    rows.append({"Feature": "HDL Cholesterol (mg/dL)", "Value": chol_hdl, "Status": assess_chol_hdl(chol_hdl)})

    trig = inputs.get("trig", np.nan)
    rows.append({"Feature": "Triglycerides (mg/dL)", "Value": trig, "Status": assess_trig(trig)})

    return pd.DataFrame(rows)

# =========================
# GAUGE + BAR PLOTS
# =========================
def risk_gauge(probability: float) -> go.Figure:
    """
    Animated gauge showing cardiovascular risk percentage.
    """
    value = float(probability) * 100
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': 'red' if value >= 50 else 'green'},
                'steps': [
                    {'range': [0, 20], 'color': 'rgba(0,128,0,0.2)'},
                    {'range': [20, 50], 'color': 'rgba(255,165,0,0.2)'},
                    {'range': [50, 100], 'color': 'rgba(255,0,0,0.2)'}
                ],
            },
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=30, b=20),
        transition={'duration': 700, 'easing': 'cubic-in-out'},
    )
    return fig


def feature_status_bar(risk_table: pd.DataFrame) -> go.Figure:
    df = risk_table.copy()
    df["PlotValue"] = df["Value"].fillna(0)
    df["Color"] = df["Status"].apply(map_status_to_color)

    fig = px.bar(
        df,
        x="Feature",
        y="PlotValue",
        color="Status",
        color_discrete_map={
            "Normal": "green",
            "Borderline": "orange",
            "Risky": "red",
            "Unknown": "gray",
        },
        title="Lab Features and Their Risk Status",
    )
    fig.update_layout(
        xaxis_title="Feature",
        yaxis_title="Value",
        xaxis_tickangle=-45,
        height=450,
        margin=dict(l=20, r=20, t=60, b=120),
    )
    return fig


def estimate_rule_based_prob(risk_table: pd.DataFrame) -> float:
    """
    Turn the risk_table (Normal / Borderline / Risky per feature)
    into a rough risk probability in [0, 1].
    """
    if risk_table is None or risk_table.empty:
        return 0.0

    risky_count = (risk_table["Status"].str.startswith("Risky")).sum()
    borderline_count = (risk_table["Status"].str.startswith("Borderline")).sum()

    # Each risky feature ~15%, borderline ~7%, cap at 90%
    rule_prob = risky_count * 0.15 + borderline_count * 0.07
    rule_prob = min(max(rule_prob, 0.0), 0.9)
    return float(rule_prob)

# =========================
# PREPROCESSING FOR MODEL
# =========================
def build_raw_input_row(patient_inputs: Dict[str, float]) -> pd.DataFrame:
    """
    Convert user-facing names into raw feature names used in training.
    Update this if your training used different names.
    """
    raw = {
        "AGE": patient_inputs.get("AGE"),
        "GENDER": patient_inputs.get("GENDER"),
        "bp": patient_inputs.get("bp"),
        "heart_rate": patient_inputs.get("heart_rate"),
        "bmi": patient_inputs.get("bmi"),
        "chol_total": patient_inputs.get("chol_total"),
        "chol_hdl": patient_inputs.get("chol_hdl"),
        "chol_ldl": patient_inputs.get("chol_ldl"),
        "trig": patient_inputs.get("trig"),
    }
    df = pd.DataFrame([raw])
    return df


def preprocess_for_model(patient_inputs: Dict[str, float]) -> np.ndarray:
    """
    1. Map UI inputs to raw training feature names.
    2. Encode GENDER with get_dummies(drop_first=True) like in training.
    3. Add any missing columns from feature_columns with 0.
    4. Reorder columns to match feature_columns.
    5. Scale using the trained scaler.
    """
    if model is None or scaler is None or feature_columns is None:
        raise RuntimeError("Artifacts not properly loaded.")

    df = build_raw_input_row(patient_inputs)

    if "GENDER" in df.columns:
        df["GENDER"] = df["GENDER"].fillna("Unknown")
        df = pd.get_dummies(df, columns=["GENDER"], drop_first=True)

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0.0

    df = df[feature_columns]
    X_scaled = scaler.transform(df.values)
    return X_scaled


def predict_risk(patient_inputs: Dict[str, float]) -> Tuple[float, int]:
    """
    Run the full pipeline for a single patient:
    - raw -> aligned with feature_columns -> scaled -> predict_proba
    """
    if model is None or scaler is None or feature_columns is None:
        return np.nan, -1

    X_scaled = preprocess_for_model(patient_inputs)

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_scaled)[:, 1][0]
    else:
        if hasattr(model, "decision_function"):
            raw = model.decision_function(X_scaled)[0]
            prob = 1 / (1 + np.exp(-raw))
        else:
            pred = model.predict(X_scaled)[0]
            prob = float(pred)

    label = int(prob >= 0.5)
    return prob, label

# =========================
# RECOMMENDATIONS
# =========================
def generate_recommendations(risk_table: pd.DataFrame) -> list:
    """
    Generate simple, educational recommendations based on each feature's status.
    Uses Status strings ("Normal", "Borderline", "Risky").
    """
    recs = []

    def get_status(feature_label: str) -> str:
        row = risk_table[risk_table["Feature"] == feature_label]
        return row["Status"].iloc(0) if hasattr(row["Status"], "iloc") else row["Status"].iloc[0] if not row.empty else ""

    def status_starts(feature_label: str, prefix: str) -> bool:
        row = risk_table[risk_table["Feature"] == feature_label]
        if row.empty:
            return False
        return row["Status"].iloc[0].startswith(prefix)

    # Age
    if status_starts("Age (years)", "Risky") or status_starts("Age (years)", "Borderline"):
        recs.append(
            "Because of your age group, it may be especially important to keep regular check-ups with your healthcare provider and discuss cardiovascular risk screening."
        )

    # Blood pressure
    if status_starts("Blood Pressure (systolic, mmHg)", "Risky"):
        recs.append(
            "Your blood pressure appears high. Reducing salt intake, staying physically active, managing stress, and discussing blood pressure control with your doctor may be helpful."
        )
    elif status_starts("Blood Pressure (systolic, mmHg)", "Borderline"):
        recs.append(
            "Your blood pressure is borderline elevated. Monitoring it regularly and maintaining a heart-healthy lifestyle can help prevent further increases."
        )

    # Heart rate
    if status_starts("Heart Rate (bpm)", "Risky"):
        recs.append(
            "Your heart rate is outside the typical resting range. You may want to discuss this with a clinician, especially if you have symptoms such as palpitations, dizziness, or shortness of breath."
        )
    elif status_starts("Heart Rate (bpm)", "Borderline"):
        recs.append(
            "Your heart rate is slightly outside the typical resting range. Tracking it over time and mentioning it during your next medical visit could be useful."
        )

    # BMI
    if status_starts("BMI (kg/m²)", "Risky"):
        recs.append(
            "Your BMI is in a higher-risk range. A balanced diet, regular physical activity, and weight management strategies—discussed with a healthcare professional—may help lower cardiovascular risk."
        )
    elif status_starts("BMI (kg/m²)", "Borderline"):
        recs.append(
            "Your BMI is borderline. Paying attention to nutrition and activity levels now can help prevent further weight-related risk in the future."
        )

    # Total Cholesterol
    if status_starts("Total Cholesterol (mg/dL)", "Risky"):
        recs.append(
            "Your total cholesterol is high. Limiting saturated and trans fats, increasing fiber intake, and checking lipids regularly with your doctor are often recommended."
        )
    elif status_starts("Total Cholesterol (mg/dL)", "Borderline"):
        recs.append(
            "Your total cholesterol is borderline. Small changes in diet and activity may help keep it from rising further."
        )

    # LDL
    if status_starts("LDL Cholesterol (mg/dL)", "Risky"):
        recs.append(
            "Your LDL ('bad') cholesterol appears high. A heart-healthy eating pattern and, in some cases, medications prescribed by a clinician may be considered to reduce cardiovascular risk."
        )
    elif status_starts("LDL Cholesterol (mg/dL)", "Borderline"):
        recs.append(
            "Your LDL cholesterol is borderline. Adjusting your diet (less saturated fat, more fiber) and staying active can help improve this value."
        )

    # HDL
    if status_starts("HDL Cholesterol (mg/dL)", "Risky"):
        recs.append(
            "Your HDL ('good') cholesterol is low. Regular physical activity, avoiding smoking, and discussing other options with your healthcare provider may help raise HDL."
        )
    # Borderline high HDL is generally protective; no concern recommendation needed.

    # Triglycerides
    if status_starts("Triglycerides (mg/dL)", "Risky"):
        recs.append(
            "Your triglycerides are high. Reducing sugary drinks, refined carbohydrates, and alcohol, and focusing on weight management are often advised for lowering triglycerides."
        )
    elif status_starts("Triglycerides (mg/dL)", "Borderline"):
        recs.append(
            "Your triglycerides are borderline. Paying attention to your sugar intake and overall diet can help keep them in a healthier range."
        )

    # Fallback if everything Normal
    if not recs:
        recs.append(
            "Your current lab indicators are within typical ranges. Continuing a balanced diet, regular exercise, and routine check-ups can help maintain your cardiovascular health."
        )

    return recs

# =========================
# STREAMLIT STATE
# =========================
if "patient_inputs" not in st.session_state:
    st.session_state.patient_inputs = None

# =========================
# SIDEBAR NAVIGATION
# =========================
pages = ["Home", "Data Entry", "Results", "Privacy"]
page = st.sidebar.radio("Navigation", pages)

# =========================
# PAGES
# =========================

# HOME
if page == "Home":
    st.markdown(
        """
        <div class="card">
          <div class="card-title">❤️ CardioPredict</div>
          <div class="card-subtitle">Personalized cardiovascular risk estimation</div>
          <p>
            <b>CardioPredict</b> lets you enter your lab results and vital signs
            to estimate your cardiovascular risk using a trained machine learning model.
          </p>
          <ul>
            <li>Enter your age, gender, and key lab values</li>
            <li>Get a predicted <b>risk percentage</b></li>
            <li>See clearly whether you are classified as <b>at risk</b> (red)
                or <b>not at high risk</b> (green)</li>
            <li>View which of your lab values are <b>normal</b>, <b>borderline</b>, or <b>risky</b>
                in a color-coded table and bar chart</li>
          </ul>
          <p style="font-size:0.9rem;color:#555;">
            ⚠️ This tool is for educational purposes only and is <b>not</b> a substitute
            for professional medical advice or diagnosis.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if model is None or scaler is None or feature_columns is None:
        st.warning("Artifacts (model/scaler/feature_columns) are not fully loaded. Check the .pkl files.")

# DATA ENTRY
elif page == "Data Entry":
    st.markdown(
        """
        <div class="card">
          <div class="card-title">🧾 Data Entry</div>
          <div class="card-subtitle">
            Please enter your information and lab results as accurately as possible.
          </div>
        """,
        unsafe_allow_html=True,
    )

    prev = st.session_state.patient_inputs or {}

    with st.form("data_entry_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Age (years)",
                min_value=1,
                max_value=120,
                value=int(prev.get("AGE", 45)),
                step=1,
            )
            gender = st.selectbox(
                "Gender",
                options=["Female", "Male", "Other"],
                index=[
                    "Female",
                    "Male",
                    "Other",
                ].index(prev.get("GENDER", "Female"))
                if prev.get("GENDER", "Female") in ["Female", "Male", "Other"]
                else 0,
            )
            bp = st.number_input(
                "Blood Pressure (systolic, mmHg)",
                min_value=70.0,
                max_value=250.0,
                value=float(prev.get("bp", 120.0)),
                step=1.0,
            )
            heart_rate = st.number_input(
                "Heart Rate (bpm)",
                min_value=30.0,
                max_value=200.0,
                value=float(prev.get("heart_rate", 75.0)),
                step=1.0,
            )

        with col2:
            bmi = st.number_input(
                "BMI (kg/m²)",
                min_value=10.0,
                max_value=60.0,
                value=float(prev.get("bmi", 25.0)),
                step=0.1,
            )
            chol_total = st.number_input(
                "Total Cholesterol (mg/dL)",
                min_value=70.0,
                max_value=400.0,
                value=float(prev.get("chol_total", 190.0)),
                step=1.0,
            )
            chol_hdl = st.number_input(
                "HDL Cholesterol (mg/dL)",
                min_value=10.0,
                max_value=120.0,
                value=float(prev.get("chol_hdl", 50.0)),
                step=1.0,
            )
            chol_ldl = st.number_input(
                "LDL Cholesterol (mg/dL)",
                min_value=30.0,
                max_value=300.0,
                value=float(prev.get("chol_ldl", 120.0)),
                step=1.0,
            )
            trig = st.number_input(
                "Triglycerides (mg/dL)",
                min_value=30.0,
                max_value=800.0,
                value=float(prev.get("trig", 150.0)),
                step=1.0,
            )

        submitted = st.form_submit_button("Save")

    if submitted:
        patient_inputs = {
            "AGE": age,
            "GENDER": gender,
            "bp": bp,
            "heart_rate": heart_rate,
            "bmi": bmi,
            "chol_total": chol_total,
            "chol_hdl": chol_hdl,
            "chol_ldl": chol_ldl,
            "trig": trig,
        }
        st.session_state.patient_inputs = patient_inputs
        st.success("Your data has been saved. Go to the **Results** page to view your risk.")

    st.markdown("</div>", unsafe_allow_html=True)

# RESULTS
elif page == "Results":
    if st.session_state.patient_inputs is None:
        st.markdown(
            """
            <div class="card">
              <div class="card-title">📊 Results</div>
              <p>No data found. Please enter your information in the <b>Data Entry</b> page first.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        patient_inputs = st.session_state.patient_inputs

        # Model-based probability
        try:
            model_prob, _ = predict_risk(patient_inputs)
        except Exception as e:
            st.markdown(
                f"""
                <div class="card">
                  <div class="card-title">📊 Results</div>
                  <p>Error during prediction: {e}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            model_prob = np.nan

        # Rule-based lab risk
        risk_table = build_risk_table(patient_inputs)
        rule_prob = estimate_rule_based_prob(risk_table)

        # Combined probability for visualization
        if np.isnan(model_prob):
            combined_prob = rule_prob
        else:
            combined_prob = max(model_prob, rule_prob)

        if np.isnan(combined_prob):
            st.markdown(
                """
                <div class="card">
                  <div class="card-title">📊 Results</div>
                  <p>Prediction is not available due to a model/preprocessing issue.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            combined_percent = combined_prob * 100
            label = 1 if combined_prob >= 0.5 else 0

            # Card: Risk summary + gauge
            st.markdown(
                """
                <div class="card">
                  <div class="card-title">📊 Results</div>
                """,
                unsafe_allow_html=True,
            )

            if label == 1:
                st.markdown(
                    f"""
                    <div style="padding:1rem;border-radius:0.8rem;background-color:#ffe5e5;color:#b00020;margin-bottom:1rem;">
                      <h3 style="margin:0 0 0.3rem 0;">⚠️ At Risk</h3>
                      <p style="margin:0;">Your combined cardiovascular risk index is <b>{combined_percent:.1f}%</b>.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style="padding:1rem;border-radius:0.8rem;background-color:#e0ffe5;color:#006400;margin-bottom:1rem;">
                      <h3 style="margin:0 0 0.3rem 0;">✅ Not at High Risk</h3>
                      <p style="margin:0;">Your combined cardiovascular risk index is <b>{combined_percent:.1f}%</b>.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.plotly_chart(risk_gauge(combined_prob), use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # Card: Lab report status (below gauge)
            st.markdown(
                """
                <div class="card">
                  <div class="card-title">🧬 Lab Report Status</div>
                  <div class="card-subtitle">
                    Your health indicator analysis based on commonly used clinical thresholds.
                  </div>
                """,
                unsafe_allow_html=True,
            )

            # Round values to 2 decimals for display
            display_table = risk_table.copy()
            display_table["Value"] = display_table["Value"].round(2)

            # Color-coded table using pandas Styler
            def style_row(row):
                status = row["Status"]
                if "Normal" in status:
                    bg = "background-color: #d4edda;"  # light green
                elif "Borderline" in status:
                    bg = "background-color: #fff3cd;"  # light yellow
                elif "Risky" in status:
                    bg = "background-color: #f8d7da;"  # light red
                else:
                    bg = ""
                return [bg] * len(row)

            styled_table = display_table.style.apply(style_row, axis=1)
            st.dataframe(styled_table, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # Card: Feature bar chart
            st.markdown(
                """
                <div class="card">
                  <div class="card-title">📈 Feature Risk Visualization</div>
                  <div class="card-subtitle">
                    Visual comparison of your lab values, colored by risk category.
                  </div>
                """,
                unsafe_allow_html=True,
            )

            fig_bar = feature_status_bar(risk_table)
            st.plotly_chart(fig_bar, use_container_width=True)

            st.caption(
                "The combined risk index uses both the trained model and rule-based lab thresholds. "
                "It is for educational use only and not a clinical diagnosis."
            )

            st.markdown("</div>", unsafe_allow_html=True)

            # Card: Personalized Recommendations
            st.markdown(
                """
                <div class="card">
                  <div class="card-title">🩺 Personalized Recommendations</div>
                  <div class="card-subtitle">
                    These general suggestions are based on which values appear normal, borderline, or risky.
                    They are not a diagnosis or a treatment plan.
                  </div>
                """,
                unsafe_allow_html=True,
            )

            recs = generate_recommendations(risk_table)
            for rec in recs:
                st.markdown(f"- {rec}")

            st.markdown(
                "<p style='font-size:0.9rem;color:#555;margin-top:0.75rem;'>"
                "Please review these points with a healthcare professional, who can give advice tailored to your full medical history."
                "</p>",
                unsafe_allow_html=True,
            )

            st.markdown("</div>", unsafe_allow_html=True)

# PRIVACY
elif page == "Privacy":
    st.markdown(
        """
        <div class="card">
          <div class="card-title">🔒 Privacy</div>
          <div class="card-subtitle">How CardioPredict handles your data</div>
          <ul>
            <li>Your inputs stay <b>only in this session</b> and are processed in memory.</li>
            <li>We do <b>not</b> store, share, or sell your personal information.</li>
            <li>Please avoid entering personally identifiable details such as your full name, address, or ID numbers.</li>
          </ul>
          <p style="font-size:0.9rem;color:#555;">
            CardioPredict is for <b>educational and informational purposes</b> only.
            It is <b>not</b> a medical device and does not provide medical diagnosis or treatment.
            Always consult a licensed healthcare professional for medical advice.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
