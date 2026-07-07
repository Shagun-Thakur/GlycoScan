"""
Diabetes Predictor & Health Dashboard
--------------------------------------
A Streamlit app that:
  1. Predicts diabetes risk using a pre-trained Random Forest model
  2. Flags individual health parameters (Glucose, Blood Pressure, BMI, Insulin, etc.)
     that fall outside typical healthy ranges, with guidance to get them checked
  3. Offers general education / lifestyle tips for people managing or at risk of diabetes

IMPORTANT: This app is for educational/informational purposes only and is NOT a
substitute for professional medical advice, diagnosis, or treatment.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title= "GlycoScan",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1b4965;
        margin-bottom: 0rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #5c677d;
        margin-top: 0rem;
    }
    .metric-card {
        background-color: #f8f9fb;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border: 1px solid #e6e9ef;
    }
    .flag-high {
        background-color: #fdecea;
        border-left: 5px solid #e63946;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.6rem;
        color: #1a1a1a;
    }
    .flag-low {
        background-color: #fff4e0;
        border-left: 5px solid #f4a300;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.6rem;
        color: #1a1a1a;
    }
    .flag-ok {
        background-color: #eaf7ee;
        border-left: 5px solid #2a9d5c;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.6rem;
        color: #1a1a1a;
    }
    .disclaimer-box {
        background-color: #f1f3f6;
        border: 1px solid #d6dae1;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        font-size: 0.88rem;
        color: #444;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Load model artifacts (cached)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "rf_diabetes_model.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.joblib"))
    return model, scaler, feature_names

@st.cache_data
def load_reference_data():
    df = pd.read_csv(os.path.join(MODEL_DIR, "diabetes.csv"))
    cols_with_invalid_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[cols_with_invalid_zeros] = df[cols_with_invalid_zeros].replace(0, np.nan)
    for col in cols_with_invalid_zeros:
        df[col] = df.groupby("Outcome")[col].transform(lambda x: x.fillna(x.median()))
    return df

model, scaler, feature_names = load_artifacts()
ref_df = load_reference_data()

# ----------------------------------------------------------------------------
# Clinical reference ranges (approximate, for general education only)
# ----------------------------------------------------------------------------
PARAM_INFO = {
    "Glucose": {
        "unit": "mg/dL",
        "ranges": [(0, 100, "Normal"), (100, 126, "Elevated (Prediabetic range)"), (126, 400, "High")],
        "tip_high": "Your glucose reading is in a high range. Elevated blood sugar is a key marker for diabetes — consider getting a fasting glucose or HbA1c test with your doctor.",
        "tip_elevated": "Your glucose is a bit above the normal range. This can be an early sign of prediabetes — it may help to monitor it periodically and review diet and activity levels.",
        "tip_ok": "Your glucose level looks within the normal range.",
    },
    "BloodPressure": {
        "unit": "mm Hg (diastolic)",
        "ranges": [(0, 80, "Normal"), (80, 90, "Elevated"), (90, 200, "High")],
        "tip_high": "Your diastolic blood pressure is high. High blood pressure increases cardiovascular risk and is common alongside diabetes — please get it checked by a healthcare professional.",
        "tip_elevated": "Your blood pressure is slightly elevated. Keeping an eye on sodium intake, stress, and activity levels may help.",
        "tip_ok": "Your blood pressure looks within a normal range.",
    },
    "BMI": {
        "unit": "kg/m²",
        "ranges": [(0, 18.5, "Underweight"), (18.5, 25, "Normal"), (25, 30, "Overweight"), (30, 70, "Obese")],
        "tip_high": "Your BMI falls in the obese range, which is strongly linked to higher diabetes risk. A conversation with a healthcare provider or dietitian about a sustainable weight-management plan could help.",
        "tip_elevated": "Your BMI is in the overweight range. Small, consistent changes in diet and physical activity can help bring this toward a healthier range.",
        "tip_low": "Your BMI is in the underweight range. Being underweight can also carry health risks — it may be worth discussing your nutrition with a professional.",
        "tip_ok": "Your BMI is within a healthy range.",
    },
    "Insulin": {
        "unit": "mu U/mL (2-hr serum)",
        "ranges": [(0, 16, "Low"), (16, 166, "Normal"), (166, 900, "High")],
        "tip_high": "Your insulin level is on the high side, which can indicate insulin resistance. It's worth discussing this reading with a doctor.",
        "tip_low": "Your insulin reading is on the lower side. This isn't necessarily a problem on its own, but worth mentioning to a doctor if you have other symptoms.",
        "tip_ok": "Your insulin level looks within a typical range.",
    },
    "SkinThickness": {
        "unit": "mm (triceps skinfold)",
        "ranges": [(0, 10, "Low"), (10, 40, "Normal"), (40, 100, "High")],
        "tip_high": "Your skinfold thickness reading is higher than typical, which can relate to body fat levels. Consider discussing overall body composition with a health professional.",
        "tip_low": "Your skinfold thickness reading is lower than typical.",
        "tip_ok": "Your skinfold thickness is within a typical range.",
    },
    "Age": {
        "unit": "years",
        "ranges": [(0, 45, "Lower risk age group"), (45, 120, "Higher risk age group")],
        "tip_high": "Diabetes risk naturally increases with age, especially after 45. Regular screening becomes more important in this age range.",
        "tip_ok": "You're in an age group generally associated with lower baseline diabetes risk.",
    },
    "DiabetesPedigreeFunction": {
        "unit": "score",
        "ranges": [(0, 0.5, "Lower genetic risk indication"), (0.5, 3, "Higher genetic risk indication")],
        "tip_high": "Your Diabetes Pedigree Function score suggests a stronger family history influence. This doesn't determine your outcome, but it's useful context for your doctor.",
        "tip_ok": "Your Diabetes Pedigree Function score suggests a lower family-history influence.",
    },
    "Pregnancies": {
        "unit": "count",
        "ranges": [(0, 100, "Informational")],
        "tip_ok": "Number of pregnancies is included as a contextual factor in this model; it isn't something to 'fix' but is part of the overall picture.",
    },
}


def classify_value(param, value):
    """Return (label, severity) for a value based on PARAM_INFO ranges."""
    info = PARAM_INFO[param]
    for low, high, label in info["ranges"]:
        if low <= value < high:
            return label, info["ranges"].index((low, high, label))
    return info["ranges"][-1][2], len(info["ranges"]) - 1


def get_tip(param, label):
    info = PARAM_INFO[param]
    label_lower = label.lower()
    if "high" in label_lower or "obese" in label_lower:
        return info.get("tip_high", info.get("tip_ok", ""))
    if "elevated" in label_lower or "overweight" in label_lower:
        return info.get("tip_elevated", info.get("tip_ok", ""))
    if "underweight" in label_lower or ("low" in label_lower and param != "Age"):
        return info.get("tip_low", info.get("tip_ok", ""))
    return info.get("tip_ok", "")


# ----------------------------------------------------------------------------
# Sidebar — user input
# ----------------------------------------------------------------------------
st.sidebar.markdown("## 🧾 Enter Your Health Details")
st.sidebar.markdown("Fill in the fields below. All fields use common clinical units.")

with st.sidebar.form("input_form"):
    pregnancies = st.number_input("Pregnancies (count)", min_value=0, max_value=20, value=1, step=1)
    glucose = st.number_input("Glucose (mg/dL)", min_value=40, max_value=300, value=110)
    blood_pressure = st.number_input("Blood Pressure — diastolic (mm Hg)", min_value=30, max_value=140, value=72)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=23)
    insulin = st.number_input("Insulin (mu U/mL, 2-hr serum)", min_value=0, max_value=850, value=80)
    bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=70.0, value=28.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.47, step=0.01)
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=33, step=1)

    submitted = st.form_submit_button("🔍 Analyze My Health")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div class='disclaimer-box'>⚠️ <b>Disclaimer:</b> This tool provides general, "
    "educational risk information only. It is not a medical diagnosis. Always consult "
    "a qualified healthcare professional about your results.</div>",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown("<div class='main-header'>🩺 GlycoScan - Female Diabetes Health Dashboard</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>A personal risk screening & health-parameter tracker for diabetes "
    "awareness, built on the Pima Indians Diabetes Database.</div>",
    unsafe_allow_html=True,
)
st.write("")

tab_predict, tab_params, tab_insights, tab_tips, tab_about = st.tabs(
    ["🔮 Risk Prediction", "📊 Parameter Check-up", "🌲 Model Insights", "💡 Health Tips", "ℹ️ About"]
)

input_dict = {
    "Pregnancies": pregnancies,
    "Glucose": glucose,
    "BloodPressure": blood_pressure,
    "SkinThickness": skin_thickness,
    "Insulin": insulin,
    "BMI": bmi,
    "DiabetesPedigreeFunction": dpf,
    "Age": age,
}
input_df = pd.DataFrame([input_dict])[feature_names]

# ----------------------------------------------------------------------------
# Tab 1 — Prediction
# ----------------------------------------------------------------------------
with tab_predict:
    if not submitted:
        st.info("👈 Enter your health details in the sidebar and click **Analyze My Health** to get started.")
    else:
        input_scaled = scaler.transform(input_df)
        proba = model.predict_proba(input_scaled)[0][1]
        pred = model.predict(input_scaled)[0]

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(proba * 100, 1),
            number={"suffix": "%", "font": {"size": 48}},
            title={"text": "Estimated Diabetes Risk", "font": {"size": 20}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1b4965"},
                "steps": [
                    {"range": [0, 30], "color": "#eaf7ee"},
                    {"range": [30, 60], "color": "#fff4e0"},
                    {"range": [60, 100], "color": "#fdecea"},
                ],
            },
        ))
        fig.update_layout(height=350, margin=dict(t=60, b=30, l=40, r=40))
        st.plotly_chart(fig, width='stretch')

        if pred == 1:
            st.error(
                f"**Higher Risk Indicated ({proba*100:.1f}% estimated probability)**\n\n"
                "The model flags a pattern in your inputs similar to individuals with diabetes "
                "in this dataset. This is **not a diagnosis** — please consult a doctor for "
                "proper testing (e.g. fasting glucose, HbA1c)."
            )
        else:
            st.success(
                f"**Lower Risk Indicated ({proba*100:.1f}% estimated probability)**\n\n"
                "Your inputs look more similar to non-diabetic patterns in this dataset. "
                "Still, regular checkups are a good idea, especially if any individual "
                "parameter is flagged in the **Parameter Check-up** tab."
            )

        st.markdown("#### Your Inputs at a Glance")
        summary_cols = st.columns(4)
        summary_items = list(input_dict.items())
        for i, (k, v) in enumerate(summary_items):
            with summary_cols[i % 4]:
                st.metric(k, v)

        st.markdown("---")
        st.caption(
            "Model: Random Forest classifier trained on the Pima Indians Diabetes Database. "
            "Estimated probability reflects patterns learned from historical data and carries "
            "inherent uncertainty."
        )
# ----------------------------------------------------------------------------
# Tab 2 — Parameter Check-up
# ----------------------------------------------------------------------------
with tab_params:
    st.markdown("### Individual Parameter Analysis")
    st.write(
        "Each health parameter you entered is compared against general clinical reference ranges. "
        "Flags here highlight which specific values may be worth getting checked — even if your "
        "overall predicted risk is low."
    )

    if not submitted:
        st.info("👈 Submit your details in the sidebar to see your parameter breakdown.")
    else:
        flagged_high = []
        flagged_ok = []

        checkup_params = ["Glucose", "BloodPressure", "BMI", "Insulin", "SkinThickness", "Age", "DiabetesPedigreeFunction"]

        for param in checkup_params:
            value = input_dict[param]
            label, severity_idx = classify_value(param, value)
            tip = get_tip(param, label)
            unit = PARAM_INFO[param]["unit"]

            is_concern = any(k in label.lower() for k in ["high", "obese", "elevated", "underweight", "higher"])

            box_class = "flag-high" if ("high" in label.lower() or "obese" in label.lower()) else (
                "flag-low" if ("elevated" in label.lower() or "underweight" in label.lower() or "higher" in label.lower()) else "flag-ok"
            )

            with st.container():
                st.markdown(
                    f"<div class='{box_class}'><b>{param}</b>: {value} {unit} &nbsp;→&nbsp; "
                    f"<b>{label}</b><br>{tip}</div>",
                    unsafe_allow_html=True,
                )

            if is_concern:
                flagged_high.append(param)
            else:
                flagged_ok.append(param)

        st.markdown("---")
        if flagged_high:
            st.warning(
                f"**Parameters worth getting checked:** {', '.join(flagged_high)}. "
                "Consider bringing these specific readings up with a healthcare provider."
            )
        else:
            st.success("All checked parameters fall within generally healthy ranges. Keep it up! 🎉")

        # # Comparison chart: user vs healthy midpoint vs dataset average (diabetic vs non-diabetic)
        # st.markdown("#### Where You Stand vs. Dataset Averages")
        # compare_cols = ["Glucose", "BloodPressure", "BMI", "Insulin", "SkinThickness"]
        # avg_no = ref_df[ref_df.Outcome == 0][compare_cols].mean()
        # avg_yes = ref_df[ref_df.Outcome == 1][compare_cols].mean()
        # you = pd.Series({c: input_dict[c] for c in compare_cols})

        # comp_df = pd.DataFrame({
        #     "You": you,
        #     "Avg (No Diabetes)": avg_no,
        #     "Avg (Diabetes)": avg_yes,
        # }).reset_index().rename(columns={"index": "Parameter"})
        # comp_melted = comp_df.melt(id_vars="Parameter", var_name="Group", value_name="Value")

        # fig2 = px.bar(
        #     comp_melted, x="Parameter", y="Value", color="Group", barmode="group",
        #     color_discrete_map={"You": "#1b4965", "Avg (No Diabetes)": "#2a9d5c", "Avg (Diabetes)": "#e63946"},
        # )
        # fig2.update_layout(height=420, margin=dict(t=30, b=10))
        # st.plotly_chart(fig2, width='stretch')

# ----------------------------------------------------------------------------
# Tab 3 — Model Insights
# ----------------------------------------------------------------------------
with tab_insights:
    st.markdown("### What Drives This Model's Predictions?")
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
    fig3 = px.bar(
        importances, x=importances.values, y=importances.index, orientation="h",
        labels={"x": "Importance", "y": "Feature"}, color=importances.values,
        color_continuous_scale="Tealgrn",
    )
    fig3.update_layout(height=420, coloraxis_showscale=False, margin=dict(t=20, b=10))
    st.plotly_chart(fig3, width='stretch')
    st.caption(
        "These importances come directly from the trained Random Forest and reflect how much each "
        "feature contributed to the model's decisions across the training data — not a statement "
        "about causation for any one individual."
    )

    st.markdown("### Dataset Snapshot")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", len(ref_df))
    c2.metric("Diabetic Cases", int(ref_df.Outcome.sum()))
    c3.metric("Non-Diabetic Cases", int((ref_df.Outcome == 0).sum()))

    # hist_df = ref_df.copy()
    # hist_df["Diabetes"] = hist_df["Outcome"].map({0: "No Diabetes", 1: "Diabetes"})
    # fig4 = px.histogram(
    #     hist_df, x="Glucose", color="Diabetes", barmode="overlay", nbins=30,
    #     color_discrete_map={"No Diabetes": "#2a9d5c", "Diabetes": "#e63946"},
    # )
    # fig4.update_layout(height=350, margin=dict(t=20, b=10))
    # st.plotly_chart(fig4, width='stretch')
# ----------------------------------------------------------------------------
# Tab 4 — Health Tips
# ----------------------------------------------------------------------------
with tab_tips:
    st.markdown("### General Lifestyle Tips for Diabetes Awareness & Management")
    st.write(
        "These are general, widely recommended lifestyle habits — they're educational in nature "
        "and not a personalized treatment plan."
    )

    tips = [
        ("🥗 Balanced Nutrition", "Favor whole grains, vegetables, and lean proteins; limit sugary drinks and refined carbs, which cause sharp glucose spikes."),
        ("🏃 Regular Physical Activity", "Aim for at least 150 minutes of moderate activity per week (e.g. brisk walking) — it improves insulin sensitivity."),
        ("⚖️ Healthy Weight Management", "Even modest weight loss (5-7% of body weight) can meaningfully reduce diabetes risk for those who are overweight."),
        ("🩸 Regular Monitoring", "If you're at elevated risk, periodic glucose/HbA1c checks help catch changes early."),
        ("😴 Sleep & Stress", "Poor sleep and chronic stress can affect blood sugar regulation — prioritize consistent sleep and stress-management routines."),
        ("🚭 Avoid Smoking & Limit Alcohol", "Both can worsen insulin resistance and cardiovascular risk, which are closely tied to diabetes complications."),
        ("👨‍⚕️ Regular Checkups", "Routine screening becomes more important with age, family history, or if any of your parameters were flagged above."),
    ]

    for title, text in tips:
        with st.expander(title):
            st.write(text)

    st.markdown("---")
    st.markdown(
        "<div class='disclaimer-box'>This section provides general wellness information. "
        "It does not replace individualized advice from a doctor, dietitian, or diabetes educator.</div>",
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# Tab 5 — About
# ----------------------------------------------------------------------------
with tab_about:
    st.markdown("### About This App")
    st.write(
        "This dashboard is built on the **Pima Indians Diabetes Database**, a well-known public "
        "health dataset of female patients of Pima Indian heritage. A **Random Forest** classifier "
        "was trained on cleaned and imputed data to estimate diabetes risk from 8 health measurements."
    )
    st.markdown("""
**What this app does:**
- Estimates an overall diabetes risk probability from your inputs
- Flags individual parameters (glucose, blood pressure, BMI, insulin, etc.) that fall outside typical healthy ranges
- Shows how your numbers compare to dataset averages for diabetic vs. non-diabetic individuals
- Offers general lifestyle guidance for diabetes awareness

**What this app does NOT do:**
- It does **not** diagnose diabetes or any medical condition
- It does **not** replace lab tests (fasting glucose, HbA1c, oral glucose tolerance test) or a doctor's evaluation
- It is trained on a specific historical population and may not generalize perfectly to all individuals
""")
    st.markdown(
        "<div class='disclaimer-box'>⚠️ <b>Medical Disclaimer:</b> This application is for "
        "educational and informational purposes only. It is not intended to diagnose, treat, cure, "
        "or prevent any disease. Always seek the advice of a physician or other qualified health "
        "provider with any questions regarding a medical condition.</div>",
        unsafe_allow_html=True,
    )
