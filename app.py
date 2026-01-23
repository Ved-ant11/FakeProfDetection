import streamlit as st
import pandas as pd
import numpy as np
import joblib
import math
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# ---------------- CONFIG ----------------
st.set_page_config(page_title="BotX-Ray Detector", layout="wide")

# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_assets():
    try:
        scaler = joblib.load("models/scaler.pkl")
        svm = joblib.load("models/svm_model.pkl")
        rf = joblib.load("models/rf_model.pkl")
        return scaler, svm, rf
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None

# ---------------- SHAP RENDERER ----------------
def st_shap(plot, height=400):
    shap_html = f"""
    <head>{shap.getjs()}</head>
    <body>{plot.html()}</body>
    """
    components.html(shap_html, height=height)

# ---------------- FEATURE ENGINEERING ----------------
def calculate_entropy(text):
    if not text:
        return 0.0
    text = str(text).lower()
    probs = [text.count(c) / len(text) for c in set(text)]
    return -sum(p * math.log2(p) for p in probs if p > 0)

# Load assets
scaler, svm_model, rf_model = load_assets()

# ---------------- UI ----------------
st.title("🛡️ BotX-Ray: Explainable Fake Profile Detection")
st.markdown("Explainable **Fake Profile Detection** using **Entropy + Behavioral Analysis**")

if scaler is None or svm_model is None or rf_model is None:
    st.error("Models not found. Ensure the 'models' folder exists with .pkl files.")
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.header("Profile Metadata")
screen_name = st.sidebar.text_input("Screen Name (Handle)", "user12345678")

col1, col2 = st.sidebar.columns(2)
followers = col1.number_input("Followers", min_value=0, value=10)
friends = col2.number_input("Following", min_value=0, value=500)

col3, col4 = st.sidebar.columns(2)
statuses = col3.number_input("Total Tweets", min_value=0, value=50)
favorites = col4.number_input("Favorites Given", min_value=0, value=0)

account_age = st.sidebar.slider("Account Age (Days)", 1, 3650, 30)

# ---------------- CALCULATION ----------------
entropy = calculate_entropy(screen_name)
reputation = followers / (followers + friends + 1)
engagement = favorites / (statuses + 1)
tweet_freq = statuses / account_age
name_len = len(screen_name)

input_data = pd.DataFrame([{
    "screen_name_entropy": entropy,
    "name_len": name_len,
    "reputation_score": reputation,
    "engagement_rate": engagement,
    "account_age_days": account_age,
    "tweet_freq": tweet_freq
}])

# ---------------- MAIN LAYOUT ----------------
left, right = st.columns([1, 2])

with left:
    st.subheader("Calculated Features")
    st.metric("Name Entropy", f"{entropy:.2f}")
    st.metric("Reputation Score", f"{reputation:.2f}")
    st.metric("Engagement Rate", f"{engagement:.3f}")

with right:
    model_choice = st.selectbox("Classifier", ["Random Forest", "SVM"])

    if st.button("Run Detection", type="primary"):
        try:
            if model_choice == "SVM":
                input_scaled = scaler.transform(input_data)
                pred = svm_model.predict(input_scaled)[0]
                prob = svm_model.predict_proba(input_scaled)[0]
            else:
                pred = rf_model.predict(input_data)[0]
                prob = rf_model.predict_proba(input_data)[0]

            confidence = prob[1] if pred == 1 else prob[0]

            if pred == 1:
                st.error(f"🚨 **FAKE PROFILE DETECTED** (Confidence: {confidence:.1%})")
            else:
                st.success(f"✅ **REAL PROFILE** (Confidence: {confidence:.1%})")

            # ---------------- SHAP EXPLANATION ----------------
            st.divider()
            st.subheader("🔍 Explainable AI Analysis")

            if model_choice == "Random Forest":
                with st.spinner("Calculating SHAP values..."):
                    explainer = shap.TreeExplainer(rf_model)
                    shap_values = explainer.shap_values(input_data)

                    # Logic to handle different SHAP output formats based on library version
                    if isinstance(shap_values, list):
                        # Multi-output: index 1 is usually the 'Fake' class
                        sv = shap_values[1][0]
                        ev = explainer.expected_value[1]
                    elif len(shap_values.shape) == 3:
                        # 3D array: (samples, features, classes)
                        sv = shap_values[0, :, 1]
                        ev = explainer.expected_value[1]
                    else:
                        # Standard 2D array
                        sv = shap_values[0]
                        ev = explainer.expected_value

                    # Render the force plot
                    p = shap.force_plot(
                        ev, 
                        sv, 
                        input_data.iloc[0],
                        matplotlib=False
                    )
                    st_shap(p)

            else:
                st.info("SHAP Analysis is optimized for Random Forest in this preview.")

        except Exception as e:
            st.error("Prediction failed ❌")
            st.exception(e)