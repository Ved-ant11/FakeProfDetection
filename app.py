import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

st.set_page_config(page_title="BotDetector - Cresci Model", layout="wide")

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

def st_shap(plot, height=400):
    shap_html = f"""
    <head>{shap.getjs()}</head>
    <body>{plot.html()}</body>
    """
    components.html(shap_html, height=height)

# Load assets
scaler, svm_model, rf_model = load_assets()

st.title("🤖 Social Bot Detection (Cresci-17)")
st.markdown("Detect fake Twitter/Social profiles using the **Cresci-2017** dataset model.")

if scaler is None or svm_model is None or rf_model is None:
    st.error("Models not found. Please ensure 'models/scaler.pkl', 'models/svm_model.pkl', and 'models/rf_model.pkl' exist.")
    st.stop()

# --- Sidebar Inputs ---
st.sidebar.header("Profile Features")

# Numeric Inputs
statuses_count = st.sidebar.number_input("Statuses Count", min_value=0, value=50)
followers_count = st.sidebar.number_input("Followers Count", min_value=0, value=100)
friends_count = st.sidebar.number_input("Friends Count (Following)", min_value=0, value=200)
favourites_count = st.sidebar.number_input("Favourites Count", min_value=0, value=10)
listed_count = st.sidebar.number_input("Listed Count", min_value=0, value=0)

st.sidebar.markdown("---")
st.sidebar.header("Profile Settings")

# Binary Inputs
geo_enabled = st.sidebar.checkbox("Geo Enabled", value=False)
default_profile = st.sidebar.checkbox("Default Profile Theme", value=True)
profile_use_bg = st.sidebar.checkbox("Use Background Image", value=True)
verified = st.sidebar.checkbox("Verified Account", value=False)
protected = st.sidebar.checkbox("Protected Account", value=False)

# Derived Feature
reputation = followers_count / (followers_count + friends_count + 1)

st.sidebar.markdown("---")
st.sidebar.header("Content Metadata (Avg per Tweet)")
avg_tweet_len = st.sidebar.number_input("Avg Tweet Length", min_value=0.0, value=50.0)
avg_hashtags = st.sidebar.number_input("Avg Hashtags", min_value=0.0, value=0.1)
avg_mentions = st.sidebar.number_input("Avg Mentions", min_value=0.0, value=0.5)
avg_urls = st.sidebar.number_input("Avg URLs", min_value=0.0, value=0.2)

# Construct Input Dataframe
# MUST match the order and names in train.py
# Construct Input Dataframe
# MUST match the order and names in train.py
feature_cols = [
    'statuses_count',
    'followers_count',
    'friends_count',
    'favourites_count',
    'listed_count',
    'geo_enabled',
    'default_profile',
    'profile_use_background_image',
    'verified',
    'protected',
    'avg_tweet_len',
    'avg_hashtags',
    'avg_mentions',
    'avg_urls',
    'reputation'
]

input_data = pd.DataFrame([{
    'statuses_count': statuses_count,
    'followers_count': followers_count,
    'friends_count': friends_count,
    'favourites_count': favourites_count,
    'listed_count': listed_count,
    'geo_enabled': int(geo_enabled),
    'default_profile': int(default_profile),
    'profile_use_background_image': int(profile_use_bg),
    'verified': int(verified),
    'protected': int(protected),
    'avg_tweet_len': avg_tweet_len,
    'avg_hashtags': avg_hashtags,
    'avg_mentions': avg_mentions,
    'avg_urls': avg_urls,
    'reputation': reputation
}])

# --- Main Layout ---
left, right = st.columns([1, 2])

with left:
    st.subheader("Profile Profile")
    st.metric("Reputation Score", f"{reputation:.4f}")
    st.write(f"**Verified:** {'Yes' if verified else 'No'}")
    st.write(f"**Geo Enabled:** {'Yes' if geo_enabled else 'No'}")
    
    model_choice = st.selectbox("Select Model", ["Random Forest", "SVM"])

    if st.button("Analyze Profile", type="primary"):
        try:
            if model_choice == "SVM":
                # SVM requires scaling
                input_scaled = scaler.transform(input_data)
                pred = svm_model.predict(input_scaled)[0]
                prob = svm_model.predict_proba(input_scaled)[0]
            else:
                # RF works on raw data (or scaled, but we trained on raw/scaled mix logic, 
                # usually RF handles raw fine. In train.py we fit RF on RAW X_train).
                # Wait, in train.py: `rf_model.fit(X_train, y_train)`. X_train was NOT scaled.
                # So we must pass raw `input_data`.
                pred = rf_model.predict(input_data)[0]
                prob = rf_model.predict_proba(input_data)[0]

            confidence = prob[1] if pred == 1 else prob[0]
            
            st.divider()
            if pred == 1:
                st.error(f"🚨 **SPAMBOT DETECTED**\n\nConfidence: {confidence:.2%}")
            else:
                st.success(f"✅ **GENUINE ACCOUNT**\n\nConfidence: {confidence:.2%}")

        except Exception as e:
            st.error("Error during prediction")
            st.exception(e)

with right:
    # SHAP Explanation (Only for RF usually easiest to impl)
    if 'pred' in locals() and model_choice == "Random Forest":
        st.subheader("🔍 Decision Explanation")
        with st.spinner("Generating explanation..."):
            try:
                explainer = shap.TreeExplainer(rf_model)
                # shap_values matching the input_data (1 row)
                shap_values = explainer.shap_values(input_data)
                
                # Check structure of shap_values
                # Random Forest Classifier (sklearn) usually returns a list of arrays (one per class)
                if isinstance(shap_values, list):
                    # Class 1 is 'Fake' (Spambot)
                    # shap_values[1] is (n_samples, n_features)
                    sv = shap_values[1][0] 
                    ev = explainer.expected_value[1]
                else:
                    # Binary classification might sometimes return just one array or 3D array
                    if len(shap_values.shape) == 3:
                         # (n_samples, n_features, n_classes)
                        sv = shap_values[0, :, 1]
                        ev = explainer.expected_value[1]
                    else:
                        # (n_samples, n_features) - likely just for the positive class or single output
                        sv = shap_values[0]
                        ev = explainer.expected_value

                # Ensure 'ev' is a scalar float, not an array or list
                if isinstance(ev, (list, np.ndarray)):
                    if len(ev) > 1: # multi-class expected value container
                         ev = ev[1] 
                    else:
                         ev = ev[0]
                
                # Force plot
                p = shap.force_plot(float(ev), sv, input_data.iloc[0], matplotlib=False, feature_names=feature_cols)
                st_shap(p)
            except Exception as e:
                st.warning(f"Could not generate SHAP plot: {e}")
                # Print debug info to console for troubleshooting
                print(f"DEBUG: SHAP Error. Type: {type(shap_values)}")
                if isinstance(shap_values, list):
                     print(f"List len: {len(shap_values)}, Shape[0]: {shap_values[0].shape}")
