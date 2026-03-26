import streamlit as st
import pandas as pd
import numpy as np
import math
import os
import json
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from collections import Counter

st.set_page_config(page_title="BotDetector - Cresci Model", layout="wide")


def username_entropy(name):
    if not isinstance(name, str) or len(name) == 0:
        return 0.0
    counts = Counter(name)
    length = len(name)
    entropy = 0.0
    for count in counts.values():
        p = count / length
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


@st.cache_resource
def load_assets():
    try:
        scaler = joblib.load("models/scaler.pkl")
        svm = joblib.load("models/svm_model.pkl")
        rf = joblib.load("models/rf_model.pkl")
        gbt = joblib.load("models/gbt_model.pkl")
        return scaler, svm, rf, gbt
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None


def st_shap(plot, height=400):
    shap_html = f"""
    <head>{shap.getjs()}</head>
    <body>{plot.html()}</body>
    """
    components.html(shap_html, height=height)


scaler, svm_model, rf_model, gbt_model = load_assets()

st.title("Social Bot Detection (Cresci-17)")
st.markdown("Detect fake Twitter/Social profiles using the **Cresci-2017** dataset model.")

if scaler is None or svm_model is None or rf_model is None or gbt_model is None:
    st.error("Models not found. Please ensure all model files exist in the models/ directory.")
    st.stop()

st.sidebar.header("Profile Features")

screen_name = st.sidebar.text_input("Username (screen_name)", value="example_user")
statuses_count = st.sidebar.number_input("Statuses Count", min_value=0, value=50)
followers_count = st.sidebar.number_input("Followers Count", min_value=0, value=100)
friends_count = st.sidebar.number_input("Friends Count (Following)", min_value=0, value=200)
favourites_count = st.sidebar.number_input("Favourites Count", min_value=0, value=10)
listed_count = st.sidebar.number_input("Listed Count", min_value=0, value=0)

st.sidebar.markdown("---")
st.sidebar.header("Profile Settings")

geo_enabled = st.sidebar.checkbox("Geo Enabled", value=False)
default_profile = st.sidebar.checkbox("Default Profile Theme", value=True)
profile_use_bg = st.sidebar.checkbox("Use Background Image", value=True)
verified = st.sidebar.checkbox("Verified Account", value=False)
protected = st.sidebar.checkbox("Protected Account", value=False)

reputation = followers_count / (followers_count + friends_count + 1)
entropy_val = username_entropy(screen_name)

st.sidebar.markdown("---")
st.sidebar.header("Content Metadata (Avg per Tweet)")
avg_tweet_len = st.sidebar.number_input("Avg Tweet Length", min_value=0.0, value=50.0)
avg_hashtags = st.sidebar.number_input("Avg Hashtags", min_value=0.0, value=0.1)
avg_mentions = st.sidebar.number_input("Avg Mentions", min_value=0.0, value=0.5)
avg_urls = st.sidebar.number_input("Avg URLs", min_value=0.0, value=0.2)

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
    'username_entropy',
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
    'username_entropy': entropy_val,
    'reputation': reputation
}])

tab_predict, tab_importance, tab_roc, tab_cv, tab_shap_summary = st.tabs([
    "Prediction", "Feature Importance", "ROC / AUC Curves", "Cross-Validation", "SHAP Summary"
])

with tab_predict:
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Profile Summary")
        st.metric("Reputation Score", f"{reputation:.4f}")
        st.metric("Username Entropy", f"{entropy_val:.4f}")
        st.write(f"**Username:** {screen_name}")
        st.write(f"**Verified:** {'Yes' if verified else 'No'}")
        st.write(f"**Geo Enabled:** {'Yes' if geo_enabled else 'No'}")

        model_choice = st.selectbox("Select Model", ["Random Forest", "SVM", "Gradient Boosting"])

        if st.button("Analyze Profile", type="primary"):
            try:
                if model_choice == "SVM":
                    input_scaled = scaler.transform(input_data)
                    pred = svm_model.predict(input_scaled)[0]
                    prob = svm_model.predict_proba(input_scaled)[0]
                elif model_choice == "Gradient Boosting":
                    pred = gbt_model.predict(input_data)[0]
                    prob = gbt_model.predict_proba(input_data)[0]
                else:
                    pred = rf_model.predict(input_data)[0]
                    prob = rf_model.predict_proba(input_data)[0]

                confidence = prob[1] if pred == 1 else prob[0]

                st.divider()
                if pred == 1:
                    st.error(f"SPAMBOT DETECTED\n\nConfidence: {confidence:.2%}")
                else:
                    st.success(f"GENUINE ACCOUNT\n\nConfidence: {confidence:.2%}")

            except Exception as e:
                st.error("Error during prediction")
                st.exception(e)

    with right:
        if 'pred' in dir() or 'pred' in locals():
            if model_choice in ["Random Forest", "Gradient Boosting"]:
                st.subheader("Decision Explanation")
                with st.spinner("Generating explanation..."):
                    try:
                        active_model = rf_model if model_choice == "Random Forest" else gbt_model
                        explainer = shap.TreeExplainer(active_model)
                        shap_values = explainer.shap_values(input_data)

                        if isinstance(shap_values, list):
                            sv = shap_values[1][0]
                            ev = explainer.expected_value[1]
                        else:
                            if len(shap_values.shape) == 3:
                                sv = shap_values[0, :, 1]
                                ev = explainer.expected_value[1]
                            else:
                                sv = shap_values[0]
                                ev = explainer.expected_value

                        if isinstance(ev, (list, np.ndarray)):
                            if len(ev) > 1:
                                ev = ev[1]
                            else:
                                ev = ev[0]

                        p = shap.force_plot(float(ev), sv, input_data.iloc[0], matplotlib=False, feature_names=feature_cols)
                        st_shap(p)
                    except Exception as e:
                        st.warning(f"Could not generate SHAP plot: {e}")

with tab_importance:
    st.subheader("Feature Importance Analysis")

    importance_path = "models/feature_importances.csv"
    if os.path.exists(importance_path):
        importance_df = pd.read_csv(importance_path)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Random Forest**")
            rf_sorted = importance_df.sort_values('rf_importance', ascending=True)
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            ax1.barh(rf_sorted['feature'], rf_sorted['rf_importance'], color='#2196F3')
            ax1.set_xlabel('Importance')
            ax1.set_title('Random Forest Feature Importance')
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)

        with col2:
            st.markdown("**Gradient Boosting**")
            gbt_sorted = importance_df.sort_values('gbt_importance', ascending=True)
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            ax2.barh(gbt_sorted['feature'], gbt_sorted['gbt_importance'], color='#FF5722')
            ax2.set_xlabel('Importance')
            ax2.set_title('Gradient Boosting Feature Importance')
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

        st.markdown("**Importance Values**")
        st.dataframe(importance_df.style.format({
            'rf_importance': '{:.4f}',
            'gbt_importance': '{:.4f}',
            'avg_importance': '{:.4f}'
        }), width='stretch')
    else:
        st.info("Feature importance data not found. Please run training first.")

with tab_roc:
    st.subheader("ROC / AUC Curves")

    roc_path = "models/roc_data.json"
    if os.path.exists(roc_path):
        with open(roc_path, 'r') as f:
            roc_data = json.load(f)

        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
        ax_roc.plot(roc_data['rf_fpr'], roc_data['rf_tpr'], color='#2196F3', lw=2,
                    label=f"Random Forest (AUC = {roc_data['rf_auc']:.4f})")
        ax_roc.plot(roc_data['svm_fpr'], roc_data['svm_tpr'], color='#4CAF50', lw=2,
                    label=f"SVM (AUC = {roc_data['svm_auc']:.4f})")
        ax_roc.plot(roc_data['gbt_fpr'], roc_data['gbt_tpr'], color='#FF5722', lw=2,
                    label=f"Gradient Boosting (AUC = {roc_data['gbt_auc']:.4f})")
        ax_roc.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Baseline')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('ROC Curves - Model Comparison')
        ax_roc.legend(loc='lower right')
        ax_roc.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_roc)
        plt.close(fig_roc)

        st.markdown("**AUC Scores**")
        auc_df = pd.DataFrame({
            'Model': ['Random Forest', 'SVM', 'Gradient Boosting'],
            'AUC Score': [roc_data['rf_auc'], roc_data['svm_auc'], roc_data['gbt_auc']]
        })
        st.dataframe(auc_df.style.format({'AUC Score': '{:.4f}'}), width='stretch')
    else:
        st.info("ROC data not found. Please run training first.")

with tab_cv:
    st.subheader("5-Fold Stratified Cross-Validation Results")

    cv_path = "models/cv_results.json"
    if os.path.exists(cv_path):
        with open(cv_path, 'r') as f:
            cv_results = json.load(f)

        cv_summary = []
        for model_name, data in cv_results.items():
            cv_summary.append({
                'Model': model_name,
                'Mean F1': data['mean'],
                'Std F1': data['std'],
                'Fold 1': data['folds'][0],
                'Fold 2': data['folds'][1],
                'Fold 3': data['folds'][2],
                'Fold 4': data['folds'][3],
                'Fold 5': data['folds'][4]
            })

        cv_df = pd.DataFrame(cv_summary)
        st.dataframe(cv_df.style.format({
            'Mean F1': '{:.4f}', 'Std F1': '{:.4f}',
            'Fold 1': '{:.4f}', 'Fold 2': '{:.4f}', 'Fold 3': '{:.4f}',
            'Fold 4': '{:.4f}', 'Fold 5': '{:.4f}'
        }), width='stretch')

        fig_cv, ax_cv = plt.subplots(figsize=(10, 5))
        models = list(cv_results.keys())
        means = [cv_results[m]['mean'] for m in models]
        stds = [cv_results[m]['std'] for m in models]
        colors = ['#2196F3', '#4CAF50', '#FF5722']
        bars = ax_cv.bar(models, means, yerr=stds, capsize=8, color=colors, edgecolor='white', linewidth=1.5)
        ax_cv.set_ylabel('F1 Score')
        ax_cv.set_title('Cross-Validation F1 Scores (Mean ± Std)')
        ax_cv.set_ylim(0, 1.1)
        ax_cv.grid(True, axis='y', alpha=0.3)
        for bar, mean, std in zip(bars, means, stds):
            ax_cv.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.02,
                       f'{mean:.4f}', ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_cv)
        plt.close(fig_cv)

        st.markdown("**Per-Fold Scores**")
        fig_folds, ax_folds = plt.subplots(figsize=(10, 5))
        fold_labels = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
        x = np.arange(len(fold_labels))
        width = 0.25
        for i, (model_name, color) in enumerate(zip(models, colors)):
            ax_folds.bar(x + i * width, cv_results[model_name]['folds'], width,
                         label=model_name, color=color, edgecolor='white')
        ax_folds.set_xlabel('Fold')
        ax_folds.set_ylabel('F1 Score')
        ax_folds.set_title('Per-Fold F1 Scores')
        ax_folds.set_xticks(x + width)
        ax_folds.set_xticklabels(fold_labels)
        ax_folds.legend()
        ax_folds.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_folds)
        plt.close(fig_folds)
    else:
        st.info("Cross-validation data not found. Please run training first.")

with tab_shap_summary:
    st.subheader("SHAP Summary Plot (Global Feature Impact)")

    shap_path = "models/shap_summary.png"
    if os.path.exists(shap_path):
        st.image(shap_path, caption="SHAP Summary Plot - Random Forest", width=800)
        st.markdown(
            "Each dot represents a sample from the test set. "
            "Position on the x-axis shows the SHAP value (impact on prediction). "
            "Color indicates feature value (red = high, blue = low)."
        )
    else:
        st.info("SHAP summary plot not found. Please run training first.")
