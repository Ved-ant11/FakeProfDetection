import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page config
st.set_page_config(page_title="Fake Profile Detector", layout="wide")

@st.cache_resource
def load_models():
    # Check if models exist
    if not os.path.exists('models/scaler.pkl'):
        return None, None, None
    
    scaler = joblib.load('models/scaler.pkl')
    svm = joblib.load('models/svm_model.pkl')
    rf = joblib.load('models/rf_model.pkl')
    return scaler, svm, rf

scaler, svm_model, rf_model = load_models()

st.title("Fake Social Media Profile Detection")
st.markdown("Use Machine Learning to determine if a profile is **Real** or **Fake**.")

if scaler is None:
    st.error("Models not found! Please run the training script first.")
    st.info("Run: `python src/train.py`")
else:
    # Sidebar for classifier selection
    st.sidebar.header("Configuration")
    model_choice = st.sidebar.selectbox("Choose Model", ["Random Forest", "SVM"])
    
    st.subheader("Profile Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        profile_pic = st.selectbox("Has Profile Picture?", ["Yes", "No"])
        nums_length_username = st.slider("Ratio of Numbers in Username", 0.0, 1.0, 0.0)
        fullname_words = st.number_input("Words in Full Name", min_value=0, value=2)
        nums_length_fullname = st.slider("Ratio of Numbers in Full Name", 0.0, 1.0, 0.0)
        
    with col2:
        name_username_match = st.selectbox("Name & Username Similar?", ["Yes", "No"])
        description_length = st.number_input("Bio Length (chars)", min_value=0, value=50)
        external_url = st.selectbox("Has External URL?", ["Yes", "No"])
        private = st.selectbox("Private Profile?", ["Yes", "No"])
        
    with col3:
        posts = st.number_input("Number of Posts", min_value=0, value=50)
        followers = st.number_input("Followers", min_value=0, value=200)
        follows = st.number_input("Following", min_value=0, value=200)

    # Convert inputs to model format
    input_data = {
        'profile_pic': 1 if profile_pic == "Yes" else 0,
        'nums_length_username': nums_length_username,
        'fullname_words': fullname_words,
        'nums_length_fullname': nums_length_fullname,
        'name_username_match': 1 if name_username_match == "Yes" else 0,
        'description_length': description_length,
        'external_url': 1 if external_url == "Yes" else 0,
        'private': 1 if private == "Yes" else 0,
        'posts': posts,
        'followers': followers,
        'follows': follows
    }
    
    if st.button("Analyze Profile", type="primary"):
        # Create DataFrame
        df = pd.DataFrame([input_data])
        
        # Scaling
        # SVM needs scaling, RF doesn't hurt.
        # Note: The scaler was fitted on 11 features. Ensure order matches.
        # Order in generates_data.py: 
        # profile_pic, nums_length_username, fullname_words, nums_length_fullname, name_username_match, 
        # description_length, external_url, private, posts, followers, follows
        
        # We must ensure the column order is identical to training
        cols = ['profile_pic', 'nums_length_username', 'fullname_words', 'nums_length_fullname', 
                'name_username_match', 'description_length', 'external_url', 'private', 
                'posts', 'followers', 'follows']
        df = df[cols]
        
        if model_choice == "SVM":
            X_scaled = scaler.transform(df)
            prediction = svm_model.predict(X_scaled)[0]
            prob = svm_model.predict_proba(X_scaled)[0]
        else:
            prediction = rf_model.predict(df)[0] # RF trained on raw data? 
            # Wait, in train.py: `rf_model.fit(X_train, y_train)` - X_train was NOT scaled for RF.
            # But the scaler is available.
            # Ideally standard practice is to scale for all, or verify.
            # In train.py I did: `X_train_scaled` for SVM, `X_train` for RF.
            # So for RF here, I should use `df` (unscaled). Correct.
            
            prob = rf_model.predict_proba(df)[0]
            
        result = "Fake" if prediction == 1 else "Real"
        color = "red" if prediction == 1 else "green"
        
        st.markdown(f"### Prediction: <span style='color:{color}'>{result}</span>", unsafe_allow_html=True)
        
        st.write(f"Probability of being Fake: {prob[1]:.2%}")
        st.write(f"Probability of being Real: {prob[0]:.2%}")
        
        # Feature importance for RF
        if model_choice == "Random Forest":
            st.divider()
            st.subheader("Why?")
            importances = rf_model.feature_importances_
            feature_imp = pd.DataFrame({'Feature': cols, 'Importance': importances})
            feature_imp = feature_imp.sort_values('Importance', ascending=False).head(3)
            st.write("Top influencing factors:")
            st.table(feature_imp)

