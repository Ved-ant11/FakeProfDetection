import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

def train_models():
    # File paths
    input_path = r"e:\botDetect\data\processed\cresci_expanded_with_content.csv"
    model_dir = r"e:\botDetect\models"
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    
    # Feature columns
    # Must match what was created in process_data.py
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
    
    target_col = 'fake'
    
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"Features: {X.columns.tolist()}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    # Important for SVM likely, less so for RF but good practice
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    print("Scaler saved.")
    
    # 1. Random Forest
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train) # Tree models don't strictly need scaling, using raw for RF (or scaled, doesn't matter much)
    
    y_pred_rf = rf_model.predict(X_test)
    print("Random Forest Results:")
    print(classification_report(y_test, y_pred_rf))
    print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    
    joblib.dump(rf_model, os.path.join(model_dir, 'rf_model.pkl'))
    
    # 2. SVM
    print("\nTraining SVM...")
    # SVM *does* need scaling
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    
    y_pred_svm = svm_model.predict(X_test_scaled)
    print("SVM Results:")
    print(classification_report(y_test, y_pred_svm))
    print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
    
    joblib.dump(svm_model, os.path.join(model_dir, 'svm_model.pkl'))
    
    print("\nAll models trained and saved.")

if __name__ == "__main__":
    train_models()