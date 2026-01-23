import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def train_models():
    # Load dataset
    data_path = 'data/raw/dataset.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run generate_data.py first.")
        return

    df = pd.read_csv(data_path)
    
    # Feature selection
    X = df.drop(['fake'], axis=1)
    y = df['fake']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Save test data for potential later verification
    test_df = pd.concat([X_test, y_test], axis=1)
    os.makedirs('data/processed', exist_ok=True)
    test_df.to_csv('data/processed/test_data.csv', index=False)

    # Preprocessing
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Scaler saved to models/scaler.pkl")

    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train) # RF works fine without scaling
    
    y_pred_rf = rf_model.predict(X_test)
    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
    print(classification_report(y_test, y_pred_rf))
    
    joblib.dump(rf_model, 'models/rf_model.pkl')
    print("Random Forest model saved to models/rf_model.pkl")

    print("\nTraining SVM...")
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train) # SVM needs scaling
    
    y_pred_svm = svm_model.predict(X_test_scaled)
    print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
    print(classification_report(y_test, y_pred_svm))
    
    joblib.dump(svm_model, 'models/svm_model.pkl')
    print("SVM model saved to models/svm_model.pkl")

if __name__ == "__main__":
    train_models()
