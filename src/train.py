import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE
import joblib
import os

def train_models():
    data_path = 'data/processed/research_data.csv'
    if not os.path.exists(data_path):
        print("Run src/process_data.py first!")
        return

    df = pd.read_csv(data_path)
    X = df.drop('fake', axis=1)
    y = df['fake']

    # 1. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Apply SMOTE (Research Requirement)
    print(f"Original Training Balance: {y_train.value_counts().to_dict()}")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"SMOTE Balanced Balance: {y_train_res.value_counts().to_dict()}")

    # 3. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')

    # 4. Train Random Forest (Ensemble)
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_res, y_train_res)
    print(f"RF F1 Score: {f1_score(y_test, rf.predict(X_test)):.4f}")
    joblib.dump(rf, 'models/rf_model.pkl')

    # 5. Train SVM
    print("Training SVM...")
    svm = SVC(kernel='rbf', probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train_res)
    print(f"SVM F1 Score: {f1_score(y_test, svm.predict(X_test_scaled)):.4f}")
    joblib.dump(svm, 'models/svm_model.pkl')
    
    # Save test data for SHAP in App
    X_test.to_csv('data/processed/X_test.csv', index=False)

if __name__ == "__main__":
    train_models()