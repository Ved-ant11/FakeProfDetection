import pandas as pd
import numpy as np
import os
import json
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def train_models():
    input_path = r"e:\botDetect\data\processed\cresci_expanded_with_content.csv"
    model_dir = r"e:\botDetect\models"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)

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

    target_col = 'fake'

    X = df[feature_cols]
    y = df[target_col]

    print(f"Features: {X.columns.tolist()}")
    print(f"Target distribution:\n{y.value_counts()}")

    print("\n--- Cross-Validation (5-Fold Stratified) ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rf_cv_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    rf_cv_scores = cross_val_score(rf_cv_pipeline, X, y, cv=cv, scoring='f1')
    print(f"Random Forest     - F1: {rf_cv_scores.mean():.4f} +/- {rf_cv_scores.std():.4f}  {rf_cv_scores}")

    svm_cv_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', probability=True, random_state=42))
    ])
    svm_cv_scores = cross_val_score(svm_cv_pipeline, X, y, cv=cv, scoring='f1')
    print(f"SVM               - F1: {svm_cv_scores.mean():.4f} +/- {svm_cv_scores.std():.4f}  {svm_cv_scores}")

    gbt_cv_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))
    ])
    gbt_cv_scores = cross_val_score(gbt_cv_pipeline, X, y, cv=cv, scoring='f1')
    print(f"Gradient Boosting - F1: {gbt_cv_scores.mean():.4f} +/- {gbt_cv_scores.std():.4f}  {gbt_cv_scores}")

    cv_results = {
        'Random Forest': {'mean': float(rf_cv_scores.mean()), 'std': float(rf_cv_scores.std()), 'folds': rf_cv_scores.tolist()},
        'SVM': {'mean': float(svm_cv_scores.mean()), 'std': float(svm_cv_scores.std()), 'folds': svm_cv_scores.tolist()},
        'Gradient Boosting': {'mean': float(gbt_cv_scores.mean()), 'std': float(gbt_cv_scores.std()), 'folds': gbt_cv_scores.tolist()}
    }
    with open(os.path.join(model_dir, 'cv_results.json'), 'w') as f:
        json.dump(cv_results, f, indent=2)
    print("Cross-validation results saved to models/cv_results.json")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\nBefore SMOTE - Training set distribution:\n{y_train.value_counts()}")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE - Training set distribution:\n{pd.Series(y_train_resampled).value_counts()}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    print("Scaler saved.")

    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_resampled, y_train_resampled)
    y_pred_rf = rf_model.predict(X_test)
    print("Random Forest Results:")
    print(classification_report(y_test, y_pred_rf))
    print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    joblib.dump(rf_model, os.path.join(model_dir, 'rf_model.pkl'))

    print("\nTraining SVM...")
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train_resampled)
    y_pred_svm = svm_model.predict(X_test_scaled)
    print("SVM Results:")
    print(classification_report(y_test, y_pred_svm))
    print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
    joblib.dump(svm_model, os.path.join(model_dir, 'svm_model.pkl'))

    print("\nTraining Gradient Boosting...")
    gbt_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    gbt_model.fit(X_train_resampled, y_train_resampled)
    y_pred_gbt = gbt_model.predict(X_test)
    print("Gradient Boosting Results:")
    print(classification_report(y_test, y_pred_gbt))
    print(f"Accuracy: {accuracy_score(y_test, y_pred_gbt):.4f}")
    joblib.dump(gbt_model, os.path.join(model_dir, 'gbt_model.pkl'))

    rf_importances = rf_model.feature_importances_
    gbt_importances = gbt_model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'rf_importance': rf_importances,
        'gbt_importance': gbt_importances
    })
    importance_df['avg_importance'] = (importance_df['rf_importance'] + importance_df['gbt_importance']) / 2
    importance_df = importance_df.sort_values('avg_importance', ascending=False)
    importance_df.to_csv(os.path.join(model_dir, 'feature_importances.csv'), index=False)
    print("\nFeature importances saved to models/feature_importances.csv")
    print(importance_df.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    rf_sorted = importance_df.sort_values('rf_importance', ascending=True)
    axes[0].barh(rf_sorted['feature'], rf_sorted['rf_importance'], color='#2196F3')
    axes[0].set_title('Random Forest Feature Importance')
    axes[0].set_xlabel('Importance')
    gbt_sorted = importance_df.sort_values('gbt_importance', ascending=True)
    axes[1].barh(gbt_sorted['feature'], gbt_sorted['gbt_importance'], color='#FF5722')
    axes[1].set_title('Gradient Boosting Feature Importance')
    axes[1].set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'feature_importances.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Feature importance chart saved.")

    print("\nGenerating ROC curves...")
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    svm_proba = svm_model.predict_proba(X_test_scaled)[:, 1]
    gbt_proba = gbt_model.predict_proba(X_test)[:, 1]

    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_proba)
    svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_proba)
    gbt_fpr, gbt_tpr, _ = roc_curve(y_test, gbt_proba)

    rf_auc = auc(rf_fpr, rf_tpr)
    svm_auc = auc(svm_fpr, svm_tpr)
    gbt_auc = auc(gbt_fpr, gbt_tpr)

    roc_data = {
        'rf_fpr': rf_fpr.tolist(), 'rf_tpr': rf_tpr.tolist(), 'rf_auc': float(rf_auc),
        'svm_fpr': svm_fpr.tolist(), 'svm_tpr': svm_tpr.tolist(), 'svm_auc': float(svm_auc),
        'gbt_fpr': gbt_fpr.tolist(), 'gbt_tpr': gbt_tpr.tolist(), 'gbt_auc': float(gbt_auc)
    }
    with open(os.path.join(model_dir, 'roc_data.json'), 'w') as f:
        json.dump(roc_data, f)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rf_fpr, rf_tpr, color='#2196F3', lw=2, label=f'Random Forest (AUC = {rf_auc:.4f})')
    ax.plot(svm_fpr, svm_tpr, color='#4CAF50', lw=2, label=f'SVM (AUC = {svm_auc:.4f})')
    ax.plot(gbt_fpr, gbt_tpr, color='#FF5722', lw=2, label=f'Gradient Boosting (AUC = {gbt_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Baseline')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - Model Comparison')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'roc_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ROC curves saved. AUC - RF: {rf_auc:.4f}, SVM: {svm_auc:.4f}, GBT: {gbt_auc:.4f}")

    print("\nGenerating SHAP summary plot...")
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif len(shap_values.shape) == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(sv, X_test, feature_names=feature_cols, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'shap_summary.png'), dpi=150, bbox_inches='tight')
    plt.close('all')
    print("SHAP summary plot saved to models/shap_summary.png")

    print("\nAll models trained and saved.")


if __name__ == "__main__":
    train_models()