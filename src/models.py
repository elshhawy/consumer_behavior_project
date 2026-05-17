import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import os
import warnings
warnings.filterwarnings('ignore')

from preprocessing import (load_data, preprocess_data,
                           plot_correlation_matrix, FIGURES_PATH, REPORTS_PATH)

# ── Define Models ───────────────────────────────────────
def get_models():
    return {
        "SVM": SVC(
           kernel='rbf',
           C=50.12,
           gamma=0.120,
           probability=True,
           random_state=42

        ),
        "XGBoost": XGBClassifier(
            max_depth=4,
            learning_rate=0.11,
            subsample=0.8,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        ),
        "CatBoost": CatBoostClassifier(
            iterations=245,
            learning_rate=0.09,
            depth=8,
            random_state=42,
            verbose=0
        ),
        "BPANN": MLPClassifier(
            hidden_layer_sizes=(100, 100, 100),
            max_iter=50,
            random_state=42
        )
    }

# ── Plot Confusion Matrix ───────────────────────────────
def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Purchase', 'Purchase'],
                yticklabels=['No Purchase', 'Purchase'])
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"{FIGURES_PATH}/confusion_matrix_{model_name.lower()}.png",
                dpi=300, bbox_inches='tight')
    plt.show()
    print(f" Confusion Matrix saved for {model_name}")

# ── Plot ROC Curves ─────────────────────────────────────
def plot_roc_curves(models_results):
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'green', 'red', 'orange']
    for (name, result), color in zip(models_results.items(), colors):
        plt.plot(result['fpr'], result['tpr'], color=color,
                 label=f"{name} (AUC = {result['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - All Models', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_PATH}/roc_curves_comparison.png",
                dpi=300, bbox_inches='tight')
    plt.show()
    print(" ROC Curves saved!")

# ── Plot Feature Importance ─────────────────────────────
def plot_feature_importance(model, feature_names):
    importance = model.get_feature_importance()
    feat_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(feat_df['Feature'], feat_df['Importance'], color='salmon')
    plt.title('CatBoost Feature Importance Ranking', fontsize=14, fontweight='bold')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(f"{FIGURES_PATH}/feature_importance.png",
                dpi=300, bbox_inches='tight')
    plt.show()
    print(" Feature Importance saved!")

# ── Train & Evaluate ────────────────────────────────────
def train_and_evaluate(models, X_train, X_test, y_train, y_test, feature_names):
    results = {}
    comparison_data = []
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f" Training {name}...")

        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        report = classification_report(y_test, y_pred,
                                       target_names=['No Purchase', 'Purchase'])
        cm = confusion_matrix(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        accuracy = accuracy_score(y_test, y_pred)

        # 10-Fold Cross Validation
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=kfold, scoring='roc_auc')

        print(f"\n Classification Report - {name}:")
        print(report)
        print(f" ROC AUC: {roc_auc:.3f}")
        print(f" Accuracy: {accuracy:.3f}")
        print(f" 10-Fold CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        # Save report
        with open(f"{REPORTS_PATH}/classification_report_{name.lower()}.txt", 'w') as f:
            f.write(f"Classification Report - {name}\n")
            f.write("="*50 + "\n")
            f.write(report)
            f.write(f"\nROC AUC: {roc_auc:.3f}")
            f.write(f"\nAccuracy: {accuracy:.3f}")
            f.write(f"\n10-Fold CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        # Confusion Matrix
        plot_confusion_matrix(cm, name)

        # Store results
        results[name] = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}
        comparison_data.append({
            'Model': name,
            'Accuracy': round(accuracy, 3),
            'ROC AUC': round(roc_auc, 3),
            'CV AUC Mean': round(cv_scores.mean(), 3),
            'CV AUC Std': round(cv_scores.std(), 3)
        })

        # Feature Importance for CatBoost
        if name == "CatBoost":
            plot_feature_importance(model, feature_names)

    # ROC Curves
    plot_roc_curves(results)

    # Save Comparison Table
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(f"{REPORTS_PATH}/models_comparison.csv", index=False)
    print("\n" + "="*50)
    print(" Final Comparison Table:")
    print(comparison_df.to_string(index=False))

    return results, comparison_df

# ── Main ────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_data(df)
    models = get_models()
    results, comparison_df = train_and_evaluate(
        models, X_train, X_test, y_train, y_test, feature_names
    )