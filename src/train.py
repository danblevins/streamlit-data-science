"""
Train and compare multiple classifiers for prompt complexity prediction.
Saves models, metrics, and test predictions for interactive visualization.
"""
import os, sys, json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.features import extract_features

os.makedirs("models", exist_ok=True)

CLASS_ORDER = ["low", "medium", "high"]


def load_and_prepare():
    df = pd.read_csv('data/prompt_examples_dataset.csv')
    X = extract_features(df)
    y = df['complexity']
    le = LabelEncoder()
    le.fit(CLASS_ORDER)
    y_enc = le.transform(y)
    return X, y, y_enc, le


def build_models():
    return {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, C=1.0, random_state=42, class_weight='balanced'))
        ]),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=15, min_samples_leaf=5,
            random_state=42, class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            random_state=42, class_weight='balanced', n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5,
            random_state=42
        ),
        'MLP': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=300,
                random_state=42, early_stopping=True, validation_fraction=0.1
            ))
        ]),
        'SVM (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=5.0, gamma='scale', random_state=42,
                        class_weight='balanced', probability=True))
        ]),
        'K-Nearest Neighbors': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=7, weights='distance'))
        ]),
    }


def cross_validate_models(models, X, y_enc):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y_enc, cv=cv, scoring="f1_macro", n_jobs=-1)
        cv_results[name] = scores
        print(f"  {name}: {scores.mean():.4f} ± {scores.std():.4f}")
    return cv_results


def evaluate_best_model(best_model, best_name, X_test, y_test_enc, y_test_str, le):
    y_pred_enc = best_model.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)

    print(f"\n=== {best_name} — Test Set Evaluation ===")
    print(classification_report(y_test_str, y_pred, target_names=CLASS_ORDER))

    acc = accuracy_score(y_test_str, y_pred)
    f1 = f1_score(y_test_str, y_pred, average="macro")
    prec = precision_score(y_test_str, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test_str, y_pred, average="macro")

    metrics = {
        "accuracy": acc,
        "macro_f1": f1,
        "macro_precision": prec,
        "macro_recall": rec,
    }

    # Confusion matrix (counts and percentages) for downstream visualization
    cm = confusion_matrix(y_test_str, y_pred, labels=CLASS_ORDER)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    print("Confusion matrix (counts):")
    print(cm)
    print("\nConfusion matrix (% by true class):")
    print(cm_pct.round(1))

    return metrics, y_pred


def main():
    print("Loading data & extracting features...")
    X, y, y_enc, le = load_and_prepare()
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test, y_train_enc, y_test_enc = train_test_split(
        X, y, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    models = build_models()

    print("\nRunning 5-fold cross-validation...")
    cv_results = cross_validate_models(models, X_train, y_train_enc)

    # Select best model by CV mean F1
    best_name = max(cv_results, key=lambda n: cv_results[n].mean())
    print(f"\nBest model: {best_name} (CV F1: {cv_results[best_name].mean():.4f})")

    # Train all models on full train set
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train_enc)
        trained_models[name] = model

    best_model = trained_models[best_name]

    # Evaluate
    metrics, y_pred = evaluate_best_model(
        best_model, best_name, X_test, y_test_enc, y_test, le
    )

    # Save test predictions for interactive visualization in the Streamlit app
    test_df = pd.DataFrame(
        {
            "true_complexity": y_test.values,
            "pred_complexity": y_pred,
        }
    )
    test_df.to_csv("models/test_predictions.csv", index=False)
    print("Saved: models/test_predictions.csv")

    # Also evaluate all models for comparison table
    all_metrics = {}
    for name, model in trained_models.items():
        yp = le.inverse_transform(model.predict(X_test))
        all_metrics[name] = {
            "Test Accuracy": accuracy_score(y_test, yp),
            "Macro F1": f1_score(y_test, yp, average="macro"),
            "CV F1 Mean": cv_results[name].mean(),
            "CV F1 Std": cv_results[name].std(),
        }

    results_df = pd.DataFrame(all_metrics).T.round(4)
    results_df.to_csv("models/model_comparison.csv")
    print("\nModel comparison:")
    print(results_df.sort_values('Macro F1', ascending=False).to_string())

    # Save best model and label encoder
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(le, "models/label_encoder.pkl")
    joblib.dump(feature_names, "models/feature_names.pkl")

    # Save all trained models
    for name, model in trained_models.items():
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(model, f"models/{safe_name}.pkl")

    # Save metrics
    with open("models/best_model_metrics.json", "w") as f:
        json.dump({"model": best_name, **metrics}, f, indent=2)

    print(f"\nAll models saved to models/")
    print(f"Best model: {best_name}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}, Macro F1: {metrics['macro_f1']:.4f}")


if __name__ == '__main__':
    main()
