#!/usr/bin/env python3
"""
ChurnScope - Training Pipeline

Usage:
    python train_pipeline.py

Steps:
1. Load and preprocess data
2. Train and evaluate pipeline
3. Save pipeline and metrics
4. Compute global SHAP importance
"""
import json
import joblib
from sklearn.model_selection import train_test_split

from src.config import DATA_PATH, MODEL_PATH, METRICS_PATH, GLOBAL_SHAP_PATH, RANDOM_STATE, TEST_SIZE
from src.preprocessing import load_and_preprocess
from src.train import train_and_evaluate, build_pipeline
from src.explainer import ShapExplainer


def main():
    print("=" * 60)
    print("ChurnScope - Training Pipeline")
    print("=" * 60)

    # Step 1: Load and preprocess
    print("\n[1/5] Loading and preprocessing data...")
    X, y = load_and_preprocess(DATA_PATH)
    print(f"    X shape: {X.shape}, y shape: {y.shape}")
    print(f"    Churn rate: {y.mean():.3f}")

    # Step 2: Split into train/test for explainer
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Step 3: Train and evaluate
    print("\n[2/5] Training pipeline...")
    pipeline, metrics = train_and_evaluate(X, y)
    print(f"    Accuracy:  {metrics['accuracy']:.4f}")
    print(f"    ROC-AUC:   {metrics['roc_auc']:.4f}")

    # Step 4: Save pipeline
    print("\n[3/5] Saving pipeline...")
    joblib.dump(pipeline, MODEL_PATH)
    print(f"    Saved to: {MODEL_PATH}")

    # Step 5: Save metrics
    print("\n[4/5] Saving metrics and classification report...")
    # Remove large arrays from metrics before saving as JSON
    metrics_json = {
        "accuracy": metrics["accuracy"],
        "roc_auc": metrics["roc_auc"],
        "classification_report": metrics["classification_report"],
        "confusion_matrix": metrics["confusion_matrix"],
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"    Saved to: {METRICS_PATH}")

    # Print classification report
    print("\nClassification Report:")
    print("-" * 50)
    from sklearn.metrics import classification_report as cr
    y_pred = pipeline.predict(X_test)
    print(cr(y_test, y_pred))

    # Step 6: Global SHAP importance
    print("\n[5/5] Computing global SHAP importance...")
    explainer = ShapExplainer(pipeline, X_train)
    global_shap = explainer.global_importance(top_n=15)
    global_shap.to_json(GLOBAL_SHAP_PATH, orient="records", indent=2)
    print(f"    Saved to: {GLOBAL_SHAP_PATH}")
    print("\nTop 10 features by SHAP importance:")
    print(global_shap.head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("Training pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()