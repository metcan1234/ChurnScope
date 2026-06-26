import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

from src import config


def build_pipeline() -> Pipeline:
    """Build an sklearn Pipeline with:
    - ColumnTransformer: StandardScaler(numeric) + OneHotEncoder(categorical)
    - GradientBoostingClassifier with GB_PARAMS

    Returns:
        Pipeline (unfitted)
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, config.NUMERIC_FEATURES),
            ("cat", categorical_transformer, config.CATEGORICAL_FEATURES),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", GradientBoostingClassifier(**config.GB_PARAMS)),
        ]
    )

    return pipeline


def train_and_evaluate(X: pd.DataFrame, y: pd.Series) -> tuple[Pipeline, dict]:
    """Split data, train pipeline, compute metrics on test set.

    Args:
        X: feature DataFrame
        y: target Series

    Returns:
        (fitted_pipeline, metrics_dict)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "classification_report": report,
        "confusion_matrix": cm,
        "y_test": y_test.tolist(),
        "y_proba": y_proba.tolist(),
    }

    return pipeline, metrics