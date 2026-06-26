import pandas as pd
import numpy as np
from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def load_and_preprocess(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load and preprocess the Telco Customer Churn dataset.

    Cleaning steps:
    1. Drop customerID column
    2. TotalCharges: convert to numeric, drop NaNs
    3. Churn: map Yes→1, No→0

    Feature engineering:
    - charges_per_month = TotalCharges / (tenure + 1)
    - is_long_term = Contract in ["One year", "Two year"]
    - service_count = sum of binary service columns

    Sets NUMERIC_FEATURES and CATEGORICAL_FEATURES in config module.

    Returns:
        X (pd.DataFrame): feature matrix
        y (pd.Series): target (binary)
    """
    df = pd.read_csv(path)

    # --- Cleaning ---
    # 1. Drop customerID
    df = df.drop(columns=["customerID"])

    # 2. TotalCharges to numeric, drop NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)

    # 3. Churn mapping
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # --- Feature Engineering ---
    # charges_per_month
    df["charges_per_month"] = df["TotalCharges"] / (df["tenure"] + 1)

    # is_long_term
    df["is_long_term"] = df["Contract"].isin(["One year", "Two year"]).astype(int)

    # service_count: sum of binary service columns
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    for col in service_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})
    df["service_count"] = df[service_cols].sum(axis=1)

    # --- Separate X and y ---
    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    # --- Populate feature lists in config ---
    global NUMERIC_FEATURES, CATEGORICAL_FEATURES
    from src import config

    config.NUMERIC_FEATURES = [
        "tenure", "MonthlyCharges", "TotalCharges",
        "charges_per_month", "service_count",
    ]

    # CATEGORICAL_FEATURES: all object dtype columns except Churn (already dropped)
    config.CATEGORICAL_FEATURES = list(X.select_dtypes(include=["object"]).columns)

    return X, y