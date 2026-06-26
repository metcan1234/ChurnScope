import os

# Paths
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "pipeline.pkl")
METRICS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "metrics.json")
GLOBAL_SHAP_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "global_shap.json")

# Training params
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Model params
GB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "random_state": 42,
    "subsample": 0.8,
}

# Feature lists (filled in preprocessing.py)
NUMERIC_FEATURES = []
CATEGORICAL_FEATURES = []