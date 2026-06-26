import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


class ShapExplainer:
    """Wrapper around SHAP TreeExplainer for the churn pipeline.

    Usage:
        explainer = ShapExplainer(pipeline, X_train)
        global_df = explainer.global_importance(top_n=15)
        local_df = explainer.local_explanation(single_row_df)
    """

    def __init__(self, pipeline: Pipeline, X_train: pd.DataFrame):
        """Initialize the explainer.

        Extracts the preprocessor, transforms X_train, builds a TreeExplainer,
        and computes global SHAP values.

        Args:
            pipeline: Fitted sklearn Pipeline with "preprocessor" and "model" steps
            X_train: Training DataFrame (raw, before preprocessing)
        """
        self.preprocessor = pipeline.named_steps["preprocessor"]
        self.model = pipeline.named_steps["model"]

        # Transform training data
        self.X_train_transformed = self.preprocessor.transform(X_train)

        # Get feature names and strip prefixes
        raw_names = self.preprocessor.get_feature_names_out()
        self.feature_names = np.array([name.split("__", 1)[1] for name in raw_names])

        # Build TreeExplainer
        self.explainer = shap.TreeExplainer(self.model)

        # Compute global SHAP values
        shap_vals = self.explainer.shap_values(self.X_train_transformed)

        # Handle both list (binary classification) and ndarray output
        if isinstance(shap_vals, list):
            # shap_values returns [negative_class, positive_class] for binary
            # We take index [1] for the positive class
            self.shap_values = shap_vals[1]
        else:
            self.shap_values = shap_vals

    def global_importance(self, top_n: int = 15) -> pd.DataFrame:
        """Return top_n features by mean |SHAP value|.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with columns ["feature", "importance"] sorted descending
        """
        mean_abs_shap = np.mean(np.abs(self.shap_values), axis=0)
        indices = np.argsort(mean_abs_shap)[::-1][:top_n]

        result = pd.DataFrame({
            "feature": self.feature_names[indices],
            "importance": mean_abs_shap[indices],
        })
        return result.reset_index(drop=True)

    def local_explanation(self, x_raw: pd.DataFrame) -> pd.DataFrame:
        """Compute SHAP values for a single raw input row.

        Args:
            x_raw: Single-row DataFrame with same columns as original X

        Returns:
            DataFrame with columns ["feature", "shap_value"] sorted by |shap_value| desc
        """
        # Transform single row
        x_transformed = self.preprocessor.transform(x_raw)

        # Compute SHAP values for this row
        shap_vals = self.explainer.shap_values(x_transformed)

        # Handle list output
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        # Build result DataFrame
        result = pd.DataFrame({
            "feature": self.feature_names,
            "shap_value": shap_vals[0],
        })
        result["abs_shap"] = np.abs(result["shap_value"])
        result = result.sort_values("abs_shap", ascending=False).reset_index(drop=True)
        result = result.drop(columns=["abs_shap"])
        return result