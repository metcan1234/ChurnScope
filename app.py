#!/usr/bin/env python3
"""
ChurnScope - Streamlit Dashboard

Usage:
    streamlit run app.py

Requires trained pipeline at models/pipeline.pkl (run train_pipeline.py first).
"""
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_curve, auc

from src.config import MODEL_PATH, METRICS_PATH, GLOBAL_SHAP_PATH
from src.explainer import ShapExplainer

# Page config
st.set_page_config(
    page_title="ChurnScope",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Shared styles
PLOTLY_TEMPLATE = "plotly_white"
PLOTLY_BGCOLOR = "white"


@st.cache_resource
def load_pipeline():
    """Load the saved pipeline and explainer."""
    import joblib
    pipeline = joblib.load(MODEL_PATH)
    return pipeline


def load_metrics():
    """Load saved metrics from JSON."""
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


def load_global_shap():
    """Load global SHAP importance from JSON."""
    import pandas as pd
    return pd.read_json(GLOBAL_SHAP_PATH)


def compute_features_for_input(input_df: pd.DataFrame) -> pd.DataFrame:
    """Compute engineered features mirroring preprocessing.py logic.

    Args:
        input_df: Single-row DataFrame with raw input values

    Returns:
        Single-row DataFrame with added engineered features
    """
    df = input_df.copy()

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

    return df


def main():
    # Load resources
    with st.spinner("Loading pipeline..."):
        pipeline = load_pipeline()
        metrics = load_metrics()
        global_shap_df = load_global_shap()

    # Initialize explainer with training data?
    # We'll compute on-the-fly for local explanations
    # Cache the explainer
    @st.cache_resource
    def get_explainer():
        import joblib
        from sklearn.model_selection import train_test_split
        from src.config import RANDOM_STATE, TEST_SIZE
        from src.preprocessing import load_and_preprocess
        from src.config import DATA_PATH

        X, y = load_and_preprocess(DATA_PATH)
        X_train, _, _, _ = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        pl = joblib.load(MODEL_PATH)
        return ShapExplainer(pl, X_train)

    explainer = get_explainer()

    # Title
    st.title("📊 ChurnScope - Müşteri Kayıp Tahmin Sistemi")
    st.markdown("---")

    # Tabs
    tab1, tab2 = st.tabs(["🔍 Müşteri Analizi", "📈 Model Performansı"])

    # =========================================
    # TAB 1 - Customer Analysis
    # =========================================
    with tab1:
        col_left, col_right = st.columns([1, 1.5])

        with col_left:
            st.subheader("Müşteri Bilgileri")

            # Account info
            st.markdown("**📋 Hesap Bilgileri**")
            tenure = st.slider("Tenure (ay)", 0, 72, 12, key="tenure")
            monthly_charges = st.slider("MonthlyCharges ($)", 0.0, 120.0, 50.0, step=0.5, key="monthly_charges")
            total_charges = st.slider("TotalCharges ($)", 0.0, 8700.0, 500.0, step=10.0, key="total_charges")
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="contract")

            # Services
            st.markdown("**🛠️ Hizmetler**")
            col_svc1, col_svc2 = st.columns(2)
            with col_svc1:
                phone_service = st.selectbox("PhoneService", ["Yes", "No"], key="phone_service")
                multiple_lines = st.selectbox("MultipleLines", ["Yes", "No"], key="multiple_lines")
                online_security = st.selectbox("OnlineSecurity", ["Yes", "No"], key="online_security")
                online_backup = st.selectbox("OnlineBackup", ["Yes", "No"], key="online_backup")
            with col_svc2:
                device_protection = st.selectbox("DeviceProtection", ["Yes", "No"], key="device_protection")
                tech_support = st.selectbox("TechSupport", ["Yes", "No"], key="tech_support")
                streaming_tv = st.selectbox("StreamingTV", ["Yes", "No"], key="streaming_tv")
                streaming_movies = st.selectbox("StreamingMovies", ["Yes", "No"], key="streaming_movies")

            # Demographics
            st.markdown("**👤 Demografik Bilgiler**")
            col_dem1, col_dem2 = st.columns(2)
            with col_dem1:
                gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
                senior_citizen = st.selectbox("SeniorCitizen", ["0", "1"], key="senior_citizen")
            with col_dem2:
                partner = st.selectbox("Partner", ["Yes", "No"], key="partner")
                dependents = st.selectbox("Dependents", ["Yes", "No"], key="dependents")

            # Payment
            st.markdown("**💳 Ödeme Bilgileri**")
            payment_method = st.selectbox(
                "PaymentMethod",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                key="payment_method",
            )
            paperless_billing = st.selectbox("PaperlessBilling", ["Yes", "No"], key="paperless_billing")

            # Predict button
            st.markdown("---")
            predict_btn = st.button("🚀 Tahmin Et", use_container_width=True, type="primary")

        # Right column - Results
        with col_right:
            if predict_btn:
                # Build input DataFrame matching original columns
                input_data = {
                    "gender": [gender],
                    "SeniorCitizen": [senior_citizen],
                    "Partner": [partner],
                    "Dependents": [dependents],
                    "tenure": [tenure],
                    "PhoneService": [phone_service],
                    "MultipleLines": [multiple_lines],
                    "OnlineSecurity": [online_security],
                    "OnlineBackup": [online_backup],
                    "DeviceProtection": [device_protection],
                    "TechSupport": [tech_support],
                    "StreamingTV": [streaming_tv],
                    "StreamingMovies": [streaming_movies],
                    "Contract": [contract],
                    "PaperlessBilling": [paperless_billing],
                    "PaymentMethod": [payment_method],
                    "MonthlyCharges": [monthly_charges],
                    "TotalCharges": [total_charges],
                }
                input_df = pd.DataFrame(input_data)

                # Compute engineered features
                input_df = compute_features_for_input(input_df)

                # Ensure columns match training order (get from pipeline's preprocessor)
                # Predict probability
                proba = pipeline.predict_proba(input_df)[0, 1]
                churn_pct = proba * 100

                # Display churn probability
                is_high_risk = churn_pct > 50
                metric_color = "inverse" if is_high_risk else "normal"
                delta_color = "off"  # no delta

                st.subheader("📊 Tahmin Sonucu")

                # Custom colored metric using markdown
                risk_label = "🔴 Yüksek Risk" if is_high_risk else "🟢 Düşük Risk"
                st.markdown(f"### {risk_label}")

                # Display probability as a large metric
                st.metric(
                    label="Kayıp Olasılığı (Churn Probability)",
                    value=f"%{churn_pct:.1f}",
                    delta=None,
                )

                # Color the metric using custom CSS
                metric_value_style = f"""
                <style>
                div[data-testid="metric-container"] {{
                    background-color: {"#ffebee" if is_high_risk else "#e8f5e9"};
                    border-radius: 10px;
                    padding: 15px;
                    border: 2px solid {"#ef5350" if is_high_risk else "#66bb6a"};
                }}
                div[data-testid="metric-container"] label {{
                    color: {"#c62828" if is_high_risk else "#2e7d32"} !important;
                    font-size: 1rem !important;
                }}
                div[data-testid="metric-container"] div[data-testid="metric-value"] {{
                    color: {"#c62828" if is_high_risk else "#2e7d32"} !important;
                    font-size: 3rem !important;
                    font-weight: bold !important;
                }}
                </style>
                """
                st.markdown(metric_value_style, unsafe_allow_html=True)

                # Prediction interpretation
                if is_high_risk:
                    st.warning("Bu müşterinin kaybetme riski yüksektir. Önleyici aksiyon alınması önerilir.")
                else:
                    st.success("Bu müşterinin kaybetme riski düşüktür.")

                # Local SHAP explanation
                st.markdown("---")
                st.subheader("🔍 SHAP Açıklaması (Katkı Faktörleri)")

                local_shap = explainer.local_explanation(input_df)

                # Top 12 features
                top_local = local_shap.head(12).copy()

                # Assign colors: positive (red) increases churn risk, negative (green) decreases
                top_local["color"] = top_local["shap_value"].apply(
                    lambda x: "#ef5350" if x > 0 else "#66bb6a"
                )
                top_local["abs_val"] = top_local["shap_value"].abs()

                # Sort by abs value for display (already sorted)
                fig_local = go.Figure()

                fig_local.add_trace(go.Bar(
                    y=top_local["feature"],
                    x=top_local["shap_value"],
                    orientation="h",
                    marker=dict(color=top_local["color"]),
                    text=top_local["shap_value"].round(3),
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>SHAP Value: %{x:.4f}<extra></extra>",
                ))

                fig_local.update_layout(
                    title="SHAP Değerleri (Kırmızı: Kaybı Artırır, Yeşil: Azaltır)",
                    xaxis_title="SHAP Value",
                    yaxis_title=None,
                    template=PLOTLY_TEMPLATE,
                    plot_bgcolor=PLOTLY_BGCOLOR,
                    paper_bgcolor=PLOTLY_BGCOLOR,
                    height=400,
                    margin=dict(l=10, r=30, t=40, b=10),
                    yaxis=dict(
                        autorange="reversed",
                    ),
                )

                st.plotly_chart(fig_local, use_container_width=True)
            else:
                # Placeholder
                st.info("👈 Sol taraftaki müşteri bilgilerini doldurun ve 'Tahmin Et' butonuna tıklayın.")
                st.markdown("""
                ### Nasıl Kullanılır?
                1. **Müşteri Bilgilerini Girin**: Sol paneldeki alanları doldurun
                2. **Tahmin Et**: Butona tıklayarak modelin tahminini görün
                3. **İçgörüler**: SHAP değerleriyle hangi faktörlerin etkili olduğunu keşfedin
                """)

    # =========================================
    # TAB 2 - Model Performance
    # =========================================
    with tab2:
        st.subheader("📈 Model Performans Metrikleri")

        # Extract metrics
        accuracy = metrics["accuracy"]
        roc_auc_val = metrics["roc_auc"]
        class_report = metrics["classification_report"]
        cm = metrics["confusion_matrix"]

        # F1-score for churn class (class "1")
        f1_churn = class_report.get("1", {}).get("f1-score", 0)

        # Row 1: Metric cards
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric(label="🎯 Accuracy", value=f"{accuracy:.3f}")
        with col_m2:
            st.metric(label="📈 ROC-AUC", value=f"{roc_auc_val:.3f}")
        with col_m3:
            st.metric(label="⭐ F1-Score (Churn)", value=f"{f1_churn:.3f}")

        # Row 2: Confusion matrix + ROC curve
        col_cm, col_roc = st.columns(2)

        with col_cm:
            st.subheader("Confusion Matrix")

            # Compute percentages for annotation
            cm_array = np.array(cm)
            cm_sum = cm_array.sum()
            cm_pct = cm_array / cm_sum * 100

            # Create annotation text
            annotations = []
            for i in range(2):
                for j in range(2):
                    annotations.append(
                        dict(
                            x=j,
                            y=i,
                            text=f"{cm_array[i][j]}<br>({cm_pct[i][j]:.1f}%)",
                            showarrow=False,
                            font=dict(size=14, color="white"),
                        )
                    )

            fig_cm = go.Figure(data=go.Heatmap(
                z=cm_array,
                x=["Tahmin: Hayır (0)", "Tahmin: Evet (1)"],
                y=["Gerçek: Hayır (0)", "Gerçek: Evet (1)"],
                colorscale="Blues",
                showscale=False,
                texttemplate="%{z}",
                textfont=dict(size=14, color="white"),
                hovertemplate="<b>%{y}</b><br><b>%{x}</b><br>Değer: %{z}<extra></extra>",
            ))

            fig_cm.update_layout(
                title="Confusion Matrix (Test Seti)",
                xaxis_title="Tahmin Edilen",
                yaxis_title="Gerçek Değer",
                template=PLOTLY_TEMPLATE,
                plot_bgcolor=PLOTLY_BGCOLOR,
                paper_bgcolor=PLOTLY_BGCOLOR,
                height=400,
                xaxis=dict(side="bottom"),
                yaxis=dict(autorange="reversed"),
                annotations=annotations,
            )

            st.plotly_chart(fig_cm, use_container_width=True)

        with col_roc:
            st.subheader("ROC Eğrisi")

            # Compute ROC curve from stored test data if available, otherwise synthetic
            # The metrics don't store full y_test/y_proba, so compute from the saved pipeline
            # Load original data to recompute ROC
            with st.spinner("ROC Eğrisi hesaplanıyor..."):
                import joblib
                from sklearn.model_selection import train_test_split
                from src.config import RANDOM_STATE, TEST_SIZE
                from src.preprocessing import load_and_preprocess
                from src.config import DATA_PATH

                X_full, y_full = load_and_preprocess(DATA_PATH)
                _, X_test_roc, _, y_test_roc = train_test_split(
                    X_full, y_full, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_full
                )
                pl = joblib.load(MODEL_PATH)
                y_scores = pl.predict_proba(X_test_roc)[:, 1]

            fpr, tpr, thresholds = roc_curve(y_test_roc, y_scores)
            roc_auc_calc = auc(fpr, tpr)

            fig_roc = go.Figure()

            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode="lines",
                name=f"ROC (AUC = {roc_auc_calc:.3f})",
                line=dict(color="#1565c0", width=3),
                hovertemplate="False Positive Rate: %{x:.3f}<br>True Positive Rate: %{y:.3f}<extra></extra>",
            ))

            # Diagonal line
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                name="Random",
                line=dict(color="gray", dash="dash", width=1),
                showlegend=False,
            ))

            fig_roc.update_layout(
                title=f"ROC Eğrisi (AUC = {roc_auc_calc:.3f})",
                xaxis_title="False Positive Rate (1 - Specificity)",
                yaxis_title="True Positive Rate (Sensitivity)",
                template=PLOTLY_TEMPLATE,
                plot_bgcolor=PLOTLY_BGCOLOR,
                paper_bgcolor=PLOTLY_BGCOLOR,
                height=400,
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
                legend=dict(x=0.7, y=0.2),
                shapes=[
                    dict(
                        type="line",
                        x0=0, y0=0, x1=1, y1=1,
                        line=dict(color="gray", dash="dash", width=1),
                    )
                ],
            )

            st.plotly_chart(fig_roc, use_container_width=True)

        # Row 3: Classification report
        st.subheader("Sınıflandırma Raporu (Classification Report)")

        # Build a nice dataframe from classification report
        report_data = []
        for class_label, class_metrics in class_report.items():
            if isinstance(class_metrics, dict):
                row = {"Class": class_label}
                row.update(class_metrics)
                report_data.append(row)

        report_df = pd.DataFrame(report_data)

        # Filter to show only meaningful rows
        report_df = report_df[report_df["Class"].isin(["0", "1", "accuracy"])]

        if "accuracy" in report_df["Class"].values:
            # For accuracy row, some fields are missing; handle gracefully
            report_df = report_df.fillna("")

        st.dataframe(report_df, use_container_width=True)

        # Row 4: Global SHAP importance
        st.subheader("Global SHAP Feature Importance (Top 15)")

        fig_global = go.Figure()

        fig_global.add_trace(go.Bar(
            y=global_shap_df["feature"],
            x=global_shap_df["importance"],
            orientation="h",
            marker=dict(color="#1565c0"),
            text=global_shap_df["importance"].round(4),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        ))

        fig_global.update_layout(
            title="Özelliklerin Global Önemi (Ortalama |SHAP|)",
            xaxis_title="Mean |SHAP Value|",
            yaxis_title=None,
            template=PLOTLY_TEMPLATE,
            plot_bgcolor=PLOTLY_BGCOLOR,
            paper_bgcolor=PLOTLY_BGCOLOR,
            height=500,
            margin=dict(l=10, r=30, t=40, b=10),
            yaxis=dict(
                autorange="reversed",
            ),
        )

        st.plotly_chart(fig_global, use_container_width=True)


if __name__ == "__main__":
    main()