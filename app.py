#!/usr/bin/env python3
"""
ChurnScope - Streamlit Dashboard
Modern, professional churn prediction UI with SHAP explanations.

Usage:
    streamlit run app.py
"""
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc

from src.config import MODEL_PATH, METRICS_PATH, GLOBAL_SHAP_PATH, DATA_PATH
from src.explainer import ShapExplainer

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnScope",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    /* ── Main container ── */
    .main > div {
        padding: 0 1rem 2rem 1rem;
    }
    
    /* ── Header ── */
    .churn-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin: 1rem 0 2rem 0;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    .churn-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: rgba(255,255,255,0.05);
        border-radius: 50%;
    }
    .churn-header::after {
        content: '';
        position: absolute;
        bottom: -30%;
        left: -10%;
        width: 300px;
        height: 300px;
        background: rgba(255,255,255,0.03);
        border-radius: 50%;
    }
    .churn-header h1 {
        color: white !important;
        font-weight: 800 !important;
        font-size: 2.2rem !important;
        margin: 0 !important;
        letter-spacing: -0.5px;
        position: relative;
        z-index: 1;
    }
    .churn-header p {
        color: rgba(255,255,255,0.85) !important;
        font-size: 1.05rem !important;
        margin: 0.3rem 0 0 0 !important;
        position: relative;
        z-index: 1;
    }
    .churn-header .subtitle-highlight {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(10px);
        padding: 0.25rem 1rem;
        border-radius: 50px;
        color: white;
        font-size: 0.85rem;
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    /* ── Section titles ── */
    .section-title {
        font-weight: 700 !important;
        font-size: 1.3rem !important;
        color: #e0e0ff !important;
        margin: 1.5rem 0 1rem 0 !important;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
    }
    
    /* ── Card containers ── */
    .card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .card:hover {
        border-color: rgba(102, 126, 234, 0.4);
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.15);
        transform: translateY(-2px);
    }
    .card-title {
        color: #b0b0ff !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 1rem !important;
    }
    
    /* ── Input labels ── */
    .stSelectbox label, .stSlider label, label {
        color: #c0c0ff !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
    }
    
    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.5) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    
    /* ── Metrics ── */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        padding: 1.2rem !important;
        transition: all 0.3s ease !important;
    }
    div[data-testid="metric-container"]:hover {
        border-color: rgba(102, 126, 234, 0.4) !important;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.15) !important;
    }
    div[data-testid="metric-container"] label {
        color: #b0b0ff !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="metric-container"] div[data-testid="metric-value"] {
        font-weight: 700 !important;
        font-size: 2rem !important;
    }
    
    /* Churn probability metric - high risk */
    .metric-high div[data-testid="metric-container"] {
        background: rgba(239, 83, 80, 0.15) !important;
        border: 2px solid rgba(239, 83, 80, 0.4) !important;
    }
    .metric-high div[data-testid="metric-container"] label {
        color: #ff8a80 !important;
    }
    .metric-high div[data-testid="metric-container"] div[data-testid="metric-value"] {
        color: #ff1744 !important;
    }
    
    /* Churn probability metric - low risk */
    .metric-low div[data-testid="metric-container"] {
        background: rgba(102, 187, 106, 0.15) !important;
        border: 2px solid rgba(102, 187, 106, 0.4) !important;
    }
    .metric-low div[data-testid="metric-container"] label {
        color: #81c784 !important;
    }
    .metric-low div[data-testid="metric-container"] div[data-testid="metric-value"] {
        color: #69f0ae !important;
    }
    
    /* ── Risk badge ── */
    .risk-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
        margin-bottom: 1rem;
    }
    .risk-badge.high {
        background: rgba(239, 83, 80, 0.2);
        color: #ff1744;
        border: 1px solid rgba(239, 83, 80, 0.4);
    }
    .risk-badge.low {
        background: rgba(102, 187, 106, 0.2);
        color: #69f0ae;
        border: 1px solid rgba(102, 187, 106, 0.4);
    }
    
    /* ── Progress bar (custom) ── */
    .churn-progress {
        width: 100%;
        height: 12px;
        background: rgba(255,255,255,0.1);
        border-radius: 50px;
        overflow: hidden;
        margin: 1rem 0;
    }
    .churn-progress-bar {
        height: 100%;
        border-radius: 50px;
        transition: width 0.8s ease;
        background: linear-gradient(90deg, #66bb6a, #ffee58, #ef5350);
        width: 0%;
    }
    
    /* ── Info/Success/Warning boxes ── */
    .stAlert {
        border-radius: 12px !important;
        border: none !important;
    }
    .stAlert > div {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem !important;
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 16px !important;
        padding: 0.3rem !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px !important;
        padding: 0.6rem 1.5rem !important;
        color: #a0a0c0 !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* ── Select boxes ── */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        color: #e0e0ff !important;
    }
    .stSelectbox > div > div:hover {
        border-color: rgba(102, 126, 234, 0.4) !important;
    }
    
    /* ── Sliders ── */
    .stSlider > div > div > div {
        color: #667eea !important;
    }
    .stSlider [data-baseweb="slider"] > div {
        background: rgba(102, 126, 234, 0.3) !important;
    }
    .stSlider [data-baseweb="slider"] > div > div {
        background: #667eea !important;
    }
    
    /* ── Dataframe ── */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    .stDataFrame td, .stDataFrame th {
        color: #c0c0ff !important;
    }
    
    /* ── Footer ── */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1.5rem;
        color: rgba(255,255,255,0.3);
        font-size: 0.8rem;
        border-top: 1px solid rgba(255,255,255,0.05);
    }
    
    /* ── Scrollbar ── */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.02);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(102, 126, 234, 0.3);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(102, 126, 234, 0.5);
    }
</style>
"""

# ─── Helpers ───────────────────────────────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_BGCOLOR = "rgba(0,0,0,0)"


@st.cache_resource
def load_pipeline():
    import joblib
    return joblib.load(MODEL_PATH)


def load_metrics():
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


def load_global_shap():
    return pd.read_json(GLOBAL_SHAP_PATH)


def compute_features_for_input(input_df: pd.DataFrame) -> pd.DataFrame:
    df = input_df.copy()
    df["charges_per_month"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["is_long_term"] = df["Contract"].isin(["One year", "Two year"]).astype(int)
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    for col in service_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})
    df["service_count"] = df[service_cols].sum(axis=1)
    return df


def create_gauge_chart(probability: float) -> go.Figure:
    """Create a modern gauge chart for churn probability."""
    color = "#ef5350" if probability > 50 else "#66bb6a"
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=probability,
        number={"suffix": "%", "font": {"size": 40, "color": color, "family": "Inter"}},
        delta={"reference": 50, "position": "top"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "rgba(255,255,255,0.3)"},
            "bar": {"color": color, "thickness": 0.7},
            "bgcolor": "rgba(255,255,255,0.05)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 33], "color": "rgba(102, 187, 106, 0.15)"},
                {"range": [33, 66], "color": "rgba(255, 238, 88, 0.15)"},
                {"range": [66, 100], "color": "rgba(239, 83, 80, 0.15)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.75,
                "value": 50,
            },
        },
    ))
    
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor=PLOTLY_BGCOLOR,
        plot_bgcolor=PLOTLY_BGCOLOR,
        font={"color": "rgba(255,255,255,0.7)"},
    )
    return fig


def create_plotly_heatmap(cm, title):
    """Create a styled confusion matrix heatmap."""
    cm_array = np.array(cm)
    fig = go.Figure(data=go.Heatmap(
        z=cm_array,
        x=["Tahmin: Hayır", "Tahmin: Evet"],
        y=["Gerçek: Hayır", "Gerçek: Evet"],
        colorscale="Blues",
        showscale=False,
        texttemplate="%{z}",
        textfont=dict(size=16, color="white", family="Inter"),
        hovertemplate="<b>%{y}</b><br><b>%{x}</b><br>Değer: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title=f"<b>{title}</b>",
        xaxis_title="Tahmin Edilen",
        yaxis_title="Gerçek Değer",
        template=PLOTLY_TEMPLATE,
        plot_bgcolor=PLOTLY_BGCOLOR,
        paper_bgcolor=PLOTLY_BGCOLOR,
        height=400,
        xaxis=dict(side="bottom", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(autorange="reversed", gridcolor="rgba(255,255,255,0.05)"),
        font={"color": "rgba(255,255,255,0.7)", "family": "Inter"},
    )
    return fig


# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    # Load resources
    with st.spinner("🔄 Model yükleniyor..."):
        pipeline = load_pipeline()
        metrics = load_metrics()
        global_shap_df = load_global_shap()

    @st.cache_resource
    def get_explainer():
        from sklearn.model_selection import train_test_split
        from src.preprocessing import load_and_preprocess
        from src.config import RANDOM_STATE, TEST_SIZE

        X, y = load_and_preprocess(DATA_PATH)
        X_train, _, _, _ = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        pl = load_pipeline()
        return ShapExplainer(pl, X_train)

    explainer = get_explainer()

    # ── Inject CSS ──
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── HEADER ──
    st.markdown("""
    <div class="churn-header">
        <h1>📊 ChurnScope</h1>
        <p>Müşteri Kayıp Tahmin Sistemi — AI destekli, SHAP ile açıklanabilir</p>
        <div class="subtitle-highlight">✨ Gradient Boosting • SHAP • Streamlit</div>
    </div>
    """, unsafe_allow_html=True)

    # ── TABS ──
    tab1, tab2 = st.tabs(["🔍 Müşteri Analizi", "📈 Model Performansı"])

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1 — Customer Analysis
    # ═══════════════════════════════════════════════════════════════════════════
    with tab1:
        col_left, col_right = st.columns([1, 1.3], gap="large")

        # ── LEFT PANEL ──────────────────────────────────────────────────────
        with col_left:
            st.markdown('<p class="section-title">📋 Müşteri Bilgileri</p>', unsafe_allow_html=True)

            # Account Info
            st.markdown("""
            <div class="card">
                <p class="card-title">📦 Hesap Bilgileri</p>
            """, unsafe_allow_html=True)
            tenure = st.slider("Tenure (ay)", 0, 72, 12, key="tenure")
            monthly_charges = st.slider("Aylık Ücret ($)", 0.0, 120.0, 50.0, step=0.5, key="monthly_charges")
            total_charges = st.slider("Toplam Ücret ($)", 0.0, 8700.0, 500.0, step=10.0, key="total_charges")
            contract = st.selectbox("Sözleşme Tipi", ["Month-to-month", "One year", "Two year"], key="contract")
            st.markdown("</div>", unsafe_allow_html=True)

            # Services
            st.markdown("""
            <div class="card">
                <p class="card-title">🛠️ Hizmetler</p>
            """, unsafe_allow_html=True)
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
            st.markdown("</div>", unsafe_allow_html=True)

            # Demographics
            st.markdown("""
            <div class="card">
                <p class="card-title">👤 Demografik Bilgiler</p>
            """, unsafe_allow_html=True)
            col_dem1, col_dem2 = st.columns(2)
            with col_dem1:
                gender = st.selectbox("Cinsiyet", ["Male", "Female"], key="gender")
                senior_citizen = st.selectbox("SeniorCitizen", ["0", "1"], key="senior_citizen")
            with col_dem2:
                partner = st.selectbox("Partner", ["Yes", "No"], key="partner")
                dependents = st.selectbox("Dependents", ["Yes", "No"], key="dependents")
            st.markdown("</div>", unsafe_allow_html=True)

            # Payment
            st.markdown("""
            <div class="card">
                <p class="card-title">💳 Ödeme Bilgileri</p>
            """, unsafe_allow_html=True)
            payment_method = st.selectbox(
                "Ödeme Yöntemi",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                key="payment_method",
            )
            paperless_billing = st.selectbox("Kağıtsız Fatura", ["Yes", "No"], key="paperless_billing")
            st.markdown("</div>", unsafe_allow_html=True)

            # Predict button
            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button("🚀 Tahmin Et", use_container_width=True, type="primary")

        # ── RIGHT PANEL ─────────────────────────────────────────────────────
        with col_right:
            if predict_btn:
                # Build input
                input_data = {
                    "gender": [gender], "SeniorCitizen": [senior_citizen],
                    "Partner": [partner], "Dependents": [dependents],
                    "tenure": [tenure], "PhoneService": [phone_service],
                    "MultipleLines": [multiple_lines], "OnlineSecurity": [online_security],
                    "OnlineBackup": [online_backup], "DeviceProtection": [device_protection],
                    "TechSupport": [tech_support], "StreamingTV": [streaming_tv],
                    "StreamingMovies": [streaming_movies], "Contract": [contract],
                    "PaperlessBilling": [paperless_billing], "PaymentMethod": [payment_method],
                    "MonthlyCharges": [monthly_charges], "TotalCharges": [total_charges],
                }
                input_df = pd.DataFrame(input_data)
                input_df = compute_features_for_input(input_df)

                proba = pipeline.predict_proba(input_df)[0, 1]
                churn_pct = proba * 100
                is_high_risk = churn_pct > 50

                st.markdown(f'<p class="section-title">📊 Tahmin Sonucu</p>', unsafe_allow_html=True)

                # Risk badge
                badge_class = "high" if is_high_risk else "low"
                badge_text = "🔴 YÜKSEK RİSK" if is_high_risk else "🟢 DÜŞÜK RİSK"
                st.markdown(
                    f'<div class="risk-badge {badge_class}">{badge_text}</div>',
                    unsafe_allow_html=True,
                )

                # Gauge
                gauge_fig = create_gauge_chart(churn_pct)
                st.plotly_chart(gauge_fig, use_container_width=True)

                # Probability metric
                metric_class = "metric-high" if is_high_risk else "metric-low"
                st.markdown(f'<div class="{metric_class}">', unsafe_allow_html=True)
                st.metric(
                    label="📉 Kayıp Olasılığı",
                    value=f"%{churn_pct:.1f}",
                    delta=f"{'🔴' if is_high_risk else '🟢'} {'Yüksek' if is_high_risk else 'Düşük'} Risk",
                )
                st.markdown('</div>', unsafe_allow_html=True)

                # Interpretation
                if is_high_risk:
                    st.warning("⚠️ Bu müşterinin kaybedilme riski **yüksek**. Önleyici aksiyon alınması önerilir. Müşteriye özel kampanya veya iyileştirme planı devreye alınabilir.")
                else:
                    st.success("✅ Bu müşterinin kaybedilme riski **düşük**. Mevcut hizmet kalitesi korunarak memnuniyet sürdürülebilir.")

                # Local SHAP
                st.markdown(f'<p class="section-title">🔍 SHAP Katkı Faktörleri</p>', unsafe_allow_html=True)

                local_shap = explainer.local_explanation(input_df)
                top_local = local_shap.head(12).copy()
                top_local["color"] = top_local["shap_value"].apply(
                    lambda x: "#ef5350" if x > 0 else "#66bb6a"
                )

                fig_local = go.Figure()
                fig_local.add_trace(go.Bar(
                    y=top_local["feature"],
                    x=top_local["shap_value"],
                    orientation="h",
                    marker=dict(
                        color=top_local["color"],
                        line=dict(color=top_local["color"], width=1),
                    ),
                    text=top_local["shap_value"].round(3),
                    textposition="outside",
                    textfont=dict(size=11, color="rgba(255,255,255,0.8)", family="Inter"),
                    hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
                ))
                fig_local.update_layout(
                    title="<b>SHAP Değerleri</b><br><span style='font-size:12px;color:#aaa'>🔴 Kaybı Artırır &nbsp;&nbsp; 🟢 Kaybı Azaltır</span>",
                    xaxis_title="SHAP Value",
                    yaxis_title=None,
                    template=PLOTLY_TEMPLATE,
                    plot_bgcolor=PLOTLY_BGCOLOR,
                    paper_bgcolor=PLOTLY_BGCOLOR,
                    height=450,
                    margin=dict(l=10, r=60, t=60, b=10),
                    yaxis=dict(autorange="reversed", gridcolor="rgba(255,255,255,0.05)"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                    font={"color": "rgba(255,255,255,0.7)", "family": "Inter"},
                )
                st.plotly_chart(fig_local, use_container_width=True)

            else:
                # Placeholder
                st.markdown(f'<p class="section-title">👋 Hoş Geldiniz</p>', unsafe_allow_html=True)
                st.markdown("""
                <div class="card" style="text-align:center;padding:3rem 2rem;">
                    <div style="font-size:4rem;margin-bottom:1rem;">📊</div>
                    <h3 style="color:#e0e0ff;font-weight:600;">Müşteri Kayıp Analizine Hoş Geldiniz</h3>
                    <p style="color:#a0a0c0;max-width:500px;margin:1rem auto;">
                        Sol paneldeki müşteri bilgilerini doldurun ve <strong>"Tahmin Et"</strong> 
                        butonuna tıklayarak yapay zeka destekli kayıp tahminini görüntüleyin.
                    </p>
                    <div style="display:flex;gap:1rem;justify-content:center;flex-wrap:wrap;margin-top:1.5rem;">
                        <div style="background:rgba(102,187,106,0.1);padding:0.8rem 1.5rem;border-radius:12px;border:1px solid rgba(102,187,106,0.2);">
                            <span style="font-size:1.5rem;">📋</span>
                            <p style="color:#81c784;font-size:0.85rem;margin:0;">Bilgileri Girin</p>
                        </div>
                        <div style="background:rgba(102,126,234,0.1);padding:0.8rem 1.5rem;border-radius:12px;border:1px solid rgba(102,126,234,0.2);">
                            <span style="font-size:1.5rem;">🤖</span>
                            <p style="color:#667eea;font-size:0.85rem;margin:0;">AI Tahmini</p>
                        </div>
                        <div style="background:rgba(255,238,88,0.1);padding:0.8rem 1.5rem;border-radius:12px;border:1px solid rgba(255,238,88,0.2);">
                            <span style="font-size:1.5rem;">🔍</span>
                            <p style="color:#ffee58;font-size:0.85rem;margin:0;">SHAP Açıklaması</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="card">
                    <p class="card-title">💡 Hızlı İpuçları</p>
                    <ul style="color:#b0b0ff;line-height:2;padding-left:1.2rem;">
                        <li><strong>Sözleşme Tipi:</strong> Aylık sözleşmelerde kayıp riski daha yüksektir</li>
                        <li><strong>Hizmet Sayısı:</strong> Daha fazla hizmet = daha düşük kayıp riski</li>
                        <li><strong>Tenure:</strong> Uzun süreli müşteriler genelde daha sadıktır</li>
                        <li><strong>Ödeme Yöntemi:</strong> Elektronik çek kullananlarda risk yükselebilir</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 2 — Model Performance
    # ═══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown(f'<p class="section-title">📈 Model Performans Metrikleri</p>', unsafe_allow_html=True)

        # Extract metrics
        accuracy = metrics["accuracy"]
        roc_auc_val = metrics["roc_auc"]
        class_report = metrics["classification_report"]
        cm = metrics["confusion_matrix"]
        f1_churn = class_report.get("1", {}).get("f1-score", 0)
        precision_churn = class_report.get("1", {}).get("precision", 0)
        recall_churn = class_report.get("1", {}).get("recall", 0)

        # ── Row 1: Metric cards ──
        col_m1, col_m2, col_m3, col_m4 = st.columns(4, gap="medium")
        with col_m1:
            st.markdown('<div style="background:rgba(102,187,106,0.1);border-radius:16px;padding:0.5rem;border:1px solid rgba(102,187,106,0.2);">', unsafe_allow_html=True)
            st.metric(label="🎯 Accuracy", value=f"{accuracy:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col_m2:
            st.markdown('<div style="background:rgba(102,126,234,0.1);border-radius:16px;padding:0.5rem;border:1px solid rgba(102,126,234,0.2);">', unsafe_allow_html=True)
            st.metric(label="📈 ROC-AUC", value=f"{roc_auc_val:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col_m3:
            st.markdown('<div style="background:rgba(255,238,88,0.1);border-radius:16px;padding:0.5rem;border:1px solid rgba(255,238,88,0.2);">', unsafe_allow_html=True)
            st.metric(label="⭐ F1-Score (Churn)", value=f"{f1_churn:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col_m4:
            st.markdown('<div style="background:rgba(239,83,80,0.1);border-radius:16px;padding:0.5rem;border:1px solid rgba(239,83,80,0.2);">', unsafe_allow_html=True)
            st.metric(label="🎯 Precision (Churn)", value=f"{precision_churn:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Row 2: Confusion Matrix + ROC ──
        col_cm, col_roc = st.columns(2, gap="large")

        with col_cm:
            fig_cm = create_plotly_heatmap(cm, "Confusion Matrix (Test Seti)")
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_roc:
            st.markdown(f'<p class="section-title" style="margin-top:0;">📉 ROC Eğrisi</p>', unsafe_allow_html=True)
            with st.spinner("🔄 ROC hesaplanıyor..."):
                from sklearn.model_selection import train_test_split
                from src.preprocessing import load_and_preprocess
                from src.config import RANDOM_STATE, TEST_SIZE

                X_full, y_full = load_and_preprocess(DATA_PATH)
                _, X_test_roc, _, y_test_roc = train_test_split(
                    X_full, y_full, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_full
                )
                y_scores = load_pipeline().predict_proba(X_test_roc)[:, 1]

            fpr, tpr, _ = roc_curve(y_test_roc, y_scores)
            roc_auc_calc = auc(fpr, tpr)

            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode="lines",
                name=f"ROC (AUC = {roc_auc_calc:.3f})",
                line=dict(color="#667eea", width=3),
                fill="tozeroy",
                fillcolor="rgba(102,126,234,0.15)",
                hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                name="Random",
                line=dict(color="rgba(255,255,255,0.2)", dash="dash", width=1),
                showlegend=False,
            ))
            fig_roc.update_layout(
                title=f"<b>ROC Eğrisi</b><br><span style='font-size:12px;color:#aaa'>AUC = {roc_auc_calc:.3f}</span>",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                template=PLOTLY_TEMPLATE,
                plot_bgcolor=PLOTLY_BGCOLOR,
                paper_bgcolor=PLOTLY_BGCOLOR,
                height=400,
                xaxis=dict(range=[0, 1], gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(range=[0, 1], gridcolor="rgba(255,255,255,0.05)"),
                font={"color": "rgba(255,255,255,0.7)", "family": "Inter"},
                legend=dict(x=0.6, y=0.2, bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(255,255,255,0.1)"),
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        # ── Row 3: Classification Report ──
        st.markdown(f'<p class="section-title">📋 Sınıflandırma Raporu</p>', unsafe_allow_html=True)

        report_data = []
        for class_label, class_metrics in class_report.items():
            if isinstance(class_metrics, dict):
                row = {"Sınıf": class_label}
                row.update(class_metrics)
                report_data.append(row)
        report_df = pd.DataFrame(report_data)
        report_df = report_df[report_df["Sınıf"].isin(["0", "1", "accuracy"])]

        # Rename columns for Turkish
        report_df = report_df.rename(columns={
            "precision": "Precision", "recall": "Recall",
            "f1-score": "F1-Score", "support": "Destek",
        })
        # Drop accuracy row's extra columns
        for col in report_df.columns:
            report_df[col] = report_df[col].apply(
                lambda x: "" if (isinstance(x, float) and np.isnan(x)) else x
            )

        st.dataframe(
            report_df.style.set_properties(**{
                "background-color": "rgba(255,255,255,0.03)",
                "color": "#c0c0ff",
                "border": "none",
                "font-family": "Inter",
            }),
            use_container_width=True,
            height=150,
        )

        # ── Row 4: Global SHAP ──
        st.markdown(f'<p class="section-title">🌍 Global SHAP Feature Importance (Top 15)</p>', unsafe_allow_html=True)

        fig_global = go.Figure()
        fig_global.add_trace(go.Bar(
            y=global_shap_df["feature"],
            x=global_shap_df["importance"],
            orientation="h",
            marker=dict(
                color="#667eea",
                line=dict(color="#764ba2", width=1),
            ),
            text=global_shap_df["importance"].round(4),
            textposition="outside",
            textfont=dict(size=11, color="rgba(255,255,255,0.8)", family="Inter"),
            hovertemplate="<b>%{y}</b><br>Önem: %{x:.4f}<extra></extra>",
        ))
        fig_global.update_layout(
            title="<b>Özelliklerin Global Önemi</b><br><span style='font-size:12px;color:#aaa'>Ortalama |SHAP| Değerine Göre</span>",
            xaxis_title="Ortalama |SHAP Değeri|",
            yaxis_title=None,
            template=PLOTLY_TEMPLATE,
            plot_bgcolor=PLOTLY_BGCOLOR,
            paper_bgcolor=PLOTLY_BGCOLOR,
            height=520,
            margin=dict(l=10, r=80, t=60, b=10),
            yaxis=dict(autorange="reversed", gridcolor="rgba(255,255,255,0.05)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            font={"color": "rgba(255,255,255,0.7)", "family": "Inter"},
        )
        st.plotly_chart(fig_global, use_container_width=True)

    # ── FOOTER ──
    st.markdown("""
    <div class="footer">
        ChurnScope v1.0 • Gradient Boosting • SHAP • Streamlit
        <br>
        Built with ❤️ for data-driven customer retention
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()