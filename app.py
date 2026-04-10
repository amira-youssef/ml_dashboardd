"""
app.py — Streamlit dashboard for ML Activities project.
Run with:  streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import joblib
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Activities Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #0f172a; }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    .metric-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 18px 22px;
        text-align: center;
        border: 1px solid #334155;
    }
    .metric-card .label { font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card .value { font-size: 28px; font-weight: 700; color: #f1f5f9; margin-top: 4px; }
    .metric-card .sub   { font-size: 12px; color: #64748b; margin-top: 2px; }
    .section-title { font-size: 22px; font-weight: 700; color: #f1f5f9; margin-bottom: 4px; }
    .section-sub   { font-size: 14px; color: #94a3b8; margin-bottom: 24px; }
    .stTabs [data-baseweb="tab"] { font-size: 14px; }
    div[data-testid="stMetric"] label { font-size: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LOAD ARTIFACTS (cached so they only load once)
# ─────────────────────────────────────────────────────────────
MODELS_DIR = Path("models")

@st.cache_resource(show_spinner="Loading models…")
def load_all():
    if not MODELS_DIR.exists():
        return None
    return {
        "cls":             joblib.load(MODELS_DIR / "cls_results.pkl"),
        "le":              joblib.load(MODELS_DIR / "label_encoder.pkl"),
        "reg":             joblib.load(MODELS_DIR / "reg_results.pkl"),
        "cluster_df":      joblib.load(MODELS_DIR / "cluster_df.pkl"),
        "kmeans":          joblib.load(MODELS_DIR / "kmeans.pkl"),
        "pca_coords":      joblib.load(MODELS_DIR / "pca_coords.pkl"),
        "cluster_features":joblib.load(MODELS_DIR / "cluster_features.pkl"),
        "sil_by_k":        joblib.load(MODELS_DIR / "sil_by_k.pkl"),
        "feature_cols":    joblib.load(MODELS_DIR / "feature_cols.pkl"),
        "cat_features":    joblib.load(MODELS_DIR / "cat_features.pkl"),
        "num_features":    joblib.load(MODELS_DIR / "num_features.pkl"),
        "ts":              joblib.load(MODELS_DIR / "ts_results.pkl"),
        "ts_data":         joblib.load(MODELS_DIR / "ts_data.pkl"),
        "master":          joblib.load(MODELS_DIR / "master.pkl"),
    }

data = load_all()

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 ML Dashboard")
    st.markdown("---")
    page = st.radio(
        "Navigate to",
        ["🏠 Overview", "🎯 Classification", "📈 Regression",
         "🔵 Clustering", "🕐 Forecasting", "🔮 Live Predictor"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("Trained on activity & participation data.")

# ─────────────────────────────────────────────────────────────
# GUARD: models not trained yet
# ─────────────────────────────────────────────────────────────
if data is None:
    st.error("⚠️ No trained models found. Please run `python train.py` first, then refresh this page.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
DARK_BG   = "#0f172a"
ACCENT    = "#6366f1"
ACCENT2   = "#22d3ee"
GRID_COLOR = "#1e293b"

def dark_fig(nrows=1, ncols=1, figsize=(10, 4)):
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor(DARK_BG)
    axes = np.array(ax).flatten() if nrows * ncols > 1 else [ax]
    for a in axes:
        a.set_facecolor(GRID_COLOR)
        a.tick_params(colors="#94a3b8")
        a.xaxis.label.set_color("#94a3b8")
        a.yaxis.label.set_color("#94a3b8")
        a.title.set_color("#f1f5f9")
        for spine in a.spines.values():
            spine.set_edgecolor("#334155")
    return fig, ax

def metric_card(label, value, sub=""):
    return f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="sub">{sub}</div>
    </div>"""

# ─────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.markdown('<div class="section-title">Project Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Summary of all trained models and dataset stats</div>', unsafe_allow_html=True)

    master = data["master"]
    cls    = data["cls"]
    reg    = data["reg"]
    ts     = data["ts"]

    # Dataset metrics
    st.markdown("#### 📦 Dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("Total records", f"{len(master):,}"), unsafe_allow_html=True)
    c2.markdown(metric_card("Features used", "13"), unsafe_allow_html=True)
    c3.markdown(metric_card("Unique activities", f"{master['activity_id'].nunique():,}"), unsafe_allow_html=True)
    c4.markdown(metric_card("Date range", f"{master['event_date'].dt.year.min()}–{master['event_date'].dt.year.max()}"), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🤖 Model Performance Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Classification**")
        rows = []
        for name, res in cls.items():
            rows.append({
                "Model": name,
                "Accuracy": f"{res['accuracy']:.3f}",
                "F1 (macro)": f"{res['f1']:.3f}",
                "ROC-AUC": f"{res['auc']:.3f}",
            })
        st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

    with col2:
        st.markdown("**Regression**")
        rows = []
        for name, res in reg.items():
            rows.append({
                "Model": name,
                "RMSE": f"{res['rmse']:.2f}",
                "MAE": f"{res['mae']:.2f}",
                "R²": f"{res['r2']:.3f}",
            })
        st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📅 Forecasting")
    col1, col2 = st.columns(2)
    for i, (name, res) in enumerate(ts.items()):
        col = col1 if i == 0 else col2
        with col:
            c1, c2, c3 = st.columns(3)
            c1.markdown(metric_card(f"{name} RMSE", f"{res['rmse']:.1f}"), unsafe_allow_html=True)
            c2.markdown(metric_card(f"{name} MAE",  f"{res['mae']:.1f}"),  unsafe_allow_html=True)
            c3.markdown(metric_card(f"{name} MAPE", f"{res['mape']:.1f}%"), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE: CLASSIFICATION
# ─────────────────────────────────────────────────────────────
elif page == "🎯 Classification":
    st.markdown('<div class="section-title">Classification — Participation Level</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Predict whether an activity will have low, medium, or high participation</div>', unsafe_allow_html=True)

    cls = data["cls"]
    le  = data["le"]

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Metrics", "🗂 Confusion Matrix", "📉 ROC Curves", "🏆 Feature Importance"])

    # ── Metrics ──
    with tab1:
        cols = st.columns(len(cls))
        for i, (name, res) in enumerate(cls.items()):
            with cols[i]:
                st.markdown(f"**{name}**")
                for label, key in [("Accuracy","accuracy"),("Precision","precision"),
                                    ("Recall","recall"),("F1 (macro)","f1"),("ROC-AUC","auc")]:
                    st.markdown(metric_card(label, f"{res[key]:.4f}"), unsafe_allow_html=True)
                    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ── Confusion Matrix ──
    with tab2:
        model_choice = st.selectbox("Model", list(cls.keys()), key="cm_model")
        res = cls[model_choice]
        cm  = confusion_matrix(res["y_test"], res["pred"])
        fig, ax = dark_fig(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(le.classes_)))
        ax.set_xticklabels(le.classes_, rotation=45, color="#e2e8f0")
        ax.set_yticks(range(len(le.classes_)))
        ax.set_yticklabels(le.classes_, color="#e2e8f0")
        ax.set_xlabel("Predicted", color="#94a3b8")
        ax.set_ylabel("Actual",    color="#94a3b8")
        ax.set_title(f"Confusion Matrix — {model_choice}", color="#f1f5f9")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "#334155",
                        fontsize=14, fontweight="bold")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        st.pyplot(fig)
        st.text(classification_report(res["y_test"], res["pred"], target_names=le.classes_))

    # ── ROC Curves ──
    with tab3:
        model_choice = st.selectbox("Model", list(cls.keys()), key="roc_model")
        res = cls[model_choice]
        y_bin = label_binarize(res["y_test"], classes=np.unique(res["y_test"]))
        fig, ax = dark_fig(figsize=(7, 5))
        colors = [ACCENT, ACCENT2, "#f59e0b"]
        for i, (cls_name, color) in enumerate(zip(le.classes_, colors)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], res["proba"][:, i])
            ax.plot(fpr, tpr, label=f"{cls_name} (AUC={auc(fpr,tpr):.2f})", color=color, linewidth=2)
        ax.plot([0,1],[0,1], linestyle="--", color="#475569")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curves — {model_choice}")
        ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
        fig.tight_layout()
        st.pyplot(fig)

    # ── Feature Importance ──
    with tab4:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression as LR
        model_choice = st.selectbox("Model", list(cls.keys()), key="fi_model")
        best = cls[model_choice]["model"]
        step = best.named_steps["model"]
        feature_names = best.named_steps["prep"].get_feature_names_out()

        if hasattr(step, "feature_importances_"):
            importances = step.feature_importances_
            title = "Feature Importances (Random Forest)"
            bar_color = ACCENT
        else:
            importances = np.abs(step.coef_).mean(axis=0)
            title = "Mean |Coefficient| (Logistic Regression)"
            bar_color = ACCENT2

        imp_df = (pd.DataFrame({"feature": feature_names, "importance": importances})
                  .sort_values("importance", ascending=False).head(15))

        fig, ax = dark_fig(figsize=(8, 5))
        bars = ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color=bar_color)
        ax.set_title(title)
        ax.set_xlabel("Importance")
        fig.tight_layout()
        st.pyplot(fig)


# ─────────────────────────────────────────────────────────────
# PAGE: REGRESSION
# ─────────────────────────────────────────────────────────────
elif page == "📈 Regression":
    st.markdown('<div class="section-title">Regression — Participant Count</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Predict the expected number of participants for an activity</div>', unsafe_allow_html=True)

    reg = data["reg"]
    tab1, tab2, tab3 = st.tabs(["📊 Metrics", "🔍 Residuals & Predictions", "🏆 Feature Importance"])

    with tab1:
        cols = st.columns(len(reg))
        for i, (name, res) in enumerate(reg.items()):
            with cols[i]:
                st.markdown(f"**{name}**")
                for label, key in [("MSE","mse"),("RMSE","rmse"),("MAE","mae"),("R²","r2")]:
                    val = f"{res[key]:.4f}" if key == "r2" else f"{res[key]:.2f}"
                    st.markdown(metric_card(label, val), unsafe_allow_html=True)
                    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    with tab2:
        model_choice = st.selectbox("Model", list(reg.keys()))
        res = reg[model_choice]
        y_true = np.array(res["y_test"])
        y_pred = np.array(res["pred"])
        residuals = y_true - y_pred

        fig, axes = dark_fig(nrows=1, ncols=2, figsize=(12, 5))
        ax1, ax2 = axes

        ax1.scatter(y_true, y_pred, alpha=0.4, color=ACCENT, s=15)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax1.plot(lims, lims, "--", color="#f59e0b", linewidth=1.5)
        ax1.set_title("Actual vs Predicted")
        ax1.set_xlabel("Actual")
        ax1.set_ylabel("Predicted")

        ax2.scatter(y_pred, residuals, alpha=0.4, color=ACCENT2, s=15)
        ax2.axhline(0, color="#f59e0b", linewidth=1.5, linestyle="--")
        ax2.set_title("Residual Plot")
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Residuals")

        fig.tight_layout()
        st.pyplot(fig)

    with tab3:
        from sklearn.ensemble import RandomForestRegressor as RFR
        model_choice = st.selectbox("Model", list(reg.keys()), key="reg_fi")
        best = reg[model_choice]["model"]
        step = best.named_steps["model"]
        feature_names = best.named_steps["prep"].get_feature_names_out()

        if hasattr(step, "feature_importances_"):
            importances = step.feature_importances_
            title, bar_color = "Feature Importances (Random Forest)", ACCENT
        else:
            importances = np.abs(step.coef_)
            title, bar_color = "Ridge Coefficients (absolute value)", ACCENT2

        imp_df = (pd.DataFrame({"feature": feature_names, "importance": importances})
                  .sort_values("importance", ascending=False).head(15))

        fig, ax = dark_fig(figsize=(8, 5))
        ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color=bar_color)
        ax.set_title(title)
        fig.tight_layout()
        st.pyplot(fig)


# ─────────────────────────────────────────────────────────────
# PAGE: CLUSTERING
# ─────────────────────────────────────────────────────────────
elif page == "🔵 Clustering":
    st.markdown('<div class="section-title">Clustering — Unit Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Group sports units by their activity dynamics</div>', unsafe_allow_html=True)

    cluster_df      = data["cluster_df"]
    pca_coords      = data["pca_coords"]
    cluster_features = data["cluster_features"]
    sil_by_k        = data["sil_by_k"]

    tab1, tab2, tab3 = st.tabs(["📍 PCA Visualization", "📊 Elbow & Silhouette", "🗂 Cluster Profiles"])

    with tab1:
        label_col = st.radio("Clustering method", ["kmeans_cluster", "agglo_cluster"],
                             format_func=lambda x: x.replace("_cluster","").upper(),
                             horizontal=True)
        labels = cluster_df[label_col].values
        n_clusters = len(np.unique(labels))
        colors_list = cm.tab10(np.linspace(0, 1, n_clusters))

        fig, ax = dark_fig(figsize=(8, 6))
        for k in range(n_clusters):
            mask = labels == k
            ax.scatter(pca_coords[mask, 0], pca_coords[mask, 1],
                       color=colors_list[k], label=f"Cluster {k}", alpha=0.7, s=30)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"PCA 2D — {label_col.replace('_cluster','').upper()}")
        ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
        fig.tight_layout()
        st.pyplot(fig)

    with tab2:
        ks  = list(sil_by_k.keys())
        sil = list(sil_by_k.values())

        fig, axes = dark_fig(nrows=1, ncols=2, figsize=(12, 4))
        axes[0].plot(ks, sil, marker="o", color=ACCENT, linewidth=2)
        axes[0].set_title("Silhouette Score by k")
        axes[0].set_xlabel("k"); axes[0].set_ylabel("Silhouette")

        # Recompute inertia for elbow
        from sklearn.cluster import KMeans as KM
        from sklearn.preprocessing import StandardScaler as SS
        from sklearn.impute import SimpleImputer as SI
        X_c = SI(strategy="median").fit_transform(cluster_df[cluster_features])
        X_cs = SS().fit_transform(X_c)
        inertias = [KM(n_clusters=k, random_state=42, n_init=10).fit(X_cs).inertia_ for k in ks]
        axes[1].plot(ks, inertias, marker="o", color=ACCENT2, linewidth=2)
        axes[1].set_title("Elbow Method")
        axes[1].set_xlabel("k"); axes[1].set_ylabel("Inertia")

        fig.tight_layout()
        st.pyplot(fig)

    with tab3:
        profile = cluster_df.groupby("kmeans_cluster")[cluster_features].mean()
        # Normalize for heatmap
        profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)

        fig, ax = dark_fig(figsize=(10, 4))
        im = ax.imshow(profile_norm.values, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(cluster_features)))
        ax.set_xticklabels(cluster_features, rotation=45, ha="right", color="#e2e8f0", fontsize=9)
        ax.set_yticks(range(len(profile.index)))
        ax.set_yticklabels([f"Cluster {i}" for i in profile.index], color="#e2e8f0")
        ax.set_title("Cluster Profile Heatmap (normalised)")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        st.pyplot(fig)

        st.markdown("**Raw cluster means**")
        st.dataframe(profile.round(2), use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PAGE: FORECASTING
# ─────────────────────────────────────────────────────────────
elif page == "🕐 Forecasting":
    st.markdown('<div class="section-title">Time Series Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">ARIMA vs SARIMA forecast of daily participant volume</div>', unsafe_allow_html=True)

    ts      = data["ts"]
    ts_data = data["ts_data"]
    train_ts, test_ts = ts_data["train"], ts_data["test"]

    tab1, tab2 = st.tabs(["📉 Forecast Chart", "📊 Metrics"])

    with tab1:
        show_models = st.multiselect("Models to display", list(ts.keys()), default=list(ts.keys()))
        history_days = st.slider("Training history shown (days)", 30, 180, 90)

        fig, ax = dark_fig(figsize=(13, 5))
        ax.plot(train_ts.index[-history_days:], train_ts.iloc[-history_days:],
                label="Train", color="#94a3b8", linewidth=1.5)
        ax.plot(test_ts.index, test_ts, label="Actual", color="#22d3ee", linewidth=2)

        palette = [ACCENT, "#f59e0b", "#10b981"]
        for i, name in enumerate(show_models):
            ax.plot(test_ts.index, ts[name]["pred"],
                    label=f"{name} forecast", linestyle="--",
                    color=palette[i % len(palette)], linewidth=2)

        ax.set_title("Participant Volume Forecast")
        ax.set_xlabel("Date"); ax.set_ylabel("Total Participants")
        ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
        fig.tight_layout()
        st.pyplot(fig)

    with tab2:
        rows = []
        for name, res in ts.items():
            rows.append({"Model": name,
                         "RMSE": f"{res['rmse']:.2f}",
                         "MAE":  f"{res['mae']:.2f}",
                         "MAPE": f"{res['mape']:.2f}%"})
        st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

        best_ts = min(ts.items(), key=lambda x: x[1]["rmse"])
        st.success(f"✅ Best model by RMSE: **{best_ts[0]}** (RMSE = {best_ts[1]['rmse']:.2f})")


# ─────────────────────────────────────────────────────────────
# PAGE: LIVE PREDICTOR
# ─────────────────────────────────────────────────────────────
elif page == "🔮 Live Predictor":
    st.markdown('<div class="section-title">Live Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Fill in activity details and get instant predictions</div>', unsafe_allow_html=True)

    master       = data["master"]
    cls          = data["cls"]
    reg          = data["reg"]
    le           = data["le"]
    feature_cols = data["feature_cols"]
    cat_features = data["cat_features"]
    num_features = data["num_features"]

    # Build dropdown options from actual data
    def opts(col):
        return sorted(master[col].dropna().unique().tolist())

    with st.form("predictor_form"):
        st.markdown("#### Activity details")
        col1, col2, col3 = st.columns(3)

        with col1:
            activity_type = st.selectbox("Activity type",    opts("activity_type"))
            country       = st.selectbox("Country",          opts("country"))
            unit_name     = st.selectbox("Unit",             opts("unit_name"))
            genre         = st.selectbox("Genre",            opts("genre"))

        with col2:
            tranche_age  = st.selectbox("Age group",         opts("tranche_age"))
            saison       = st.selectbox("Season",            opts("saison"))
            categorie    = st.selectbox("Category",          opts("categorie"))

        with col3:
            month        = st.slider("Month",        1, 12, 6)
            year         = st.slider("Year",         int(master["year"].min()),
                                                     int(master["year"].max()),
                                                     int(master["year"].median()))
            dayofweek    = st.slider("Day of week (0=Mon)", 0, 6, 2)
            duration     = st.number_input("Duration (days)", min_value=0.0,
                                           max_value=float(master["duration"].max()), value=3.0)
            act_budget   = st.number_input("Activity budget", min_value=0.0,
                                           max_value=float(master["activity_budget"].max()), value=500.0)
            unit_budget  = st.number_input("Unit budget", min_value=0.0,
                                           max_value=float(master["unit_budget"].max()), value=1000.0)

        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([{
            "activity_type":  activity_type,
            "country":        country,
            "unit_name":      unit_name,
            "genre":          genre,
            "tranche_age":    tranche_age,
            "saison":         saison,
            "categorie":      categorie,
            "month":          month,
            "year":           year,
            "dayofweek":      dayofweek,
            "duration":       duration,
            "activity_budget": act_budget,
            "unit_budget":    unit_budget,
        }])

        st.markdown("---")
        st.markdown("#### 🎯 Predictions")
        col1, col2 = st.columns(2)

        # Classification predictions
        with col1:
            st.markdown("**Participation class**")
            for name, res in cls.items():
                pred_class = le.inverse_transform(res["model"].predict(input_df))[0]
                proba      = res["model"].predict_proba(input_df)[0]
                color = {"low": "#ef4444", "medium": "#f59e0b", "high": "#22c55e"}.get(pred_class, ACCENT)
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="label">{name}</div>
                        <div class="value" style="color:{color}">{pred_class.upper()}</div>
                        <div class="sub">Confidence: {max(proba)*100:.1f}%</div>
                    </div>""", unsafe_allow_html=True
                )
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Regression predictions
        with col2:
            st.markdown("**Estimated participant count**")
            for name, res in reg.items():
                pred_count = max(0, res["model"].predict(input_df)[0])
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="label">{name}</div>
                        <div class="value">{pred_count:.0f}</div>
                        <div class="sub">participants expected</div>
                    </div>""", unsafe_allow_html=True
                )
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
