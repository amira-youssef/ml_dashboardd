"""
train.py — Run this ONCE to fit all models and save them to the models/ folder.
Usage:  python train.py
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, davies_bouldin_score
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────
print("Loading data...")
BASE_DIR = Path("data")

dim_unit        = pd.read_csv(BASE_DIR / "dim_unit.csv")
dim_temps       = pd.read_csv(BASE_DIR / "dim_temps.csv")
dim_country     = pd.read_csv(BASE_DIR / "dim_country.csv")
dim_adherent    = pd.read_csv(BASE_DIR / "dim_adherent.csv")
fact_activities = pd.read_csv(BASE_DIR / "fact_activities.csv")
dim_budget      = pd.read_csv(BASE_DIR / "dim_budget.csv")
dim_activity    = pd.read_csv(BASE_DIR / "dim_activity.csv")
fact_finance    = pd.read_csv(BASE_DIR / "fact_finance.csv")

# ─────────────────────────────────────────────────────────────
# 2. BUILD MASTER TABLE
# ─────────────────────────────────────────────────────────────
print("Building master table...")
dim_unit_clean = dim_unit.rename(columns={"budjet": "unit_budget", "unite_name": "unit_name"})

master = (
    fact_activities
    .merge(dim_activity, on="activity_id", how="left", suffixes=("_fact", "_dim"))
    .merge(dim_temps[["time_id", "date", "jour", "mois", "annee", "saison"]], on="time_id", how="left")
    .merge(dim_unit_clean, on="unite_id", how="left")
    .merge(dim_country, on="country_id", how="left")
)

master["event_date"]      = pd.to_datetime(master["date_y"])
master["participants"]    = master["nb_participants"].combine_first(master["nbr_participant"])
master["activity_budget"] = master["budget_activite"].combine_first(master["budjet"])
master["duration"]        = master["durees_activite"].combine_first(master["duration_days"])
master["month"]           = master["event_date"].dt.month
master["year"]            = master["event_date"].dt.year
master["dayofweek"]       = master["event_date"].dt.dayofweek

# Outlier capping (IQR)
for col in ["participants", "activity_budget", "duration"]:
    q1, q3 = master[col].quantile(0.25), master[col].quantile(0.75)
    iqr = q3 - q1
    master[col] = master[col].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)

# ─────────────────────────────────────────────────────────────
# 3. CLASSIFICATION
# ─────────────────────────────────────────────────────────────
print("Training classification models...")

feature_cols  = ["activity_type", "country", "unit_name", "genre", "tranche_age",
                 "saison", "categorie", "month", "year", "dayofweek",
                 "duration", "activity_budget", "unit_budget"]
cat_features  = ["activity_type", "country", "unit_name", "genre",
                 "tranche_age", "saison", "categorie"]
num_features  = ["month", "year", "dayofweek", "duration", "activity_budget", "unit_budget"]

class_df = master.dropna(subset=["participants"]).copy()
class_df["participation_class"] = pd.qcut(
    class_df["participants"], q=3, labels=["low", "medium", "high"]
)

label_encoder = LabelEncoder()
y_cls = label_encoder.fit_transform(class_df["participation_class"])
X_cls = class_df[feature_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_features)
])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cls_models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        {"model__C": [0.5, 1.0, 2.0]}
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"),
        {"model__n_estimators": [100, 150], "model__max_depth": [None, 12]}
    )
}

cls_results = {}
for name, (model, params) in cls_models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    search = GridSearchCV(pipe, param_grid=params, cv=skf,
                         scoring="f1_macro", n_jobs=-1)
    search.fit(X_train, y_train)
    pred  = search.predict(X_test)
    proba = search.predict_proba(X_test)
    cls_results[name] = {
        "model":     search.best_estimator_,
        "accuracy":  accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, average="macro"),
        "recall":    recall_score(y_test, pred, average="macro"),
        "f1":        f1_score(y_test, pred, average="macro"),
        "auc":       roc_auc_score(y_test, proba, multi_class="ovr", average="macro"),
        "pred":      pred,
        "proba":     proba,
        "y_test":    y_test,
    }
    print(f"  {name} — F1: {cls_results[name]['f1']:.4f}")

# ─────────────────────────────────────────────────────────────
# 4. REGRESSION
# ─────────────────────────────────────────────────────────────
print("Training regression models...")

reg_df = master.dropna(subset=["participants"]).copy()
X_reg, y_reg = reg_df[feature_cols], reg_df["participants"]
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
reg_models = {
    "Ridge": (Ridge(), {"model__alpha": [0.5, 1.0, 2.0]}),
    "Random Forest Regressor": (
        RandomForestRegressor(random_state=42, n_jobs=-1),
        {"model__n_estimators": [100, 150], "model__max_depth": [None, 12]}
    )
}

reg_results = {}
for name, (model, params) in reg_models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    search = GridSearchCV(pipe, param_grid=params, cv=kf,
                         scoring="neg_root_mean_squared_error", n_jobs=-1)
    search.fit(X_train_r, y_train_r)
    pred = search.predict(X_test_r)
    reg_results[name] = {
        "model": search.best_estimator_,
        "mse":   mean_squared_error(y_test_r, pred),
        "rmse":  mean_squared_error(y_test_r, pred) ** 0.5,
        "mae":   mean_absolute_error(y_test_r, pred),
        "r2":    r2_score(y_test_r, pred),
        "pred":  pred,
        "y_test": y_test_r,
    }
    print(f"  {name} — R²: {reg_results[name]['r2']:.4f}")

# ─────────────────────────────────────────────────────────────
# 5. CLUSTERING
# ─────────────────────────────────────────────────────────────
print("Running clustering...")

cluster_df = (
    master.groupby(["unite_id", "unit_name", "genre", "tranche_age", "saison"], as_index=False)
    .agg(
        nb_events=("activity_id", "count"),
        total_participants=("participants", "sum"),
        avg_participants=("participants", "mean"),
        avg_budget=("activity_budget", "mean"),
        avg_duration=("duration", "mean"),
        total_budget=("activity_budget", "sum"),
        active_countries=("country_id", "nunique"),
    )
)

cluster_features = ["nb_events", "total_participants", "avg_participants",
                    "avg_budget", "avg_duration", "total_budget", "active_countries"]

X_cluster = SimpleImputer(strategy="median").fit_transform(cluster_df[cluster_features])
cluster_scaler = StandardScaler()
X_cluster_scaled = cluster_scaler.fit_transform(X_cluster)

# Choose best k
sil_by_k = {}
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    sil_by_k[k] = silhouette_score(X_cluster_scaled, km.fit_predict(X_cluster_scaled))
best_k = max(sil_by_k, key=sil_by_k.get)

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_cluster_scaled)
agglo = AgglomerativeClustering(n_clusters=best_k)
agglo_labels = agglo.fit_predict(X_cluster_scaled)

cluster_df["kmeans_cluster"] = kmeans_labels
cluster_df["agglo_cluster"]  = agglo_labels

pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(X_cluster_scaled)

print(f"  Best k={best_k} — KMeans silhouette: {silhouette_score(X_cluster_scaled, kmeans_labels):.4f}")

# ─────────────────────────────────────────────────────────────
# 6. TIME SERIES
# ─────────────────────────────────────────────────────────────
print("Fitting time series models...")

ts_df = master.dropna(subset=["event_date", "participants"]).copy()
daily = (
    ts_df.groupby("event_date")
    .agg(total_participants=("participants", "sum"))
    .sort_index()
)
full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
daily = daily.reindex(full_idx, fill_value=0)
series = daily["total_participants"]

train_ts = series.iloc[:-90]
test_ts  = series.iloc[-90:]

def ts_metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(1, np.abs(y_true)))) * 100
    return {"rmse": rmse, "mae": mae, "mape": mape}

arima_model  = ARIMA(train_ts, order=(2, 1, 1)).fit()
arima_pred   = arima_model.forecast(steps=len(test_ts))
sarima_model = ARIMA(train_ts, order=(2, 1, 1), seasonal_order=(1, 1, 1, 7)).fit()
sarima_pred  = sarima_model.forecast(steps=len(test_ts))

ts_results = {
    "ARIMA":  {**ts_metrics(test_ts, arima_pred),  "pred": arima_pred},
    "SARIMA": {**ts_metrics(test_ts, sarima_pred), "pred": sarima_pred},
}
print(f"  ARIMA  RMSE: {ts_results['ARIMA']['rmse']:.2f}")
print(f"  SARIMA RMSE: {ts_results['SARIMA']['rmse']:.2f}")

# ─────────────────────────────────────────────────────────────
# 7. SAVE ALL ARTIFACTS
# ─────────────────────────────────────────────────────────────
print("Saving artifacts to models/...")
Path("models").mkdir(exist_ok=True)

joblib.dump(cls_results,     "models/cls_results.pkl")
joblib.dump(label_encoder,   "models/label_encoder.pkl")
joblib.dump(reg_results,     "models/reg_results.pkl")
joblib.dump(cluster_df,      "models/cluster_df.pkl")
joblib.dump(cluster_scaler,  "models/cluster_scaler.pkl")
joblib.dump(kmeans,          "models/kmeans.pkl")
joblib.dump(pca_coords,      "models/pca_coords.pkl")
joblib.dump(cluster_features,"models/cluster_features.pkl")
joblib.dump(sil_by_k,        "models/sil_by_k.pkl")
joblib.dump(feature_cols,    "models/feature_cols.pkl")
joblib.dump(cat_features,    "models/cat_features.pkl")
joblib.dump(num_features,    "models/num_features.pkl")
joblib.dump(ts_results,      "models/ts_results.pkl")
joblib.dump({"train": train_ts, "test": test_ts}, "models/ts_data.pkl")
joblib.dump(master,          "models/master.pkl")

print("✅ All artifacts saved. You can now run: streamlit run app.py")
