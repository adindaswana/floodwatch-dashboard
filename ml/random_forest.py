# ==============================================================================
# ml/random_forest.py — Model 1: Random Forest Klasifikasi Risiko Banjir
# ==============================================================================
# Target  : status_risiko (rendah / sedang / tinggi / kritis) — 4 kelas
# Fitur   : avg_rainfall, lag, rolling avg, elevasi, NDVI, slope, soil, musim, landcover
# Library : scikit-learn
# Output  : model (.pkl) + laporan evaluasi
# ==============================================================================

import os, logging, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score)
from sklearn.inspection import permutation_importance

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PATHS

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# --- Kolom fitur yang digunakan ---
FITUR_COLS = [
    "avg_rainfall", "max_rainfall", "avg_temperature",
    "elevation", "ndvi", "slope", "soil_moisture",
    "month", "musim_encoded",
    "hujan_lag1", "hujan_lag3", "hujan_lag7",
    "rolling_avg_3", "rolling_avg_7", "rolling_avg_14",
    "lc_built_up", "lc_tree_cover", "lc_cropland",
    "lc_permanent_water_bodies", "lc_unknown",
    "risiko_index",
]
TARGET_COL = "status_risiko"
ORDER_KELAS = ["rendah", "sedang", "tinggi", "kritis"]


def load_features(path: str = None) -> pd.DataFrame:
    """Load feature set dari CSV."""
    if path is None:
        feat_dir = PATHS["features"]
        files = sorted([f for f in os.listdir(feat_dir) if f.endswith(".csv")])
        if not files:
            raise FileNotFoundError(f"Tidak ada file fitur di {feat_dir}")
        path = os.path.join(feat_dir, files[-1])  # file terbaru
    log.info(f"Load feature set: {path}")
    df = pd.read_csv(path)
    log.info(f"  Shape: {df.shape}")
    return df


def prepare_data(df: pd.DataFrame):
    """Persiapkan X dan y untuk training."""
    fitur_ada = [c for c in FITUR_COLS if c in df.columns]
    log.info(f"Fitur digunakan ({len(fitur_ada)}): {fitur_ada}")

    X = df[fitur_ada].copy()
    y = df[TARGET_COL].copy()

    # Drop baris dengan NaN
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]
    log.info(f"  Setelah drop NaN: {len(X)} baris")

    # Encode label (sudah string, perlu jadi int untuk beberapa metrik)
    le = LabelEncoder()
    le.classes_ = np.array(ORDER_KELAS)
    y_enc = le.transform(y.values)

    return X, y, y_enc, le, fitur_ada


def train_random_forest(df: pd.DataFrame = None, path_fitur: str = None):
    """
    Training Random Forest untuk klasifikasi risiko banjir.

    Args:
        df: DataFrame fitur (jika sudah di-load). Jika None, load dari file.
        path_fitur: Path file CSV fitur (opsional).

    Returns:
        dict berisi model, encoder, hasil evaluasi, dan path model tersimpan.
    """
    if df is None:
        df = load_features(path_fitur)

    log.info("=" * 55)
    log.info("TRAINING — Random Forest Klasifikasi Risiko")
    log.info("=" * 55)

    X, y, y_enc, le, fitur_ada = prepare_data(df)

    # --- Distribusi kelas ---
    log.info("Distribusi kelas:")
    for kls in ORDER_KELAS:
        n = (y == kls).sum()
        log.info(f"  {kls:<10}: {n:,} ({n/len(y)*100:.1f}%)")

    # --- Split train/test stratified ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y  # stratified agar proporsi kelas seimbang
    )
    log.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # --- Model ---
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",   # tangani imbalanced class
        random_state=42,
        n_jobs=-1,
    )

    log.info("Training Random Forest...")
    model.fit(X_train, y_train)
    log.info("Training selesai!")

    # --- Evaluasi ---
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1_w   = f1_score(y_test, y_pred, average="weighted", labels=ORDER_KELAS, zero_division=0)
    f1_m   = f1_score(y_test, y_pred, average="macro",    labels=ORDER_KELAS, zero_division=0)

    log.info(f"\n=== HASIL EVALUASI RANDOM FOREST ===")
    log.info(f"Accuracy        : {acc:.4f} ({acc*100:.2f}%)")
    log.info(f"F1 Weighted     : {f1_w:.4f}")
    log.info(f"F1 Macro        : {f1_m:.4f}")
    log.info("\nClassification Report:")
    report = classification_report(y_test, y_pred, labels=ORDER_KELAS, zero_division=0)
    log.info("\n" + report)

    # --- Cross Validation (5-fold) ---
    log.info("Cross Validation (5-fold, scoring=f1_weighted)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)
    log.info(f"CV F1 Scores : {[round(s,4) for s in cv_scores]}")
    log.info(f"CV Mean±Std  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # --- Feature Importance ---
    feat_imp = pd.DataFrame({
        "fitur":      fitur_ada,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    log.info("\nTop 10 Feature Importance:")
    log.info(feat_imp.head(10).to_string(index=False))

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred, labels=ORDER_KELAS)

    # --- Simpan plot ---
    os.makedirs(PATHS["models"], exist_ok=True)
    _plot_confusion_matrix(cm, ORDER_KELAS)
    _plot_feature_importance(feat_imp)

    # --- Simpan model ---
    model_path = os.path.join(PATHS["models"], "random_forest.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "label_encoder": le,
                     "fitur_cols": fitur_ada}, f)
    log.info(f"\\nModel tersimpan: {model_path}")

    # --- Simpan meta json ---
    import json
    meta = {
        "fitur_cols": fitur_ada,
        "accuracy":   round(acc, 4),
        "f1_weighted": round(f1_w, 4),
        "f1_macro":    round(f1_m, 4),
        "cv_mean":    round(cv_scores.mean(), 4),
        "cv_std":     round(cv_scores.std(), 4),
    }
    with open(os.path.join(PATHS["models"], "rf_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Metadata tersimpan: {os.path.join(PATHS['models'], 'rf_meta.json')}")

    return {
        "model":         model,
        "label_encoder": le,
        "fitur_cols":    fitur_ada,
        "accuracy":      acc,
        "f1_weighted":   f1_w,
        "f1_macro":      f1_m,
        "cv_mean":       cv_scores.mean(),
        "cv_std":        cv_scores.std(),
        "feat_importance": feat_imp,
        "confusion_matrix": cm,
        "model_path":    model_path,
    }


def predict_risiko(model_path: str, X_baru: pd.DataFrame) -> pd.Series:
    """
    Prediksi status risiko banjir menggunakan model yang sudah disimpan.

    Args:
        model_path: Path file .pkl model.
        X_baru: DataFrame dengan kolom fitur yang sama saat training.

    Returns:
        Series prediksi label risiko.
    """
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    model    = bundle["model"]
    fitur    = bundle["fitur_cols"]
    X_input  = X_baru[[c for c in fitur if c in X_baru.columns]]
    y_pred   = model.predict(X_input)
    y_prob   = model.predict_proba(X_input)

    result = pd.DataFrame({
        "prediksi_risiko": y_pred,
        "prob_rendah":  y_prob[:, 0].round(3),
        "prob_sedang":  y_prob[:, 1].round(3),
        "prob_tinggi":  y_prob[:, 2].round(3),
        "prob_kritis":  y_prob[:, 3].round(3),
    })
    return result


def _plot_confusion_matrix(cm: np.ndarray, kelas: list):
    """Plot dan simpan confusion matrix."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=kelas, yticklabels=kelas, ax=ax)
    ax.set_title("Confusion Matrix — Random Forest", fontsize=13)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    plt.tight_layout()
    path = os.path.join(PATHS["models"], "rf_confusion_matrix.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Plot confusion matrix: {path}")


def _plot_feature_importance(feat_imp: pd.DataFrame, top_n: int = 15):
    """Plot feature importance horizontal bar chart."""
    top = feat_imp.head(top_n)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["fitur"][::-1], top["importance"][::-1], color="#1F4E79")
    ax.set_title(f"Top {top_n} Feature Importance — Random Forest", fontsize=13)
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PATHS["models"], "rf_feature_importance.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Plot feature importance: {path}")


if __name__ == "__main__":
    hasil = train_random_forest()
    print(f"\nAccuracy : {hasil['accuracy']:.4f}")
    print(f"F1 (W)   : {hasil['f1_weighted']:.4f}")
    print(f"CV Mean  : {hasil['cv_mean']:.4f} ± {hasil['cv_std']:.4f}")
