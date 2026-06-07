# ==============================================================================
# ml/xgboost_model.py — Model 3: XGBoost Deteksi Anomali Cuaca Ekstrem
# ==============================================================================
# Target  : label_banjir (0/1) — deteksi kondisi cuaca ekstrem
# Sumber  : master_cuaca_training (data harian: suhu, hujan, kelembapan, angin)
# Library : xgboost + shap (explainability)
# Output  : model (.json) + SHAP plot + ROC curve
# ==============================================================================

import os, logging, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve,
                              average_precision_score, confusion_matrix,
                              accuracy_score, f1_score, precision_recall_curve)
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PATHS

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Fitur dari data cuaca harian (master_cuaca)
FITUR_CUACA = [
    "suhu_celcius", "kelembapan_persen",
    "curah_hujan_mm", "kecepatan_angin_kmh",
]

# Fitur tambahan dari combine (jika tersedia)
FITUR_COMBINE = [
    "avg_rainfall", "max_rainfall", "rolling_avg_3", "rolling_avg_7",
    "elevation", "slope", "ndvi", "soil_moisture", "musim_encoded",
    "hujan_lag1", "hujan_lag3", "risiko_index",
]

TARGET_COL = "label_banjir"


def load_cuaca_harian(path: str = None) -> pd.DataFrame:
    """Load data cuaca harian yang sudah di-transform."""
    if path is None:
        p = os.path.join(PATHS["processed"])
        files = sorted([f for f in os.listdir(p) if "cuaca" in f and f.endswith(".csv")])
        if not files:
            raise FileNotFoundError(f"Tidak ada file cuaca di {p}")
        path = os.path.join(p, files[-1])
    log.info(f"Load cuaca harian: {path}")
    df = pd.read_csv(path)
    log.info(f"  Shape: {df.shape}")
    return df


def load_features_combine(path: str = None) -> pd.DataFrame:
    """Load feature set dari combine dataset."""
    if path is None:
        feat_dir = PATHS["features"]
        files = sorted([f for f in os.listdir(feat_dir) if f.endswith(".csv")])
        if not files:
            return pd.DataFrame()
        path = os.path.join(feat_dir, files[-1])
    return pd.read_csv(path)


def prepare_data_xgb(df_cuaca: pd.DataFrame,
                     df_combine: pd.DataFrame = None):
    """
    Gabungkan data cuaca harian dengan combine features (jika tersedia).
    Pilih fitur yang relevan dan bersih.

    Returns:
        X, y, fitur_cols
    """
    df = df_cuaca.copy()

    # Tambah fitur dari combine jika tersedia dan cukup baris
    if df_combine is not None and not df_combine.empty and len(df_combine) > 50:
        fitur_combine_ada = [c for c in FITUR_COMBINE if c in df_combine.columns]
        if fitur_combine_ada and TARGET_COL in df_combine.columns:
            # Gunakan dataset combine sebagai sumber utama jika lebih besar
            if len(df_combine) > len(df_cuaca):
                log.info("  Menggunakan dataset combine sebagai sumber utama XGBoost")
                df = df_combine.copy()

    # Tentukan fitur yang tersedia
    semua_fitur = FITUR_CUACA + FITUR_COMBINE
    fitur_ada   = [c for c in semua_fitur if c in df.columns]

    if not fitur_ada:
        raise ValueError("Tidak ada fitur yang ditemukan di dataset!")

    log.info(f"Fitur XGBoost ({len(fitur_ada)}): {fitur_ada}")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Kolom target '{TARGET_COL}' tidak ditemukan.")

    X = df[fitur_ada].copy()
    y = df[TARGET_COL].astype(int)

    # Drop NaN
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]
    log.info(f"  Setelah drop NaN: {len(X):,} baris")
    log.info(f"  Distribusi: 0={int((y==0).sum()):,}  1={int((y==1).sum()):,}")

    return X, y, fitur_ada


def train_xgboost(df_cuaca: pd.DataFrame = None,
                  df_combine: pd.DataFrame = None):
    """
    Training XGBoost untuk deteksi anomali cuaca ekstrem / prediksi banjir.

    Keunggulan vs Random Forest:
    - Lebih cepat pada data besar
    - SHAP values untuk explainability per prediksi
    - Penanganan imbalanced class via scale_pos_weight

    Returns:
        dict berisi model, evaluasi, shap_values, path model.
    """
    log.info("=" * 55)
    log.info("TRAINING — XGBoost Deteksi Anomali Cuaca")
    log.info("=" * 55)

    # Load data jika belum disediakan
    if df_cuaca is None:
        df_cuaca = load_cuaca_harian()
    if df_combine is None:
        df_combine = load_features_combine()

    X, y, fitur_ada = prepare_data_xgb(df_cuaca, df_combine)

    # --- Split stratified ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    log.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # --- Hitung scale_pos_weight untuk imbalanced ---
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    spw   = n_neg / max(n_pos, 1)
    log.info(f"scale_pos_weight: {spw:.2f} (neg={n_neg}, pos={n_pos})")

    # --- Model XGBoost ---
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        scale_pos_weight=spw,
        use_label_encoder=False,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    # --- Training dengan early stopping ---
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False,
    )
    log.info("Training XGBoost selesai!")

    # --- Evaluasi ---
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc   = accuracy_score(y_test, y_pred)
    f1    = f1_score(y_test, y_pred, zero_division=0)
    auc   = roc_auc_score(y_test, y_prob)
    ap    = average_precision_score(y_test, y_prob)
    report = classification_report(y_test, y_pred,
                                   target_names=["normal","anomali_banjir"],
                                   zero_division=0)

    log.info(f"\n=== HASIL EVALUASI XGBOOST ===")
    log.info(f"Accuracy          : {acc:.4f}")
    log.info(f"F1 Score          : {f1:.4f}")
    log.info(f"AUC-ROC           : {auc:.4f}")
    log.info(f"Avg Precision (AP): {ap:.4f}")
    log.info(f"\n{report}")

    # --- Cross Validation ---
    log.info("Cross Validation (5-fold, AUC)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    log.info(f"CV AUC: {[round(s,4) for s in cv_auc]}")
    log.info(f"CV Mean±Std: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

    # --- Feature Importance ---
    feat_imp = pd.DataFrame({
        "fitur":      fitur_ada,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    log.info("\nTop 10 Feature Importance (XGBoost):")
    log.info(feat_imp.head(10).to_string(index=False))

    # --- Plot ---
    os.makedirs(PATHS["models"], exist_ok=True)
    _plot_roc_curve(y_test, y_prob, auc)
    _plot_pr_curve(y_test, y_prob, ap)
    _plot_xgb_feature_importance(feat_imp)

    # --- SHAP Explainability ---
    shap_vals = None
    try:
        import shap
        log.info("Menghitung SHAP values...")
        explainer  = shap.TreeExplainer(model)
        X_sample   = X_test.sample(min(500, len(X_test)), random_state=42)
        shap_vals  = explainer.shap_values(X_sample)
        _plot_shap(shap_vals, X_sample, fitur_ada)
        log.info("SHAP values selesai.")
    except ImportError:
        log.warning("Library 'shap' tidak terinstall. Skip SHAP analysis.")
        log.info("Install: pip install shap")
    except Exception as e:
        log.warning(f"SHAP error: {e}")

    # --- Simpan model ---
    model_path = os.path.join(PATHS["models"], "xgboost_model.json")
    model.save_model(model_path)

    meta = {
        "fitur_cols": fitur_ada,
        "accuracy":   round(acc, 4),
        "f1":         round(f1,  4),
        "auc_roc":    round(auc, 4),
        "avg_precision": round(ap, 4),
        "cv_auc_mean": round(cv_auc.mean(), 4),
        "cv_auc_std":  round(cv_auc.std(),  4),
    }
    with open(os.path.join(PATHS["models"], "xgboost_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"Model XGBoost tersimpan: {model_path}")

    return {
        "model":       model,
        "accuracy":    acc,
        "f1":          f1,
        "auc":         auc,
        "avg_precision": ap,
        "cv_auc_mean": cv_auc.mean(),
        "feat_importance": feat_imp,
        "shap_values": shap_vals,
        "model_path":  model_path,
    }


def predict_anomali(model_path: str, X_baru: pd.DataFrame,
                    threshold: float = 0.5) -> pd.DataFrame:
    """
    Prediksi skor anomali cuaca menggunakan model XGBoost.

    Args:
        model_path: Path file .json model.
        X_baru    : DataFrame fitur cuaca baru.
        threshold : Ambang batas skor anomali (default 0.5).

    Returns:
        DataFrame berisi skor_anomali, prediksi, dan level_alert.
    """
    model = XGBClassifier()
    model.load_model(model_path)

    with open(os.path.join(PATHS["models"], "xgboost_meta.json")) as f:
        meta = json.load(f)

    fitur = [c for c in meta["fitur_cols"] if c in X_baru.columns]
    X     = X_baru[fitur]

    skor = model.predict_proba(X)[:, 1]
    pred = (skor >= threshold).astype(int)

    return pd.DataFrame({
        "skor_anomali": skor.round(4),
        "is_anomali":   pred.astype(bool),
        "level_alert":  pd.cut(skor,
                               bins=[0, 0.3, 0.5, 0.75, 1.01],
                               labels=["Normal","Waspada","Siaga","Awas"]),
    })


# --- PLOT HELPERS ---
def _plot_roc_curve(y_true, y_prob, auc_score):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#1F4E79", lw=2, label=f"AUC = {auc_score:.4f}")
    ax.plot([0,1],[0,1], "k--", lw=1, alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — XGBoost", fontsize=12)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(PATHS["models"], "xgb_roc_curve.png"), dpi=120)
    plt.close(fig)


def _plot_pr_curve(y_true, y_prob, ap_score):
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, color="#0D6E6E", lw=2, label=f"AP = {ap_score:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — XGBoost", fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(PATHS["models"], "xgb_pr_curve.png"), dpi=120)
    plt.close(fig)


def _plot_xgb_feature_importance(feat_imp: pd.DataFrame, top_n: int = 15):
    top = feat_imp.head(top_n)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["fitur"][::-1], top["importance"][::-1], color="#0D6E6E")
    ax.set_title(f"Top {top_n} Feature Importance — XGBoost", fontsize=12)
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(PATHS["models"], "xgb_feature_importance.png"), dpi=120)
    plt.close(fig)


def _plot_shap(shap_values, X_sample, fitur_ada):
    try:
        import shap
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(shap_values, X_sample,
                          feature_names=fitur_ada,
                          plot_type="bar", show=False)
        plt.title("SHAP Feature Importance — XGBoost")
        plt.tight_layout()
        path = os.path.join(PATHS["models"], "xgb_shap_summary.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        log.info(f"Plot SHAP: {path}")
    except Exception as e:
        log.warning(f"Plot SHAP gagal: {e}")


if __name__ == "__main__":
    hasil = train_xgboost()
    print(f"\nAccuracy : {hasil['accuracy']:.4f}")
    print(f"F1       : {hasil['f1']:.4f}")
    print(f"AUC-ROC  : {hasil['auc']:.4f}")
