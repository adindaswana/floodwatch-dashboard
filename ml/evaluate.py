# ==============================================================================
# ml/evaluate.py — Evaluasi & Perbandingan Ketiga Model ML
# ==============================================================================
# Menghasilkan tabel perbandingan + plot metrik semua model
# ==============================================================================

import os, json, logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PATHS

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def baca_meta_model() -> pd.DataFrame:
    """Baca metadata hasil evaluasi semua model dari file JSON."""
    models_dir = PATHS["models"]
    rows = []

    # Random Forest — dari file pkl (ekstrak metadata manual)
    rf_pkl = os.path.join(models_dir, "random_forest.pkl")
    if os.path.exists(rf_pkl):
        rows.append({
            "Model":     "Random Forest",
            "Tugas":     "Klasifikasi Risiko (4 kelas)",
            "Accuracy":  None,
            "F1":        None,
            "AUC-ROC":   None,
            "CV Score":  None,
            "Status":    "Tersimpan",
        })

    # LSTM — dari lstm_meta.json
    lstm_meta = os.path.join(models_dir, "lstm_meta.json")
    if os.path.exists(lstm_meta):
        with open(lstm_meta) as f:
            m = json.load(f)
        rows.append({
            "Model":    "LSTM/GRU",
            "Tugas":    "Prediksi Time-Series (biner)",
            "Accuracy": m.get("accuracy"),
            "F1":       m.get("f1"),
            "AUC-ROC":  m.get("auc"),
            "CV Score": None,
            "Status":   "Tersimpan",
        })

    # XGBoost — dari xgboost_meta.json
    xgb_meta = os.path.join(models_dir, "xgboost_meta.json")
    if os.path.exists(xgb_meta):
        with open(xgb_meta) as f:
            m = json.load(f)
        rows.append({
            "Model":    "XGBoost",
            "Tugas":    "Deteksi Anomali Cuaca",
            "Accuracy": m.get("accuracy"),
            "F1":       m.get("f1"),
            "AUC-ROC":  m.get("auc_roc"),
            "CV Score": m.get("cv_auc_mean"),
            "Status":   "Tersimpan",
        })

    return pd.DataFrame(rows)


def plot_perbandingan_model(hasil_rf: dict, hasil_xgb: dict):
    """
    Buat bar chart perbandingan metrik: Accuracy, F1, AUC.
    Hanya model yang sudah ditraining yang ditampilkan.
    """
    data = {}

    if hasil_rf:
        data["Random Forest"] = {
            "Accuracy":  hasil_rf.get("accuracy", 0),
            "F1 Weighted": hasil_rf.get("f1_weighted", 0),
            "CV F1":     hasil_rf.get("cv_mean", 0),
        }

    if hasil_xgb:
        data["XGBoost"] = {
            "Accuracy":  hasil_xgb.get("accuracy", 0),
            "F1 Weighted": hasil_xgb.get("f1", 0),
            "AUC-ROC":   hasil_xgb.get("auc", 0),
        }

    if not data:
        log.warning("Tidak ada model yang bisa dibandingkan.")
        return

    df_plot = pd.DataFrame(data).T.fillna(0)
    metrics = df_plot.columns.tolist()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df_plot))
    bar_w = 0.25
    colors = ["#1F4E79", "#0D6E6E", "#7F4F00"]

    for i, metric in enumerate(metrics):
        bars = ax.bar(x + i * bar_w, df_plot[metric], bar_w,
                      label=metric, color=colors[i % len(colors)])
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.text(b.get_x() + b.get_width()/2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + bar_w)
    ax.set_xticklabels(df_plot.index, fontsize=11)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_title("Perbandingan Metrik Model Machine Learning", fontsize=13)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(PATHS["models"], "perbandingan_model.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Plot perbandingan: {path}")
    return path


def cetak_laporan_evaluasi(hasil_rf: dict = None,
                            hasil_lstm: dict = None,
                            hasil_xgb: dict = None):
    """Cetak tabel ringkasan evaluasi semua model ke log."""
    log.info("\n" + "=" * 65)
    log.info("LAPORAN EVALUASI MKPL — SISTEM DETEKSI DINI BANJIR")
    log.info("=" * 65)
    log.info(f"{'Model':<20} {'Accuracy':>10} {'F1':>10} {'AUC':>10} {'CV':>10}")
    log.info("-" * 65)

    def _baris(nama, hasil, f1_key="f1_weighted", auc_key="auc", cv_key="cv_mean"):
        if hasil is None:
            log.info(f"  {nama:<18} {'(belum ditraining)'}")
            return
        acc = hasil.get("accuracy",  "-")
        f1  = hasil.get(f1_key,      "-")
        auc = hasil.get(auc_key,     "-")
        cv  = hasil.get(cv_key,      "-")
        fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
        log.info(f"  {nama:<18} {fmt(acc):>10} {fmt(f1):>10} {fmt(auc):>10} {fmt(cv):>10}")

    _baris("Random Forest",  hasil_rf,   f1_key="f1_weighted", auc_key="auc", cv_key="cv_mean")
    _baris("LSTM/GRU",       hasil_lstm, f1_key="f1",          auc_key="auc", cv_key="cv_mean")
    _baris("XGBoost",        hasil_xgb,  f1_key="f1",          auc_key="auc", cv_key="cv_auc_mean")

    log.info("=" * 65)
    log.info("Keterangan:")
    log.info("  Random Forest  → Klasifikasi risiko 4 kelas")
    log.info("  LSTM/GRU       → Prediksi banjir berbasis time-series")
    log.info("  XGBoost        → Deteksi anomali cuaca ekstrem + SHAP")
    log.info("=" * 65)


def run_all_ml(df_fitur: pd.DataFrame,
               df_cuaca: pd.DataFrame,
               jalankan_lstm: bool = True) -> dict:
    """
    Jalankan training semua model ML secara berurutan.

    Args:
        df_fitur   : Feature set dari ETL pipeline (combine + feature eng).
        df_cuaca   : Data cuaca harian (dari master_cuaca).
        jalankan_lstm: Set False untuk skip LSTM (lebih cepat, butuh GPU).

    Returns:
        dict berisi semua hasil model.
    """
    from ml.random_forest import train_random_forest
    from ml.xgboost_model import train_xgboost

    log.info("=" * 55)
    log.info("TAHAP 4: MACHINE LEARNING")
    log.info("=" * 55)

    # --- Model 1: Random Forest ---
    log.info("\n[ML 1/3] Random Forest...")
    hasil_rf = train_random_forest(df_fitur)

    # --- Model 2: LSTM (opsional, butuh tensorflow) ---
    hasil_lstm = None
    if jalankan_lstm:
        log.info("\n[ML 2/3] LSTM/GRU...")
        try:
            from ml.lstm_model import train_lstm
            hasil_lstm = train_lstm(df_fitur)
        except ImportError:
            log.warning("TensorFlow tidak tersedia. LSTM dilewati.")
        except Exception as e:
            log.warning(f"LSTM error: {e}. Dilewati.")
    else:
        log.info("[ML 2/3] LSTM dilewati (jalankan_lstm=False)")

    # --- Model 3: XGBoost ---
    log.info("\n[ML 3/3] XGBoost...")
    hasil_xgb = train_xgboost(df_cuaca, df_fitur)

    # --- Evaluasi & perbandingan ---
    cetak_laporan_evaluasi(hasil_rf, hasil_lstm, hasil_xgb)
    plot_perbandingan_model(hasil_rf, hasil_xgb)

    return {
        "random_forest": hasil_rf,
        "lstm":          hasil_lstm,
        "xgboost":       hasil_xgb,
    }
