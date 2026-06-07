# ==============================================================================
# ml/lstm_model.py — Model 2: LSTM/GRU Prediksi Deret Waktu
# ==============================================================================
# Target  : label_banjir (0/1) berbasis pola time-series curah hujan
# Arsitektur: Input → LSTM(64) → Dropout(0.2) → GRU(32) → Dense(1, sigmoid)
# Library : TensorFlow/Keras
# Window  : 7 bulan sebelumnya sebagai sequence input
# ==============================================================================

import os, logging
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

FITUR_COLS = [
    "avg_rainfall", "hujan_lag1", "hujan_lag3", "hujan_lag7",
    "rolling_avg_3", "rolling_avg_7", "rolling_avg_14",
    "avg_temperature", "soil_moisture", "slope", "ndvi",
    "musim_encoded",
]
TARGET_COL = "label_banjir"
WINDOW     = 7      # Panjang sequence (bulan)


def load_features(path: str = None) -> pd.DataFrame:
    if path is None:
        feat_dir = PATHS["features"]
        files = sorted([f for f in os.listdir(feat_dir) if f.endswith(".csv")])
        if not files:
            raise FileNotFoundError(f"Tidak ada file fitur di {feat_dir}")
        path = os.path.join(feat_dir, files[-1])
    df = pd.read_csv(path)
    return df


def buat_sequences(df: pd.DataFrame, window: int = WINDOW):
    """
    Buat sequence time-series per lokasi (kabupaten+kecamatan).

    Untuk setiap lokasi:
      - Urutkan berdasarkan year + month
      - Buat sliding window: X[i] = fitur t-(window-1)..t, y[i] = label_banjir[t]

    Returns:
        X: array (n_samples, window, n_fitur)
        y: array (n_samples,) — label biner
    """
    fitur_ada = [c for c in FITUR_COLS if c in df.columns]
    group_cols = [c for c in ["kabupaten","kecamatan"] if c in df.columns]

    X_all, y_all = [], []

    for _, grup in df.groupby(group_cols):
        g = grup.sort_values(["year","month"]).reset_index(drop=True)

        # Drop baris dengan NaN pada fitur
        g_fitur = g[fitur_ada].ffill().fillna(0)
        g_label = g[TARGET_COL].astype(int).values

        n = len(g)
        if n < window + 1:
            continue   # Skip lokasi yang datanya terlalu sedikit

        for i in range(window, n):
            X_all.append(g_fitur.iloc[i-window:i].values)
            y_all.append(g_label[i])

    if not X_all:
        raise ValueError("Tidak ada sequence yang bisa dibuat. "
                         "Coba kurangi WINDOW atau tambah data.")

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.float32)
    log.info(f"Sequences: X={X.shape}, y={y.shape}")
    log.info(f"  Distribusi label: 0={int((y==0).sum())}, 1={int((y==1).sum())}")
    return X, y, fitur_ada


def bangun_model(n_fitur: int, window: int = WINDOW):
    """
    Bangun arsitektur LSTM + GRU.

    Arsitektur:
      Input(window, n_fitur)
      → LSTM(64, return_sequences=True)
      → Dropout(0.2)
      → GRU(32)
      → Dropout(0.2)
      → Dense(16, relu)
      → Dense(1, sigmoid)   ← output probabilitas banjir
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    except ImportError:
        log.error("TensorFlow tidak terinstall. Jalankan: pip install tensorflow")
        raise

    tf.random.set_seed(42)

    model = Sequential([
        Input(shape=(window, n_fitur)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )

    model.summary(print_fn=log.info)
    return model


def train_lstm(df: pd.DataFrame = None, path_fitur: str = None):
    """
    Training model LSTM/GRU untuk prediksi banjir berbasis time-series.

    Returns:
        dict berisi model, history, hasil evaluasi, path model.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
    except ImportError:
        log.error("tensorflow atau scikit-learn belum terinstall.")
        raise

    if df is None:
        df = load_features(path_fitur)

    log.info("=" * 55)
    log.info("TRAINING — LSTM/GRU Time-Series Prediksi Banjir")
    log.info("=" * 55)

    # --- Buat sequence ---
    X, y, fitur_ada = buat_sequences(df, WINDOW)
    n_fitur = X.shape[2]

    # --- Split: 70% train / 15% val / 15% test ---
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.18, random_state=42, stratify=y_tmp)

    log.info(f"Train:{len(X_train):,}  Val:{len(X_val):,}  Test:{len(X_test):,}")

    # --- Bangun model ---
    model = bangun_model(n_fitur, WINDOW)

    # --- Callbacks ---
    os.makedirs(PATHS["models"], exist_ok=True)
    ckpt_path = os.path.join(PATHS["models"], "lstm_best.keras")

    callbacks = [
        EarlyStopping(monitor="val_auc", patience=10, restore_best_weights=True,
                      mode="max", verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5,
                          min_lr=1e-6, verbose=1),
        ModelCheckpoint(ckpt_path, monitor="val_auc", save_best_only=True,
                        mode="max", verbose=0),
    ]

    # --- Class weight (imbalanced) ---
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    class_weight = {0: 1.0, 1: n_neg / max(n_pos, 1)}
    log.info(f"Class weights: {class_weight}")

    # --- Training ---
    log.info("Mulai training LSTM...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    # --- Evaluasi pada test set ---
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    from sklearn.metrics import (accuracy_score, f1_score,
                                 roc_auc_score, classification_report)
    acc   = accuracy_score(y_test, y_pred)
    f1    = f1_score(y_test, y_pred, zero_division=0)
    auc   = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred,
                                   target_names=["tidak_banjir","banjir"],
                                   zero_division=0)

    log.info(f"\n=== HASIL EVALUASI LSTM ===")
    log.info(f"Accuracy : {acc:.4f}")
    log.info(f"F1 Score : {f1:.4f}")
    log.info(f"AUC-ROC  : {auc:.4f}")
    log.info(f"\n{report}")

    # --- Plot loss & accuracy ---
    _plot_training_history(history)

    # --- Simpan model final ---
    model_path = os.path.join(PATHS["models"], "lstm_model.keras")
    model.save(model_path)
    log.info(f"Model LSTM tersimpan: {model_path}")

    # Simpan metadata
    import json
    meta = {"fitur_cols": fitur_ada, "window": WINDOW,
            "accuracy": acc, "f1": f1, "auc": auc}
    with open(os.path.join(PATHS["models"], "lstm_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "model":      model,
        "history":    history.history,
        "accuracy":   acc,
        "f1":         f1,
        "auc":        auc,
        "fitur_cols": fitur_ada,
        "model_path": model_path,
    }


def predict_banjir(model_path: str, X_seq: np.ndarray,
                   threshold: float = 0.5) -> dict:
    """
    Prediksi probabilitas banjir dari sequence input.

    Args:
        model_path: Path file .keras model.
        X_seq: Array shape (n_samples, window, n_fitur).
        threshold: Ambang batas probabilitas (default 0.5).

    Returns:
        Dict berisi prob_banjir dan prediksi biner.
    """
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        log.error(f"Gagal load model LSTM: {e}")
        raise

    prob = model.predict(X_seq, verbose=0).flatten()
    pred = (prob >= threshold).astype(int)

    return {
        "prob_banjir": prob.round(4),
        "prediksi":    pred,
        "level_siaga": pd.cut(prob,
                              bins=[0, 0.3, 0.5, 0.75, 1.01],
                              labels=["Normal","Waspada","Siaga","Awas"]),
    }


def _plot_training_history(history):
    """Plot loss dan AUC training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history.history["loss"],     label="Train Loss", color="#1F4E79")
    axes[0].plot(history.history["val_loss"], label="Val Loss",   color="#E74C3C", linestyle="--")
    axes[0].set_title("Training Loss", fontsize=12)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Crossentropy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # AUC
    if "auc" in history.history:
        axes[1].plot(history.history["auc"],     label="Train AUC", color="#1F4E79")
        axes[1].plot(history.history["val_auc"], label="Val AUC",   color="#E74C3C", linestyle="--")
        axes[1].set_title("AUC-ROC", fontsize=12)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("AUC")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PATHS["models"], "lstm_training_history.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Plot training history: {path}")


if __name__ == "__main__":
    hasil = train_lstm()
    print(f"\nAccuracy: {hasil['accuracy']:.4f}")
    print(f"F1      : {hasil['f1']:.4f}")
    print(f"AUC     : {hasil['auc']:.4f}")
