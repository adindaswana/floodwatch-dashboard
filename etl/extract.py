# ==============================================================================
# etl/extract.py — Extract dari 3 File Dataset Nyata
# Sumber:
#   1. Data_Bencana.xlsx           → BNPB DIBI (raw kejadian banjir)
#   2. data_banjir_combine_final.csv → Dataset ML gabungan (cuaca+geospasial)
#   3. master_cuaca_training_banjir_copy.csv → Data cuaca per-3jam + label
# ==============================================================================

import os, logging
import pandas as pd
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FILES, COLS_BNPB, COLS_DROP_COMBINE, COLS_DROP_CUACA

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ==============================================================================
# 1. EXTRACT — Data_Bencana.xlsx (BNPB)
# ==============================================================================
def extract_bnpb() -> pd.DataFrame:
    """
    Load Data_Bencana.xlsx dari BNPB DIBI.
    - Row 0 = header sebenarnya (row pertama setelah judul)
    - Hanya ambil kolom yang relevan (buang Kronologi, Penyebab, dll.)
    - Filter hanya kejadian BANJIR

    Returns: DataFrame BNPB mentah (belum dibersihkan)
    """
    path = FILES["bnpb_raw"]
    log.info(f"[EXTRACT] BNPB ← {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan: {path}\n"
                                f"Salin file ke: {path}")

    # Baca dengan header di baris pertama, baris 0 = judul (skip)
    df = pd.read_excel(path, header=0)
    # Row pertama adalah header asli (No., Tanggal Kejadian, dst.)
    df.columns = df.iloc[0]
    df = df.drop(index=0).reset_index(drop=True)

    log.info(f"  Raw shape: {df.shape}, kolom: {list(df.columns)}")

    # Hanya ambil kolom yang diperlukan
    cols_ada = [c for c in COLS_BNPB if c in df.columns]
    df = df[cols_ada].copy()

    # Filter hanya BANJIR
    if "Kejadian" in df.columns:
        df = df[df["Kejadian"].str.upper().str.contains("BANJIR", na=False)]

    log.info(f"  Setelah filter BANJIR: {len(df)} baris")
    return df.reset_index(drop=True)


# ==============================================================================
# 2. EXTRACT — data_banjir_combine_final.csv (ML Dataset)
# ==============================================================================
def extract_combine() -> pd.DataFrame:
    """
    Load data_banjir_combine_final.csv.
    - Buang kolom: map_image (path file TIF, tidak berguna untuk ML)
                   NAME_3_clean (duplikat NAME_3, banyak null)
    - Standarisasi nama kolom

    Returns: DataFrame ML dataset gabungan
    """
    path = FILES["combine_ml"]
    log.info(f"[EXTRACT] Combine ML ← {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    df = pd.read_csv(path, low_memory=False)
    log.info(f"  Raw shape: {df.shape}")

    # Drop kolom tidak berguna
    df = df.drop(columns=[c for c in COLS_DROP_COMBINE if c in df.columns])
    log.info(f"  Drop kolom {COLS_DROP_COMBINE} → shape: {df.shape}")

    # Standarisasi nama kolom
    rename_map = {
        "NAME_2": "kabupaten",
        "NAME_3": "kecamatan",
        "lat":    "latitude",
        "long":   "longitude",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    log.info(f"  Kolom final: {list(df.columns)}")
    return df.reset_index(drop=True)


# ==============================================================================
# 3. EXTRACT — master_cuaca_training_banjir_copy.csv (Cuaca + Label)
# ==============================================================================
def extract_cuaca() -> pd.DataFrame:
    """
    Load master_cuaca_training_banjir_copy.csv.
    - Data cuaca per 3 jam (time-series)
    - Buang kolom: kategori_wilayah (redundan)

    Returns: DataFrame cuaca mentah (granularitas per 3 jam)
    """
    path = FILES["cuaca_train"]
    log.info(f"[EXTRACT] Cuaca Training ← {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    df = pd.read_csv(path, low_memory=False)
    log.info(f"  Raw shape: {df.shape}")

    # Drop kolom tidak berguna
    df = df.drop(columns=[c for c in COLS_DROP_CUACA if c in df.columns])
    log.info(f"  Drop kolom {COLS_DROP_CUACA} → shape: {df.shape}")
    log.info(f"  Kolom final: {list(df.columns)}")

    return df.reset_index(drop=True)


# ==============================================================================
# FUNGSI UTAMA EXTRACT
# ==============================================================================
def run_extract() -> dict:
    """Jalankan extract semua sumber, kembalikan dict DataFrame."""
    log.info("=" * 55)
    log.info("TAHAP 1: EXTRACT")
    log.info("=" * 55)

    df_bnpb    = extract_bnpb()
    df_combine = extract_combine()
    df_cuaca   = extract_cuaca()

    log.info(f"\nRingkasan Extract:")
    log.info(f"  BNPB    : {len(df_bnpb):,} baris")
    log.info(f"  Combine : {len(df_combine):,} baris")
    log.info(f"  Cuaca   : {len(df_cuaca):,} baris")

    return {"bnpb": df_bnpb, "combine": df_combine, "cuaca": df_cuaca}
