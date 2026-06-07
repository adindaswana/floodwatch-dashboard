# ==============================================================================
# etl/transform.py v2 — Cleaning + Feature Engineering (dari file nyata)
# ==============================================================================

import os, logging
import numpy as np
import pandas as pd
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURE_CONFIG, PATHS

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Mapping nama bulan Indonesia
NAMA_BULAN = {1:"Januari",2:"Februari",3:"Maret",4:"April",5:"Mei",
              6:"Juni",7:"Juli",8:"Agustus",9:"September",
              10:"Oktober",11:"November",12:"Desember"}

NAMA_HARI  = {0:"Senin",1:"Selasa",2:"Rabu",3:"Kamis",
              4:"Jumat",5:"Sabtu",6:"Minggu"}

# Mapping pulau
PULAU_MAP = {
    "Aceh":"Sumatera","Sumatera Utara":"Sumatera","Sumatera Barat":"Sumatera",
    "Riau":"Sumatera","Jambi":"Sumatera","Sumatera Selatan":"Sumatera",
    "Bengkulu":"Sumatera","Lampung":"Sumatera","Bangka Belitung":"Sumatera",
    "Kepulauan Riau":"Sumatera",
    "Dki Jakarta":"Jawa","Jakarta":"Jawa","Jawa Barat":"Jawa",
    "Jawa Tengah":"Jawa","Jawa Timur":"Jawa","Banten":"Jawa",
    "Daerah Istimewa Yogyakarta":"Jawa","Yogyakarta":"Jawa",
    "Bali":"Bali",
    "Nusa Tenggara Barat":"Nusa Tenggara","Nusa Tenggara Timur":"Nusa Tenggara",
    "Kalimantan Barat":"Kalimantan","Kalimantan Tengah":"Kalimantan",
    "Kalimantan Selatan":"Kalimantan","Kalimantan Timur":"Kalimantan",
    "Kalimantan Utara":"Kalimantan",
    "Sulawesi Utara":"Sulawesi","Sulawesi Tengah":"Sulawesi",
    "Sulawesi Selatan":"Sulawesi","Sulawesi Tenggara":"Sulawesi",
    "Gorontalo":"Sulawesi","Sulawesi Barat":"Sulawesi",
    "Maluku":"Maluku","Maluku Utara":"Maluku",
    "Papua":"Papua","Papua Barat":"Papua",
}


def _get_musim(bulan: int) -> str:
    if bulan in [11, 12, 1, 2, 3]:  return "Hujan"
    elif bulan in [5, 6, 7, 8, 9]:  return "Kemarau"
    return "Peralihan"


def _get_risiko(hujan_mm: float) -> str:
    t = FEATURE_CONFIG["risk_thresholds"]
    if hujan_mm <= t["rendah"]:  return "rendah"
    elif hujan_mm <= t["sedang"]: return "sedang"
    elif hujan_mm <= t["tinggi"]: return "tinggi"
    return "kritis"


# ==============================================================================
# TRANSFORM 1: BNPB
# ==============================================================================
def transform_bnpb(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bersihkan Data_Bencana.xlsx:
    - Parse tanggal
    - Standarisasi provinsi/kabupaten (Title Case)
    - Kolom numerik: isi NaN→0, hapus outlier negatif
    - Hapus duplikat
    - Tambah kolom bantu: tahun, bulan, musim, terjadi_banjir=True
    """
    log.info(f"[TRANSFORM] BNPB: {len(df)} baris masuk")

    # --- 1. Rename kolom ke snake_case ---
    df = df.rename(columns={
        "Tanggal Kejadian": "tanggal",
        "Kejadian":         "kejadian",
        "Kabupaten":        "kabupaten",
        "Provinsi":         "provinsi",
        "Meninggal":        "korban_meninggal",
        "Hilang":           "korban_hilang",
        "Terluka":          "korban_terluka",
        "Rumah Rusak":      "rumah_rusak",
        "Rumah Terendam":   "rumah_terendam",
        "Fasum Rusak":      "fasum_rusak",
    })

    # --- 2. Parse tanggal ---
    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce", dayfirst=True)
    n_before = len(df)
    df = df.dropna(subset=["tanggal"])
    log.info(f"  Drop tanggal null: -{n_before - len(df)} baris")

    # --- 3. Standarisasi teks lokasi ---
    for col in ["provinsi", "kabupaten", "kejadian"]:
        if col in df.columns:
            df[col] = (df[col].astype(str).str.strip()
                       .str.title()
                       .replace({"Nan": None, "None": None}))

    # --- 4. Kolom numerik: NaN→0, clip negatif ---
    num_cols = ["korban_meninggal","korban_hilang","korban_terluka",
                "rumah_rusak","rumah_terendam","fasum_rusak"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0).astype(int)

    # --- 5. Hapus duplikat ---
    n_before = len(df)
    df = df.drop_duplicates(subset=["tanggal","provinsi","kabupaten"])
    log.info(f"  Drop duplikat: -{n_before - len(df)} baris")

    # --- 6. Fitur tambahan ---
    df["tahun"]          = df["tanggal"].dt.year
    df["bulan"]          = df["tanggal"].dt.month
    df["musim"]          = df["bulan"].apply(_get_musim)
    df["terjadi_banjir"] = True

    log.info(f"  BNPB selesai: {len(df)} baris")
    return df.reset_index(drop=True)


# ==============================================================================
# TRANSFORM 2: Combine ML Dataset
# ==============================================================================
def transform_combine(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bersihkan data_banjir_combine_final.csv:
    - Standarisasi landcover_class (ada inkonsistensi case)
    - Clip nilai NDVI ke range valid (-1..1)
    - Clip elevation & slope ke nilai logis
    - Buat kolom tanggal dari year+month
    - Tambah label banjir boolean
    """
    log.info(f"[TRANSFORM] Combine ML: {len(df)} baris masuk")

    # --- 1. Standarisasi landcover_class (case inconsistent) ---
    df["landcover_class"] = (df["landcover_class"].str.strip()
                             .str.title()
                             .replace({
                                 "Tree Cover": "Tree Cover",
                                 "Built-Up":   "Built-Up",
                                 "Permanent Water Bodies": "Permanent Water Bodies",
                             }))
    log.info(f"  landcover unik setelah standarisasi: {df['landcover_class'].nunique()}")

    # --- 2. Standarisasi nama kabupaten & kecamatan ---
    for col in ["kabupaten", "kecamatan"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    # --- 3. Clip nilai tidak logis ---
    df["ndvi"]         = df["ndvi"].clip(-1.0, 1.0)
    df["elevation"]    = df["elevation"].clip(lower=0)
    df["slope"]        = df["slope"].clip(lower=0, upper=90)
    df["soil_moisture"]= df["soil_moisture"].clip(0, 100)
    df["avg_rainfall"] = df["avg_rainfall"].clip(lower=0)
    df["max_rainfall"] = df["max_rainfall"].clip(lower=0)
    df["avg_temperature"] = df["avg_temperature"].clip(10, 50)

    # --- 4. Buat kolom tanggal dari year + month ---
    df["tanggal"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str) + "-01",
        format="%Y-%m-%d", errors="coerce"
    )
    df["musim"] = df["month"].apply(_get_musim)

    # --- 5. Label banjir → boolean ---
    df["label_banjir"] = df["banjir"].astype(bool)

    # --- 6. Status risiko dari avg_rainfall ---
    df["status_risiko"] = df["avg_rainfall"].apply(_get_risiko)

    log.info(f"  Combine selesai: {len(df)} baris")
    return df.reset_index(drop=True)


# ==============================================================================
# TRANSFORM 3: Master Cuaca
# ==============================================================================
def transform_cuaca(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bersihkan master_cuaca_training_banjir_copy.csv:
    - Parse waktu_lokal → datetime
    - Agregasi dari per-3jam → harian (per lokasi)
    - Clip nilai cuaca ke range logis
    - Tambah fitur musim
    - Label banjir → boolean
    """
    log.info(f"[TRANSFORM] Cuaca: {len(df)} baris masuk")

    # --- 1. Parse datetime ---
    df["waktu_lokal"] = pd.to_datetime(df["waktu_lokal"], errors="coerce")
    df = df.dropna(subset=["waktu_lokal"])
    df["tanggal"] = df["waktu_lokal"].dt.date

    # --- 2. Standarisasi lokasi ---
    for col in ["provinsi","kota_kabupaten","kecamatan","kelurahan_desa"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    # --- 3. Clip nilai cuaca ---
    df["suhu_celcius"]       = pd.to_numeric(df["suhu_celcius"], errors="coerce").clip(10, 50)
    df["kelembapan_persen"]  = pd.to_numeric(df["kelembapan_persen"], errors="coerce").clip(0, 100)
    df["curah_hujan_mm"]     = pd.to_numeric(df["curah_hujan_mm"], errors="coerce").clip(lower=0)
    df["kecepatan_angin_kmh"]= pd.to_numeric(df["kecepatan_angin_kmh"], errors="coerce").clip(lower=0)

    # --- 4. Agregasi harian per lokasi ---
    group_cols = ["provinsi","kota_kabupaten","kecamatan","kelurahan_desa",
                  "latitude","longitude","tanggal","label_banjir"]
    group_cols = [c for c in group_cols if c in df.columns]

    agg = {
        "suhu_celcius":        "mean",
        "kelembapan_persen":   "mean",
        "curah_hujan_mm":      "sum",    # Total hujan sehari
        "kecepatan_angin_kmh": "mean",
        "kondisi_cuaca":       lambda x: x.mode()[0] if len(x) > 0 else None,
    }
    agg_valid = {k: v for k, v in agg.items() if k in df.columns}

    df_harian = (df.groupby(group_cols, as_index=False)
                   .agg(agg_valid)
                   .round(3))

    # --- 5. Fitur tambahan ---
    df_harian["tanggal"] = pd.to_datetime(df_harian["tanggal"])
    df_harian["tahun"]   = df_harian["tanggal"].dt.year
    df_harian["bulan"]   = df_harian["tanggal"].dt.month
    df_harian["musim"]   = df_harian["bulan"].apply(_get_musim)
    df_harian["label_banjir"] = df_harian["label_banjir"].astype(bool)

    log.info(f"  Cuaca harian selesai: {len(df_harian)} baris (dari {len(df)} per-3jam)")
    return df_harian.reset_index(drop=True)


# ==============================================================================
# FEATURE ENGINEERING (pada dataset combine — dataset utama ML)
# ==============================================================================
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Buat fitur ML dari dataset combine yang sudah di-transform.
    Fitur yang dibuat:
      - Lag curah hujan: t-1, t-3, t-7
      - Rolling average: 3, 7, 14 hari
      - Fitur temporal: bulan_encoded, musim_encoded
      - Fitur interaksi: hujan × kelembapan, risiko_index
      - Label: status_risiko (4 kelas)
    """
    log.info(f"[FEATURE ENG] Input: {len(df)} baris")

    # Deteksi nama kolom tahun & bulan (combine pakai year/month, cuaca pakai tahun/bulan)
    tahun_col = "tahun" if "tahun" in df.columns else "year"
    bulan_col = "bulan" if "bulan" in df.columns else "month"

    sort_cols = [c for c in ["kabupaten","kecamatan", tahun_col, bulan_col] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    group = ["kabupaten", "kecamatan"] if "kecamatan" in df.columns else ["kabupaten"]

    # --- Lag avg_rainfall ---
    for lag in FEATURE_CONFIG["lag_days"]:
        col_name = f"hujan_lag{lag}"
        df[col_name] = (df.groupby(group)["avg_rainfall"]
                          .shift(lag)
                          .round(3))
        log.info(f"  Fitur lag t-{lag} dibuat")

    # --- Rolling average avg_rainfall ---
    for w in FEATURE_CONFIG["rolling_windows"]:
        col_name = f"rolling_avg_{w}"
        df[col_name] = (df.groupby(group)["avg_rainfall"]
                          .transform(lambda x: x.rolling(w, min_periods=1).mean())
                          .round(3))
        log.info(f"  Rolling avg {w} hari dibuat")

    # --- Encode musim ---
    musim_map = {"Kemarau": 0, "Peralihan": 1, "Hujan": 2}
    df["musim_encoded"] = df["musim"].map(musim_map).fillna(1)

    # --- Landcover one-hot (top 5 kelas) ---
    top_lc = df["landcover_class"].value_counts().head(5).index.tolist()
    for lc in top_lc:
        col = "lc_" + lc.lower().replace(" ", "_").replace("-", "_")
        df[col] = (df["landcover_class"] == lc).astype(int)

    # --- Fitur interaksi ---
    df["risiko_index"] = (
        df["avg_rainfall"] * 0.4
        + df["soil_moisture"] * 0.2
        + df["slope"] * 0.15
        - df["elevation"] * 0.1
        + df["ndvi"].abs() * 0.15
    ).round(3)

    # --- Isi NaN lag/rolling dengan 0 (awal sequence) ---
    lag_rolling_cols = [c for c in df.columns
                        if c.startswith("hujan_lag") or c.startswith("rolling_avg")]
    df[lag_rolling_cols] = df[lag_rolling_cols].fillna(0)

    # Simpan ke processed
    os.makedirs(PATHS["features"], exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out = f"{PATHS['features']}features_{ts}.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    log.info(f"  Features disimpan: {out}")
    log.info(f"[FEATURE ENG] Selesai: {len(df)} baris, {len(df.columns)} kolom")
    return df


# ==============================================================================
# FUNGSI UTAMA TRANSFORM
# ==============================================================================
def run_transform(raw: dict) -> dict:
    """Jalankan semua transform, kembalikan dict DataFrame."""
    log.info("=" * 55)
    log.info("TAHAP 2: TRANSFORM")
    log.info("=" * 55)

    df_bnpb    = transform_bnpb(raw["bnpb"])
    df_combine = transform_combine(raw["combine"])
    df_cuaca   = transform_cuaca(raw["cuaca"])
    df_fitur   = feature_engineering(df_combine.copy())

    # Simpan processed
    os.makedirs(PATHS["processed"], exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    df_bnpb.to_csv(f"{PATHS['processed']}bnpb_clean_{ts}.csv", index=False)
    df_combine.to_csv(f"{PATHS['processed']}combine_clean_{ts}.csv", index=False)
    df_cuaca.to_csv(f"{PATHS['processed']}cuaca_clean_{ts}.csv", index=False)

    log.info(f"\nRingkasan Transform:")
    log.info(f"  BNPB clean    : {len(df_bnpb):,} baris, {df_bnpb.shape[1]} kolom")
    log.info(f"  Combine clean : {len(df_combine):,} baris, {df_combine.shape[1]} kolom")
    log.info(f"  Cuaca harian  : {len(df_cuaca):,} baris, {df_cuaca.shape[1]} kolom")
    log.info(f"  Feature set   : {len(df_fitur):,} baris, {df_fitur.shape[1]} kolom")

    return {"bnpb": df_bnpb, "combine": df_combine,
            "cuaca": df_cuaca, "fitur": df_fitur}
