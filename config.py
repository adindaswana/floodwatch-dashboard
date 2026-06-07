# ==============================================================================
# config.py — Konfigurasi Pipeline v2 (berdasarkan dataset nyata)
# ==============================================================================

# --- MySQL Workbench ---
DB_CONFIG = {
    "host":     "localhost",
    "port":     3306,
    "user":     "root",
    "password": "ad1nd4",   # ⚠️ Ganti dengan password MySQL Anda
    "database": "db_banjir_v2",
    "charset":  "utf8mb4",
}

DB_URL = (
    f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    f"?charset={DB_CONFIG['charset']}"
)

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Path Dataset (file asli yang diunggah) ---
FILES = {
    "bnpb_raw":      os.path.join(BASE_DIR, "data", "raw", "Data_Bencana.xlsx"),
    "combine_ml":    os.path.join(BASE_DIR, "data", "raw", "data_banjir_combine_final.csv"),
    "cuaca_train":   os.path.join(BASE_DIR, "data", "raw", "master_cuaca_training_banjir_copy.csv"),
}

# --- Kolom yang DIPAKAI (kolom tidak berguna akan di-drop) ---
COLS_BNPB = [
    "Tanggal Kejadian", "Kejadian", "Kabupaten",
    "Provinsi", "Meninggal", "Hilang",
    "Terluka", "Rumah Rusak", "Rumah Terendam", "Fasum Rusak",
]

# Kolom yang di-drop dari data_banjir_combine_final
COLS_DROP_COMBINE = ["map_image", "NAME_3_clean"]

# Kolom yang di-drop dari master_cuaca
COLS_DROP_CUACA = ["kategori_wilayah"]  # redundan (gabungan kota+stasiun)

# --- Feature Engineering ---
FEATURE_CONFIG = {
    "lag_days":        [1, 3, 7],
    "rolling_windows": [3, 7, 14],
    "risk_thresholds": {           # mm/hari
        "rendah":  20,
        "sedang":  50,
        "tinggi":  100,
    },
}

PATHS = {
    "raw":       os.path.join(BASE_DIR, "data", "raw"),
    "processed": os.path.join(BASE_DIR, "data", "processed"),
    "features":  os.path.join(BASE_DIR, "data", "features"),
    "logs":      os.path.join(BASE_DIR, "logs"),
    "models":    os.path.join(BASE_DIR, "models"),
}
