# ==============================================================================
# etl/load.py v2 — Load ke MySQL Workbench
# Alur: Staging → Dimensi → Fakta
# Fix: tidak ada TINYINT(1), gunakan BOOLEAN
# ==============================================================================

import logging
import numpy as np
import pandas as pd
from datetime import date
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_CONFIG, DB_URL

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PULAU_MAP = {
    "Aceh":"Sumatera","Sumatera Utara":"Sumatera","Sumatera Barat":"Sumatera",
    "Riau":"Sumatera","Jambi":"Sumatera","Sumatera Selatan":"Sumatera",
    "Bengkulu":"Sumatera","Lampung":"Sumatera","Bangka Belitung":"Sumatera",
    "Kepulauan Riau":"Sumatera",
    "Dki Jakarta":"Jawa","Jakarta":"Jawa","Jawa Barat":"Jawa",
    "Jawa Tengah":"Jawa","Jawa Timur":"Jawa","Banten":"Jawa",
    "Daerah Istimewa Yogyakarta":"Jawa","Yogyakarta":"Jawa",
    "Bali":"Bali","Nusa Tenggara Barat":"Nusa Tenggara",
    "Nusa Tenggara Timur":"Nusa Tenggara","Kalimantan Barat":"Kalimantan",
    "Kalimantan Tengah":"Kalimantan","Kalimantan Selatan":"Kalimantan",
    "Kalimantan Timur":"Kalimantan","Kalimantan Utara":"Kalimantan",
    "Sulawesi Utara":"Sulawesi","Sulawesi Tengah":"Sulawesi",
    "Sulawesi Selatan":"Sulawesi","Sulawesi Tenggara":"Sulawesi",
    "Gorontalo":"Sulawesi","Sulawesi Barat":"Sulawesi",
    "Maluku":"Maluku","Maluku Utara":"Maluku",
    "Papua":"Papua","Papua Barat":"Papua",
}

NAMA_BULAN = {1:"Januari",2:"Februari",3:"Maret",4:"April",5:"Mei",
              6:"Juni",7:"Juli",8:"Agustus",9:"September",
              10:"Oktober",11:"November",12:"Desember"}
NAMA_HARI  = {0:"Senin",1:"Selasa",2:"Rabu",3:"Kamis",
              4:"Jumat",5:"Sabtu",6:"Minggu"}


# ==============================================================================
# KONEKSI
# ==============================================================================
def buat_koneksi(db_url: str = None):
    if db_url is None:
        db_url = DB_URL
    try:
        engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=3600, echo=False,
                               connect_args={"connect_timeout": 30, "charset": "utf8mb4"})
        with engine.connect() as conn:
            ver = conn.execute(text("SELECT VERSION()")).scalar()
            log.info(f"MySQL terhubung ✓  versi: {ver}")
            log.info(f"Database: {DB_CONFIG['database']} @ {DB_CONFIG['host']}:{DB_CONFIG['port']}")
        return engine
    except SQLAlchemyError as e:
        log.error(f"Koneksi MySQL gagal: {e}")
        log.info("Petunjuk:\n  1. Pastikan MySQL Workbench berjalan\n"
                 "  2. Sesuaikan config.py (user/password/database)\n"
                 "  3. Jalankan sql/create_schema.sql di MySQL Workbench\n"
                 "  4. pip install pymysql")
        raise


# ==============================================================================
# HELPER
# ==============================================================================
def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Ganti NaN/inf/nat dengan None agar kompatibel MySQL."""
    df = df.copy()
    df = df.replace({np.nan: None, np.inf: None, -np.inf: None})
    for col in df.select_dtypes(include=["datetime64[ns]","datetime64[ns, UTC]"]).columns:
        df[col] = df[col].dt.date
    return df


def _insert(df: pd.DataFrame, tabel: str, engine, ignore_dup: bool = True,
            chunk: int = 1000) -> int:
    """
    Batch INSERT ke MySQL.
    ignore_dup=True → INSERT IGNORE (skip duplikat berdasarkan UNIQUE KEY)
    """
    if df.empty:
        log.warning(f"  [{tabel}] DataFrame kosong, dilewati.")
        return 0

    df = _clean(df)
    total = 0
    kw = "INSERT IGNORE" if ignore_dup else "INSERT"
    cols = ", ".join(f"`{c}`" for c in df.columns)
    ph   = ", ".join(f":{c}" for c in df.columns)
    sql  = f"{kw} INTO `{tabel}` ({cols}) VALUES ({ph})"

    with engine.begin() as conn:
        for i in range(0, len(df), chunk):
            rows = df.iloc[i:i+chunk].to_dict(orient="records")
            result = conn.execute(text(sql), rows)
            total += result.rowcount

    log.info(f"  [{tabel}] {total:,} baris diinsert")
    return total


def _fetch(engine, sql: str) -> list:
    with engine.connect() as conn:
        return conn.execute(text(sql)).fetchall()


# ==============================================================================
# STAGING
# ==============================================================================
def load_stg_bnpb(df: pd.DataFrame, engine) -> int:
    cols = {"tanggal","kejadian","kabupaten","provinsi",
            "korban_meninggal","korban_hilang","korban_terluka",
            "rumah_rusak","rumah_terendam","fasum_rusak"}
    rename = {"korban_meninggal":"meninggal","korban_hilang":"hilang",
              "korban_terluka":"terluka"}
    df2 = df[[c for c in df.columns if c in cols]].rename(columns=rename)
    return _insert(df2, "stg_bnpb", engine)


def load_stg_combine(df: pd.DataFrame, engine) -> int:
    cols = {"kabupaten","kecamatan","avg_rainfall","max_rainfall",
            "avg_temperature","elevation","landcover_class","ndvi",
            "slope","soil_moisture","year","month","label_banjir",
            "latitude","longitude"}
    rename = {"year":"tahun","month":"bulan","latitude":"lat","longitude":"lon"}
    df2 = df[[c for c in df.columns if c in cols]].rename(columns=rename)
    return _insert(df2, "stg_combine", engine)


def load_stg_cuaca(df: pd.DataFrame, engine) -> int:
    cols = {"provinsi","kota_kabupaten","kecamatan","kelurahan_desa",
            "latitude","longitude","tanggal","suhu_celcius",
            "kelembapan_persen","curah_hujan_mm","kondisi_cuaca",
            "kecepatan_angin_kmh","label_banjir"}
    df2 = df[[c for c in df.columns if c in cols]].copy()
    if "tanggal" in df2.columns:
        df2["tanggal"] = pd.to_datetime(df2["tanggal"], errors="coerce")
        df2.rename(columns={"tanggal":"waktu_lokal"}, inplace=True)
    return _insert(df2, "stg_cuaca", engine)


# ==============================================================================
# DIM WAKTU
# ==============================================================================
def load_dim_waktu(df_list: list, engine) -> dict:
    """Kumpulkan semua tanggal dari semua DataFrame, generate dim_waktu."""
    log.info("  Generating dim_waktu...")
    tanggal_set = set()
    for df in df_list:
        for col in df.columns:
            if "tanggal" in col.lower() or col == "waktu_lokal":
                try:
                    dates = pd.to_datetime(df[col], errors="coerce").dropna()
                    tanggal_set.update(dates.dt.date.tolist())
                except Exception:
                    pass

    if not tanggal_set:
        log.warning("  Tidak ada tanggal ditemukan.")
        return {}

    records = []
    for t in sorted(tanggal_set):
        dt = pd.Timestamp(t)
        records.append({
            "tanggal":    t,
            "hari":       dt.day,
            "bulan":      dt.month,
            "tahun":      dt.year,
            "kuartal":    dt.quarter,
            "nama_bulan": NAMA_BULAN.get(dt.month, ""),
            "nama_hari":  NAMA_HARI.get(dt.dayofweek, ""),
            "musim":      ("Hujan" if dt.month in [11,12,1,2,3]
                           else "Kemarau" if dt.month in [5,6,7,8,9]
                           else "Peralihan"),
            "semester":   1 if dt.month <= 6 else 2,
            "minggu_ke":  int(dt.strftime("%W")),
            "is_libur":   False,
        })

    _insert(pd.DataFrame(records), "dim_waktu", engine)

    rows = _fetch(engine, "SELECT id_waktu, tanggal FROM dim_waktu")
    mapping = {str(r[1]): r[0] for r in rows}
    log.info(f"  dim_waktu mapping: {len(mapping)} tanggal")
    return mapping


# ==============================================================================
# DIM LOKASI
# ==============================================================================
def load_dim_lokasi(df_list: list, engine) -> dict:
    """
    Kumpulkan kombinasi lokasi unik dari semua DataFrame.
    Sumber lokasi:
      - BNPB: provinsi + kabupaten
      - Combine: kabupaten + kecamatan + lat/lon
      - Cuaca: provinsi + kota_kabupaten + kecamatan + lat/lon
    """
    log.info("  Generating dim_lokasi...")
    records = {}

    for df in df_list:
        prov_col = next((c for c in ["provinsi"] if c in df.columns), None)
        kab_col  = next((c for c in ["kabupaten","kota_kabupaten","NAME_2"] if c in df.columns), None)
        kec_col  = next((c for c in ["kecamatan","NAME_3"] if c in df.columns), None)
        lat_col  = next((c for c in ["latitude","lat"] if c in df.columns), None)
        lon_col  = next((c for c in ["longitude","lon","long"] if c in df.columns), None)

        for _, row in df.iterrows():
            prov = str(row.get(prov_col, "") or "").strip() if prov_col else ""
            kab  = str(row.get(kab_col, "")  or "").strip() if kab_col  else ""
            kec  = str(row.get(kec_col, "")  or "").strip() if kec_col  else ""
            lat  = row.get(lat_col) if lat_col else None
            lon  = row.get(lon_col) if lon_col else None

            key = (prov, kab, kec)
            if key not in records:
                records[key] = {
                    "provinsi":  prov or None,
                    "kabupaten": kab  or None,
                    "kecamatan": kec  or None,
                    "latitude":  lat,
                    "longitude": lon,
                    "pulau":     PULAU_MAP.get(prov, "Lainnya"),
                    "zona_risiko": "Sedang",
                }

    df_lok = pd.DataFrame(list(records.values())).dropna(subset=["provinsi"])
    _insert(df_lok, "dim_lokasi", engine)

    rows = _fetch(engine,
        "SELECT id_lokasi, COALESCE(provinsi,''), "
        "COALESCE(kabupaten,''), COALESCE(kecamatan,'') FROM dim_lokasi")
    mapping = {(r[1], r[2], r[3]): r[0] for r in rows}
    log.info(f"  dim_lokasi mapping: {len(mapping)} lokasi")
    return mapping


# ==============================================================================
# DIM CUACA
# ==============================================================================
def load_dim_cuaca(df_cuaca: pd.DataFrame, map_lokasi: dict, engine) -> dict:
    """Load cuaca harian ke dim_cuaca dengan resolusi id_lokasi."""
    log.info("  Loading dim_cuaca...")

    if df_cuaca.empty:
        return {}

    df = df_cuaca.copy()

    # Resolve id_lokasi
    def _resolve_lok(row):
        prov = str(row.get("provinsi", "") or "")
        kab  = str(row.get("kota_kabupaten", "") or "")
        kec  = str(row.get("kecamatan", "") or "")
        return (map_lokasi.get((prov, kab, kec))
                or map_lokasi.get((prov, kab, ""))
                or map_lokasi.get((prov, "", "")))

    df["id_lokasi"] = df.apply(_resolve_lok, axis=1)

    col_cuaca = {"tanggal":"tanggal_cuaca","suhu_celcius":"suhu_celcius",
                 "kelembapan_persen":"kelembapan_persen",
                 "curah_hujan_mm":"curah_hujan_mm",
                 "kecepatan_angin_kmh":"kecepatan_angin",
                 "kondisi_cuaca":"kondisi_cuaca"}
    df = df.rename(columns={k:v for k,v in col_cuaca.items() if k in df.columns})
    keep = ["tanggal_cuaca","id_lokasi","suhu_celcius","kelembapan_persen",
            "curah_hujan_mm","kecepatan_angin","kondisi_cuaca"]
    df2 = df[[c for c in keep if c in df.columns]].dropna(subset=["tanggal_cuaca"])
    _insert(df2, "dim_cuaca", engine)

    rows = _fetch(engine,
        "SELECT id_cuaca, COALESCE(id_lokasi,0), tanggal_cuaca FROM dim_cuaca")
    mapping = {(str(r[1]), str(r[2])): r[0] for r in rows}
    log.info(f"  dim_cuaca mapping: {len(mapping)} record")
    return mapping


# ==============================================================================
# DIM LINGKUNGAN
# ==============================================================================
def load_dim_lingkungan(df_combine: pd.DataFrame, map_lokasi: dict, engine) -> dict:
    """Load fitur geospasial ke dim_lingkungan (per lokasi unik)."""
    log.info("  Loading dim_lingkungan...")

    geo_cols = ["kabupaten","kecamatan","elevation","slope","ndvi",
                "soil_moisture","landcover_class"]
    df = df_combine[[c for c in geo_cols if c in df_combine.columns]].copy()

    # Ambil nilai representatif per lokasi (median fitur numerik)
    group = [c for c in ["kabupaten","kecamatan"] if c in df.columns]
    num_agg = {c:"median" for c in ["elevation","slope","ndvi","soil_moisture"] if c in df.columns}
    str_agg = {c: lambda x: x.mode()[0] if len(x)>0 else None
               for c in ["landcover_class"] if c in df.columns}
    df_uniq = df.groupby(group).agg({**num_agg, **str_agg}).reset_index()

    # Resolve id_lokasi by kabupaten fallback
    kab_map = {}
    for (prov, kab, kec), id_lok in map_lokasi.items():
        if kab:
            k_clean = str(kab).lower().strip()
            if k_clean not in kab_map or not kec:
                kab_map[k_clean] = id_lok

    def _lok(row):
        kab = str(row.get("kabupaten","") or "").strip().lower()
        return kab_map.get(kab)

    df_uniq["id_lokasi"] = df_uniq.apply(_lok, axis=1)
    df_load = df_uniq[["id_lokasi","elevation","slope","ndvi",
                        "soil_moisture","landcover_class"]].dropna(subset=["id_lokasi"])
    _insert(df_load, "dim_lingkungan", engine)

    rows = _fetch(engine, "SELECT id_lingkungan, COALESCE(id_lokasi,0) FROM dim_lingkungan")
    mapping = {str(r[1]): r[0] for r in rows}
    log.info(f"  dim_lingkungan mapping: {len(mapping)} lokasi")
    return mapping


# ==============================================================================
# FACT BANJIR
# ==============================================================================
def load_fact_banjir(df_fitur: pd.DataFrame, df_bnpb: pd.DataFrame,
                     map_waktu: dict, map_lokasi: dict,
                     map_cuaca: dict, map_ling: dict,
                     engine) -> int:
    """
    Load ke fact_banjir dengan resolusi semua foreign key.
    Basis: df_fitur (combine + feature engineering)
    Enrich: data dampak dari BNPB (join by kabupaten + bulan + tahun)
    """
    log.info("  Loading fact_banjir...")

    df = df_fitur.copy()

    # --- Resolve id_waktu ---
    def _wkt(row):
        t = row.get("tanggal")
        if pd.isna(t) or t is None: return None
        t = pd.Timestamp(t).date()
        return map_waktu.get(str(t))

    df["id_waktu"] = df.apply(_wkt, axis=1)

    # --- Resolve id_lokasi ---
    kab_map = {}
    for (prov, kab, kec), id_lok in map_lokasi.items():
        if kab:
            k_clean = str(kab).lower().strip()
            # prefer general locations over specific kecamatans
            if k_clean not in kab_map or not kec:
                kab_map[k_clean] = id_lok

    def _lok(row):
        kab = str(row.get("kabupaten","") or "").strip().lower()
        return kab_map.get(kab)

    df["id_lokasi"] = df.apply(_lok, axis=1)

    # --- Resolve id_cuaca ---
    def _cuaca(row):
        id_lok = row.get("id_lokasi")
        t      = row.get("tanggal")
        if pd.isna(id_lok) or pd.isna(t) or id_lok is None or t is None: return None
        t_str = str(pd.Timestamp(t).date())
        lok_str = str(int(id_lok))
        return map_cuaca.get((lok_str, t_str))

    df["id_cuaca"] = df.apply(_cuaca, axis=1)

    # --- Resolve id_lingkungan ---
    def _ling(row):
        id_lok = row.get("id_lokasi")
        if pd.isna(id_lok) or id_lok is None: return None
        return map_ling.get(str(int(id_lok)))

    df["id_lingkungan"] = df.apply(_ling, axis=1)

    # --- Enrich dampak dari BNPB (join by kabupaten + tahun + bulan) ---
    if not df_bnpb.empty:
        df_bnpb_agg = (df_bnpb
            .groupby(["kabupaten","tahun","bulan"], as_index=False)
            .agg({"korban_meninggal":"sum","korban_hilang":"sum",
                  "korban_terluka":"sum","rumah_rusak":"sum",
                  "rumah_terendam":"sum","fasum_rusak":"sum"})
        )
        df_bnpb_agg["kabupaten"] = df_bnpb_agg["kabupaten"].str.title()
        df_bnpb_agg["_kab_key"]  = df_bnpb_agg["kabupaten"].str.lower().str.strip()
        df["_kab_key"] = df["kabupaten"].str.lower().str.strip()

        df = df.merge(df_bnpb_agg.rename(columns={"tahun":"year","bulan":"month"}),
                      on=["_kab_key","year","month"],
                      how="left", suffixes=("","_bnpb"))
        df.drop(columns=["_kab_key"], inplace=True)

        df["terjadi_banjir"] = df["korban_meninggal"].notna()
        for c in ["korban_meninggal","korban_hilang","korban_terluka",
                  "rumah_rusak","rumah_terendam","fasum_rusak"]:
            if c in df.columns:
                df[c] = df[c].fillna(0).astype(int)
    else:
        df["terjadi_banjir"]   = False
        df["korban_meninggal"] = 0
        df["korban_hilang"]    = 0
        df["korban_terluka"]   = 0
        df["rumah_rusak"]      = 0
        df["rumah_terendam"]   = 0
        df["fasum_rusak"]      = 0

    # --- Enrich Detail Cuaca dari dim_cuaca ---
    if not df["id_cuaca"].isna().all():
        df_dim_cuaca = pd.read_sql(
            "SELECT id_cuaca, curah_hujan_mm, suhu_celcius, kelembapan_persen, kecepatan_angin, kondisi_cuaca FROM dim_cuaca", 
            engine.connect()
        )
        # Drop columns in `df` if they already exist so we don't duplicate them during merge
        cuaca_cols_to_merge = ["curah_hujan_mm", "suhu_celcius", "kelembapan_persen", "kecepatan_angin", "kondisi_cuaca"]
        for c in cuaca_cols_to_merge:
            if c in df.columns:
                df.drop(columns=[c], inplace=True)
                
        df = df.merge(df_dim_cuaca, on="id_cuaca", how="left")
    
    # Initialize if they don't exist
    for col in ["suhu_celcius", "curah_hujan_mm", "kelembapan_persen", "kondisi_cuaca", "kecepatan_angin"]:
        if col not in df.columns:
            df[col] = np.nan
            
    # Fallback to feature metrics if daily metrics are missing for the row
    if "avg_temperature" in df.columns:
        df["suhu_celcius"] = df["suhu_celcius"].fillna(df["avg_temperature"])
    if "avg_rainfall" in df.columns:
        df["curah_hujan_mm"] = df["curah_hujan_mm"].fillna(df["avg_rainfall"])
    if "soil_moisture" in df.columns:
        df["kelembapan_persen"] = df["kelembapan_persen"].fillna(df["soil_moisture"])
    if "kondisi_cuaca" in df.columns:
        df["kondisi_cuaca"] = df["kondisi_cuaca"].fillna("Normal/Tidak Diketahui")

    # --- Pilih kolom fact ---
    fact_cols = [
        "id_waktu","id_lokasi","id_cuaca","id_lingkungan",
        "curah_hujan_mm","avg_rainfall","max_rainfall",
        "suhu_celcius","kelembapan_persen","kecepatan_angin","kondisi_cuaca",
        "hujan_lag1","hujan_lag3","hujan_lag7",
        "rolling_avg_3","rolling_avg_7","rolling_avg_14",
        "elevation","slope","ndvi","soil_moisture","landcover_class",
        "terjadi_banjir","korban_meninggal","korban_hilang",
        "korban_terluka","rumah_rusak","rumah_terendam","fasum_rusak",
        "status_risiko","label_banjir",
    ]
    df_load = df[[c for c in fact_cols if c in df.columns]].copy()

    # Isi default
    for bc in ["terjadi_banjir","label_banjir"]:
        if bc in df_load.columns:
            df_load[bc] = df_load[bc].fillna(False).astype(bool)

    if "status_risiko" not in df_load.columns:
        df_load["status_risiko"] = "rendah"

    df_load["sumber_data"] = "BNPB+Combine+Cuaca"

    # Drop baris tanpa FK wajib
    n_before = len(df_load)
    df_load = df_load.dropna(subset=["id_waktu","id_lokasi"])
    log.info(f"  Drop baris tanpa FK: {n_before - len(df_load)}")

    return _insert(df_load, "fact_banjir", engine)


# ==============================================================================
# VERIFIKASI
# ==============================================================================
def verifikasi(engine) -> dict:
    tabel_list = ["stg_bnpb","stg_combine","stg_cuaca",
                  "dim_waktu","dim_lokasi","dim_cuaca","dim_lingkungan","fact_banjir"]
    counts = {}
    with engine.connect() as conn:
        for t in tabel_list:
            try:
                n = conn.execute(text(f"SELECT COUNT(*) FROM `{t}`")).scalar()
                counts[t] = n
                log.info(f"  {t:<25}: {n:,} baris")
            except Exception:
                counts[t] = "Error"
                log.warning(f"  {t}: tabel tidak ditemukan")
    return counts


# ==============================================================================
# FUNGSI UTAMA LOAD
# ==============================================================================
def run_load(transformed: dict, engine=None) -> dict:
    if engine is None:
        engine = buat_koneksi()

    log.info("=" * 55)
    log.info("TAHAP 3: LOAD → MySQL Workbench")
    log.info("=" * 55)

    df_bnpb    = transformed["bnpb"].copy()
    df_combine = transformed["combine"].copy()
    df_cuaca   = transformed["cuaca"].copy()
    df_fitur   = transformed["fitur"].copy()

    # Create dummy date for combine & fitur if missing (set day to 1)
    for df in [df_combine, df_fitur]:
        if "year" in df.columns and "month" in df.columns and "tanggal" not in df.columns:
            df["tanggal"] = pd.to_datetime(df[["year", "month"]].assign(day=1))

    # 1. Staging
    log.info("\n[1/5] Load Staging...")
    load_stg_bnpb(df_bnpb, engine)
    load_stg_combine(df_combine, engine)
    load_stg_cuaca(df_cuaca, engine)

    # 2. Dimensi Waktu
    log.info("\n[2/5] Load dim_waktu...")
    map_waktu = load_dim_waktu([df_bnpb, df_combine, df_cuaca], engine)

    # 3. Dimensi Lokasi
    log.info("\n[3/5] Load dim_lokasi...")
    map_lokasi = load_dim_lokasi([df_bnpb, df_combine, df_cuaca], engine)

    # 4. Dimensi Cuaca & Lingkungan
    log.info("\n[4/5] Load dim_cuaca & dim_lingkungan...")
    map_cuaca = load_dim_cuaca(df_cuaca, map_lokasi, engine)
    map_ling  = load_dim_lingkungan(df_combine, map_lokasi, engine)

    # 5. Fact
    log.info("\n[5/5] Load fact_banjir...")
    n_fact = load_fact_banjir(df_fitur, df_bnpb,
                              map_waktu, map_lokasi,
                              map_cuaca, map_ling, engine)

    # Verifikasi
    log.info("\n[VERIFIKASI] Jumlah record per tabel:")
    counts = verifikasi(engine)
    engine.dispose()

    return {"fact_banjir": n_fact, "verifikasi": counts}
