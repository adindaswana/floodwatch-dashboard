# ==============================================================================
# dashboard/dashboard.py - Dashboard Interaktif Deteksi Dini Banjir
# ==============================================================================
# Visualisasi hasil ETL + ML menggunakan Plotly (offline, no server)
# Menyimpan semua chart sebagai file HTML interaktif
# ==============================================================================

import os, logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False
    logging.warning("Plotly tidak terinstall. Install: pip install plotly")

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PATHS

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = "dashboard/output"
WARNA = {
    "rendah":  "#27AE60",
    "sedang":  "#F39C12",
    "tinggi":  "#E67E22",
    "kritis":  "#E74C3C",
    "biru":    "#1F4E79",
    "teal":    "#0D6E6E",
}


def _load_data():
    """Load data processed dari file terbaru."""
    proc = PATHS["processed"]
    feat = PATHS["features"]

    df_bnpb, df_combine, df_fitur = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for prefix, target in [("bnpb","df_bnpb"),
                            ("combine","df_combine")]:
        files = sorted([f for f in os.listdir(proc)
                        if f.startswith(prefix) and f.endswith(".csv")])
        if files:
            locals()[target] = pd.read_csv(os.path.join(proc, files[-1]))

    files_feat = sorted([f for f in os.listdir(feat) if f.endswith(".csv")])
    if files_feat:
        df_fitur = pd.read_csv(os.path.join(feat, files_feat[-1]))

    return (locals().get("df_bnpb",   pd.DataFrame()),
            locals().get("df_combine", pd.DataFrame()),
            df_fitur)


def load_all_data():
    """Load semua data yang diperlukan dashboard."""
    proc_dir = PATHS["processed"]
    feat_dir = PATHS["features"]
    result = {}

    for prefix in ["bnpb", "combine", "cuaca"]:
        files = sorted([f for f in os.listdir(proc_dir)
                        if f.startswith(prefix) and f.endswith(".csv")])
        if files:
            result[prefix] = pd.read_csv(os.path.join(proc_dir, files[-1]))
        else:
            result[prefix] = pd.DataFrame()

    files_feat = sorted([f for f in os.listdir(feat_dir) if f.endswith(".csv")])
    if files_feat:
        result["fitur"] = pd.read_csv(os.path.join(feat_dir, files_feat[-1]))
    else:
        result["fitur"] = pd.DataFrame()

    return result


# ==============================================================================
# CHART 1 - Frekuensi Banjir per Provinsi (dari BNPB)
# ==============================================================================
def chart_banjir_per_provinsi(df_bnpb: pd.DataFrame):
    """Bar chart horizontal: Top 15 provinsi frekuensi banjir."""
    if df_bnpb.empty or "provinsi" not in df_bnpb.columns:
        log.warning("Data BNPB kosong atau tidak ada kolom provinsi.")
        return

    top = (df_bnpb.groupby("provinsi")
                  .size()
                  .reset_index(name="jumlah_banjir")
                  .sort_values("jumlah_banjir", ascending=False)
                  .head(15))

    if PLOTLY_OK:
        fig = px.bar(top, x="jumlah_banjir", y="provinsi",
                     orientation="h",
                     color="jumlah_banjir",
                     color_continuous_scale=["#85C1E9","#1F4E79"],
                     title="Top 15 Provinsi Frekuensi Banjir (BNPB DIBI)",
                     labels={"jumlah_banjir":"Jumlah Kejadian","provinsi":"Provinsi"})
        fig.update_layout(yaxis={"categoryorder":"total ascending"},
                          coloraxis_showscale=False, height=500)
        _simpan_html(fig, "chart_banjir_provinsi.html")
    else:
        _bar_chart_matplotlib(top, "jumlah_banjir", "provinsi",
                              "Top 15 Provinsi - Frekuensi Banjir",
                              "chart_banjir_provinsi.png")


# ==============================================================================
# CHART 2 - Tren Banjir Tahunan
# ==============================================================================
def chart_tren_tahunan(df_bnpb: pd.DataFrame):
    """Line chart tren jumlah kejadian banjir per tahun."""
    if df_bnpb.empty or "tahun" not in df_bnpb.columns:
        return

    tren = (df_bnpb.groupby("tahun")
                   .agg(jumlah=("tahun","count"),
                        korban=("korban_meninggal","sum"))
                   .reset_index())

    if PLOTLY_OK:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=tren["tahun"], y=tren["jumlah"],
                             name="Kejadian Banjir", marker_color=WARNA["biru"]),
                      secondary_y=False)
        fig.add_trace(go.Scatter(x=tren["tahun"], y=tren["korban"],
                                 name="Korban Meninggal",
                                 line=dict(color=WARNA["kritis"], width=2),
                                 mode="lines+markers"),
                      secondary_y=True)
        fig.update_layout(title="Tren Banjir Tahunan - Kejadian & Korban",
                          xaxis_title="Tahun", height=420,
                          legend=dict(x=0.01, y=0.99))
        fig.update_yaxes(title_text="Jumlah Kejadian", secondary_y=False)
        fig.update_yaxes(title_text="Korban Meninggal", secondary_y=True)
        _simpan_html(fig, "chart_tren_tahunan.html")


# ==============================================================================
# CHART 3 - Distribusi Status Risiko (dari Feature Set)
# ==============================================================================
def chart_distribusi_risiko(df_fitur: pd.DataFrame):
    """Pie chart distribusi status risiko banjir."""
    if df_fitur.empty or "status_risiko" not in df_fitur.columns:
        return

    dist = (df_fitur["status_risiko"]
                    .value_counts()
                    .reindex(["rendah","sedang","tinggi","kritis"], fill_value=0)
                    .reset_index())
    dist.columns = ["risiko", "jumlah"]

    if PLOTLY_OK:
        fig = px.pie(dist, values="jumlah", names="risiko",
                     title="Distribusi Status Risiko Banjir",
                     color="risiko",
                     color_discrete_map=WARNA,
                     hole=0.35)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        _simpan_html(fig, "chart_distribusi_risiko.html")


# ==============================================================================
# CHART 4 - Curah Hujan vs Label Banjir (Boxplot)
# ==============================================================================
def chart_hujan_vs_banjir(df_fitur: pd.DataFrame):
    """Boxplot curah hujan rata-rata: banjir vs tidak banjir."""
    if df_fitur.empty or "avg_rainfall" not in df_fitur.columns:
        return

    df = df_fitur[["avg_rainfall","label_banjir"]].dropna().copy()
    df["Status"] = df["label_banjir"].map({True:"Banjir", False:"Tidak Banjir"})

    if PLOTLY_OK:
        fig = px.box(df, x="Status", y="avg_rainfall",
                     color="Status",
                     color_discrete_map={"Banjir": WARNA["kritis"],
                                         "Tidak Banjir": WARNA["biru"]},
                     title="Distribusi Curah Hujan: Banjir vs Tidak Banjir",
                     labels={"avg_rainfall":"Curah Hujan Rata-rata (mm)"})
        fig.update_layout(height=420, showlegend=False)
        _simpan_html(fig, "chart_hujan_vs_banjir.html")


# ==============================================================================
# CHART 5 - Heatmap Korelasi Fitur
# ==============================================================================
def chart_heatmap_korelasi(df_fitur: pd.DataFrame):
    """Heatmap korelasi antar fitur numerik."""
    num_cols = ["avg_rainfall","max_rainfall","avg_temperature",
                "elevation","ndvi","slope","soil_moisture",
                "rolling_avg_7","risiko_index"]
    cols_ada  = [c for c in num_cols if c in df_fitur.columns]

    if len(cols_ada) < 3:
        return

    corr = df_fitur[cols_ada].corr().round(2)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.3,
                annot_kws={"size": 8})
    ax.set_title("Heatmap Korelasi Fitur Numerik", fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "chart_heatmap_korelasi.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Heatmap tersimpan: {path}")


# ==============================================================================
# CHART 6 - Scatter: Elevation vs Avg Rainfall, warna = label_banjir
# ==============================================================================
def chart_scatter_elevation_hujan(df_fitur: pd.DataFrame):
    """Scatter plot elevasi vs curah hujan, diwarnai status banjir."""
    if not all(c in df_fitur.columns
               for c in ["elevation","avg_rainfall","label_banjir"]):
        return

    df = df_fitur[["elevation","avg_rainfall","label_banjir",
                   "kabupaten","status_risiko"]].dropna().sample(
                       min(3000, len(df_fitur)), random_state=42)
    df["Status"] = df["label_banjir"].map({True:"Banjir", False:"Tidak Banjir"})

    if PLOTLY_OK:
        fig = px.scatter(df, x="elevation", y="avg_rainfall",
                         color="Status",
                         color_discrete_map={"Banjir":      WARNA["kritis"],
                                             "Tidak Banjir": WARNA["biru"]},
                         hover_data=["kabupaten","status_risiko"],
                         title="Elevasi vs Curah Hujan (sample 3.000 titik)",
                         labels={"elevation":"Elevasi (m)",
                                 "avg_rainfall":"Curah Hujan Rata-rata (mm)"},
                         opacity=0.6, height=450)
        _simpan_html(fig, "chart_scatter_elevation_hujan.html")


# ==============================================================================
# CHART 7 - Cuaca Harian (dari master_cuaca)
# ==============================================================================
def chart_cuaca_harian(df_cuaca: pd.DataFrame):
    """Line chart curah hujan harian dari data cuaca training."""
    if df_cuaca.empty or "curah_hujan_mm" not in df_cuaca.columns:
        return

    tanggal_col = "tanggal" if "tanggal" in df_cuaca.columns else None
    if not tanggal_col:
        return

    df = df_cuaca.copy()
    df["tanggal"] = pd.to_datetime(df[tanggal_col], errors="coerce")
    daily = (df.groupby("tanggal")
               .agg({"curah_hujan_mm":"mean",
                     "suhu_celcius":"mean",
                     "kelembapan_persen":"mean"})
               .reset_index()
               .sort_values("tanggal"))

    if PLOTLY_OK:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("Curah Hujan Harian (mm)",
                                            "Suhu & Kelembapan"),
                            vertical_spacing=0.12)
        fig.add_trace(go.Bar(x=daily["tanggal"], y=daily["curah_hujan_mm"],
                             name="Curah Hujan", marker_color=WARNA["biru"]),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=daily["tanggal"], y=daily["suhu_celcius"],
                                 name="Suhu (°C)",
                                 line=dict(color=WARNA["kritis"], width=1.5)),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=daily["tanggal"], y=daily["kelembapan_persen"],
                                 name="Kelembapan (%)",
                                 line=dict(color=WARNA["teal"], width=1.5)),
                      row=2, col=1)
        fig.update_layout(title="Data Cuaca Harian - Master Cuaca Training",
                          height=500)
        _simpan_html(fig, "chart_cuaca_harian.html")


# ==============================================================================
# HELPER
# ==============================================================================
def _simpan_html(fig, nama_file: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, nama_file)
    fig.write_html(path, include_plotlyjs="cdn")
    log.info(f"  Chart tersimpan: {path}")


def _bar_chart_matplotlib(df, x_col, y_col, title, filename):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(df[y_col], df[x_col], color=WARNA["biru"])
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(x_col)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    log.info(f"  Chart tersimpan: {path}")


# ==============================================================================
# FUNGSI UTAMA DASHBOARD
# ==============================================================================
def run_dashboard(data: dict = None):
    """
    Generate semua chart dashboard.

    Args:
        data: dict berisi df keys: 'bnpb', 'combine', 'cuaca', 'fitur'.
              Jika None, load otomatis dari folder processed/features.
    """
    log.info("=" * 55)
    log.info("TAHAP 5: DASHBOARD INTERAKTIF")
    log.info("=" * 55)

    if data is None:
        data = load_all_data()

    df_bnpb    = data.get("bnpb",    pd.DataFrame())
    df_combine = data.get("combine", pd.DataFrame())
    df_cuaca   = data.get("cuaca",   pd.DataFrame())
    df_fitur   = data.get("fitur",   pd.DataFrame())

    log.info("Membuat charts...")
    chart_banjir_per_provinsi(df_bnpb)
    chart_tren_tahunan(df_bnpb)
    chart_distribusi_risiko(df_fitur if not df_fitur.empty else df_combine)
    chart_hujan_vs_banjir(df_fitur if not df_fitur.empty else df_combine)
    chart_heatmap_korelasi(df_fitur if not df_fitur.empty else df_combine)
    chart_scatter_elevation_hujan(df_fitur if not df_fitur.empty else df_combine)
    chart_cuaca_harian(df_cuaca)

    out = os.path.abspath(OUTPUT_DIR)
    files = [f for f in os.listdir(out) if f.endswith((".html",".png"))]
    log.info(f"\nDashboard selesai! {len(files)} chart tersimpan di: {out}")
    for f in sorted(files):
        log.info(f"  {f}")
    return out


if __name__ == "__main__":
    run_dashboard()
