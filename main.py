#!/usr/bin/env python3
# ==============================================================================
# main.py v2 — Pipeline Lengkap: ETL + ML + Dashboard
# ==============================================================================
# Penggunaan:
#   python main.py                         → Semua tahap
#   python main.py --tahap etl             → Hanya ETL
#   python main.py --tahap ml              → Hanya ML
#   python main.py --tahap dashboard       → Hanya Dashboard
#   python main.py --skip-load             → ETL tanpa MySQL
#   python main.py --skip-lstm             → Skip LSTM (lebih cepat)
# ==============================================================================

import os, sys, time, argparse, logging
import pandas as pd
from datetime import datetime

# Ensure working directory is the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            f"logs/pipeline_{datetime.now().strftime('%Y%m%d')}.log",
            encoding="utf-8"),
    ]
)
log = logging.getLogger(__name__)


def buat_folder():
    for d in ["data/raw","data/processed","data/features",
              "logs","models","dashboard/output","ml"]:
        os.makedirs(d, exist_ok=True)


def pastikan_file_ada():
    files = {
        "data/raw/Data_Bencana.xlsx":
            "BNPB DIBI — download dari dibi.bnpb.go.id",
        "data/raw/data_banjir_combine_final.csv":
            "Dataset ML gabungan",
        "data/raw/master_cuaca_training_banjir_copy.csv":
            "Data cuaca training",
    }
    semua_ada = True
    for path, sumber in files.items():
        if not os.path.exists(path):
            log.error(f"File tidak ditemukan: {path}  (sumber: {sumber})")
            semua_ada = False
    return semua_ada


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline MKPL Deteksi Dini Banjir v2")
    parser.add_argument("--tahap",
                        choices=["etl","ml","dashboard","semua"],
                        default="semua")
    parser.add_argument("--skip-load",  action="store_true")
    parser.add_argument("--skip-lstm",  action="store_true")
    args = parser.parse_args()

    buat_folder()
    t0 = time.time()
    log.info("=" * 60)
    log.info("PIPELINE MKPL v2 — SISTEM DETEKSI DINI BANJIR")
    log.info(f"Mulai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    transformed = {}
    hasil_ml    = {}

    try:
        # ETL
        if args.tahap in ("etl","semua"):
            if not pastikan_file_ada():
                sys.exit(1)
            from etl.extract   import run_extract
            from etl.transform import run_transform
            raw         = run_extract()
            transformed = run_transform(raw)
            if not args.skip_load:
                from etl.load import run_load
                run_load(transformed)

        # ML
        if args.tahap in ("ml","semua"):
            if not transformed:
                feat_dir = "data/features"
                proc_dir = "data/processed"
                feat_f   = sorted([f for f in os.listdir(feat_dir) if f.endswith(".csv")])
                cuaca_f  = sorted([f for f in os.listdir(proc_dir)
                                   if "cuaca" in f and f.endswith(".csv")])
                if not feat_f:
                    log.error("Tidak ada feature set. Jalankan ETL dulu.")
                    sys.exit(1)
                transformed = {
                    "fitur": pd.read_csv(os.path.join(feat_dir, feat_f[-1])),
                    "cuaca": pd.read_csv(os.path.join(proc_dir, cuaca_f[-1]))
                              if cuaca_f else pd.DataFrame(),
                    "bnpb": pd.DataFrame(), "combine": pd.DataFrame(),
                }
            from ml.evaluate import run_all_ml
            hasil_ml = run_all_ml(
                df_fitur=transformed["fitur"],
                df_cuaca=transformed.get("cuaca", pd.DataFrame()),
                jalankan_lstm=not args.skip_lstm,
            )

        # Dashboard
        if args.tahap in ("dashboard","semua"):
            from dashboard.dashboard import run_dashboard
            run_dashboard({
                "bnpb":    transformed.get("bnpb",    pd.DataFrame()),
                "combine": transformed.get("combine", pd.DataFrame()),
                "cuaca":   transformed.get("cuaca",   pd.DataFrame()),
                "fitur":   transformed.get("fitur",   pd.DataFrame()),
            })

        dur = time.time() - t0
        log.info(f"\nPIPELINE SELESAI — {dur:.1f} detik ({dur/60:.1f} menit)")

    except (FileNotFoundError, KeyboardInterrupt, Exception) as e:
        log.error(f"Error: {e}", exc_info=isinstance(e, Exception)
                  and not isinstance(e, (FileNotFoundError, KeyboardInterrupt)))
        sys.exit(1)


if __name__ == "__main__":
    main()
