"""Orkestrator lokal: menjalankan seluruh tahap berurutan dan berhenti bila satu tahap gagal.

Urutan dan dependensinya sama dengan DAG Airflow di dags/used_car_pipeline_dag.py.
Contoh:
    python run_pipeline.py                 # semua tahap
    python run_pipeline.py --from s05      # mulai dari tahap tertentu
    python run_pipeline.py --only s08      # satu tahap saja
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
STAGES = [
    ("s01", "s01_ingest_raw.py", "Ingestion ke Raw Zone"),
    ("s02", "s02_raw_to_curated.py", "Raw Zone ke Curated Zone (validasi, quarantine)"),
    ("s03", "s03_curated_to_serving.py", "Curated Zone ke Serving Layer"),
    ("s04", "s04_prepare_ml_dataset.py", "Menyiapkan dataset ML"),
    ("s05", "s05_train_sklearn.py", "Model tanpa Spark MLlib (scikit-learn)"),
    ("s06", "s06_train_spark_mllib.py", "Model dengan Spark MLlib"),
    ("s07", "s07_evaluate.py", "Evaluasi dan perbandingan model"),
    ("s08", "s08_graph_analytics.py", "Graph analytics"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="s01")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    keys = [s[0] for s in STAGES]
    todo = [s for s in STAGES if s[0] == args.only] if args.only else STAGES[keys.index(args.start):]
    log = []
    for key, script, desc in todo:
        print(f"\n=== {key}: {desc} ===", flush=True)
        t0 = time.perf_counter()
        rc = subprocess.call([sys.executable, script], cwd=SRC)
        dt = time.perf_counter() - t0
        log.append(f"{key}\t{desc}\t{'OK' if rc == 0 else 'GAGAL'}\t{dt:.1f}s")
        if rc != 0:
            print(f"\nTahap {key} gagal (exit {rc}). Tahap berikutnya tidak dijalankan.")
            break
    (SRC.parent / "outputs" / "pipeline_run_log.tsv").write_text("\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    main()
