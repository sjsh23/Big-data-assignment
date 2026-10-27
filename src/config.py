"""Konfigurasi bersama: path lakehouse, SparkSession, dan helper baca/tulis tabel."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"                 # Raw Zone (pengganti object storage di laptop)
CURATED = ROOT / "lakehouse" / "curated"    # Curated Zone (Delta Lake)
SERVING = ROOT / "lakehouse" / "serving"    # Serving Layer (tabel agregat)
ML_DIR = ROOT / "lakehouse" / "ml"          # dataset siap model (Parquet)
OUT = ROOT / "outputs"                      # metrik, tabel hasil, dan gambar

SOURCE_FILE = "used_cars_with_images.csv"
# Hasil script penggabungan dataset Assignment I. data/source dipakai bila ada (struktur zip lama).
SOURCE_PATH = next((d / SOURCE_FILE for d in (ROOT / "data" / "processed", ROOT / "data" / "source")
                    if (d / SOURCE_FILE).exists()), ROOT / "data" / "processed" / SOURCE_FILE)
# Folder dataset Kaggle asli. image_path di CSV relatif terhadap IMAGE_ROOT.
IMAGE_ROOT = ROOT / "car damages images"
LISTING_SOURCE_DIR = ROOT / "Uk selling cars"
COLLECTION_YEAR = 2020   # tahun listing paling baru yang valid di dataset
SEED = 42

# USE_DELTA=0 untuk fallback ke Parquet biasa bila jar Delta gagal diunduh
USE_DELTA = os.environ.get("USE_DELTA", "1") == "1"
DELTA_PKG_NOTE = "delta-spark 3.2.x untuk Spark 3.5"
GRAPHFRAMES_PKG = "graphframes:graphframes:0.8.4-spark3.5-s_2.12"

for p in (RAW, CURATED, SERVING, ML_DIR, OUT):
    p.mkdir(parents=True, exist_ok=True)

# Pastikan worker Spark memakai Python yang sama dengan driver (penting di Windows)
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)


def uri(path: Path) -> str:
    """Path lokal ke URI file:/// tanpa percent-encoding (aman di Windows maupun Linux)."""
    p = Path(path).resolve().as_posix()
    return "file://" + p if p.startswith("/") else "file:///" + p


def get_spark(app_name: str, with_graphframes: bool = False):
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "6g"))
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.session.timeZone", "Asia/Jakarta")
        .config("spark.ui.showConsoleProgress", "false")
    )
    extra = [GRAPHFRAMES_PKG] if with_graphframes else []
    if extra:
        builder = builder.config("spark.jars.repositories", "https://repos.spark-packages.org")

    if USE_DELTA:
        from delta import configure_spark_with_delta_pip
        builder = (
            builder.config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        )
        builder = configure_spark_with_delta_pip(builder, extra_packages=extra)
    elif extra:
        builder = builder.config("spark.jars.packages", ",".join(extra))

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    port = spark.sparkContext.uiWebUrl.rsplit(":", 1)[-1] if spark.sparkContext.uiWebUrl else "4040"
    print(f"Spark UI: http://localhost:{port}  (hanya aktif selama script ini berjalan)", flush=True)
    if os.environ.get("SPARK_UI_HOLD") == "1":
        # Tahan SparkSession di akhir script supaya Spark UI masih bisa dibuka dan di-screenshot.
        original_stop = spark.stop

        def stop_with_hold():
            input(f"\nSelesai. Spark UI masih terbuka di http://localhost:{port}. Tekan Enter untuk menutup... ")
            original_stop()

        spark.stop = stop_with_hold
    return spark


def write_table(df, path: Path, partition_by=None, mode="overwrite"):
    fmt = "delta" if USE_DELTA else "parquet"
    w = df.write.format(fmt).mode(mode)
    if USE_DELTA:
        w = w.option("overwriteSchema", "true")
    if partition_by:
        w = w.partitionBy(*partition_by)
    w.save(uri(path))


def read_table(spark, path: Path, version=None):
    if USE_DELTA:
        r = spark.read.format("delta")
        if version is not None:
            r = r.option("versionAsOf", version)
        return r.load(uri(path))
    return spark.read.parquet(uri(path))


DAMAGE_CLASSES = ["Broken part", "Cracked", "Corrosion", "Dent", "Flaking",
                  "Missing part", "Paint chip", "Scratch"]
PART_CLASSES = ["Back-bumper", "Back-door", "Back-wheel", "Back-window", "Back-windshield",
                "Fender", "Front-bumper", "Front-door", "Front-wheel", "Front-window",
                "Grille", "Headlight", "Hood", "License-plate", "Mirror", "Quarter-panel",
                "Rocker-panel", "Roof", "Tail-light", "Trunk", "Windshield"]


def label_col(prefix: str, label: str) -> str:
    """'Broken part' -> 'dmg_broken_part', 'Back-bumper' -> 'part_back_bumper'."""
    return prefix + "_" + label.lower().replace(" ", "_").replace("-", "_")
