"""Tahap 2, Raw Zone ke Curated Zone dengan Spark.

Membaca CSV dengan skema eksplisit, merapikan teks, menjalankan aturan validasi,
memberi penanda data_issue, menulis Curated Zone (Delta, partisi per brand) dan
quarantine untuk baris bermasalah. Bila persentase baris bermasalah melewati ambang,
tahap ini gagal supaya tahap berikutnya tidak berjalan.
"""
import json
from datetime import datetime

from pyspark.sql import functions as F
from pyspark.sql import types as T

from config import COLLECTION_YEAR, CURATED, IMAGE_ROOT, OUT, RAW, get_spark, uri, write_table

SCHEMA = T.StructType([
    T.StructField("car_id", T.LongType()),
    T.StructField("brand", T.StringType()),
    T.StructField("model", T.StringType()),
    T.StructField("year", T.IntegerType()),
    T.StructField("price", T.IntegerType()),
    T.StructField("transmission", T.StringType()),
    T.StructField("mileage", T.IntegerType()),
    T.StructField("fuelType", T.StringType()),
    T.StructField("tax", T.IntegerType()),
    T.StructField("mpg", T.DoubleType()),
    T.StructField("engineSize", T.DoubleType()),
    T.StructField("data_issue_source", T.StringType()),  # kolom bawaan file, diganti hasil rule di bawah
    T.StructField("image_id", T.StringType()),
    T.StructField("image_path", T.StringType()),
    T.StructField("image_source", T.StringType()),
    T.StructField("damage_labels", T.StringType()),
    T.StructField("part_labels", T.StringType()),
])

MAX_ISSUE_PCT = 5.0  # ambang: lebih dari 5% baris bermasalah berarti ada yang salah di sumber


def main():
    spark = get_spark("raw_to_curated")
    raw = (spark.read.option("header", True).schema(SCHEMA)
           .csv(uri(RAW / "source=kaggle_combined"))
           .withColumn("source_file", F.input_file_name()))

    df = (raw.withColumn("model", F.trim("model"))
             .withColumn("brand", F.trim("brand"))
             .drop("data_issue_source"))

    # Aturan validasi. Dua aturan pertama dari Assignment I, dua terakhir melengkapi Tabel 14.
    issue = (F.when((F.col("engineSize") == 0) & (F.col("fuelType") != "Electric"), "zero_engine_size")
              .when(F.col("year") > COLLECTION_YEAR, "year_after_collection")
              .when(F.col("year") < 1990, "year_before_1990")
              .when(F.col("mpg") < 10, "mpg_below_10"))
    df = (df.withColumn("data_issue", issue)
            .withColumn("ingest_ts", F.lit(datetime.now().isoformat(timespec="seconds"))))

    total = df.count()
    issues = df.filter(F.col("data_issue").isNotNull())
    n_issue = issues.count()
    pct = 100 * n_issue / total
    by_rule = {r["data_issue"]: r["count"] for r in issues.groupBy("data_issue").count().collect()}
    null_pct = {c: round(100 * df.filter(F.col(c).isNull()).count() / total, 2)
                for c in ["price", "mileage", "mpg", "damage_labels", "part_labels"]}
    report = {"rows_in": total, "rows_flagged": n_issue, "pct_flagged": round(pct, 3),
              "by_rule": by_rule, "null_pct": null_pct, "threshold_pct": MAX_ISSUE_PCT}

    # Prinsip pointer: tabel hanya menyimpan image_path, jadi cek apakah file gambarnya benar-benar ada.
    paths = [r["image_path"] for r in df.select("image_path").distinct().collect()]
    if IMAGE_ROOT.exists():
        missing = [p for p in paths if not (IMAGE_ROOT / p).exists()]
        report["image_check"] = {"unique_paths": len(paths), "missing": len(missing),
                                 "missing_examples": missing[:5]}
    else:
        report["image_check"] = f"dilewati, folder {IMAGE_ROOT.name} tidak ditemukan"
    (OUT / "quality_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    if pct > MAX_ISSUE_PCT:
        raise SystemExit(f"Baris bermasalah {pct:.2f}% melewati ambang {MAX_ISSUE_PCT}%, pipeline dihentikan")

    write_table(df, CURATED / "listings", partition_by=["brand"])
    write_table(issues, CURATED / "quarantine")
    print(f"Curated Zone: {total:,} baris, {n_issue:,} ditandai dan disalin ke quarantine")
    spark.stop()


if __name__ == "__main__":
    main()
