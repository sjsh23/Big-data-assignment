"""Tahap 4, menyiapkan dataset ML dari Curated Zone (dikerjakan dengan Spark).

Langkah:
  1. buang baris yang punya data_issue
  2. fitur turunan: age = 2020 - year, log_price sebagai target
  3. label anotasi dipecah menjadi 29 indikator biner (8 kerusakan, 21 part)
  4. pembagian train / val / test berdasarkan image_id (bukan per baris)
     hash(image_id) mod 10: 0..6 train, 7 val, 8..9 test
Hasil ditulis sebagai Parquet supaya bisa dibaca pandas (scikit-learn) maupun Spark MLlib.
"""
import json

from pyspark.sql import functions as F

from config import (COLLECTION_YEAR, CURATED, DAMAGE_CLASSES, ML_DIR, OUT, PART_CLASSES,
                    get_spark, label_col, read_table, uri)


def main():
    spark = get_spark("prepare_ml_dataset")
    cur = read_table(spark, CURATED / "listings")
    clean = cur.filter(F.col("data_issue").isNull())

    df = (clean
          .withColumn("age", F.lit(COLLECTION_YEAR) - F.col("year"))
          .withColumn("log_price", F.log("price")))

    dmg = F.split(F.coalesce(F.col("damage_labels"), F.lit("")), r"\|")
    prt = F.split(F.coalesce(F.col("part_labels"), F.lit("")), r"\|")
    for c in DAMAGE_CLASSES:
        df = df.withColumn(label_col("dmg", c), F.array_contains(dmg, c).cast("int"))
    for c in PART_CLASSES:
        df = df.withColumn(label_col("part", c), F.array_contains(prt, c).cast("int"))

    bucket = F.pmod(F.xxhash64("image_id"), F.lit(10))
    df = df.withColumn("split", F.when(bucket <= 6, "train").when(bucket == 7, "val").otherwise("test"))

    keep = (["car_id", "image_id", "split", "price", "log_price", "age", "mileage", "tax", "mpg",
             "engineSize", "brand", "model", "transmission", "fuelType"]
            + [label_col("dmg", c) for c in DAMAGE_CLASSES]
            + [label_col("part", c) for c in PART_CLASSES])
    df = df.select(*keep)
    df.coalesce(1).write.mode("overwrite").parquet(uri(ML_DIR / "ml_dataset"))

    # ringkasan untuk laporan (subbab 4.3)
    pdf = df.toPandas()
    summary = {
        "rows": int(len(pdf)),
        "images": int(pdf.image_id.nunique()),
        "split_rows": pdf.split.value_counts().to_dict(),
        "split_images": pdf.groupby("split").image_id.nunique().to_dict(),
        "images_shared_across_splits": int((pdf.groupby("image_id").split.nunique() > 1).sum()),
        "price_skew": round(float(pdf.price.skew()), 2),
        "log_price_skew": round(float(pdf.log_price.skew()), 2),
        "mileage_skew": round(float(pdf.mileage.skew()), 2),
        "n_categories": {c: int(pdf[c].nunique()) for c in ["brand", "model", "transmission", "fuelType"]},
        "price_mean_by_split": pdf.groupby("split").price.mean().round(0).to_dict(),
    }
    (OUT / "ml_dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    spark.stop()


if __name__ == "__main__":
    main()
