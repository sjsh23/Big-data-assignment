"""Tahap 3, Curated Zone ke Serving Layer: tabel agregat untuk dashboard."""
from pyspark.sql import functions as F

from config import CURATED, OUT, SERVING, USE_DELTA, get_spark, read_table, write_table


def main():
    spark = get_spark("curated_to_serving")
    cur = read_table(spark, CURATED / "listings").filter(F.col("data_issue").isNull())

    price_by_brand_year = (cur.groupBy("brand", "year")
                           .agg(F.count("*").alias("jumlah"),
                                F.round(F.avg("price"), 0).alias("harga_rata"),
                                F.round(F.avg("mileage"), 0).alias("mileage_rata"))
                           .orderBy("brand", "year"))
    write_table(price_by_brand_year, SERVING / "price_by_brand_year")

    damage_dist = (cur.filter(F.col("damage_labels").isNotNull())
                   .withColumn("damage", F.explode(F.split("damage_labels", r"\|")))
                   .groupBy("damage").agg(F.count("*").alias("jumlah_listing"),
                                          F.countDistinct("image_id").alias("jumlah_gambar"))
                   .orderBy(F.desc("jumlah_gambar")))
    write_table(damage_dist, SERVING / "damage_distribution")

    n1, n2 = price_by_brand_year.count(), damage_dist.count()
    print(f"Serving Layer: price_by_brand_year {n1} baris, damage_distribution {n2} baris")
    price_by_brand_year.toPandas().to_csv(OUT / "serving_price_by_brand_year.csv", index=False)

    if USE_DELTA:  # bukti versioning Delta Lake untuk subbab compliance dan lineage
        from delta.tables import DeltaTable
        from config import uri
        hist = DeltaTable.forPath(spark, uri(CURATED / "listings")).history()
        hist.select("version", "timestamp", "operation").show(truncate=False)
    spark.stop()


if __name__ == "__main__":
    main()
