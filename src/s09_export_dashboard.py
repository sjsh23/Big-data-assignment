"""Tahap 9, export data untuk dashboard Power BI.

Membaca Curated Zone dan Serving Layer (Delta Lake) dengan Spark, lalu menggabungkannya
dengan hasil s05 sampai s08 di outputs/. Hasilnya CSV rapi di dashboard/data/ yang dibaca
Power BI. Setelah tahap ini dijalankan ulang, cukup tekan Refresh di Power BI.

Tabel yang ditulis (model star schema, lihat subbab 4.2.3):
  dim_brand.csv                    dimensi brand, penghubung slicer ke semua tabel
  dim_listing.csv                  satu baris per listing di Curated Zone (termasuk yang ditandai)
  fact_prediction.csv              prediksi test set, satu baris per listing test
  serving_price_by_brand_year.csv  Serving Layer: harga dan mileage rata-rata per brand per tahun
  serving_damage_distribution.csv  Serving Layer: jumlah listing dan gambar per jenis kerusakan
  model_comparison.csv             metrik test set semua model (s07)
  feature_importance.csv           permutation importance scikit-learn dan importance GBT Spark
  graph_pairs.csv                  pasangan kerusakan dan part dengan jumlah gambar dan lift (s08)
  graph_nodes.csv                  simpul graph dengan PageRank dan komunitas Louvain (s08)
  kpi_summary.csv                  ringkasan quality gate (s02) dan struktur graph (s08)
"""
import json
from datetime import datetime

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

from config import COLLECTION_YEAR, CURATED, ML_DIR, OUT, ROOT, SERVING, get_spark, read_table, uri

DASH = ROOT / "dashboard" / "data"

PRICE_BANDS = [0, 10000, 20000, 30000, 50000, np.inf]
PRICE_LABELS = ["di bawah 10rb", "10rb sampai 20rb", "20rb sampai 30rb", "30rb sampai 50rb", "di atas 50rb"]
AGE_BANDS = [-1, 2, 5, 9, np.inf]
AGE_LABELS = ["0-2 tahun", "3-5 tahun", "6-9 tahun", "10+ tahun"]


def banded(series, bins, labels):
    # Batas kanan inklusif, sama dengan pembagian rentang harga di s07 (Tabel 24 laporan).
    cat = pd.cut(series, bins=bins, labels=labels)
    order = cat.cat.codes + 1
    return cat.astype(str), order


def save(df, name):
    DASH.mkdir(parents=True, exist_ok=True)
    df.to_csv(DASH / name, index=False, encoding="utf-8")
    print(f"  {name:36s} {len(df):>7,} baris")
    return len(df)


def export_lakehouse(spark):
    """Bagian yang dibaca langsung dari tabel Delta di lakehouse."""
    listings = read_table(spark, CURATED / "listings").select(
        "car_id", "brand", "model", "year", "price", "mileage", "fuelType", "transmission",
        "engineSize", "mpg", "tax", "data_issue", "image_id", "image_source").toPandas()

    split = (spark.read.parquet(uri(ML_DIR / "ml_dataset")).select("car_id", "split").toPandas()
             if (ML_DIR / "ml_dataset").exists() else pd.DataFrame(columns=["car_id", "split"]))
    listings = listings.merge(split, on="car_id", how="left")

    listings["age"] = COLLECTION_YEAR - listings["year"]
    listings["status"] = np.where(listings["data_issue"].isna(), "Valid", "Ditandai")
    listings["data_issue"] = listings["data_issue"].fillna("valid")
    listings["price_band"], listings["price_band_order"] = banded(listings["price"], PRICE_BANDS, PRICE_LABELS)
    listings["age_band"], listings["age_band_order"] = banded(listings["age"], AGE_BANDS, AGE_LABELS)
    # Baris dengan tahun di atas 2020 (data_issue year_after_collection) punya umur negatif
    # dan tidak masuk rentang mana pun. Beri label eksplisit supaya tidak muncul sebagai blank.
    invalid_age = listings["age"] < 0
    listings.loc[invalid_age, "age_band"] = "Tahun tidak valid"
    listings.loc[invalid_age, "age_band_order"] = 0
    listings = listings.sort_values("car_id")

    pby = read_table(spark, SERVING / "price_by_brand_year").toPandas()
    pby["age"] = COLLECTION_YEAR - pby["year"]
    pby = pby.sort_values(["brand", "year"])

    dmg = read_table(spark, SERVING / "damage_distribution").toPandas().sort_values(
        "jumlah_gambar", ascending=False)
    return listings, pby, dmg


def export_predictions(listings):
    sk = pd.read_csv(OUT / "predictions_sklearn.csv")
    sp = pd.read_csv(OUT / "predictions_spark.csv")
    df = (sk[["car_id", "price", "sk_Gradient Boosting"]]
          .merge(sp[["car_id", "spark_GBT"]], on="car_id", how="left")
          .rename(columns={"price": "actual_price", "sk_Gradient Boosting": "pred_sklearn_gb",
                           "spark_GBT": "pred_spark_gbt"}))
    for tag, col in (("sklearn", "pred_sklearn_gb"), ("spark", "pred_spark_gbt")):
        err = df[col] - df["actual_price"]
        df[f"error_{tag}"] = err.round(1)
        df[f"abs_error_{tag}"] = err.abs().round(1)
        df[f"ape_pct_{tag}"] = (100 * err.abs() / df["actual_price"]).round(3)
        df[f"within_10pct_{tag}"] = (df[f"ape_pct_{tag}"] <= 10).astype(int)
    df["pred_sklearn_gb"] = df["pred_sklearn_gb"].round(0)
    df["pred_spark_gbt"] = df["pred_spark_gbt"].round(0)

    missing = set(df.car_id) - set(listings.car_id)
    if missing:
        raise SystemExit(f"{len(missing)} car_id di prediksi tidak ada di Curated Zone, cek ulang s04 sampai s06")
    return df.sort_values("car_id")


def export_models():
    mc = pd.read_csv(OUT / "model_comparison.csv").rename(
        columns={"Fit (detik)": "fit_seconds", "MAPE_pct": "MAPE_pct"})
    mc["model_label"] = mc["Library"] + " " + mc["Model"]
    mc["is_best"] = ((mc["Family"] != "Ablation") & (mc["RMSE"] == mc.loc[mc.Family != "Ablation", "RMSE"].min())).astype(int)
    mc = mc.drop(columns=["Params"])

    rows = []
    for method, path in (("Permutation importance (scikit-learn GB)", OUT / "sklearn_permutation_importance.csv"),
                         ("Split importance (Spark GBT)", OUT / "spark_gbt_importance.csv")):
        if path.exists():
            f = pd.read_csv(path)
            f["method"] = method
            rows.append(f)
    fi = pd.concat(rows, ignore_index=True)
    is_label = fi["feature"].str.startswith(("dmg_", "part_"))
    fi["kelompok"] = np.where(is_label, "Label gambar (29 fitur)", "Atribut kendaraan")
    fi["importance"] = fi["importance"].round(5)
    return mc, fi


def export_graph():
    dp = pd.read_csv(OUT / "graph_damage_part_pairs.csv").rename(columns={"damage": "from_label", "part": "to_label"})
    dp["pair_type"], dp["to_type"] = "Damage-Part", "Part"
    dd = pd.read_csv(OUT / "graph_damage_damage_pairs.csv").rename(columns={"damage_a": "from_label", "damage_b": "to_label"})
    dd["pair_type"], dd["to_type"] = "Damage-Damage", "Damage"
    pairs = pd.concat([dp, dd], ignore_index=True)
    pairs["from_type"] = "Damage"
    pairs["lift"] = pairs["lift"].round(3)
    pairs["in_assoc_graph"] = ((pairs["images"] >= 20) & (pairs["lift"] >= 1.05)).astype(int)
    pairs["pair_label"] = pairs["from_label"] + " dan " + pairs["to_label"]
    pairs = pairs[["pair_type", "from_type", "from_label", "to_type", "to_label", "pair_label",
                   "images", "lift", "in_assoc_graph"]].sort_values("lift", ascending=False)

    nodes = pd.read_csv(OUT / "graph_nodes_metrics.csv")
    deg = pd.read_csv(OUT / "graph_label_degree.csv").rename(columns={"images": "jumlah_gambar"})
    nodes = nodes.merge(deg[["name", "jumlah_gambar"]], on="name", how="left")
    # Nomor komunitas dimulai dari 1, sama dengan penomoran di Tabel 26 laporan.
    nodes["community"] = (nodes["louvain_community"] + 1).astype("Int64")
    nodes["community_label"] = np.where(nodes["community"].isna(), "Tanpa asosiasi",
                                        "Komunitas " + nodes["community"].astype(str))
    nodes = nodes[["name", "type", "jumlah_gambar", "strength", "pagerank_weighted",
                   "pagerank_graphframes", "assoc_degree", "community", "community_label"]]
    return pairs, nodes


def export_kpi(listings, pred):
    q = json.loads((OUT / "quality_report.json").read_text())
    img = q.get("image_check", {}) if isinstance(q.get("image_check"), dict) else {}
    rows = [
        ("Baris masuk Curated Zone", q["rows_in"]),
        ("Baris ditandai", q["rows_flagged"]),
        ("Persen ditandai", q["pct_flagged"]),
        ("Ambang quality gate (persen)", q["threshold_pct"]),
        ("Baris bersih untuk ML", int((listings.status == "Valid").sum())),
        ("Path gambar unik", img.get("unique_paths")),
        ("Path gambar hilang", img.get("missing")),
        ("Listing di test set", len(pred)),
    ]
    rows += [(f"Ditandai: {k}", v) for k, v in sorted(q["by_rule"].items(), key=lambda kv: -kv[1])]
    g = json.loads((OUT / "graph_summary.json").read_text())
    rows += [
        ("Graph: gambar dengan label kerusakan dan part", g["images_with_damage_and_part"]),
        ("Graph frekuensi: edge", g["freq_graph"]["edges"]),
        ("Graph frekuensi: modularity Louvain", g["freq_graph"]["louvain_modularity"]),
        ("Graph asosiasi: edge", g["assoc_graph"]["edges"]),
        ("Graph asosiasi: komunitas Louvain", g["assoc_graph"]["louvain_communities"]),
        ("Graph asosiasi: modularity Louvain", g["assoc_graph"]["louvain_modularity"]),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def main():
    spark = get_spark("export_dashboard")
    listings, pby, dmg = export_lakehouse(spark)
    spark.stop()

    pred = export_predictions(listings)
    mc, fi = export_models()
    pairs, nodes = export_graph()
    kpi = export_kpi(listings, pred)
    brands = pd.DataFrame({"brand": sorted(listings["brand"].unique())})

    print(f"Menulis CSV untuk Power BI ke {DASH}")
    counts = {
        "dim_brand.csv": save(brands, "dim_brand.csv"),
        "dim_listing.csv": save(listings, "dim_listing.csv"),
        "fact_prediction.csv": save(pred, "fact_prediction.csv"),
        "serving_price_by_brand_year.csv": save(pby, "serving_price_by_brand_year.csv"),
        "serving_damage_distribution.csv": save(dmg, "serving_damage_distribution.csv"),
        "model_comparison.csv": save(mc, "model_comparison.csv"),
        "feature_importance.csv": save(fi, "feature_importance.csv"),
        "graph_pairs.csv": save(pairs, "graph_pairs.csv"),
        "graph_nodes.csv": save(nodes, "graph_nodes.csv"),
        "kpi_summary.csv": save(kpi, "kpi_summary.csv"),
    }
    info = {"exported_at": datetime.now().isoformat(timespec="seconds"), "rows": counts,
            "catatan": "Angka desimal memakai titik. Di Power BI set locale import ke English (United States)."}
    (DASH / "_export_info.json").write_text(json.dumps(info, indent=2))

    within = pred["within_10pct_sklearn"].mean() * 100
    print(f"\nCek cepat: MAPE GB scikit-learn {pred['ape_pct_sklearn'].mean():.2f}%, "
          f"{within:.1f}% prediksi test set selisihnya maksimal 10%")


if __name__ == "__main__":
    main()
