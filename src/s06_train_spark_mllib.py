"""Tahap 6, pengembangan model DENGAN Spark MLlib.

Algoritma setara dengan versi scikit-learn: LinearRegression, RandomForestRegressor,
GBTRegressor. Pipeline MLlib: StringIndexer -> OneHotEncoder -> VectorAssembler
(-> StandardScaler untuk model linear) -> model. Pemilihan hyperparameter memakai split
'val' yang sama dengan scikit-learn supaya perbandingannya adil.
"""
import json
import time
from itertools import product

import numpy as np
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.regression import GBTRegressor, LinearRegression, RandomForestRegressor
from pyspark.sql import functions as F

from common_ml import CATEGORICAL, LABEL_FEATURES, NUMERIC, TARGET, regression_metrics
from config import ML_DIR, OUT, SEED, get_spark, uri


def stages_for(model, scale: bool):
    idx = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in CATEGORICAL]
    ohe = OneHotEncoder(inputCols=[f"{c}_idx" for c in CATEGORICAL],
                        outputCols=[f"{c}_ohe" for c in CATEGORICAL], handleInvalid="keep")
    asm = VectorAssembler(inputCols=NUMERIC + [f"{c}_ohe" for c in CATEGORICAL] + LABEL_FEATURES,
                          outputCol="features_raw")
    if scale:
        sc = StandardScaler(inputCol="features_raw", outputCol="features", withMean=False, withStd=True)
        return idx + [ohe, asm, sc, model]
    asm.setOutputCol("features")
    return idx + [ohe, asm, model]


MODELS = {
    "Linear Regression": dict(
        build=lambda p: LinearRegression(labelCol=TARGET, featuresCol="features",
                                         regParam=p["regParam"], elasticNetParam=0.0),
        grid={"regParam": [0.0, 0.001, 0.01]}, scale=True),
    "Random Forest": dict(
        build=lambda p: RandomForestRegressor(labelCol=TARGET, featuresCol="features", numTrees=60,
                                              maxDepth=p["maxDepth"], featureSubsetStrategy="onethird",
                                              subsamplingRate=0.8, maxMemoryInMB=512,
                                              cacheNodeIds=True, seed=SEED),
        grid={"maxDepth": [10, 15]}, scale=False),
    "GBT": dict(
        build=lambda p: GBTRegressor(labelCol=TARGET, featuresCol="features", maxIter=100,
                                     maxDepth=p["maxDepth"], stepSize=0.1, maxMemoryInMB=512,
                                     cacheNodeIds=True, seed=SEED),
        grid={"maxDepth": [5, 8]}, scale=False),
}


def fit_eval(name, params, train, test):
    spec = MODELS[name]
    pipe = Pipeline(stages=stages_for(spec["build"](params), spec["scale"]))
    t0 = time.perf_counter()
    model = pipe.fit(train)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = (model.transform(test).select("car_id", "price", F.exp("prediction").alias("pred"))
           .toPandas().sort_values("car_id"))
    pred_s = time.perf_counter() - t0
    return model, out, fit_s, pred_s


def importances(model):
    """featureImportances (vektor) dipetakan kembali ke nama kolom asli."""
    rf = model.stages[-1]
    attrs = model.transform(train_ref.limit(1)).schema["features"].metadata["ml_attr"]["attrs"]
    names = {}
    for group in attrs.values():
        for a in group:
            names[a["idx"]] = a["name"]
    vec = rf.featureImportances.toArray()
    rows = []
    for i, v in enumerate(vec):
        n = names.get(i, str(i))
        base = next((c for c in CATEGORICAL if n.startswith(f"{c}_ohe")), n)
        rows.append((base, v))
    return (pd.DataFrame(rows, columns=["feature", "importance"])
            .groupby("feature", as_index=False).importance.sum()
            .sort_values("importance", ascending=False))


def main():
    global train_ref
    spark = get_spark("train_spark_mllib")
    spark.sparkContext.setCheckpointDir(uri(OUT / "checkpoints"))
    df = spark.read.parquet(uri(ML_DIR / "ml_dataset")).cache()
    train = df.filter("split = 'train'").cache()
    val = df.filter("split = 'val'").cache()
    test = df.filter("split = 'test'").cache()
    trainval = df.filter("split in ('train', 'val')").cache()
    train_ref = train
    print(f"train {train.count():,}, val {val.count():,}, test {test.count():,}")

    results, tuning, preds = {}, {}, None
    for name, spec in MODELS.items():
        keys = list(spec["grid"])
        rows = []
        for values in product(*[spec["grid"][k] for k in keys]):
            params = dict(zip(keys, values))
            _, out, fit_s, _ = fit_eval(name, params, train, val)
            m = regression_metrics(out.price, out.pred)
            rows.append({"params": params, **m, "fit_seconds": round(fit_s, 1)})
            print(f"  {name} {params}: val RMSE {m['RMSE']}, R2 {m['R2']}  ({fit_s:.0f}s)")
        tuning[name] = rows
        best_params = min(rows, key=lambda r: r["RMSE"])["params"]
        model, out, fit_s, pred_s = fit_eval(name, best_params, trainval, test)
        m = regression_metrics(out.price, out.pred)
        results[name] = {"library": "Spark MLlib", "best_params": best_params, "test": m,
                         "fit_seconds": round(fit_s, 1), "predict_seconds": round(pred_s, 2)}
        print(f"[test] {name}: {m}  fit {fit_s:.1f}s")
        preds = out[["car_id", "price"]] if preds is None else preds
        preds[f"spark_{name}"] = out.pred.values
        if name == "GBT":
            importances(model).to_csv(OUT / "spark_gbt_importance.csv", index=False)
            model.write().overwrite().save(uri(ML_DIR / "spark_gbt_model"))

    preds.to_csv(OUT / "predictions_spark.csv", index=False)
    (OUT / "metrics_spark.json").write_text(json.dumps(results, indent=2))
    (OUT / "tuning_spark.json").write_text(json.dumps(tuning, indent=2))
    spark.stop()


if __name__ == "__main__":
    main()
