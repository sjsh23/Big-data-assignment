"""Tahap 5, pengembangan model TANPA Spark MLlib (scikit-learn, satu mesin).

Tiga algoritma: Ridge (baseline linear), Random Forest, Histogram Gradient Boosting.
Hyperparameter dipilih dengan validation set (split 'val', sudah dipisah per image_id),
lalu model final dilatih ulang pada train + val dan diuji sekali pada test.
"""
import json
import time
from itertools import product

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common_ml import CATEGORICAL, LABEL_FEATURES, NUMERIC, TARGET, regression_metrics
from config import ML_DIR, OUT, SEED

FEATURES = NUMERIC + CATEGORICAL + LABEL_FEATURES


def make_preprocessor(dense: bool, scale: bool, features):
    num = [c for c in NUMERIC if c in features]
    cat = [c for c in CATEGORICAL if c in features]
    lab = [c for c in LABEL_FEATURES if c in features]
    return ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense), cat),
        ("lab", "passthrough", lab),
    ], sparse_threshold=0.0 if dense else 0.3)


MODELS = {
    "Ridge": dict(
        build=lambda p: Ridge(alpha=p["alpha"]),
        grid={"alpha": [0.1, 1.0, 10.0]}, dense=False, scale=True),
    "Random Forest": dict(
        build=lambda p: RandomForestRegressor(n_estimators=100, max_depth=p["max_depth"],
                                              max_features=0.33, n_jobs=-1, random_state=SEED),
        grid={"max_depth": [15, 25]}, dense=False, scale=False),
    "Gradient Boosting": dict(
        build=lambda p: HistGradientBoostingRegressor(max_iter=500, learning_rate=p["learning_rate"],
                                                      max_leaf_nodes=p["max_leaf_nodes"],
                                                      early_stopping=False, random_state=SEED),
        grid={"learning_rate": [0.05, 0.1], "max_leaf_nodes": [31, 63]}, dense=True, scale=False),
}


def pipeline(name, params, features):
    spec = MODELS[name]
    return Pipeline([("prep", make_preprocessor(spec["dense"], spec["scale"], features)),
                     ("model", spec["build"](params))])


def fit_eval(name, params, train, test, features):
    pipe = pipeline(name, params, features)
    t0 = time.perf_counter()
    pipe.fit(train[features], train[TARGET])
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = np.exp(pipe.predict(test[features]))
    pred_s = time.perf_counter() - t0
    return pipe, pred, fit_s, pred_s


def tune(name, train, val):
    spec = MODELS[name]
    keys = list(spec["grid"])
    rows = []
    for values in product(*[spec["grid"][k] for k in keys]):
        params = dict(zip(keys, values))
        _, pred, fit_s, _ = fit_eval(name, params, train, val, FEATURES)
        m = regression_metrics(val["price"], pred)
        rows.append({"params": params, **m, "fit_seconds": round(fit_s, 1)})
        print(f"  {name} {params}: val RMSE {m['RMSE']}, R2 {m['R2']}")
    best = min(rows, key=lambda r: r["RMSE"])
    return best["params"], rows


def main():
    df = pd.read_parquet(ML_DIR / "ml_dataset")
    train, val, test = (df[df.split == s].copy() for s in ("train", "val", "test"))
    trainval = pd.concat([train, val])
    print(f"train {len(train):,}, val {len(val):,}, test {len(test):,}")

    results, tuning, preds = {}, {}, {"car_id": test.car_id.values, "brand": test.brand.values,
                                      "price": test.price.values}
    for name in MODELS:
        print(f"[tuning] {name}")
        best_params, rows = tune(name, train, val)
        tuning[name] = rows
        pipe, pred, fit_s, pred_s = fit_eval(name, best_params, trainval, test, FEATURES)
        m = regression_metrics(test["price"], pred)
        results[name] = {"library": "scikit-learn", "best_params": best_params, "test": m,
                         "fit_seconds": round(fit_s, 1), "predict_seconds": round(pred_s, 2)}
        preds[f"sk_{name}"] = pred
        print(f"[test] {name}: {m}  fit {fit_s:.1f}s")
        if name == "Gradient Boosting":
            best_pipe = pipe

    # Ablation: apakah 29 fitur label gambar menambah informasi?
    base_feats = NUMERIC + CATEGORICAL
    gb_params = results["Gradient Boosting"]["best_params"]
    _, pred_nolab, fit_s, _ = fit_eval("Gradient Boosting", gb_params, trainval, test, base_feats)
    results["Gradient Boosting (tanpa fitur label gambar)"] = {
        "library": "scikit-learn", "best_params": gb_params,
        "test": regression_metrics(test["price"], pred_nolab), "fit_seconds": round(fit_s, 1)}

    # Permutation importance pada level kolom asli (sampel 5.000 baris test)
    sample = test.sample(5000, random_state=SEED)
    imp = permutation_importance(best_pipe, sample[FEATURES], sample[TARGET], n_repeats=3,
                                 random_state=SEED, scoring="neg_root_mean_squared_error", n_jobs=1)
    imp_df = (pd.DataFrame({"feature": FEATURES, "importance": imp.importances_mean})
              .sort_values("importance", ascending=False))
    imp_df.to_csv(OUT / "sklearn_permutation_importance.csv", index=False)

    pd.DataFrame(preds).to_csv(OUT / "predictions_sklearn.csv", index=False)
    (OUT / "metrics_sklearn.json").write_text(json.dumps(results, indent=2))
    (OUT / "tuning_sklearn.json").write_text(json.dumps(tuning, indent=2))
    print(imp_df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
