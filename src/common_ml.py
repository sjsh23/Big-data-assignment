"""Definisi fitur dan metrik yang dipakai bersama oleh scikit-learn dan Spark MLlib."""
import numpy as np

from config import DAMAGE_CLASSES, PART_CLASSES, label_col

NUMERIC = ["age", "mileage", "tax", "mpg", "engineSize"]
CATEGORICAL = ["brand", "model", "transmission", "fuelType"]
LABEL_FEATURES = ([label_col("dmg", c) for c in DAMAGE_CLASSES]
                  + [label_col("part", c) for c in PART_CLASSES])
TARGET = "log_price"


def regression_metrics(y_true_price, y_pred_price):
    """Metrik dihitung pada skala harga asli (GBP), bukan skala log."""
    y = np.asarray(y_true_price, dtype=float)
    p = np.asarray(y_pred_price, dtype=float)
    err = p - y
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "RMSE": round(float(np.sqrt(np.mean(err ** 2))), 1),
        "MAE": round(float(np.mean(np.abs(err))), 1),
        "MAPE_pct": round(float(np.mean(np.abs(err) / y) * 100), 2),
        "R2": round(1 - ss_res / ss_tot, 4),
    }
