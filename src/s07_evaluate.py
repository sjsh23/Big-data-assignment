"""Tahap 7, evaluasi gabungan: tabel perbandingan dan gambar untuk laporan."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_ml import regression_metrics
from config import OUT

BLUE, ORANGE = "#2a78d6", "#eb6834"          # scikit-learn, Spark MLlib
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
FAMILY = {"Ridge": "Linear", "Linear Regression": "Linear", "Random Forest": "Random Forest",
          "Gradient Boosting": "Gradient Boosting", "GBT": "Gradient Boosting"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})


def load():
    sk = json.loads((OUT / "metrics_sklearn.json").read_text())
    sp = json.loads((OUT / "metrics_spark.json").read_text())
    rows = []
    for lib, d in (("scikit-learn", sk), ("Spark MLlib", sp)):
        for name, r in d.items():
            rows.append({"Library": lib, "Model": name, "Family": FAMILY.get(name, "Ablation"),
                         **r["test"], "Fit (detik)": r["fit_seconds"],
                         "Params": json.dumps(r["best_params"])})
    return pd.DataFrame(rows)


def fig_comparison(tab):
    main = tab[tab.Family != "Ablation"]
    fams = ["Linear", "Random Forest", "Gradient Boosting"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, metric, title in ((axes[0], "RMSE", "RMSE pada test set (GBP)"),
                              (axes[1], "Fit (detik)", "Waktu training final (detik)")):
        x = np.arange(len(fams))
        for i, (lib, col) in enumerate((("scikit-learn", BLUE), ("Spark MLlib", ORANGE))):
            vals = [main[(main.Library == lib) & (main.Family == f)][metric].iloc[0] for f in fams]
            bars = ax.bar(x + (i - 0.5) * 0.36, vals, 0.34, color=col, label=lib,
                          edgecolor="white", linewidth=1.5)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:,.0f}",
                        ha="center", va="bottom", fontsize=8.5, color=INK2)
        ax.set_xticks(x, fams)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_model_comparison.png", dpi=200)
    plt.close(fig)


def fig_actual_vs_pred(pred_col, label):
    p = pd.read_csv(OUT / "predictions_sklearn.csv")
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    hb = ax.hexbin(p.price, p[pred_col], gridsize=60, bins="log", cmap="Blues", mincnt=1,
                   xscale="log", yscale="log")
    lims = [p.price.min(), p.price.max()]
    ax.plot(lims, lims, color=INK2, linewidth=1, linestyle="--")
    ax.set_xlabel("Harga aktual (GBP, skala log)")
    ax.set_ylabel("Harga prediksi (GBP, skala log)")
    ax.set_title(f"Aktual vs prediksi, {label}", fontsize=11, color=INK, loc="left")
    fig.colorbar(hb, ax=ax, label="jumlah listing (log)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_actual_vs_predicted.png", dpi=200)
    plt.close(fig)


def error_by_band(pred_col):
    p = pd.read_csv(OUT / "predictions_sklearn.csv")
    bands = pd.cut(p.price, [0, 10000, 20000, 30000, 50000, 1e9],
                   labels=["< 10rb", "10rb-20rb", "20rb-30rb", "30rb-50rb", "> 50rb"])
    rows = []
    for band, g in p.groupby(bands, observed=True):
        m = regression_metrics(g.price, g[pred_col])
        rows.append({"Rentang harga (GBP)": str(band), "Jumlah": len(g), **m})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "error_by_price_band.csv", index=False)
    by_brand = []
    for brand, g in p.groupby("brand"):
        by_brand.append({"brand": brand, "Jumlah": len(g), **regression_metrics(g.price, g[pred_col])})
    pd.DataFrame(by_brand).sort_values("MAPE_pct").to_csv(OUT / "error_by_brand.csv", index=False)
    return out


def fig_importance():
    imp = pd.read_csv(OUT / "sklearn_permutation_importance.csv").head(12)[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.barh(imp.feature, imp.importance, color=BLUE, height=0.6, edgecolor="white")
    for y, v in enumerate(imp.importance):
        ax.text(v, y, f" {v:.3f}", va="center", fontsize=8.5, color=INK2)
    ax.set_xlabel("Kenaikan RMSE log-price saat kolom diacak")
    ax.set_title("Permutation importance, Gradient Boosting (scikit-learn)", fontsize=11, color=INK, loc="left")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "fig_feature_importance.png", dpi=200)
    plt.close(fig)


def main():
    tab = load()
    tab.to_csv(OUT / "model_comparison.csv", index=False)
    print(tab.drop(columns="Params").to_string(index=False))

    main_models = tab[tab.Family != "Ablation"]
    best = main_models.sort_values("RMSE").iloc[0]
    print(f"\nModel terbaik: {best.Model} ({best.Library}), RMSE {best.RMSE}, R2 {best.R2}")

    fig_comparison(tab)
    fig_actual_vs_pred("sk_Gradient Boosting", "Gradient Boosting (scikit-learn)")
    print(error_by_band("sk_Gradient Boosting").to_string(index=False))
    fig_importance()

    sk_pred = pd.read_csv(OUT / "predictions_sklearn.csv")
    sp_pred = pd.read_csv(OUT / "predictions_spark.csv")
    merged = sk_pred.merge(sp_pred, on=["car_id", "price"])
    corr = np.corrcoef(merged["sk_Gradient Boosting"], merged["spark_GBT"])[0, 1]
    print(f"Korelasi prediksi GB scikit-learn vs GBT Spark: {corr:.4f}")


if __name__ == "__main__":
    main()
