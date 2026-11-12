"""Versi web dashboard Final Project (Streamlit), meniru dashboard Power BI.

Membaca CSV hasil tahap s09 di dashboard/data, sehingga angka di sini sama dengan
Dashboard_Final_Project.pbix. Menjalankan secara lokal dari folder project:
    streamlit run dashboard/app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).resolve().parent / "data"

# Palet bawaan Power BI (urutan sama dengan warna legend di Power BI Desktop)
PBI = ["#118DFF", "#12239E", "#E66C37", "#6B007B", "#E044A7", "#744EC2", "#D9B300", "#D64550", "#197278", "#1AAB40"]
BLUE, NAVY = PBI[0], PBI[1]
INK, MUTED, GRID = "#252423", "#605E5C", "#E1DFDD"
FONT = "Segoe UI, Segoe UI Web, -apple-system, Helvetica Neue, Arial, sans-serif"

st.set_page_config(page_title="Dashboard Big Data Assignment", layout="wide", initial_sidebar_state="collapsed")

st.markdown(f"""
<style>
  html, body, [class*="css"], .stApp {{ font-family: {FONT}; }}
  .stApp {{ background: #EAEAEA; }}
  header[data-testid="stHeader"], footer, #MainMenu, [data-testid="stToolbar"] {{ display: none; }}
  .block-container {{ padding: 1.2rem 1.6rem 1rem 1.6rem; max-width: 1500px; }}
  /* Kotak visual putih seperti container visual Power BI */
  div[class*="st-key-pbi_box_"] {{
      background: #FFFFFF; border: 1px solid #E6E6E6 !important; border-radius: 4px;
      box-shadow: 0 1px 2px rgba(0,0,0,.06); padding: 10px 12px; }}
  .pbi-title {{ font-size: 14px; color: {INK}; margin: 0 0 0 2px; line-height: 1.25; }}
  .pbi-sub {{ font-size: 12px; color: {MUTED}; margin: 2px 0 4px 2px; }}
  .pbi-card {{ background: #FFFFFF; border: 1px solid #E6E6E6; border-radius: 4px;
      box-shadow: 0 1px 2px rgba(0,0,0,.06); padding: 12px 14px 12px 14px; height: 86px; }}
  .pbi-card .lbl {{ font-size: 12px; color: {MUTED}; }}
  .pbi-card .val {{ font-size: 22px; font-weight: 600; color: {INK}; margin-top: 6px; }}
  /* Slicer dropdown */
  div[data-testid="stSelectbox"] label p {{ font-size: 11px !important; color: {INK}; font-weight: 600; }}
  div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
      min-height: 30px; border-radius: 2px; border-color: #B3B0AD; background: #FFFFFF; font-size: 12px; }}
  /* Tab halaman seperti page tab Power BI */
  div[role="tablist"] {{ gap: 0 !important; background: #F3F2F1; border-bottom: 1px solid #D2D0CE; }}
  .stTabs [role="tab"] {{ padding: 8px 18px !important; margin: 0 !important; background: #F3F2F1;
      border-right: 1px solid #D2D0CE !important; border-radius: 0 !important; height: auto !important; }}
  .stTabs [role="tab"] p {{ font-size: 13px !important; color: {INK} !important; }}
  .stTabs [role="tab"][aria-selected="true"] {{ background: #FFFFFF; }}
  .stTabs [role="tab"][aria-selected="true"] p {{ font-weight: 600; }}
  div[data-baseweb="tab-highlight"], div[data-testid="stTabsHighlight"] {{ background-color: #117865 !important; }}
  /* Tabel gaya Power BI */
  table.pbi {{ border-collapse: collapse; width: 100%; font-size: 12px; color: {INK}; }}
  table.pbi th {{ text-align: left; font-weight: 400; color: {INK}; padding: 6px 8px; border-bottom: 1px solid #252423; white-space: nowrap; }}
  table.pbi td {{ padding: 4px 8px; border-bottom: 1px solid #EDEBE9; white-space: nowrap; }}
  table.pbi td.num, table.pbi th.num {{ text-align: right; }}
  table.pbi th, table.pbi td {{ border-left: none !important; border-right: none !important; border-top: none !important; }}
  table.pbi tr {{ background: transparent !important; }}
  table.heat {{ font-size: 11px; }}
  table.heat td {{ text-align: right; border-bottom: none !important; padding: 4px 5px; }}
  table.heat th {{ padding: 6px 5px; }}
  table.heat td.rowh, table.heat th.rowh {{ text-align: left; border-right: 1px solid #252423 !important; }}
  .scroll {{ overflow: auto; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data
@st.cache_data
def load():
    rd = lambda n, **k: pd.read_csv(DATA / n, **k)
    d = {
        "brand": rd("dim_brand.csv"),
        "listing": rd("dim_listing.csv", keep_default_na=False, na_values=[""]),
        "pred": rd("fact_prediction.csv"),
        "serving": rd("serving_price_by_brand_year.csv"),
        "models": rd("model_comparison.csv"),
        "fi": rd("feature_importance.csv"),
        "pairs": rd("graph_pairs.csv"),
        "nodes": rd("graph_nodes.csv"),
        "kpi": rd("kpi_summary.csv"),
    }
    d["listing"]["data_issue"] = d["listing"]["data_issue"].fillna("valid")
    d["pred_full"] = d["pred"].merge(
        d["listing"][["car_id", "brand", "price_band", "price_band_order", "age_band", "age_band_order"]], on="car_id")
    return d


D = load()


# ---------------------------------------------------------------- helpers
def k_fmt(v):
    """40000 -> 40K, gaya label sumbu Power BI."""
    if v == 0:
        return "0K"
    return f"{v/1000:,.0f}K" if abs(v) >= 1000 else f"{v:g}"


_box_n = [0]


def box(parent=None):
    """Container visual putih. Key unik dipakai CSS untuk memberi latar putih."""
    _box_n[0] += 1
    return (parent or st).container(border=True, key=f"pbi_box_{_box_n[0]}")


def card(col, label, value):
    col.markdown(f'<div class="pbi-card"><div class="lbl">{label}</div><div class="val">{value}</div></div>',
                 unsafe_allow_html=True)


def head(title, sub=None):
    st.markdown(f'<div class="pbi-title">{title}</div>' + (f'<div class="pbi-sub">{sub}</div>' if sub else ""),
                unsafe_allow_html=True)


def base_layout(fig, height, legend=False):
    fig.update_layout(
        height=height, margin=dict(l=8, r=12, t=8, b=8), paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family=FONT, size=11, color=MUTED), showlegend=legend,
        hoverlabel=dict(bgcolor="white", font=dict(family=FONT, size=12, color=INK), bordercolor="#C8C6C4"),
        legend=dict(orientation="h", yanchor="top", y=-0.12, x=0, font=dict(size=11, color=INK)),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=GRID, ticks="", tickfont=dict(color=MUTED))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, griddash="dot", zeroline=False, ticks="", tickfont=dict(color=MUTED))
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def slicer(col, label, options, key):
    return col.selectbox(label, ["All"] + list(options), key=key)


def pick(df, column, value):
    return df if value == "All" else df[df[column] == value]


def html_table(df, num_cols=(), fmt=None, height=None):
    fmt = fmt or {}
    th = "".join(f'<th class="{"num" if c in num_cols else ""}">{c}</th>' for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        tds = "".join(f'<td class="{"num" if c in num_cols else ""}">{fmt[c](r[c]) if c in fmt else r[c]}</td>'
                      for c in df.columns)
        rows.append(f"<tr>{tds}</tr>")
    style = f' style="max-height:{height}px"' if height else ""
    st.markdown(f'<div class="scroll"{style}><table class="pbi"><thead><tr>{th}</tr></thead>'
                f'<tbody>{"".join(rows)}</tbody></table></div>', unsafe_allow_html=True)


def mix(c1, c2, t):
    a = np.array([int(c1[i:i + 2], 16) for i in (1, 3, 5)])
    b = np.array([int(c2[i:i + 2], 16) for i in (1, 3, 5)])
    r = (a + (b - a) * min(max(t, 0), 1)).round().astype(int)
    return "#" + "".join(f"{x:02X}" for x in r)


def heat_color(v, lo=0.75, mid=1.0, hi=1.25, c_lo="#F4B6B6", c_mid="#FFFFFF", c_hi="#9DC3E6"):
    if pd.isna(v):
        return "#FFFFFF"
    return mix(c_mid, c_lo, (mid - v) / (mid - lo)) if v < mid else mix(c_mid, c_hi, (v - mid) / (hi - mid))


# ---------------------------------------------------------------- halaman
tabs = st.tabs(["Pasar", "Akurasi estimasi harga (insight machine learning)",
                "Faktor harga dan perbandingan model (insight machine learning)",
                "Pola kerusakan (insight graph analytics)"])

# ===== Halaman 1: Pasar
with tabs[0]:
    L = D["listing"]
    top = st.container()  # card di atas slicer, sama dengan tata letak Power BI
    c = st.columns(3)
    b = slicer(c[0], "brand", D["brand"].brand, "p1_brand")
    f = slicer(c[1], "fuelType", sorted(L.fuelType.unique()), "p1_fuel")
    t = slicer(c[2], "transmission", sorted(L.transmission.unique()), "p1_trans")

    lst = pick(pick(pick(L, "brand", b), "fuelType", f), "transmission", t)
    valid = lst[lst.status == "Valid"]
    n_brand = 9 if b == "All" else 1  # dim_brand hanya difilter oleh slicer brand
    cc = top.columns(4)
    card(cc[0], "Listing Valid", f"{len(valid):,}")
    card(cc[1], "Harga Median", f"{valid.price.median():,.0f}" if len(valid) else "(Blank)")
    card(cc[2], "Mileage Median", f"{valid.mileage.median():,.0f}" if len(valid) else "(Blank)")
    card(cc[3], "Jumlah Brand", f"{n_brand}")

    left, right = st.columns(2)
    with box(left):
        head("Harga rata-rata per umur kendaraan (tahun)", "Harga Rata Serving by age and brand")
        s = pick(D["serving"], "brand", b)
        s = s[s.jumlah >= 20]  # sama dengan filter visual jumlah >= 20 di Power BI
        fig = go.Figure()
        for i, br in enumerate(D["brand"].brand):
            g = s[s.brand == br].set_index("age").harga_rata.sort_index()  # satu baris per brand per umur
            if len(g):
                fig.add_trace(go.Scatter(x=g.index, y=g.values, name=br, mode="lines", line=dict(width=2, color=PBI[i], shape="spline"),
                                         hovertemplate=f"<b>{br}</b><br>age %{{x}}<br>Harga Rata Serving %{{y:,.0f}}<extra></extra>"))
        ticks = [0, 10000, 20000, 30000, 40000]
        fig.update_yaxes(tickvals=ticks, ticktext=[k_fmt(v) for v in ticks], range=[0, 41000])
        fig.update_xaxes(dtick=2)
        show(base_layout(fig, 470, legend=True))

    with box(right):
        head("Harga Median per brand (GBP)", "Harga Median by brand")
        m = valid.groupby("brand").price.median().sort_values()
        fig = go.Figure(go.Bar(x=m.values, y=m.index, orientation="h", marker_color=BLUE,
                               hovertemplate="%{y}<br>Harga Median %{x:,.0f}<extra></extra>"))
        ticks = [0, 5000, 10000, 15000, 20000]
        fig.update_xaxes(tickvals=ticks, ticktext=[k_fmt(v) for v in ticks], showgrid=True, gridcolor=GRID, griddash="dot")
        fig.update_yaxes(showgrid=False)
        show(base_layout(fig, 470))

# ===== Halaman 2: Akurasi estimasi harga
with tabs[1]:
    P = D["pred_full"]
    top = st.container()
    c = st.columns(2)
    b = slicer(c[0], "brand", D["brand"].brand, "p2_brand")
    ages = (D["listing"][D["listing"].age_band != "Tahun tidak valid"]
            .drop_duplicates("age_band").sort_values("age_band_order").age_band)
    a = slicer(c[1], "age_band", ages, "p2_age")
    p = pick(pick(P, "brand", b), "age_band", a)

    cc = top.columns(4)
    card(cc[0], "Jumlah Prediksi", f"{len(p):,}")
    card(cc[1], "MAE GB", f"{p.abs_error_sklearn.mean():,.1f}" if len(p) else "(Blank)")
    card(cc[2], "MAPE GB", f"{p.ape_pct_sklearn.mean():.2f}%" if len(p) else "(Blank)")
    card(cc[3], "Prediksi Dalam 10 Persen", f"{100*p.within_10pct_sklearn.mean():.1f}%" if len(p) else "(Blank)")

    with box():
        head("Error estimasi per rentang harga (MAPE)", "MAPE GB, Jumlah Prediksi and MAE GB by price_band")
        g = (p.groupby(["price_band_order", "price_band"])
             .agg(mape=("ape_pct_sklearn", "mean"), n=("car_id", "size"), mae=("abs_error_sklearn", "mean"))
             .reset_index().sort_values("price_band_order"))
        fig = go.Figure(go.Bar(x=g.price_band, y=g.mape / 100, marker_color=BLUE, width=0.45,
                               customdata=np.stack([g.n, g.mae], axis=1),
                               hovertemplate="%{x}<br>MAPE GB %{y:.2%}<br>Jumlah Prediksi %{customdata[0]:,}"
                                             "<br>MAE GB %{customdata[1]:,.1f}<extra></extra>"))
        fig.update_yaxes(tickformat=".0%", range=[0, 0.10], dtick=0.02)
        show(base_layout(fig, 300))

    left, right = st.columns([1, 2.4])
    with box(left):
        head("Harga aktual vs harga prediksi (test set)", "Sum of actual_price and Sum of pred_sklearn_gb by car_id")
        fig = go.Figure(go.Scattergl(x=p.actual_price, y=p.pred_sklearn_gb, mode="markers",
                                     marker=dict(color=BLUE, size=6), customdata=p.car_id,
                                     hovertemplate="car_id %{customdata}<br>actual_price %{x:,}<br>pred_sklearn_gb %{y:,}<extra></extra>"))
        ticks = [0, 50000, 100000, 150000]
        fig.update_xaxes(range=[0, 160000], tickvals=ticks, ticktext=[k_fmt(v) for v in ticks],
                         title=dict(text="Sum of actual_price", font=dict(size=11, color=MUTED)))
        fig.update_yaxes(range=[0, 160000], tickvals=ticks, ticktext=[k_fmt(v) for v in ticks],
                         title=dict(text="Sum of pred_sklearn_gb", font=dict(size=11, color=MUTED)))
        show(base_layout(fig, 300))
    with box(right):
        head("Error estimasi per brand (MAPE)", "MAPE GB by brand")
        m = p.groupby("brand").ape_pct_sklearn.mean().sort_values() / 100
        fig = go.Figure(go.Bar(x=m.values, y=m.index, orientation="h", marker_color=BLUE,
                               hovertemplate="%{y}<br>MAPE GB %{x:.2%}<extra></extra>"))
        fig.update_xaxes(tickformat=".0%", showgrid=True, gridcolor=GRID, griddash="dot", rangemode="tozero")
        fig.update_yaxes(showgrid=False)
        show(base_layout(fig, 300))

# ===== Halaman 3: Faktor harga dan perbandingan model
with tabs[2]:
    fi = D["fi"][D["fi"].method.str.startswith("Permutation")]
    left, right = st.columns(2)
    with left:
        with box():
            kel = fi.groupby("kelompok").importance.sum().sort_values().reset_index()
            kel.columns = ["kelompok", "Sum of importance"]
            html_table(kel, num_cols=["Sum of importance"], fmt={"Sum of importance": lambda v: f"{round(v, 3) + 0.0:.3f}"})
        with box():
            head("Faktor yang paling memengaruhi estimasi harga (permutation importance)", "Sum of importance by feature")
            a = fi[fi.kelompok == "Atribut kendaraan"].sort_values("importance")
            fig = go.Figure(go.Bar(x=a.importance, y=a.feature, orientation="h", marker_color=BLUE,
                                   hovertemplate="%{y}<br>Sum of importance %{x:.3f}<extra></extra>"))
            fig.update_xaxes(tickformat=".2f", showgrid=True, gridcolor=GRID, griddash="dot", rangemode="tozero")
            fig.update_yaxes(showgrid=False)
            show(base_layout(fig, 520))
    with right:
        with box():
            mt = D["models"].sort_values("RMSE")[["model_label", "RMSE", "MAPE_pct", "R2", "fit_seconds"]]
            mt.columns = ["model_label", "RMSE (GBP)", "MAPE (%)", "R2", "Waktu training (detik)"]
            html_table(mt, num_cols=mt.columns[1:],
                       fmt={"RMSE (GBP)": lambda v: f"{v:,.2f}", "MAPE (%)": lambda v: f"{v:.2f}",
                            "R2": lambda v: f"{v:.4f}", "Waktu training (detik)": lambda v: f"{v:.1f}"})
        with box():
            head("RMSE scikit-learn vs Spark MLlib per keluarga algoritma", "Sum of RMSE by Family and Library")
            mc = D["models"][D["models"].Family != "Ablation"]
            fams = ["Linear", "Random Forest", "Gradient Boosting"]
            fig = go.Figure()
            for lib, col in (("scikit-learn", BLUE), ("Spark MLlib", NAVY)):
                v = mc[mc.Library == lib].set_index("Family").reindex(fams).RMSE
                fig.add_trace(go.Bar(x=fams, y=v.values, name=lib, marker_color=col,
                                     hovertemplate=f"{lib}<br>%{{x}}<br>Sum of RMSE %{{y:,.1f}}<extra></extra>"))
            fig.update_layout(barmode="group", bargap=0.45, bargroupgap=0.05)
            fig.update_yaxes(tickformat=",", rangemode="tozero")
            show(base_layout(fig, 455, legend=True))

# ===== Halaman 4: Pola kerusakan
with tabs[3]:
    pr = D["pairs"]
    with box():
        dp = pr[pr.pair_type == "Damage-Part"].pivot_table(index="from_label", columns="to_label", values="lift")
        th = '<th class="rowh">from_label</th>' + "".join(f'<th class="num">{c}</th>' for c in dp.columns)
        body = ""
        for idx, row in dp.iterrows():
            # +1e-9 supaya 1.125 dibulatkan ke 1.13 seperti Power BI (bukan 1.12)
            tds = "".join(f'<td style="background:{heat_color(v)}">{v + 1e-9:.2f}</td>' for v in row.values)
            body += f'<tr><td class="rowh">{idx}</td>{tds}</tr>'
        st.markdown(f'<div class="scroll"><table class="pbi heat"><thead><tr>{th}</tr></thead>'
                    f'<tbody>{body}</tbody></table></div>', unsafe_allow_html=True)

    left, mid, right = st.columns([2.1, 1.1, 1.7])
    with box(left):
        head("Lift", "by pair_label")
        top = pr[pr.in_assoc_graph == 1].nlargest(10, "lift").sort_values("lift")
        fig = go.Figure(go.Bar(x=top.lift, y=top.pair_label, orientation="h", marker_color=BLUE,
                               customdata=top.images,
                               hovertemplate="%{y}<br>Lift %{x:.3f}<br>Jumlah gambar %{customdata}<extra></extra>"))
        fig.update_xaxes(tickformat=".1f", showgrid=True, gridcolor=GRID, griddash="dot", rangemode="tozero")
        fig.update_yaxes(showgrid=False)
        show(base_layout(fig, 300))
    kpi = D["kpi"].set_index("metric").value
    with mid:
        card(mid, "Modularity graph asosiasi", f"{kpi['Graph asosiasi: modularity Louvain']:.3f}")
        st.write("")
        card(mid, "Modularity graph frekuensi", f"{kpi['Graph frekuensi: modularity Louvain']:.3f}")
    with box(right):
        n = D["nodes"].copy()
        n["urut"] = n.community.fillna(99)
        n = n.sort_values(["urut", "name"])[["community_label", "name", "jumlah_gambar", "type"]]
        n.columns = ["community_label", "name", "Jumlah gambar", "type"]
        html_table(n, num_cols=["Jumlah gambar"], height=300)
