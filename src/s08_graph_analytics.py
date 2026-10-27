"""Tahap 8, graph analytics.

1. Property graph dari Curated Zone (GraphFrames di atas Spark):
   vertex  : Brand, Model, Listing, Image, Damage, Part
   edge    : Listing-OF_MODEL->Model, Model-BY_BRAND->Brand, Listing-HAS_IMAGE->Image,
             Image-SHOWS_DAMAGE->Damage, Image-SHOWS_PART->Part
2. Motif finding untuk menurunkan graph co-occurrence kerusakan dan part
   (berapa gambar yang memuat kerusakan d sekaligus part p).
3. Analitik pada graph co-occurrence: degree, PageRank, Label Propagation (GraphFrames),
   lalu weighted PageRank, Louvain, dan lift (NetworkX) serta visualisasi.
Bila GraphFrames tidak tersedia, langkah 2 dihitung dengan join Spark SQL yang setara.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

from config import CURATED, OUT, SEED, get_spark, read_table, uri

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
MIN_SUPPORT = 20  # minimal jumlah gambar agar sebuah pasangan dianggap
MIN_LIFT = 1.05   # lift di bawah ini dianggap praktis independen


def build_property_graph(spark):
    cur = read_table(spark, CURATED / "listings").filter(F.col("data_issue").isNull())
    cur = cur.withColumn("model_id", F.concat(F.lit("model:"), F.col("brand"), F.lit("|"), F.col("model")))
    imgs = cur.select("image_id", "damage_labels", "part_labels").dropDuplicates(["image_id"])

    dmg = (imgs.filter(F.col("damage_labels").isNotNull())
           .select("image_id", F.explode(F.split("damage_labels", r"\|")).alias("label")))
    prt = (imgs.filter(F.col("part_labels").isNotNull())
           .select("image_id", F.explode(F.split("part_labels", r"\|")).alias("label")))

    v = (cur.select(F.concat(F.lit("brand:"), "brand").alias("id"), F.lit("Brand").alias("type"), F.col("brand").alias("name"))
         .union(cur.select(F.col("model_id").alias("id"), F.lit("Model"), F.col("model")))
         .union(cur.select(F.concat(F.lit("listing:"), F.col("car_id").cast("string")), F.lit("Listing"), F.col("car_id").cast("string")))
         .union(imgs.select(F.concat(F.lit("image:"), "image_id"), F.lit("Image"), F.col("image_id")))
         .union(dmg.select(F.concat(F.lit("damage:"), "label"), F.lit("Damage"), F.col("label")))
         .union(prt.select(F.concat(F.lit("part:"), "label"), F.lit("Part"), F.col("label")))
         .dropDuplicates(["id"]))
    e = (cur.select(F.concat(F.lit("listing:"), F.col("car_id").cast("string")).alias("src"), F.col("model_id").alias("dst"), F.lit("OF_MODEL").alias("relationship"))
         .union(cur.select("model_id", F.concat(F.lit("brand:"), "brand"), F.lit("BY_BRAND")).dropDuplicates())
         .union(cur.select(F.concat(F.lit("listing:"), F.col("car_id").cast("string")), F.concat(F.lit("image:"), "image_id"), F.lit("HAS_IMAGE")))
         .union(dmg.select(F.concat(F.lit("image:"), "image_id"), F.concat(F.lit("damage:"), "label"), F.lit("SHOWS_DAMAGE")))
         .union(prt.select(F.concat(F.lit("image:"), "image_id"), F.concat(F.lit("part:"), "label"), F.lit("SHOWS_PART"))))
    return v.cache(), e.cache()


def cooccurrence_graphframes(g):
    pairs = (g.find("(i)-[e1]->(d); (i)-[e2]->(p)")
             .filter("e1.relationship = 'SHOWS_DAMAGE' AND e2.relationship = 'SHOWS_PART'")
             .groupBy(F.col("d.name").alias("damage"), F.col("p.name").alias("part"))
             .agg(F.countDistinct("i.id").alias("images")))
    dd = (g.find("(i)-[e1]->(a); (i)-[e2]->(b)")
          .filter("e1.relationship = 'SHOWS_DAMAGE' AND e2.relationship = 'SHOWS_DAMAGE' AND a.id < b.id")
          .groupBy(F.col("a.name").alias("damage_a"), F.col("b.name").alias("damage_b"))
          .agg(F.countDistinct("i.id").alias("images")))
    return pairs, dd


def cooccurrence_sql(e):
    sd = e.filter("relationship = 'SHOWS_DAMAGE'").select(F.col("src").alias("i"), F.regexp_replace("dst", "^damage:", "").alias("damage"))
    sp = e.filter("relationship = 'SHOWS_PART'").select(F.col("src").alias("i"), F.regexp_replace("dst", "^part:", "").alias("part"))
    pairs = sd.join(sp, "i").groupBy("damage", "part").agg(F.countDistinct("i").alias("images"))
    a = sd.withColumnRenamed("damage", "damage_a")
    b = sd.withColumnRenamed("damage", "damage_b")
    dd = (a.join(b, "i").filter("damage_a < damage_b")
          .groupBy("damage_a", "damage_b").agg(F.countDistinct("i").alias("images")))
    return pairs, dd


def main():
    try:
        spark = get_spark("graph_analytics", with_graphframes=True)
    except Exception as ex:  # jar GraphFrames gagal diunduh: lanjut tanpa GraphFrames
        print(f"Jar GraphFrames gagal dimuat ({type(ex).__name__}), Spark dijalankan tanpa GraphFrames")
        spark = get_spark("graph_analytics", with_graphframes=False)
    spark.sparkContext.setCheckpointDir(uri(OUT / "checkpoints"))
    v, e = build_property_graph(spark)

    stats = {"vertices": v.groupBy("type").count().toPandas().set_index("type")["count"].to_dict(),
             "edges": e.groupBy("relationship").count().toPandas().set_index("relationship")["count"].to_dict()}
    try:
        from graphframes import GraphFrame
        g = GraphFrame(v, e)
        pairs, dd = cooccurrence_graphframes(g)
        engine = "GraphFrames"
        indeg = (g.inDegrees.join(v, "id").filter("type in ('Damage', 'Part')")
                 .select("type", "name", F.col("inDegree").alias("images")).toPandas())
    except Exception as ex:  # fallback bila jar GraphFrames tidak tersedia
        print(f"GraphFrames tidak tersedia ({type(ex).__name__}), memakai join Spark SQL")
        g, engine = None, "Spark SQL join"
        pairs, dd = cooccurrence_sql(e)
        indeg = (e.groupBy("dst").agg(F.count("*").alias("images")).join(v, v.id == F.col("dst"))
                 .filter("type in ('Damage', 'Part')").select("type", "name", "images").toPandas())
    stats["motif_engine"] = engine

    pairs_pd = pairs.toPandas()
    dd_pd = dd.toPandas()
    sd = e.filter("relationship = 'SHOWS_DAMAGE'").select(F.col("src").alias("i"),
                                                         F.regexp_replace("dst", "^damage:", "").alias("name"))
    sp = e.filter("relationship = 'SHOWS_PART'").select(F.col("src").alias("i"),
                                                       F.regexp_replace("dst", "^part:", "").alias("name"))
    both = sd.select("i").distinct().join(sp.select("i").distinct(), "i")
    n_both = both.count()
    n_dmg_img = sd.select("i").distinct().count()
    stats["images_with_damage_and_part"] = n_both
    stats["images_with_damage"] = n_dmg_img

    # Lift = P(a dan b) / (P(a) P(b)). Untuk pasangan damage-part, peluang dihitung di dalam
    # gambar yang punya kedua jenis label; untuk damage-damage, di dalam gambar berlabel kerusakan.
    pd_both = sd.join(both, "i").groupBy("name").count().toPandas().set_index("name")["count"]
    pp_both = sp.join(both, "i").groupBy("name").count().toPandas().set_index("name")["count"]
    pd_all = sd.groupBy("name").count().toPandas().set_index("name")["count"]
    pairs_pd["lift"] = [(r.images / n_both) / ((pd_both[r.damage] / n_both) * (pp_both[r.part] / n_both))
                        for r in pairs_pd.itertuples()]
    dd_pd["lift"] = [(r.images / n_dmg_img) / ((pd_all[r.damage_a] / n_dmg_img) * (pd_all[r.damage_b] / n_dmg_img))
                     for r in dd_pd.itertuples()]
    pairs_pd = pairs_pd.sort_values("images", ascending=False)
    dd_pd = dd_pd.sort_values("images", ascending=False)
    pairs_pd.to_csv(OUT / "graph_damage_part_pairs.csv", index=False)
    dd_pd.to_csv(OUT / "graph_damage_damage_pairs.csv", index=False)
    indeg.sort_values(["type", "images"], ascending=[True, False]).to_csv(OUT / "graph_label_degree.csv", index=False)

    # (a) Graph frekuensi: bobot = jumlah gambar. Menunjukkan simpul paling sentral.
    G = nx.Graph()
    for r in pairs_pd.itertuples():
        G.add_edge(f"D:{r.damage}", f"P:{r.part}", weight=r.images)
    for r in dd_pd.itertuples():
        G.add_edge(f"D:{r.damage_a}", f"D:{r.damage_b}", weight=r.images)
    pr_w = nx.pagerank(G, weight="weight", alpha=0.85)
    strength = dict(G.degree(weight="weight"))
    comms_f = nx.community.louvain_communities(G, weight="weight", seed=SEED)
    mod_f = nx.community.modularity(G, comms_f, weight="weight")

    # (b) Graph asosiasi: hanya pasangan dengan dukungan >= MIN_SUPPORT gambar dan lift >= MIN_LIFT,
    #     bobot = lift - 1. Menunjukkan pasangan yang muncul bersama lebih sering dari kebetulan.
    A = nx.Graph()
    A.add_nodes_from(G.nodes)
    for r in pairs_pd.itertuples():
        if r.images >= MIN_SUPPORT and r.lift >= MIN_LIFT:
            A.add_edge(f"D:{r.damage}", f"P:{r.part}", weight=r.lift - 1, lift=r.lift)
    for r in dd_pd.itertuples():
        if r.images >= MIN_SUPPORT and r.lift >= MIN_LIFT:
            A.add_edge(f"D:{r.damage_a}", f"D:{r.damage_b}", weight=r.lift - 1, lift=r.lift)
    A_core = A.subgraph([n for n in A.nodes if A.degree(n) > 0]).copy()
    comms = nx.community.louvain_communities(A_core, weight="weight", seed=SEED)
    comms = sorted(comms, key=lambda c: (-len(c), sorted(c)))
    comm_of = {n: i for i, c in enumerate(comms) for n in c}
    mod_a = nx.community.modularity(A_core, comms, weight="weight")

    nodes = pd.DataFrame({"node": list(G.nodes)})
    nodes["type"] = nodes.node.str[0].map({"D": "Damage", "P": "Part"})
    nodes["name"] = nodes.node.str[2:]
    nodes["strength"] = nodes.node.map(strength)
    nodes["pagerank_weighted"] = nodes.node.map(pr_w).round(4)
    nodes["assoc_degree"] = nodes.node.map(dict(A.degree))
    nodes["louvain_community"] = nodes.node.map(comm_of)

    if g is not None:  # PageRank (graph frekuensi) dan Label Propagation (graph asosiasi) di GraphFrames
        from graphframes import GraphFrame
        vv = spark.createDataFrame(nodes[["node"]].rename(columns={"node": "id"}))
        e_f = pd.DataFrame([(a, b) for a, b in G.edges] + [(b, a) for a, b in G.edges], columns=["src", "dst"])
        pr = GraphFrame(vv, spark.createDataFrame(e_f)).pageRank(resetProbability=0.15, maxIter=20).vertices.toPandas()
        e_a = pd.DataFrame([(a, b) for a, b in A.edges] + [(b, a) for a, b in A.edges], columns=["src", "dst"])
        lpa = GraphFrame(vv, spark.createDataFrame(e_a)).labelPropagation(maxIter=10).toPandas()
        nodes = nodes.merge(pr.rename(columns={"id": "node", "pagerank": "pagerank_graphframes"}), on="node", how="left")
        nodes = nodes.merge(lpa.rename(columns={"id": "node", "label": "lpa_label"}), on="node", how="left")
        stats["lpa_communities"] = int(nodes[nodes.assoc_degree > 0].lpa_label.nunique())  # tanpa simpul terisolasi

    nodes = nodes.sort_values("pagerank_weighted", ascending=False)
    nodes.to_csv(OUT / "graph_nodes_metrics.csv", index=False)
    stats.update({
        "freq_graph": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
                       "density": round(nx.density(G), 3), "louvain_communities": len(comms_f),
                       "louvain_modularity": round(mod_f, 3)},
        "assoc_graph": {"min_support": MIN_SUPPORT, "min_lift": MIN_LIFT, "edges": A.number_of_edges(),
                        "isolated_nodes": sorted(n[2:] for n in A.nodes if A.degree(n) == 0),
                        "louvain_communities": len(comms), "louvain_modularity": round(mod_a, 3),
                        "communities": {i: sorted(n[2:] for n in c) for i, c in enumerate(comms)}},
    })
    (OUT / "graph_summary.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(nodes.to_string(index=False))
    print(pairs_pd[pairs_pd.images >= MIN_SUPPORT].sort_values("lift", ascending=False).head(10).to_string(index=False))
    print(dd_pd.sort_values("lift", ascending=False).to_string(index=False))

    draw_network(A, comm_of, strength)
    draw_lift_heatmap(pairs_pd)
    spark.stop()


def draw_network(A, comm_of, strength):
    pos = nx.spring_layout(A, weight="weight", seed=SEED, k=0.9, iterations=200)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    widths = [1 + 6 * A[u][v]["weight"] for u, v in A.edges]
    nx.draw_networkx_edges(A, pos, ax=ax, width=widths, edge_color="#b9b8b0", alpha=0.9)
    smax = max(strength.values())
    for shape, prefix in (("s", "D:"), ("o", "P:")):
        ns = [n for n in A.nodes if n.startswith(prefix)]
        nx.draw_networkx_nodes(A, pos, nodelist=ns, ax=ax, node_shape=shape,
                               node_size=[120 + 900 * strength[n] / smax for n in ns],
                               node_color=[CAT[comm_of[n] % len(CAT)] if n in comm_of else "#c3c2b7" for n in ns],
                               edgecolors="white", linewidths=1.5)
    nx.draw_networkx_labels(A, pos, labels={n: n[2:] for n in A.nodes}, font_size=7.5, font_color=INK, ax=ax)
    ax.set_title(f"Graph asosiasi (lift >= {MIN_LIFT}, dukungan >= {MIN_SUPPORT} gambar). Kotak = kerusakan, lingkaran = part,\n"
                 "warna = komunitas Louvain, abu-abu = tanpa asosiasi, ukuran = frekuensi, tebal garis = lift",
                 fontsize=9.5, color=INK, loc="left")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "fig_graph_network.png", dpi=200)
    plt.close(fig)


def draw_lift_heatmap(pairs_pd):
    m = pairs_pd.pivot(index="damage", columns="part", values="lift")
    sup = pairs_pd.pivot(index="damage", columns="part", values="images")
    m = m.where(sup >= 20)
    order_d = sup.sum(axis=1).sort_values(ascending=False).index
    order_p = sup.sum(axis=0).sort_values(ascending=False).index
    m = m.loc[order_d, order_p]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    im = ax.imshow(m.values, cmap="RdBu_r", vmin=0.4, vmax=1.6, aspect="auto")
    ax.set_xticks(range(len(m.columns)), m.columns, rotation=55, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(m.index)), m.index, fontsize=9)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            val = m.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(val - 1) > 0.4 else INK)
    fig.colorbar(im, ax=ax, label="lift (1 = independen)")
    ax.set_title("Lift kerusakan terhadap part (sel kosong: dukungan < 20 gambar)", fontsize=10.5,
                 color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_graph_lift_heatmap.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
