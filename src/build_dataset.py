"""Merge the UK used-car listings with the Car Parts & Car Damages image pool.

Every car row gets exactly one randomly assigned image (balanced, with
replacement). The pairing is synthetic: the image carries no information about
the car. See README.md.

Run from the project root:
    python src/build_dataset.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
CAR_DIR = ROOT / "Uk selling cars"
IMG_ROOT = ROOT / "car damages images"  # image_path values are relative to this
OUT = ROOT / "data" / "processed"

BRAND_FILES = {
    "audi.csv": "Audi",
    "bmw.csv": "BMW",
    "ford.csv": "Ford",
    "hyundi.csv": "Hyundai",
    "merc.csv": "Mercedes",
    "skoda.csv": "Skoda",
    "toyota.csv": "Toyota",
    "vauxhall.csv": "Vauxhall",
    "vw.csv": "Volkswagen",
}
# Model subsets of merc/ford and their raw versions -- excluded to avoid double counting.
EXCLUDED_FILES = {"cclass.csv", "focus.csv", "unclean cclass.csv", "unclean focus.csv"}
CAR_COLS = ["model", "year", "price", "transmission", "mileage",
            "fuelType", "tax", "mpg", "engineSize"]

DROP_EXACT_DUPLICATES = True
COLLECTION_YEAR = 2020  # listings were scraped in 2020
IMG_EXTS = {".jpg", ".jpeg", ".png"}
DAMAGE_CLASSES = {"Missing part", "Broken part", "Scratch", "Cracked",
                  "Dent", "Flaking", "Paint chip", "Corrosion"}


# --------------------------------------------------------------------------- #
# Step 2: combine brand CSVs
# --------------------------------------------------------------------------- #
def load_cars() -> tuple[pd.DataFrame, dict]:
    found = {p.name for p in CAR_DIR.glob("*.csv")}
    unknown = found - set(BRAND_FILES) - EXCLUDED_FILES
    assert not unknown, f"Unexpected CSVs in {CAR_DIR}: {sorted(unknown)}"
    missing = set(BRAND_FILES) - found
    assert not missing, f"Missing brand CSVs: {sorted(missing)}"

    frames, rows_per_file = [], {}
    for fname, brand in BRAND_FILES.items():
        df = pd.read_csv(CAR_DIR / fname)
        df.columns = df.columns.str.strip()
        df = df.rename(columns={"tax(£)": "tax"})  # harmonise Hyundai
        assert list(df.columns) == CAR_COLS, f"{fname} columns: {list(df.columns)}"
        df["model"] = df["model"].str.strip()      # model values have leading spaces
        df.insert(0, "brand", brand)               # brand only exists in the filename
        rows_per_file[fname] = len(df)
        frames.append(df)

    cars = pd.concat(frames, ignore_index=True)
    info = {"rows_per_file": rows_per_file, "rows_raw": len(cars),
            "excluded_files": sorted(EXCLUDED_FILES & found)}

    if DROP_EXACT_DUPLICATES:
        before = len(cars)
        cars = cars.drop_duplicates(ignore_index=True)
        info["exact_duplicates_dropped"] = before - len(cars)
        info["rows_after_dedup_per_brand"] = cars["brand"].value_counts().sort_index().to_dict()
    else:
        info["exact_duplicates_dropped"] = 0

    # Flag (not drop) suspicious values so the cleaning decision stays traceable.
    issues = pd.Series("", index=cars.index)
    issues[cars["year"] > COLLECTION_YEAR] += "year_after_collection;"
    issues[(cars["engineSize"] == 0) & (cars["fuelType"] != "Electric")] += "zero_engine_size;"
    cars["data_issue"] = issues.str.rstrip(";")
    info["flagged_rows"] = {
        "year_after_collection": int((cars["year"] > COLLECTION_YEAR).sum()),
        "zero_engine_size": int(((cars["engineSize"] == 0) & (cars["fuelType"] != "Electric")).sum()),
    }
    return cars, info


# --------------------------------------------------------------------------- #
# Step 3: shuffle
# --------------------------------------------------------------------------- #
def shuffle_cars(cars: pd.DataFrame, seed: int) -> pd.DataFrame:
    cars = cars.sample(frac=1, random_state=seed).reset_index(drop=True)
    cars.insert(0, "car_id", range(len(cars)))
    return cars


# --------------------------------------------------------------------------- #
# Step 4: image index
# --------------------------------------------------------------------------- #
def annotation_kind(subset_dir: Path) -> str:
    """Decide from meta.json whether a subset annotates damages or parts.

    The folder names in the Kaggle download are swapped relative to their
    annotations, so the folder name cannot be trusted.
    """
    classes = {c["title"] for c in json.loads((subset_dir / "meta.json").read_text("utf-8"))["classes"]}
    return "damage" if classes <= DAMAGE_CLASSES else "part"


def load_images() -> tuple[pd.DataFrame, dict]:
    subsets = sorted(p for p in IMG_ROOT.iterdir() if (p / "meta.json").exists())
    assert subsets, f"No annotated subsets under {IMG_ROOT}"

    records = []
    for subset in subsets:
        kind = annotation_kind(subset)
        # Only img/ folders: masks_human / masks_machine hold PNG masks, not photos.
        for img in sorted(subset.glob("*/img/*")):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            ann = img.parent.parent / "ann" / f"{img.name}.json"
            objects = json.loads(ann.read_text("utf-8"))["objects"] if ann.exists() else []
            records.append({
                "file": img,
                "sha1": hashlib.sha1(img.read_bytes()).hexdigest(),
                "kind": kind,
                "labels": sorted({o["classTitle"] for o in objects}),
                "has_ann": ann.exists(),
            })
    files = pd.DataFrame(records)

    # Unreadable images would break any downstream loader, so fail early.
    corrupt = []
    for f in files["file"]:
        try:
            with Image.open(f) as im:
                im.verify()
        except Exception:
            corrupt.append(f.relative_to(IMG_ROOT).as_posix())
    assert not corrupt, f"Corrupt images: {corrupt[:10]}"

    # The same photo appears in both subsets (once per annotation type).
    # Collapse to one row per unique photo so reuse stays balanced and a
    # group split cannot leak one photo through two different paths.
    images = []
    for sha, g in files.groupby("sha1", sort=False):
        g = g.sort_values("file")
        damage = sorted({l for ls in g.loc[g["kind"] == "damage", "labels"] for l in ls})
        part = sorted({l for ls in g.loc[g["kind"] == "part", "labels"] for l in ls})
        kinds = set(g["kind"])
        images.append({
            "image_id": sha[:12],
            "image_path": g["file"].iloc[0].relative_to(IMG_ROOT).as_posix(),
            "image_source": "both" if len(kinds) == 2 else kinds.pop(),
            "damage_labels": "|".join(damage),
            "part_labels": "|".join(part),
        })
    images = pd.DataFrame(images).sort_values("image_path", ignore_index=True)
    assert images["image_id"].is_unique, "image_id prefix collision; lengthen the hash prefix"

    info = {
        "image_files_per_subset": {s.name: int((files["file"].map(lambda f: s in f.parents)).sum())
                                   for s in subsets},
        "annotation_kind_per_subset": {s.name: annotation_kind(s) for s in subsets},
        "image_files_total": len(files),
        "images_missing_annotation": int((~files["has_ann"]).sum()),
        "unique_images": len(images),
        "image_source_counts": images["image_source"].value_counts().to_dict(),
    }
    return images, info


# --------------------------------------------------------------------------- #
# Step 5: balanced random assignment (Option B)
# --------------------------------------------------------------------------- #
def assign_images(cars: pd.DataFrame, images: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n, m = len(cars), len(images)
    idx = np.concatenate([
        np.tile(np.arange(m), n // m),             # every image, full rounds
        rng.choice(m, size=n % m, replace=False),  # remainder, no repeats
    ])
    rng.shuffle(idx)
    return cars.join(images.iloc[idx].reset_index(drop=True))


def build(seed: int) -> tuple[pd.DataFrame, dict]:
    cars, car_info = load_cars()
    images, img_info = load_images()
    table = assign_images(shuffle_cars(cars, seed), images, seed)
    return table, {"cars": car_info, "images": img_info, "_images": images}


# --------------------------------------------------------------------------- #
# Step 7: validation
# --------------------------------------------------------------------------- #
def validate(table: pd.DataFrame, meta: dict, images: pd.DataFrame) -> dict:
    car_info = meta["cars"]
    n, m = len(table), len(images)

    expected_rows = sum(car_info["rows_per_file"].values()) - car_info["exact_duplicates_dropped"]
    assert n == expected_rows, f"row count {n} != {expected_rows}"

    expected_brands = car_info.get("rows_after_dedup_per_brand") or {
        BRAND_FILES[f]: c for f, c in car_info["rows_per_file"].items()}
    assert table["brand"].value_counts().sort_index().to_dict() == expected_brands

    assert table["image_path"].notna().all(), "null image_path"
    missing = [p for p in images["image_path"] if not (IMG_ROOT / p).is_file()]
    assert not missing, f"missing image files: {missing[:5]}"
    assert table["car_id"].is_unique and table["car_id"].tolist() == list(range(n))

    reuse = table["image_id"].value_counts()
    assert len(reuse) == m, "some images were never assigned"
    assert set(reuse.unique()) <= {n // m, -(-n // m)}, f"unbalanced reuse: {reuse.min()}..{reuse.max()}"

    # Brand runs: a random order of these brand proportions gives
    # about n * (1 - sum p^2) runs; sorted-by-brand data would give 9.
    runs = int((table["brand"] != table["brand"].shift()).sum())
    p = table["brand"].value_counts(normalize=True)
    expected_runs = n * (1 - (p ** 2).sum())
    assert runs > 0.95 * expected_runs, f"only {runs} brand runs (expected ~{expected_runs:.0f})"

    return {"rows": n, "unique_images": m, "reuse_min": int(reuse.min()),
            "reuse_max": int(reuse.max()), "brand_runs": runs,
            "brand_runs_expected_if_random": round(expected_runs)}


# --------------------------------------------------------------------------- #
# Step 6: save
# --------------------------------------------------------------------------- #
OUTPUT_COLS = ["car_id", "brand", *CAR_COLS, "data_issue",
               "image_id", "image_path", "image_source", "damage_labels", "part_labels"]


def main() -> None:
    table, meta = build(SEED)
    images = meta.pop("_images")
    checks = validate(table, meta, images)

    # Same seed must give an identical table.
    again, _ = build(SEED)
    assert table[OUTPUT_COLS].equals(again[OUTPUT_COLS]), "build is not reproducible"

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "used_cars_with_images.csv"
    table[OUTPUT_COLS].to_csv(csv_path, index=False, lineterminator="\n")
    table[OUTPUT_COLS].to_parquet(OUT / "used_cars_with_images.parquet", index=False)

    meta.update({
        "seed": SEED,
        "img_root": IMG_ROOT.relative_to(ROOT).as_posix(),
        "car_dir": CAR_DIR.relative_to(ROOT).as_posix(),
        "assignment": "balanced random with replacement (Option B)",
        "validation": checks,
        "output_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    })
    (OUT / "build_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps(meta, indent=2))
    print(f"\nWrote {csv_path.relative_to(ROOT)} ({len(table):,} rows)")


if __name__ == "__main__":
    main()
