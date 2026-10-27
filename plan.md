# Plan: Merge Used Car Data with Car Parts & Damages Images

## Goal

Produce one shuffled table containing every used car listing from all brands, where each row has a brand column and exactly one image randomly assigned from the Car Parts and Car Damages dataset.

**Sources**

| # | Dataset | Type | Role |
|---|---------|------|------|
| A | `humansintheloop/car-parts-and-car-damages` | Images + annotation JSON | Image pool |
| B | `adityadesai13/used-car-dataset-ford-and-mercedes` | One CSV per brand | Base table (one row per car) |

**Output**

`data/processed/used_cars_with_images.csv` (plus a `.parquet` copy). Each row holds the car attributes, a `brand` column, and an `image_path` pointing to an image file. Images stay as files on disk and are referenced by relative path. They are not embedded in the table.

---

## 0. Important caveat: what random pairing means

The two datasets have no shared key, so the join is **synthetic**. A randomly assigned image carries **no information** about the car in that row. The price, mileage, and year are not related to the dent or bumper in the picture, and the image may not even show the same make or model.

This matters in three ways:

1. **As a pipeline or demo exercise** (building a multimodal data loader, a dashboard, or a database schema), this approach is fine.
2. **For prediction** (for example, predicting price from tabular data plus image), image features are pure noise by construction. Expect no gain from adding them. Adding a control experiment (tabular-only vs. tabular+image) will show this, and it is worth stating in any write-up.
3. **Image reuse.** The image pool (on the order of ~2k images) is much smaller than the car table (~100k rows), so each image appears about 50 times. If you ever split into train and test sets, split **by image** (see Step 7). Otherwise the same image leaks across splits.

If a meaningful link is needed later, it has to come from a real relationship. The random assignment cannot supply one.

---

## 1. Setup and inventory (verify before coding)

```
project/
├── data/
│   ├── raw/
│   │   ├── used_cars/            # dataset B, unzipped
│   │   └── car_parts_damages/    # dataset A, unzipped
│   └── processed/
├── src/
│   └── build_dataset.py
└── plan.md
```

Download both datasets, either manually or with `kaggle datasets download -d <slug> --unzip`. Then **inspect the actual contents** before writing merge code, because the logic depends on them.

**Dataset B checks**

- [ ] List all CSV files. Expected: one file per brand (`audi`, `bmw`, `ford`, `hyundi`, `merc`, `skoda`, `toyota`, `vauxhall`, `vw`), plus some extras.
- [ ] Identify **non-brand / duplicate files** and exclude them:
  - `cclass.csv` and `focus.csv` are model subsets that overlap with `merc.csv` and `ford.csv`. They also have fewer columns (no `tax` or `mpg`).
  - Any file named `unclean*` is a raw, messy version.
- [ ] Compare column headers across files. `hyundi.csv` is known to use `tax(£)` instead of `tax`.
- [ ] Record the row count per file. You will need these counts for validation.

**Dataset A checks**

- [ ] Map the folder tree: which folder holds the damage images, which holds the part images, and where the annotations live.
- [ ] Identify the annotation format. It is likely Supervisely-style (`ann/*.json` + `meta.json`) or COCO (`annotations.json`); confirm which.
- [ ] Count images per subset (damages vs. parts).
- [ ] Check for corrupt or unreadable images (open each with PIL once).

---

## 2. Combine all brand CSVs into one table

```python
import pandas as pd
from pathlib import Path

SEED = 42
CAR_DIR = Path("data/raw/used_cars")

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

frames = []
for fname, brand in BRAND_FILES.items():
    df = pd.read_csv(CAR_DIR / fname)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"tax(£)": "tax"})         # harmonise Hyundai
    df["model"] = df["model"].str.strip()             # model values have leading spaces
    df.insert(0, "brand", brand)                      # brand only exists in the filename
    frames.append(df)

cars = pd.concat(frames, ignore_index=True)
```

**Decisions to record**

- **Brand column.** Brand is derived from the filename, because the CSVs have no brand column. Without this step, the brand information is lost after concatenation.
- **Column alignment.** All nine brand files should share the same columns: `model, year, price, transmission, mileage, fuelType, tax, mpg, engineSize`. Assert this rather than letting `concat` silently create NaN columns.
- **Exact duplicate rows.** Some exist. Decide whether to drop them with `cars.drop_duplicates()` and log how many were removed. Two identical listings are plausible in real data, but they also inflate reuse.
- **Obvious bad values.** Check for things like a `year` beyond the collection date (a known outlier exists) and `engineSize == 0` on non-electric cars. Flag or drop them, but keep this step separate from the merge so it stays traceable.

---

## 3. Shuffle so rows are no longer grouped by brand

```python
cars = cars.sample(frac=1, random_state=SEED).reset_index(drop=True)
cars.insert(0, "car_id", range(len(cars)))
```

`sample(frac=1)` returns every row exactly once in a random order. A fixed `random_state` makes the order reproducible. `car_id` is assigned **after** shuffling so it serves as a stable row key for the final dataset.

---

## 4. Build the image index

```python
IMG_DIR = Path("data/raw/car_parts_damages")
EXTS = {".jpg", ".jpeg", ".png"}

img_paths = sorted(p for p in IMG_DIR.rglob("*") if p.suffix.lower() in EXTS)
images = pd.DataFrame({
    "image_path": [p.relative_to(IMG_DIR).as_posix() for p in img_paths]
})

# Adjust this rule to the real folder names found in Step 1
images["image_source"] = images["image_path"].str.lower().map(
    lambda s: "damage" if "damage" in s else "part"
)
images = images.reset_index(drop=True)
```

`sorted()` matters here. `rglob` order depends on the filesystem, so the same seed could otherwise give different assignments on different machines.

**Optional: attach annotation labels.** For each image, parse its annotation file and store the list of class names, for example `["Dent", "Scratch"]` or `["Front-bumper", "Hood"]`, as an `image_labels` column.

- Supervisely format: read `ann/<image_name>.json` and collect `obj["classTitle"]` for each item in `objects`.
- COCO format: map `image_id` to its `category_id` values, then to category names.

This makes the image column more useful than a bare path, for example to filter to "damaged cars only".

---

## 5. Randomly assign one image per car row

With N cars (~100k) and M images (~2k), sampling **must be with replacement**. There are two options.

**Option A: independent random draw (simplest)**

```python
rng = np.random.default_rng(SEED)
idx = rng.integers(0, M, size=N)
```

Reuse counts per image will vary around N/M, so some images end up used noticeably more than others.

**Option B: balanced random assignment (recommended)**

Every image is used either ⌊N/M⌋ or ⌈N/M⌉ times, and the pairing is still random.

```python
import numpy as np

rng = np.random.default_rng(SEED)
N, M = len(cars), len(images)

idx = np.concatenate([
    np.tile(np.arange(M), N // M),                  # every image, full rounds
    rng.choice(M, size=N % M, replace=False),       # remainder, no repeats
])
rng.shuffle(idx)

cars = cars.join(images.iloc[idx].reset_index(drop=True))
```

Option B is preferred because it prevents a handful of images from dominating by chance, and it makes the reuse count easy to state in a report.

**Optional variant: stratified assignment.** If you want a fixed ratio of damage vs. part images (for example 50/50 regardless of the pool sizes), run Option B separately on each subset for the matching share of rows, then shuffle.

---

## 6. Save outputs

```python
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

cols = ["car_id", "brand", "model", "year", "price", "transmission",
        "mileage", "fuelType", "tax", "mpg", "engineSize",
        "image_path", "image_source"]          # + "image_labels" if added

cars[cols].to_csv(OUT / "used_cars_with_images.csv", index=False)
cars[cols].to_parquet(OUT / "used_cars_with_images.parquet", index=False)
```

- Store **relative paths**, together with an `IMG_ROOT` config value, so the dataset still works if the folder moves.
- Do **not** embed image bytes in the CSV. At ~100k rows with repeated images, that would inflate the file size by about 50× for no benefit.
- If a self-contained bundle is needed, copy the image folder into `data/processed/images/` next to the table and zip both together.
- Save a small `build_meta.json` containing the seed, source file list, row counts, number of duplicates dropped, image count, and min/max reuse.

---

## 7. Validation checklist

Run these as assertions at the end of `build_dataset.py`.

- [ ] `len(cars)` equals the sum of the brand CSV row counts, minus any rows dropped on purpose.
- [ ] `brand` value counts match the per-file row counts.
- [ ] No nulls in `image_path`.
- [ ] Every `image_path` exists on disk.
- [ ] Reuse per image is within {⌊N/M⌋, ⌈N/M⌉} (Option B).
- [ ] Rows are no longer grouped by brand: the number of brand "runs" (consecutive same-brand stretches) should be close to N, not 9.
- [ ] Re-running with the same seed produces an identical file (compare hashes).
- [ ] Spot-check around 10 random rows by opening the image and reading the row.

**If the data is used for modelling:** split with `GroupShuffleSplit` (or `GroupKFold`) using `groups=cars["image_path"]`, so no image appears in both the train and test sets.

---

## 8. Deliverables

1. `src/build_dataset.py`, a single reproducible script covering Steps 2–7
2. `data/processed/used_cars_with_images.csv` and `.parquet`
3. `data/processed/build_meta.json`
4. A short README note on the synthetic pairing caveat (Step 0)
