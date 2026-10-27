# Used cars + car parts/damages images

One shuffled table of every UK used-car listing (9 brands), where each row is paired with one image from the Car Parts and Car Damages dataset.

```
python src/build_dataset.py
```

This writes the following files to `data/processed/`:

| File | Contents |
|---|---|
| `used_cars_with_images.csv` / `.parquet` | 97,712 rows: car attributes, `brand`, and the assigned image |
| `build_meta.json` | Seed, source row counts, rows dropped or flagged, image counts, reuse range, validation results, CSV hash |

The script checks everything in plan Step 7 as assertions, and it fails loudly if any check does not hold. Running it again with the same seed produces a byte-identical CSV.

## ⚠️ The image pairing is synthetic

The two datasets share no key, so each image was assigned to its car **at random**. An image says nothing about the car in its row: its price, mileage and year are unrelated to the damage shown, and the photo often shows a different make altogether (for example, a Lexus next to a Skoda listing).

- **Suitable for:** pipeline, data-loader, schema or dashboard exercises.
- **Not suitable for:** learning anything about the cars from the images. Image features are noise by construction. A tabular-only vs. tabular+image comparison will show no gain, and a write-up should say so.
- **Image reuse:** 1,371 unique photos cover 97,712 rows, so each photo is used 71 or 72 times. When splitting into train and test sets, group by `image_id` so the same photo never lands in both:

  ```python
  from sklearn.model_selection import GroupShuffleSplit
  train_idx, test_idx = next(GroupShuffleSplit(test_size=0.2, random_state=0)
                             .split(df, groups=df["image_id"]))
  ```

## Columns

| Column | Meaning |
|---|---|
| `car_id` | Row key, assigned after shuffling |
| `brand` | Taken from the source file name (the CSVs have no brand column) |
| `model` … `engineSize` | Original listing fields. Hyundai's `tax(£)` is renamed to `tax`, and whitespace is stripped from `model` |
| `data_issue` | Empty, or `year_after_collection` / `zero_engine_size`. These rows are flagged but kept |
| `image_id` | First 12 hex digits of the image's SHA-1. Use this to group splits |
| `image_path` | Path relative to `car damages images/` |
| `image_source` | `damage`, `part`, or `both`: which annotation sets cover the photo |
| `damage_labels` / `part_labels` | `\|`-separated class names from the annotations |

## Data decisions (differences from `plan.md`)

- **Excluded CSVs:** `cclass.csv`, `focus.csv` and the two `unclean *.csv` files. They are subsets or raw copies of `merc.csv` and `ford.csv`.
- **Exact duplicates:** 1,475 duplicate car rows were dropped (99,187 → 97,712). To keep them, set `DROP_EXACT_DUPLICATES = False`.
- **Flagged rows:** 1 row with year 2060 and 266 non-electric cars with `engineSize == 0`.
- **Only `img/` folders are used.** The `masks_human/` and `masks_machine/` folders hold PNG segmentation masks, not photos.
- **The folder names are swapped relative to their contents.** `Car damages dataset/` is annotated with *car parts* (Hood, Bumper, …), and `Car parts dataset/` is annotated with *damage types* (Dent, Scratch, …). The script decides each subset's type from its `meta.json` classes instead of trusting the folder name.
- **Photos shared between subsets are merged.** 441 photos appear byte-for-byte in both folders (once per annotation type). Photos are deduplicated by content hash, leaving 1,812 files → 1,371 unique images, and they are counted once. Otherwise they would be assigned twice as often, and a split by path would leak them.
- **Image assignment:** balanced random with replacement (plan Option B). Every image is used ⌊N/M⌋ or ⌈N/M⌉ times.
