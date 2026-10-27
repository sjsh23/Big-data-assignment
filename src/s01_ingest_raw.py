"""Tahap 1, Ingestion ke Raw Zone.

Menyalin file sumber apa adanya ke Raw Zone dengan partisi source dan ingest_date,
lalu menulis manifest (jumlah baris, checksum, lisensi). File di Raw Zone tidak diubah lagi.
"""
import hashlib
import json
import shutil
import sys
from datetime import date
from pathlib import Path

from config import RAW, SOURCE_FILE, SOURCE_PATH


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(src: Path):
    if not src.exists():
        sys.exit(f"File sumber tidak ditemukan: {src}")
    target_dir = RAW / "source=kaggle_combined" / f"ingest_date={date.today().isoformat()}"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / SOURCE_FILE
    if target.exists():
        print(f"Sudah ada, tidak ditimpa (Raw Zone immutable): {target}")
    else:
        shutil.copy2(src, target)
        target.chmod(0o444)  # read only, meniru sifat immutable object storage

    with open(target, encoding="utf-8") as f:
        n_rows = sum(1 for _ in f) - 1
    manifest = {
        "file": SOURCE_FILE,
        "rows": n_rows,
        "sha256": sha256(target),
        "sources": {
            "listing": "Desai (2020), Kaggle, lisensi tidak dinyatakan",
            "images": "Humans in the Loop (2023), Kaggle, CC0 1.0",
        },
    }
    (target_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Raw Zone: {target}  ({n_rows:,} baris)")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else SOURCE_PATH)
