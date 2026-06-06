"""
download_data.py
----------------
Downloads the RetailRocket e-commerce dataset from Kaggle and places
the CSV files in the expected data/raw/ folder.

Requirements:
    pip install kagglehub

Kaggle authentication:
    Option 1 — API token (recommended):
        1. Go to https://www.kaggle.com/settings → "API" → "Create New Token"
        2. Save the downloaded kaggle.json to:
               ~/.kaggle/kaggle.json          (Linux / Mac)
               C:\\Users\\<user>\\.kaggle\\kaggle.json  (Windows)
        3. Run this script — no further setup needed.

    Option 2 — environment variables:
        export KAGGLE_USERNAME=your_username
        export KAGGLE_KEY=your_api_key

Usage:
    python download_data.py
"""

import os
import shutil
import sys

# ── target directory ─────────────────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.abspath(__file__))
RAW_DIR  = os.path.join(ROOT, "data", "raw")

# ── expected files ───────────────────────────────────────────────────────────
EXPECTED_FILES = [
    "events.csv",
    "item_properties_part1.csv",
    "item_properties_part2.csv",
    "category_tree.csv",
]


def download():
    # ── check kagglehub is installed ─────────────────────────────────────────
    try:
        import kagglehub
    except ImportError:
        print("[ERROR] kagglehub is not installed.")
        print("        Run:  pip install kagglehub")
        sys.exit(1)

    # ── skip if data already exists ──────────────────────────────────────────
    if all(os.path.exists(os.path.join(RAW_DIR, f)) for f in EXPECTED_FILES):
        print("[download] All data files already present in data/raw/ — skipping download.")
        return

    os.makedirs(RAW_DIR, exist_ok=True)

    # ── download ─────────────────────────────────────────────────────────────
    print("[download] Downloading RetailRocket dataset from Kaggle …")
    try:
        path = kagglehub.dataset_download("retailrocket/ecommerce-dataset")
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print(
            "\nMake sure your Kaggle API credentials are set up.\n"
            "See the instructions at the top of this file."
        )
        sys.exit(1)

    print(f"[download] Dataset downloaded to: {path}")

    # ── copy CSVs to data/raw/ ───────────────────────────────────────────────
    print(f"[download] Copying CSV files to {RAW_DIR}/ …")
    copied = []

    for root_dir, _, files in os.walk(path):
        for fname in files:
            if fname.endswith(".csv"):
                src = os.path.join(root_dir, fname)
                dst = os.path.join(RAW_DIR, fname)
                shutil.copy2(src, dst)
                copied.append(fname)
                print(f"           ✓  {fname}")

    if not copied:
        print(f"[WARNING] No CSV files found in {path}")
        sys.exit(1)

    # ── verify expected files ────────────────────────────────────────────────
    missing = [f for f in EXPECTED_FILES if not os.path.exists(os.path.join(RAW_DIR, f))]
    if missing:
        print(f"\n[WARNING] These expected files were not found: {missing}")
        print("          The dataset structure may have changed on Kaggle.")
    else:
        print(f"\n[download] All {len(EXPECTED_FILES)} files ready in data/raw/ ✓")


if __name__ == "__main__":
    download()
