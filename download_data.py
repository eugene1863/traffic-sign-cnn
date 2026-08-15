"""Fetches the GTSRB (German Traffic Sign Recognition Benchmark) dataset
into data/, laid out the way src/data.py expects:

    data/Train/<0..42>/*.png
    data/Test/*.png
    data/Test.csv

Primary path: Kaggle's "gtsrb-german-traffic-sign" dataset via kagglehub
(requires a Kaggle account + API token, see README.md "Getting the data").
If kagglehub isn't available or auth isn't configured, this prints manual
download instructions instead of failing silently.
"""

import os
import shutil
import sys

from src import config

KAGGLE_DATASET = "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"

MANUAL_INSTRUCTIONS = f"""
Automatic download didn't work. To get the data manually:

1. Create a free Kaggle account and API token:
   https://www.kaggle.com/settings -> "Create New Token" (downloads kaggle.json)
2. Place kaggle.json at ~/.kaggle/kaggle.json (chmod 600) or set
   KAGGLE_USERNAME / KAGGLE_KEY environment variables.
3. Re-run: python download_data.py

Or download directly from the browser:
   https://www.kaggle.com/datasets/{KAGGLE_DATASET}
   -> unzip it so you end up with:
      {config.DATA_DIR}\\Train\\0\\...png ... Train\\42\\...png
      {config.DATA_DIR}\\Test\\...png
      {config.DATA_DIR}\\Test.csv

Any other folder-per-class image dataset also works with make_train_val_datasets()
in src/data.py -- GTSRB's Test.csv holdout format is only needed by evaluate.py.
"""


def main() -> None:
    os.makedirs(config.DATA_DIR, exist_ok=True)

    try:
        import kagglehub
    except ImportError:
        print("kagglehub not installed (pip install kagglehub).")
        print(MANUAL_INSTRUCTIONS)
        sys.exit(1)

    try:
        path = kagglehub.dataset_download(KAGGLE_DATASET)
    except Exception as exc:  # noqa: BLE001 - surface any auth/network error to the user
        print(f"kagglehub download failed: {exc}")
        print(MANUAL_INSTRUCTIONS)
        sys.exit(1)

    print(f"Downloaded to cache: {path}")
    for name in ("Train", "Test", "Test.csv", "Meta"):
        src_path = os.path.join(path, name)
        dst_path = os.path.join(config.DATA_DIR, name)
        if not os.path.exists(src_path):
            continue
        if os.path.exists(dst_path):
            print(f"  skip (already exists): {dst_path}")
            continue
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)
        print(f"  copied {name} -> {dst_path}")

    print("Done. Data is ready under", config.DATA_DIR)


if __name__ == "__main__":
    main()
