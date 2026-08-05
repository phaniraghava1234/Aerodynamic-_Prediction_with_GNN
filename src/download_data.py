"""Download the pre-processed AirfRANS dataset into data/.

Run once:  python src/download_data.py
Downloads ~9 GB and unzips into data/Dataset/.
"""
from pathlib import Path

import airfrans as af

ROOT = Path(__file__).resolve().parents[1] / "data"

if __name__ == "__main__":
    ROOT.mkdir(exist_ok=True)
    if (ROOT / "Dataset").exists():
        print(f"{ROOT / 'Dataset'} already exists, skipping download.")
    else:
        af.dataset.download(root=str(ROOT), file_name="Dataset", unzip=True, OpenFOAM=False)
        print(f"Dataset downloaded to {ROOT / 'Dataset'}")
