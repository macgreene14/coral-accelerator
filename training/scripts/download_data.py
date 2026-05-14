"""Download PV thermal datasets for training.

Sources
-------
1. InfraredSolarModules (Kaggle) — ~2,400 images, Pascal VOC bounding boxes
   Classes: hotspot, bypass diode failure, multi-hotspot, shadowing, soiling
2. Thermographic PV Systems (Kaggle) — ~700 images, binary anomaly/normal
3. PV module inspection dataset (Zenodo record 3894823) — ~800 images, polygon masks

Prerequisites
-------------
  ~/.kaggle/kaggle.json  — Kaggle API credentials
  Run: pip install kagglehub
"""

import argparse
import shutil
import sys
import tarfile
from pathlib import Path

import requests


KAGGLE_DATASETS = [
    {
        "id": "afsharshamsi/infrared-solar-modules",
        "dest": "infrared_solar_modules",
    },
    {
        "id": "marcosgabriel/thermographic-images-of-photovoltaic-systems",
        "dest": "thermographic_pv",
    },
]

ZENODO_URL = "https://zenodo.org/record/3894823/files/module_images.tar.gz"


def download_kaggle(dataset_id: str, output_dir: Path) -> Path:
    """Download a Kaggle dataset via kagglehub and copy to output_dir."""
    import kagglehub

    print(f"Downloading Kaggle dataset: {dataset_id}")
    cache_path = kagglehub.dataset_download(dataset_id)
    dest = output_dir / dataset_id.split("/")[1]
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(cache_path, dest)
    img_count = sum(1 for _ in dest.rglob("*.jpg")) + sum(1 for _ in dest.rglob("*.png"))
    print(f"  {dest.name}: {img_count} images")
    return dest


def download_zenodo(url: str, output_dir: Path) -> Path:
    """Download and extract a Zenodo dataset archive."""
    dest = output_dir / "pvdn"
    dest.mkdir(parents=True, exist_ok=True)
    archive = output_dir / "pvdn.tar.gz"

    if not archive.exists():
        print(f"Downloading: {url}")
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(archive, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {pct:.1f}%", end="", flush=True)
        print()

    print(f"Extracting to {dest}")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest)
    archive.unlink()
    img_count = sum(1 for _ in dest.rglob("*.jpg")) + sum(1 for _ in dest.rglob("*.png"))
    print(f"  pvdn: {img_count} images")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PV thermal training datasets")
    parser.add_argument("--output", type=Path, required=True, help="Directory for raw data")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["kaggle", "zenodo", "all"],
        default=["all"],
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    sources = set(args.sources)
    if "all" in sources:
        sources = {"kaggle", "zenodo"}

    if "kaggle" in sources:
        try:
            for ds in KAGGLE_DATASETS:
                download_kaggle(ds["id"], args.output)
        except Exception as exc:
            print(f"ERROR downloading Kaggle datasets: {exc}", file=sys.stderr)
            print("Ensure ~/.kaggle/kaggle.json exists. Get credentials at kaggle.com/account", file=sys.stderr)
            sys.exit(1)

    if "zenodo" in sources:
        try:
            download_zenodo(ZENODO_URL, args.output)
        except Exception as exc:
            print(f"WARNING: Zenodo download failed: {exc}", file=sys.stderr)
            print("Continuing without PVDN dataset.", file=sys.stderr)

    print(f"\nDownload complete. Raw data: {args.output}")


if __name__ == "__main__":
    main()
