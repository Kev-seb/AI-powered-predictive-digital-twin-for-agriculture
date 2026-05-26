"""
download_dataset.py
-------------------
Utility script to help download and extract UAV multispectral datasets.
"""

import os
import argparse
import urllib.request
import zipfile
from pathlib import Path

def download_file(url: str, dest: Path) -> None:
    print(f"Downloading from {url} to {dest}...")
    urllib.request.urlretrieve(url, dest)
    print("Download complete.")

def extract_zip(zip_path: Path, extract_to: Path) -> None:
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")

def main():
    parser = argparse.ArgumentParser(description="Download UAV multispectral dataset")
    parser.add_argument("--url", type=str, required=True, help="URL of the dataset zip file")
    parser.add_argument("--dest", type=str, default="data/raw", help="Destination folder")
    args = parser.parse_args()

    dest_dir = Path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = dest_dir / "dataset.zip"
    download_file(args.url, zip_path)
    extract_zip(zip_path, dest_dir)
    
    print("Dataset ready.")

if __name__ == "__main__":
    main()
