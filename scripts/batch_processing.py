"""
batch_processing.py
-------------------
Batch process a directory of raw multispectral images into preprocessed arrays and indices.
"""

import argparse
from pathlib import Path
from src.core.multispectral_loader import load_multispectral_tiff
from src.indices.indices import compute_all_indices
from src.core.utils import setup_logger
from loguru import logger

def main():
    parser = argparse.ArgumentParser(description="Batch process multispectral TIFFs")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing raw TIFFs")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed results")
    args = parser.parse_args()

    setup_logger(log_dir=args.output_dir)
    
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tiffs = list(in_dir.glob("*.tif")) + list(in_dir.glob("*.tiff"))
    logger.info(f"Found {len(tiffs)} TIFF files in {in_dir}")

    for tif in tiffs:
        logger.info(f"Processing {tif.name}...")
        try:
            ms_img = load_multispectral_tiff(tif)
            indices = compute_all_indices(ms_img.green, ms_img.red, ms_img.red_edge, ms_img.nir)
            logger.info(f"Computed {len(indices)} indices for {tif.name}")
            # In a real pipeline, save the indices to disk here.
        except Exception as e:
            logger.error(f"Failed to process {tif.name}: {e}")

if __name__ == "__main__":
    main()
