"""
generate_indices.py
-------------------
Command-line tool to quickly generate a specific vegetation index from a multispectral image.
"""

import argparse
from src.core.multispectral_loader import load_multispectral_tiff
from src.indices.indices import compute_all_indices
from src.core.utils import save_array_as_png

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--index", type=str, required=True, choices=["NDVI", "NDRE", "GNDVI", "MSAVI2", "EVI"])
    args = parser.parse_args()

    ms_img = load_multispectral_tiff(args.input)
    indices = compute_all_indices(ms_img.green, ms_img.red, ms_img.red_edge, ms_img.nir)
    
    if args.index in indices:
        # Save as a colored heatmap
        save_array_as_png(indices[args.index], args.output, cmap="RdYlGn")
        print(f"Saved {args.index} heatmap to {args.output}")
    else:
        print(f"Index {args.index} not computed.")

if __name__ == "__main__":
    main()
