"""
convert_tiff.py
---------------
Convert raw UAV imagery into standardized false-color composites (RGB) for visualization.
"""

import argparse
from pathlib import Path
from src.core.multispectral_loader import load_multispectral_tiff
from src.core.utils import save_array_as_png

def main():
    parser = argparse.ArgumentParser(description="Convert TIFF to false-color PNG")
    parser.add_argument("--input", type=str, required=True, help="Input TIFF file")
    parser.add_argument("--output", type=str, required=True, help="Output PNG file")
    parser.add_argument("--type", type=str, choices=["cir", "veg"], default="cir", help="Composite type")
    args = parser.parse_args()

    ms_img = load_multispectral_tiff(args.input)
    
    if args.type == "cir":
        composite = ms_img.false_color_cir()
    else:
        composite = ms_img.false_color_vegetation()
        
    save_array_as_png(composite, args.output)
    print(f"Saved {args.type} composite to {args.output}")

if __name__ == "__main__":
    main()
