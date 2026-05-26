"""
prepare_dataset.py
------------------
Parses the raw multi-stage paddy dataset (single-band TIFFs),
stacks them, extracts 224x224 patches, and saves them as .npy
into the `data/processed/classification/` directory.
"""

import os
from pathlib import Path
import numpy as np
import tifffile

def process_raw_dataset(raw_dir: str, out_dir: str, patch_size: int = 224):
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    
    stage_mapping = {
        "Nursery": "Nursery",
        "Vegetative": "Vegetative",
        "Booting": "Vegetative",
        "Flowering": "Flowering",
        "Mature": "Mature"
    }
    
    for stage in set(stage_mapping.values()):
        (out_path / stage).mkdir(parents=True, exist_ok=True)
        
    stage_dirs = []
    for root, dirs, files in os.walk(raw_path):
        for d in dirs:
            for key in stage_mapping.keys():
                if key in d:
                    stage_dirs.append((Path(root) / d, stage_mapping[key]))
                    break
                    
    print(f"Found {len(stage_dirs)} stage directories.")
    
    patch_count = 0
    max_patches_per_stage = 500 # limit patches so training is fast for demo
    stage_counts = {s: 0 for s in set(stage_mapping.values())}
    
    for s_dir, target_stage in stage_dirs:
        if stage_counts[target_stage] >= max_patches_per_stage:
            continue
            
        print(f"Processing {s_dir.name} -> {target_stage}")
        all_g = list(s_dir.glob("*_MS_G.TIF")) + list(s_dir.glob("*_MS_G.tif"))
        prefixes = [str(p).replace("_MS_G.TIF", "").replace("_MS_G.tif", "") for p in all_g]
        
        for prefix in prefixes:
            if stage_counts[target_stage] >= max_patches_per_stage:
                break
                
            try:
                g = tifffile.imread(f"{prefix}_MS_G.TIF").astype(np.float32)
                r = tifffile.imread(f"{prefix}_MS_R.TIF").astype(np.float32)
                re = tifffile.imread(f"{prefix}_MS_RE.TIF").astype(np.float32)
                nir = tifffile.imread(f"{prefix}_MS_NIR.TIF").astype(np.float32)
                
                stack = np.stack([g, r, re, nir], axis=0) # (4, H, W)
                
                for i in range(4):
                    p98 = np.percentile(stack[i], 98)
                    if p98 > 0:
                        stack[i] = np.clip(stack[i] / p98, 0, 1)
                        
                _, H, W = stack.shape
                stride = patch_size * 2 # larger stride to get diverse patches
                for y in range(0, H - patch_size + 1, stride):
                    for x in range(0, W - patch_size + 1, stride):
                        if stage_counts[target_stage] >= max_patches_per_stage:
                            break
                        patch = stack[:, y:y+patch_size, x:x+patch_size]
                        out_file = out_path / target_stage / f"patch_{patch_count:06d}.npy"
                        np.save(out_file, patch)
                        patch_count += 1
                        stage_counts[target_stage] += 1
                        
            except Exception as e:
                pass
                
    print(f"Saved {patch_count} patches to {out_path}")
    print(f"Counts per stage: {stage_counts}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="data/raw/paddy_dataset")
    parser.add_argument("--out_dir", type=str, default="data/processed/classification")
    args = parser.parse_args()
    
    process_raw_dataset(args.raw_dir, args.out_dir)
