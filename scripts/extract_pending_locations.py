#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract pending locations and create directory structure with symlinks
"""

import os
import csv
from pathlib import Path
from tqdm import tqdm

# Paths
BASE_DIR = Path("/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/temporal-analysis/data_all")
FOLDER_TPL = "all_kilns_y_{y}_z_17_buf_25m"
YEARS = [2014, 2016, 2018, 2020, 2022, 2024]

PROCESSED_CSV = Path(
    "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/temporal-analysis/"
    "gemini-3m-pro_kiln_sentinelklindb_and_beyond_with_bbox_from_csv.csv"
)

PENDING_BASE_DIR = Path(
    "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/temporal-analysis/"
    "data_pending_only"
)

BBOX_MASTER_CSV = Path(
    "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/temporal-analysis/"
    "data_all/lat_lon_all_locations_with_bbox.csv"
)

print("="*60)
print("STEP 1: Extract processed locations from CSV")
print("="*60)

# Read processed locations (handle bad lines)
processed_locs = set()
with open(PROCESSED_CSV, 'r') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        try:
            if 'lat_lon' in row and row['lat_lon']:
                processed_locs.add(str(row['lat_lon']))
        except Exception as e:
            print(f"Skipping bad row {i}: {e}")
            continue

print(f"Processed locations: {len(processed_locs)}")

print("\n" + "="*60)
print("STEP 2: Get all locations from bbox CSV")
print("="*60)

# Get all locations from bbox CSV
all_locs = set()
import pandas as pd
df_bbox = pd.read_csv(BBOX_MASTER_CSV)
all_locs = set(df_bbox['lat_lon'].astype(str).tolist())
print(f"Total locations: {len(all_locs)}")

# Find pending
pending_locs = sorted(list(all_locs - processed_locs))
print(f"Pending locations: {len(pending_locs)}")

if len(pending_locs) == 0:
    print("\n✓ No pending locations! All done!")
    exit(0)

# Save pending locations list
pending_list_file = Path(
    "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/temporal-analysis/"
    "pending_locations.txt"
)
with open(pending_list_file, 'w') as f:
    for loc in pending_locs:
        f.write(f"{loc}\n")
print(f"\nSaved pending list to: {pending_list_file}")

print("\n" + "="*60)
print("STEP 3: Create symlinks for pending locations")
print("="*60)

# Create directory structure
for year in YEARS:
    year_dir = PENDING_BASE_DIR / f"all_kilns_y_{year}_z_17_buf_25m" / "images"
    year_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created: {year_dir}")

# Create symlinks
symlinks_created = 0
files_missing = 0

for year in tqdm(YEARS, desc="Years"):
    source_dir = BASE_DIR / FOLDER_TPL.format(y=year)
    target_dir = PENDING_BASE_DIR / FOLDER_TPL.format(y=year) / "images"

    # Check both root and images subdirectory
    source_paths = [source_dir, source_dir / "images"]

    for loc in pending_locs:
        lat, lon = loc.split("_")
        filename = f"{lat}_{lon}_{year}.png"

        # Find the source file
        source_file = None
        for src_path in source_paths:
            potential_file = src_path / filename
            if potential_file.exists():
                source_file = potential_file
                break

        if source_file is None:
            files_missing += 1
            continue

        target_file = target_dir / filename

        # Create symlink
        if not target_file.exists():
            os.symlink(source_file, target_file)
            symlinks_created += 1

print(f"\n✓ Symlinks created: {symlinks_created}")
print(f"✗ Files missing: {files_missing}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Processed locations: {len(processed_locs)}")
print(f"Total locations: {len(all_locs)}")
print(f"Pending locations: {len(pending_locs)}")
print(f"Symlinks created: {symlinks_created}")
print(f"\nPending data directory: {PENDING_BASE_DIR}")
print("\nNext step: Run gemini3-pro_bbox.py with:")
print(f"  BASE_DIR = '{PENDING_BASE_DIR}'")
print(f"  N_LOCATIONS = {len(pending_locs)}")
