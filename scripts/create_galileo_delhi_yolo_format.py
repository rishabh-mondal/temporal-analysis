#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert Galileo inference results to YOLO-format CSV
Format: filename, 2014, 2016, 2018, 2020, 2022, 2024
Labels: 0=negetive, 1=FCBK, 2=Zigzag
"""

import pandas as pd
from pathlib import Path
import os

# Input directory with Galileo inference CSVs
INFERENCE_DIR = Path("/home/suruchi.hardaha/IJCAI_2026/results/foundation-model/galileo_2025_merge01/inference")

# Output file
OUTPUT_CSV = Path(
    "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/temporal-analysis/"
    "galileo_prediction_delhi_by_year.csv"
)

# Years to process
YEARS = [2014, 2016, 2018, 2020, 2022, 2024]

# Label mapping
LABEL_MAP = {
    0: "negetive",
    1: "FCBK",
    2: "Zigzag"
}

print("="*80)
print("CONVERTING GALILEO INFERENCE TO YOLO FORMAT")
print("="*80)

# Dictionary to store predictions by filename
predictions_by_file = {}

# Process each year
for year in YEARS:
    csv_file = INFERENCE_DIR / f"delhi_airshed_y_{year}_z_17_buf_25m_top1.csv"

    if not csv_file.exists():
        print(f"WARNING: {csv_file} not found, skipping...")
        continue

    print(f"\nProcessing {year}...")
    df = pd.read_csv(csv_file)
    print(f"  Loaded {len(df)} rows")

    # Extract filename and prediction
    for _, row in df.iterrows():
        # Extract lat_lon from image_path
        # Format: /path/to/28.205600_77.105800_2014.png
        img_path = row['image_path']
        filename = os.path.basename(img_path)

        # Remove year from filename: 28.205600_77.105800_2014.png -> 28.205600_77.105800.jpg
        # We'll use .jpg to match YOLO format
        lat_lon_year = filename.replace('.png', '')
        parts = lat_lon_year.rsplit('_', 1)  # Split from right to remove year
        if len(parts) == 2:
            lat_lon = parts[0]
            filename_base = f"{lat_lon}.jpg"
        else:
            print(f"  WARNING: Could not parse filename: {filename}")
            continue

        # Get prediction label
        pred_label = int(row['pred_label'])
        pred_class = LABEL_MAP.get(pred_label, "negetive")

        # Store prediction
        if filename_base not in predictions_by_file:
            predictions_by_file[filename_base] = {}

        predictions_by_file[filename_base][year] = pred_class

    print(f"  Processed {len(df)} predictions")

print(f"\nTotal unique locations: {len(predictions_by_file)}")

# Create output DataFrame
rows = []
for filename in sorted(predictions_by_file.keys()):
    row = {'filename': filename}
    for year in YEARS:
        # Default to "negetive" if year not found
        row[str(year)] = predictions_by_file[filename].get(year, "negetive")
    rows.append(row)

# Create DataFrame
output_df = pd.DataFrame(rows)

# Ensure column order
column_order = ['filename'] + [str(y) for y in YEARS]
output_df = output_df[column_order]

# Save to CSV
output_df.to_csv(OUTPUT_CSV, index=False)

print(f"\n{'='*80}")
print(f"OUTPUT SAVED: {OUTPUT_CSV}")
print(f"{'='*80}")
print(f"Total locations: {len(output_df)}")
print(f"Columns: {output_df.columns.tolist()}")

# Print statistics
print("\nStatistics by year:")
for year in YEARS:
    year_col = str(year)
    value_counts = output_df[year_col].value_counts()
    print(f"\n{year}:")
    for label, count in value_counts.items():
        print(f"  {label}: {count}")

# Sample rows
print("\nSample rows (first 10):")
print(output_df.head(10).to_string(index=False))

print("\nSample rows (last 5):")
print(output_df.tail(5).to_string(index=False))
