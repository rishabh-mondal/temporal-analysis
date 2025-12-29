#!/usr/bin/env python3
"""
Convert Terramind model predictions to year-wise CSV format.

Input: Multiple CSV files (one per year) with predictions
Output: Single CSV with columns: filename, 2014, 2016, 2018, 2020, 2022, 2024

Prediction mapping:
- 0: negative
- 1: FCBK
- 2: Zigzag
"""

import pandas as pd
import os
from pathlib import Path

# Paths
terramind_dir = "/home/suruchi.hardaha/IJCAI_2026/results/foundation-model/terramind_2025_merge01/inference-single-instance"
output_path = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/temporal-analysis/terramind_prediction_by_year.csv"

# Years to process
years = [2014, 2016, 2018, 2020, 2022, 2024]

# Label mapping
label_map = {
    0: "negetive",
    1: "FCBK",
    2: "Zigzag"
}

print("Loading Terramind predictions...")

# Dictionary to store predictions by filename
predictions = {}

for year in years:
    csv_file = os.path.join(terramind_dir, f"infer_{year}_unique.csv")

    if not os.path.exists(csv_file):
        print(f"Warning: {csv_file} not found, skipping...")
        continue

    print(f"Processing {year}...")
    df = pd.read_csv(csv_file)

    # Extract filename and prediction
    for _, row in df.iterrows():
        image_path = row['image_path']
        pred_label = int(row['pred_label'])

        # Extract filename from path
        # Example: /path/to/28.205600_77.105800_2014.png -> 28.205600_77.105800.png
        filename = os.path.basename(image_path)
        # Remove year suffix
        base_filename = filename.replace(f"_{year}.png", ".png")

        # Initialize if not exists
        if base_filename not in predictions:
            predictions[base_filename] = {str(y): "negetive" for y in years}
            predictions[base_filename]['filename'] = base_filename

        # Store prediction
        predictions[base_filename][str(year)] = label_map.get(pred_label, "negetive")

print(f"\nTotal unique locations: {len(predictions)}")

# Convert to DataFrame
result_df = pd.DataFrame.from_dict(predictions, orient='index')

# Reorder columns
columns = ['filename'] + [str(y) for y in years]
result_df = result_df[columns]

# Sort by filename
result_df = result_df.sort_values('filename').reset_index(drop=True)

# Save to CSV
result_df.to_csv(output_path, index=False)
print(f"\nSaved predictions to: {output_path}")
print(f"Shape: {result_df.shape}")

# Show sample
print("\nSample predictions:")
print(result_df.head(10))

# Show statistics
print("\nPrediction statistics by year:")
for year in years:
    year_col = str(year)
    print(f"\n{year}:")
    print(result_df[year_col].value_counts())
