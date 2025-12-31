#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create cleaned CSV from TeoChat Lucknow predictions
Similar format to GPT cleaned CSV: lat_lon, presence, appearance_year,
shape_transition_year_before, shape_transition_year_after

Note: TeoChat has multi-line JSON and different column structure
"""

import pandas as pd
from pathlib import Path

# Input and output paths
INPUT_CSV = Path(
    "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/temporal-analysis/"
    "lucknow_result_teochat.csv"
)

OUTPUT_CSV = Path(
    "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/temporal-analysis/"
    "model_prediction_csv/teochat_kiln_stats_lucknow_cleaned.csv"
)

# Create output directory if it doesn't exist
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

print(f"Reading: {INPUT_CSV}")

# TeoChat CSV has multi-line JSON in raw_response column
# Need to read it carefully
try:
    # Try reading with standard pandas first
    df = pd.read_csv(INPUT_CSV, on_bad_lines='skip', engine='python')
    print(f"Loaded with pandas: {len(df)} rows")
except Exception as e:
    print(f"Standard pandas read failed: {e}")
    print("Using line-by-line parsing...")

    # Line-by-line parsing
    rows = []
    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        header = f.readline().strip()

        current_row_parts = []
        for line in f:
            current_row_parts.append(line)

            # Row ends with just "}" on a line
            if line.strip() == '}':
                full_row = ''.join(current_row_parts)
                parts = full_row.split(',', maxsplit=9)  # Split only first 9 commas

                if len(parts) >= 9:
                    try:
                        lat_lon = parts[0].strip().strip('"')
                        bbox = parts[1].strip().strip('"')
                        presence = parts[2].strip()
                        shape_start = parts[3].strip().strip('"')
                        shape_end = parts[4].strip().strip('"')
                        demolished = parts[5].strip()
                        appearance_year = parts[6].strip()
                        shape_change_year = parts[7].strip()
                        demolished_year = parts[8].strip()

                        rows.append({
                            'lat_lon': lat_lon,
                            'presence': presence,
                            'appearance_year': appearance_year,
                            'shape_start': shape_start,
                            'shape_end': shape_end,
                            'shape_change_year': shape_change_year
                        })
                    except:
                        pass

                current_row_parts = []

    df = pd.DataFrame(rows)
    print(f"Loaded with line parser: {len(df)} rows")

print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Map TeoChat columns to standard format
# TeoChat uses: presence, appearance_year, shape_change_year
# Standard uses: presence, appearance_year, shape_transition_year_before, shape_transition_year_after

cleaned_df = pd.DataFrame()
cleaned_df["lat_lon"] = df["lat_lon"]

# Convert presence to binary (handle True/False strings)
def to_binary(val):
    if pd.isna(val) or val == '':
        return 0
    val_str = str(val).strip().lower()
    return 1 if val_str in ['true', '1', 'yes'] else 0

cleaned_df["presence"] = df["presence"].apply(to_binary)

# Ensure appearance_year is integer
cleaned_df["appearance_year"] = pd.to_numeric(
    df["appearance_year"], errors="coerce"
).fillna(0).astype(int)

# TeoChat has shape_change_year instead of transition_year_before/after
# We'll use shape_change_year as transition_year_after
# Set transition_year_before to 0 (or we could infer from shape_change_year - 2)
cleaned_df["shape_transition_year_before"] = 0

cleaned_df["shape_transition_year_after"] = pd.to_numeric(
    df["shape_change_year"], errors="coerce"
).fillna(0).astype(int)

# Save cleaned CSV
cleaned_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved cleaned CSV: {OUTPUT_CSV}")
print(f"Cleaned rows: {len(cleaned_df)}")
print("\nSample rows:")
print(cleaned_df.head(10))

# Print statistics
print("\n" + "="*60)
print("STATISTICS")
print("="*60)
print(f"Total locations: {len(cleaned_df)}")
print(f"Presence=1: {(cleaned_df['presence'] == 1).sum()}")
print(f"Presence=0: {(cleaned_df['presence'] == 0).sum()}")
print(f"With appearance year: {(cleaned_df['appearance_year'] > 0).sum()}")
print(f"With transition: {(cleaned_df['shape_transition_year_after'] > 0).sum()}")
print("\nAppearance year distribution:")
app_dist = cleaned_df[cleaned_df['appearance_year'] > 0]['appearance_year'].value_counts().sort_index()
if len(app_dist) > 0:
    print(app_dist)
else:
    print("No appearance years detected")
print("\nTransition year distribution:")
trans_dist = cleaned_df[cleaned_df['shape_transition_year_after'] > 0]['shape_transition_year_after'].value_counts().sort_index()
if len(trans_dist) > 0:
    print(trans_dist)
else:
    print("No transitions detected")
