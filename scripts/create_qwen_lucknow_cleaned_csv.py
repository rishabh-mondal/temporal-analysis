#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create cleaned CSV from Qwen Lucknow predictions
Similar format to GPT cleaned CSV: lat_lon, presence, appearance_year,
shape_transition_year_before, shape_transition_year_after
"""

import pandas as pd
from pathlib import Path

# Input and output paths
INPUT_CSV = Path(
    "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/temporal-analysis/"
    "qwen_kiln_change_results_lucknow_all_loc.csv"
)

OUTPUT_CSV = Path(
    "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/temporal-analysis/"
    "model_prediction_csv/qwen_kiln_stats_lucknow_cleaned.csv"
)

# Create output directory if it doesn't exist
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# Read the Qwen Lucknow predictions
print(f"Reading: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)
print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Select and rename columns to match GPT format
cleaned_df = df[[
    "lat_lon",
    "presence",
    "appearance_year",
    "shape_transition_year_before",
    "shape_transition_year_after"
]].copy()

# Convert presence to binary (1/0)
cleaned_df["presence"] = cleaned_df["presence"].astype(int)

# Ensure appearance_year is integer
cleaned_df["appearance_year"] = pd.to_numeric(
    cleaned_df["appearance_year"], errors="coerce"
).fillna(0).astype(int)

# Ensure transition years are integers
cleaned_df["shape_transition_year_before"] = pd.to_numeric(
    cleaned_df["shape_transition_year_before"], errors="coerce"
).fillna(0).astype(int)

cleaned_df["shape_transition_year_after"] = pd.to_numeric(
    cleaned_df["shape_transition_year_after"], errors="coerce"
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
print(cleaned_df[cleaned_df['appearance_year'] > 0]['appearance_year'].value_counts().sort_index())
print("\nTransition year distribution:")
trans_dist = cleaned_df[cleaned_df['shape_transition_year_after'] > 0]['shape_transition_year_after'].value_counts().sort_index()
if len(trans_dist) > 0:
    print(trans_dist)
else:
    print("No transitions detected")
