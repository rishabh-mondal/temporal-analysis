#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create cleaned CSV from Cosmos Reason2-8B Lucknow predictions
Similar format to GPT cleaned CSV: lat_lon, presence, appearance_year,
shape_transition_year_before, shape_transition_year_after

Note: Cosmos CSV has some parse failures and multi-line JSON
"""

import pandas as pd
import json
from pathlib import Path

# Input and output paths
INPUT_CSV = Path(
    "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/temporal-analysis/"
    "cosmos/cosmos_reason2_8b_kiln_change_results_lucknow_all_loc_bbox.csv"
)

OUTPUT_CSV = Path(
    "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/temporal-analysis/"
    "model_prediction_csv/cosmos_kiln_stats_lucknow_cleaned.csv"
)

# Create output directory if it doesn't exist
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# Read the Cosmos Lucknow predictions with special handling for multi-line JSON
print(f"Reading: {INPUT_CSV}")

# Read line by line to handle multi-line JSON properly
rows = []
with open(INPUT_CSV, 'r', encoding='utf-8') as f:
    # Skip header
    header = f.readline()

    current_row_parts = []
    for line in f:
        current_row_parts.append(line)

        # Check if this completes a row (ends with parse_fail or ok)
        if line.strip().endswith(',parse_fail') or line.strip().endswith(',ok'):
            full_row = ''.join(current_row_parts)

            # Split by comma, but be careful with JSON content
            parts = full_row.split(',')

            if len(parts) >= 21:
                try:
                    lat = parts[0].strip()
                    lon = parts[1].strip()
                    lat_lon = parts[2].strip()
                    presence = parts[3].strip()
                    appearance_year = parts[4].strip()

                    # Skip rows with parse_fail status
                    if 'parse_fail' in full_row:
                        current_row_parts = []
                        continue

                    # Try to extract from raw_output if main fields are empty
                    if not presence or not appearance_year:
                        # Look for JSON in raw_output
                        try:
                            # Find JSON block
                            json_start = full_row.find('```json')
                            if json_start != -1:
                                json_content = full_row[json_start + 7:]
                                json_end = json_content.find('```')
                                if json_end != -1:
                                    json_str = json_content[:json_end]
                                    data = json.loads(json_str)

                                    presence = str(data.get('presence', ''))
                                    appearance_year = str(data.get('appearance_year', 0))
                                    shape_transition_year_before = str(data.get('shape_transition_year_before', 0))
                                    shape_transition_year_after = str(data.get('shape_transition_year_after', 0))

                                    rows.append({
                                        'lat_lon': lat_lon,
                                        'presence': presence,
                                        'appearance_year': appearance_year,
                                        'shape_transition_year_before': shape_transition_year_before,
                                        'shape_transition_year_after': shape_transition_year_after
                                    })
                        except:
                            pass
                    else:
                        # Use direct fields
                        shape_transition_year_before = parts[9].strip() if len(parts) > 9 else '0'
                        shape_transition_year_after = parts[10].strip() if len(parts) > 10 else '0'

                        rows.append({
                            'lat_lon': lat_lon,
                            'presence': presence,
                            'appearance_year': appearance_year,
                            'shape_transition_year_before': shape_transition_year_before,
                            'shape_transition_year_after': shape_transition_year_after
                        })
                except Exception as e:
                    print(f"Error processing row: {e}")
                    pass

            current_row_parts = []

# Create DataFrame
df = pd.DataFrame(rows)

if len(df) == 0:
    print("WARNING: No valid rows found. Trying alternative parsing...")
    # Fallback: try reading with pandas directly, skip bad lines
    try:
        df_raw = pd.read_csv(INPUT_CSV, on_bad_lines='skip', engine='python')
        # Filter only OK status
        df_raw = df_raw[df_raw['status'] == 'ok']

        df = df_raw[[
            "lat_lon",
            "presence",
            "appearance_year",
            "shape_transition_year_before",
            "shape_transition_year_after"
        ]].copy()
    except Exception as e:
        print(f"Fallback parsing failed: {e}")
        raise

print(f"Total rows extracted: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Clean data
# Convert presence to binary (handle True/False strings)
def to_binary(val):
    if pd.isna(val) or val == '':
        return 0
    val_str = str(val).strip().lower()
    return 1 if val_str in ['true', '1', 'yes'] else 0

df["presence"] = df["presence"].apply(to_binary)

# Ensure appearance_year is integer
df["appearance_year"] = pd.to_numeric(
    df["appearance_year"], errors="coerce"
).fillna(0).astype(int)

# Ensure transition years are integers
df["shape_transition_year_before"] = pd.to_numeric(
    df["shape_transition_year_before"], errors="coerce"
).fillna(0).astype(int)

df["shape_transition_year_after"] = pd.to_numeric(
    df["shape_transition_year_after"], errors="coerce"
).fillna(0).astype(int)

# Save cleaned CSV
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved cleaned CSV: {OUTPUT_CSV}")
print(f"Cleaned rows: {len(df)}")
print("\nSample rows:")
print(df.head(10))

# Print statistics
print("\n" + "="*60)
print("STATISTICS")
print("="*60)
print(f"Total locations: {len(df)}")
print(f"Presence=1: {(df['presence'] == 1).sum()}")
print(f"Presence=0: {(df['presence'] == 0).sum()}")
print(f"With appearance year: {(df['appearance_year'] > 0).sum()}")
print(f"With transition: {(df['shape_transition_year_after'] > 0).sum()}")
print("\nAppearance year distribution:")
app_dist = df[df['appearance_year'] > 0]['appearance_year'].value_counts().sort_index()
if len(app_dist) > 0:
    print(app_dist)
else:
    print("No appearance years detected")
print("\nTransition year distribution:")
trans_dist = df[df['shape_transition_year_after'] > 0]['shape_transition_year_after'].value_counts().sort_index()
if len(trans_dist) > 0:
    print(trans_dist)
else:
    print("No transitions detected")
