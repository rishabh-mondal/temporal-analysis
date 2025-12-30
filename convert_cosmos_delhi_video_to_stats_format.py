import pandas as pd
import numpy as np

# Read the Cosmos predictions CSV (only the columns we need, skip raw_output to avoid parsing issues)
cosmos_df = pd.read_csv(
    '/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/temporal-analysis/cosmos_reason2_8b_kiln_change_results_delhi_video_bbox.csv',
    usecols=['lat_lon', 'presence', 'appearance_year', 'shape_transition_year_before', 'shape_transition_year_after']
)

print(f"Loaded {len(cosmos_df)} records from Cosmos predictions")

# Create the new dataframe with the required columns
stats_df = pd.DataFrame()

# Map the columns
stats_df['lat_lon'] = cosmos_df['lat_lon']

# Map presence - handle True/False strings and convert to 1/0
def convert_presence(val):
    if pd.isna(val) or val == '':
        return 0
    if isinstance(val, str):
        if val.lower() == 'true':
            return 1
        elif val.lower() == 'false':
            return 0
    if isinstance(val, (int, float)):
        return int(val)
    return 0

stats_df['presence'] = cosmos_df['presence'].apply(convert_presence)

# Map appearance_year (0 if not present, otherwise the year)
def convert_year(val):
    if pd.isna(val) or val == '':
        return 0
    try:
        year = int(float(val))
        return year if year > 0 else 0
    except (ValueError, TypeError):
        return 0

stats_df['appearance_year'] = cosmos_df.apply(
    lambda row: 0 if row['presence'] == False or convert_presence(row['presence']) == 0
    else convert_year(row['appearance_year']),
    axis=1
)

# Map shape_transition_year_before and shape_transition_year_after
stats_df['shape_transition_year_before'] = cosmos_df['shape_transition_year_before'].apply(convert_year)
stats_df['shape_transition_year_after'] = cosmos_df['shape_transition_year_after'].apply(convert_year)

# Save to CSV
output_path = '/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/temporal-analysis/model_prediction_csv/cosmos_kiln_stats_delhi_video_bbox_cleaned.csv'
stats_df.to_csv(output_path, index=False)

print(f"\nConversion complete! Output saved to: {output_path}")
print(f"\nTotal records: {len(stats_df)}")
print(f"\nFirst few rows:")
print(stats_df.head(10))
print(f"\nLast few rows:")
print(stats_df.tail(10))
print(f"\nSummary statistics:")
print(f"Presence = 1: {stats_df['presence'].sum()}")
print(f"Presence = 0: {(stats_df['presence'] == 0).sum()}")
print(f"Kilns with transitions: {((stats_df['shape_transition_year_before'] != 0) | (stats_df['shape_transition_year_after'] != 0)).sum()}")
print(f"\nAppearance year distribution:")
print(stats_df[stats_df['presence'] == 1]['appearance_year'].value_counts().sort_index())
