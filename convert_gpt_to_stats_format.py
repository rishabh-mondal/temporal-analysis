import pandas as pd

# Read the GPT predictions CSV
gpt_df = pd.read_csv('/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/temporal-analysis/gpt5_kiln_change_results_delhi_all_loc_bbox.csv')

# Create the new dataframe with the required columns
stats_df = pd.DataFrame()

# Map the columns
stats_df['lat_lon'] = gpt_df['lat_lon']

# Map presence (True -> 1, False -> 0)
stats_df['presence'] = gpt_df['presence'].astype(int)

# Map appearance_year (0 if not present, otherwise the year)
stats_df['appearance_year'] = gpt_df.apply(
    lambda row: 0 if not row['presence'] else row['appearance_year'],
    axis=1
)

# Map shape_transition_year_before and shape_transition_year_after
stats_df['shape_transition_year_before'] = gpt_df['shape_transition_year_before'].fillna(0).astype(int)
stats_df['shape_transition_year_after'] = gpt_df['shape_transition_year_after'].fillna(0).astype(int)

# Save to CSV
output_path = '/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/temporal-analysis/model_prediction_csv/gpt_kiln_stats_delhi_all_loc_cleaned.csv'
stats_df.to_csv(output_path, index=False)

print(f"Conversion complete! Output saved to: {output_path}")
print(f"\nTotal records: {len(stats_df)}")
print(f"\nFirst few rows:")
print(stats_df.head(10))
print(f"\nSummary statistics:")
print(f"Presence = 1: {stats_df['presence'].sum()}")
print(f"Presence = 0: {(stats_df['presence'] == 0).sum()}")
print(f"Kilns with transitions: {((stats_df['shape_transition_year_before'] != 0) | (stats_df['shape_transition_year_after'] != 0)).sum()}")
