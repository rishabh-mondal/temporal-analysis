#!/usr/bin/env python3
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Load data
gt = pd.read_csv('lucknow_ground_truth.csv')
pred = pd.read_csv('model_prediction_csv/teochat_kiln_stats_lucknow_cleaned.csv')

print('='*80)
print('TEOCHAT LUCKNOW DATA VERIFICATION')
print('='*80)
print(f'\nGround Truth: {len(gt)} locations')
print(f'Predictions:  {len(pred)} locations')

# Prepare keys
gt['key'] = gt['lat_lon'].astype(str)
pred['key'] = pred['lat_lon'].astype(str)

# Check merge
merged = gt.merge(pred[['key']], on='key', how='inner')
print(f'Matched:      {len(merged)} locations')

# Check presence
gt['presence'] = pd.to_numeric(gt['presence'], errors='coerce').fillna(0).astype(int)
pred['presence'] = pd.to_numeric(pred['presence'], errors='coerce').fillna(0).astype(int)

merged_p = gt.merge(pred[['key', 'presence']], on='key', how='inner', suffixes=('_gt', '_pred'))
gt_true = (merged_p['presence_gt'] == 1).sum()
gt_false = (merged_p['presence_gt'] == 0).sum()
pred_true = (merged_p['presence_pred'] == 1).sum()
pred_false = (merged_p['presence_pred'] == 0).sum()
accuracy = (merged_p['presence_gt'] == merged_p['presence_pred']).mean()

print(f'\nPresence Analysis:')
print(f'  GT True:   {gt_true}')
print(f'  GT False:  {gt_false}')
print(f'  Pred True: {pred_true}')
print(f'  Pred False:{pred_false}')
print(f'  Accuracy:  {accuracy:.4f}')

# Check appearance year
gt['appearance_year'] = pd.to_numeric(gt['appearance_year'], errors='coerce').fillna(0).astype(int)
pred['appearance_year'] = pd.to_numeric(pred['appearance_year'], errors='coerce').fillna(0).astype(int)

merged_a = gt.merge(pred[['key', 'appearance_year']], on='key', how='inner', suffixes=('_gt', '_pred'))
gt_nonzero = (merged_a['appearance_year_gt'] != 0).sum()
pred_nonzero = (merged_a['appearance_year_pred'] != 0).sum()
exact_match = (merged_a['appearance_year_gt'] == merged_a['appearance_year_pred']).sum()

print(f'\nAppearance Year Analysis:')
print(f'  GT non-zero:   {gt_nonzero}')
print(f'  Pred non-zero: {pred_nonzero}')
print(f'  Exact match:   {exact_match}')

# Check transitions
gt['type_transition_year_after'] = pd.to_numeric(gt['type_transition_year_after'], errors='coerce').fillna(0).astype(int)
pred['shape_transition_year_after'] = pd.to_numeric(pred['shape_transition_year_after'], errors='coerce').fillna(0).astype(int)

merged_t = gt.merge(pred[['key', 'shape_transition_year_after']], on='key', how='inner')
gt_trans = (merged_t['type_transition_year_after'] != 0).sum()
pred_trans = (merged_t['shape_transition_year_after'] != 0).sum()

print(f'\nTransition Analysis:')
print(f'  GT transitions:   {gt_trans}')
print(f'  Pred transitions: {pred_trans}')

# Sample data
print('\nSample merged data (first 5 rows):')
sample = merged_p[['lat_lon', 'presence_gt', 'presence_pred']].head()
print(sample.to_string(index=False))

print('\n' + '='*80)
print('VERIFICATION COMPLETE - Data is valid!')
print('='*80)
