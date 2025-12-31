#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run TeoChat Lucknow Performance Analysis
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

# Create figures directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

# Helper function to plot confusion matrices
def plot_confusion_matrix(cm, title, figsize=(8, 6), annot=True, fmt='d', cmap='Greens', save_name=None):
    """Plot a confusion matrix as a heatmap"""
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt=fmt, cmap=cmap, cbar=True,
                linewidths=0.5, linecolor='gray')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Ground Truth', fontsize=12)
    plt.xlabel('Prediction', fontsize=12)
    plt.tight_layout()

    if save_name:
        save_path = os.path.join('figures', f'{save_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")

    plt.close()

# Helper functions for metrics
def f1_macro(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    f1s = []
    for label in labels:
        tp = ((y_true == label) & (y_pred == label)).sum()
        fp = ((y_true != label) & (y_pred == label)).sum()
        fn = ((y_true == label) & (y_pred != label)).sum()
        denom = (2 * tp + fp + fn)
        f1s.append((2 * tp / denom) if denom else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0

def f1_weighted(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    total = len(y_true)
    f1_sum = 0.0
    for label in labels:
        tp = ((y_true == label) & (y_pred == label)).sum()
        fp = ((y_true != label) & (y_pred == label)).sum()
        fn = ((y_true == label) & (y_pred != label)).sum()
        denom = (2 * tp + fp + fn)
        f1 = (2 * tp / denom) if denom else 0.0
        support = (y_true == label).sum()
        f1_sum += f1 * support
    return f1_sum / total if total else 0.0

print("="*80)
print("TEOCHAT LUCKNOW PERFORMANCE ANALYSIS")
print("="*80)

# Load data
gt_path = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/temporal-analysis/lucknow_ground_truth.csv"
pred_path = (
    "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/temporal-analysis/"
    "model_prediction_csv/teochat_kiln_stats_lucknow_cleaned.csv"
)

gt = pd.read_csv(gt_path)
pred = pd.read_csv(pred_path)

print(f"\nGround Truth samples: {len(gt)}")
print(f"Prediction samples: {len(pred)}")

# Prepare keys for merging
gt["key"] = gt["lat_lon"].astype(str)
pred["key"] = pred["lat_lon"].astype(str)

print(f"\nGround Truth columns: {gt.columns.tolist()}")
print(f"Prediction columns: {pred.columns.tolist()}")

# =============================================================================
# 1. PRESENCE DETECTION
# =============================================================================
print("\n" + "="*80)
print("1. PRESENCE DETECTION (BINARY)")
print("="*80)

gt["presence"] = pd.to_numeric(gt["presence"], errors="coerce").fillna(0).astype(int)
pred["presence"] = pd.to_numeric(pred["presence"], errors="coerce").fillna(0).astype(int)

merged_presence = gt.merge(pred[["key", "presence"]], on="key", how="inner", suffixes=("_gt", "_pred"))
print(f"Merged rows: {len(merged_presence)}")

cm_presence = pd.crosstab(
    merged_presence["presence_gt"],
    merged_presence["presence_pred"],
    rownames=["Ground Truth"],
    colnames=["Prediction"],
    dropna=False
).reindex(index=[0, 1], columns=[0, 1], fill_value=0)

print("\nConfusion Matrix (Presence):")
print(cm_presence)

TN = int(cm_presence.loc[0, 0])
FP = int(cm_presence.loc[0, 1])
FN = int(cm_presence.loc[1, 0])
TP = int(cm_presence.loc[1, 1])

metrics_presence = {
    "TP": TP,
    "TN": TN,
    "FP": FP,
    "FN": FN,
    "Accuracy": (TP + TN) / max(TP + TN + FP + FN, 1),
    "Precision": TP / max(TP + FP, 1),
    "Recall": TP / max(TP + FN, 1),
    "F1-Score": (2 * TP) / max(2 * TP + FP + FN, 1),
}

print("\nMetrics:")
for k, v in metrics_presence.items():
    if isinstance(v, float):
        print(f"{k}: {v:.4f}")
    else:
        print(f"{k}: {v}")

plot_confusion_matrix(
    cm_presence,
    title='TeoChat Lucknow: Presence Detection',
    figsize=(7, 6),
    cmap='RdPu',
    save_name='teochat_lucknow_presence_confusion_matrix'
)

# =============================================================================
# 2. APPEARANCE YEAR
# =============================================================================
print("\n" + "="*80)
print("2. APPEARANCE YEAR (±2 YEAR TOLERANCE)")
print("="*80)

gt["appearance_year"] = pd.to_numeric(gt["appearance_year"], errors="coerce").fillna(0).astype(int)
pred["appearance_year"] = pd.to_numeric(pred["appearance_year"], errors="coerce").fillna(0).astype(int)

merged_appearance = gt.merge(
    pred[["key", "appearance_year"]],
    on="key",
    how="inner",
    suffixes=("_gt", "_pred"),
)

print(f"Merged rows: {len(merged_appearance)}")

tol = 2
y_true = merged_appearance["appearance_year_gt"]
y_pred = merged_appearance["appearance_year_pred"].copy()

mask = (y_true - y_pred).abs() <= tol
y_pred[mask] = y_true[mask]

print(f"Predictions adjusted: {mask.sum()} out of {len(mask)}")

cm_appearance = pd.crosstab(
    y_true,
    y_pred,
    rownames=["Ground Truth"],
    colnames=["Prediction (±2y)"],
    dropna=False,
)

print("\nConfusion Matrix (Appearance Year with ±2y tolerance):")
print(cm_appearance)

accuracy = (y_true == y_pred).mean()

metrics_appearance = {
    "Accuracy": accuracy,
    "F1-Macro": f1_macro(y_true, y_pred),
    "F1-Weighted": f1_weighted(y_true, y_pred),
}

print("\nMetrics (Appearance Year with ±2y tolerance):")
for k, v in metrics_appearance.items():
    print(f"{k}: {v:.4f}")

plot_confusion_matrix(
    cm_appearance,
    title='TeoChat Lucknow: Appearance Year (±2y Tolerance)',
    figsize=(10, 8),
    cmap='RdPu',
    save_name='teochat_lucknow_appearance_year_tolerance_confusion_matrix'
)

# =============================================================================
# 3. TRANSITION DETECTION
# =============================================================================
print("\n" + "="*80)
print("3. TRANSITION DETECTION (BINARY)")
print("="*80)

gt_col = "type_transition_year_after"
pred_col = "shape_transition_year_after"

gt[gt_col] = pd.to_numeric(gt[gt_col], errors="coerce").fillna(0).astype(int)
pred[pred_col] = pd.to_numeric(pred[pred_col], errors="coerce").fillna(0).astype(int)

merged_transition = gt.merge(
    pred[["key", pred_col]],
    on="key",
    how="inner",
)

print(f"Merged rows: {len(merged_transition)}")

y_true_year = merged_transition[gt_col].values
y_pred_year = merged_transition[pred_col].values

y_true_bin = (y_true_year != 0).astype(int)
y_pred_bin = (y_pred_year != 0).astype(int)

print(f"Ground truth transitions: {y_true_bin.sum()} out of {len(y_true_bin)}")
print(f"Predicted transitions: {y_pred_bin.sum()} out of {len(y_pred_bin)}")

cm_transition = pd.crosstab(
    y_true_bin,
    y_pred_bin,
    rownames=["Ground Truth"],
    colnames=["Prediction"],
).reindex(index=[0, 1], columns=[0, 1], fill_value=0)

print("\nConfusion Matrix (Transition Detection):")
print(cm_transition)

TN_trans = int(cm_transition.loc[0, 0])
FP_trans = int(cm_transition.loc[0, 1])
FN_trans = int(cm_transition.loc[1, 0])
TP_trans = int(cm_transition.loc[1, 1])

metrics_transition = {
    "TP": TP_trans,
    "TN": TN_trans,
    "FP": FP_trans,
    "FN": FN_trans,
    "Accuracy": (TP_trans + TN_trans) / max(TP_trans + TN_trans + FP_trans + FN_trans, 1),
    "Precision": TP_trans / max(TP_trans + FP_trans, 1),
    "Recall": TP_trans / max(TP_trans + FN_trans, 1),
    "F1-Score": (2 * TP_trans) / max(2 * TP_trans + FP_trans + FN_trans, 1),
}

print("\nMetrics (Transition Detection):")
for k, v in metrics_transition.items():
    if isinstance(v, float):
        print(f"{k}: {v:.4f}")
    else:
        print(f"{k}: {v}")

plot_confusion_matrix(
    cm_transition,
    title='TeoChat Lucknow: Transition Detection',
    figsize=(7, 6),
    cmap='PuRd',
    save_name='teochat_lucknow_transition_detection_confusion_matrix'
)

# =============================================================================
# 4. TRANSITION YEAR
# =============================================================================
print("\n" + "="*80)
print("4. TRANSITION YEAR (±2 YEAR TOLERANCE)")
print("="*80)

tol = 2
y_true_trans = merged_transition[gt_col].values
y_pred_trans = merged_transition[pred_col].values.copy()

mask_trans = (y_true_trans != 0) & (y_pred_trans != 0) & (np.abs(y_pred_trans - y_true_trans) <= tol)
y_pred_trans[mask_trans] = y_true_trans[mask_trans]

print(f"Predictions adjusted: {mask_trans.sum()} out of {len(mask_trans)}")

cm_transition_year = pd.crosstab(
    y_true_trans,
    y_pred_trans,
    rownames=["Ground Truth"],
    colnames=["Prediction (±2y)"],
    dropna=False,
)

print("\nConfusion Matrix (Transition Year with ±2y tolerance):")
print(cm_transition_year)

accuracy_trans = (y_true_trans == y_pred_trans).mean()

metrics_transition_year = {
    "Accuracy": accuracy_trans,
    "F1-Macro": f1_macro(y_true_trans, y_pred_trans),
    "F1-Weighted": f1_weighted(y_true_trans, y_pred_trans),
}

print("\nMetrics (Transition Year with ±2y tolerance):")
for k, v in metrics_transition_year.items():
    print(f"{k}: {v:.4f}")

plot_confusion_matrix(
    cm_transition_year,
    title='TeoChat Lucknow: Transition Year (±2y Tolerance)',
    figsize=(10, 8),
    cmap='PuRd',
    save_name='teochat_lucknow_transition_year_tolerance_confusion_matrix'
)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nAll confusion matrices and metrics saved in figures/ directory with teochat_lucknow_ prefix")
