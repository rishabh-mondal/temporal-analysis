#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score


# --------------------
# PATHS
# --------------------
GT_CSV = Path(
    "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/"
    "temporal-analysis/delhi_airshed_GrountdTruth.csv"
)

PRED_CSV = Path(
    "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/"
    "temporal-analysis/yolo_prediction.csv"
)

SAVE_DIR = Path("./yolo_eval_results")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------
# LOAD + MERGE
# --------------------
gt = pd.read_csv(GT_CSV)
pred = pd.read_csv(PRED_CSV)

df = gt.merge(pred, on="filename", how="inner")
print(f"Rows evaluated: {len(df)}")


# --------------------
# METRIC FUNCTION
# --------------------
def evaluate(gt_col: str, pred_col: str):
    y_true = df[gt_col].astype(str)
    y_pred = df[pred_col].astype(str)

    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro")
    weighted = f1_score(y_true, y_pred, average="weighted")

    return acc, macro, weighted


# --------------------
# 1. APPEARANCE YEAR
# --------------------
acc_a, macro_a, w_a = evaluate(
    "Year_made_Category",
    "appearance_year"
)

print("\nAppearance Year")
print(f"Accuracy    : {acc_a:.4f}")
print(f"Macro F1    : {macro_a:.4f}")
print(f"Weighted F1 : {w_a:.4f}")


# --------------------
# 2. TRANSITION YEAR
# --------------------
acc_t, macro_t, w_t = evaluate(
    "fcb_to_zigzag_Category",
    "Transition_year"
)

print("\nTransition Year")
print(f"Accuracy    : {acc_t:.4f}")
print(f"Macro F1    : {macro_t:.4f}")
print(f"Weighted F1 : {w_t:.4f}")


# --------------------
# 3. CLASS NAME
# --------------------
acc_c, macro_c, w_c = evaluate(
    "class_name_x",
    "class_name_y"
)

print("\nClass Name")
print(f"Accuracy    : {acc_c:.4f}")
print(f"Macro F1    : {macro_c:.4f}")
print(f"Weighted F1 : {w_c:.4f}")


# --------------------
# SAVE METRICS
# --------------------
metrics = pd.DataFrame(
    [
        ["appearance_year", acc_a, macro_a, w_a],
        ["transition_year", acc_t, macro_t, w_t],
        ["class_name", acc_c, macro_c, w_c],
    ],
    columns=["task", "accuracy", "macro_f1", "weighted_f1"]
)

metrics.to_csv(SAVE_DIR / "metrics.csv", index=False)


# --------------------
# YEAR-WISE BAR PLOTS (GT vs PRED)
# --------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# ===== 1) APPEARANCE YEAR =====
gt_app = df["Year_made_Category"].astype(str)
pred_app = df["appearance_year"].astype(str)

gt_app_counts = gt_app.value_counts().sort_index()
pred_app_counts = pred_app.value_counts().sort_index()

years_app = sorted(set(gt_app_counts.index) | set(pred_app_counts.index))
gt_app_vals = [gt_app_counts.get(y, 0) for y in years_app]
pred_app_vals = [pred_app_counts.get(y, 0) for y in years_app]

x = range(len(years_app))

axes[0].bar(x, gt_app_vals, width=0.4, label="GT", align="center")
axes[0].bar(x, pred_app_vals, width=0.4, label="Pred", align="edge")
axes[0].set_xticks(x)
axes[0].set_xticklabels(years_app, rotation=45)
axes[0].set_title("Appearance Year")
axes[0].set_xlabel("Year Category")
axes[0].set_ylabel("Count")
axes[0].legend()


# ===== 2) TRANSITION YEAR =====
gt_tr = df["fcb_to_zigzag_Category"].astype(str)
pred_tr = df["Transition_year"].astype(str)

gt_tr_counts = gt_tr.value_counts().sort_index()
pred_tr_counts = pred_tr.value_counts().sort_index()

years_tr = sorted(set(gt_tr_counts.index) | set(pred_tr_counts.index))
gt_tr_vals = [gt_tr_counts.get(y, 0) for y in years_tr]
pred_tr_vals = [pred_tr_counts.get(y, 0) for y in years_tr]

x = range(len(years_tr))

axes[1].bar(x, gt_tr_vals, width=0.4, label="GT", align="center")
axes[1].bar(x, pred_tr_vals, width=0.4, label="Pred", align="edge")
axes[1].set_xticks(x)
axes[1].set_xticklabels(years_tr, rotation=45)
axes[1].set_title("Transition Year (FCB → Zigzag)")
axes[1].set_xlabel("Year Category")
axes[1].legend()

plt.tight_layout()
plt.savefig(SAVE_DIR / "appearance_and_transition_year_distribution.pdf", dpi=300)
plt.show()