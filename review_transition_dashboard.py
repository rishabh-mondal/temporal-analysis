#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


# =========================================================
# CONFIG
# =========================================================
PRED_CSV = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/temporal-analysis/vlm_kiln_change_results_753_loc.csv"
GT_CSV   = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/temporal-analysis/images_753_with_Year_Categories_fixed_zero.csv"

BASE_DATA_DIR = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/temporal-analysis/data"
FOLDER_TPL = "delhi_airshed_y_{y}_z_17_buf_25m"
YEARS = [2014, 2016, 2018, 2020, 2022, 2024, 2025]


# =========================================================
# STREAMLIT COMPAT
# =========================================================
def st_image_compat(img, caption=None):
    try:
        st.image(img, use_container_width=True, caption=caption)
    except TypeError:
        st.image(img, use_column_width=True, caption=caption)


# =========================================================
# HELPERS (ROBUST)
# =========================================================
def clean_float(x):
    if x is None:
        return None
    x = re.sub(r"[^0-9.]", "", str(x))
    x = re.sub(r"\.+$", "", x)
    if x == "":
        return None
    return float(x)


def make_key(lat, lon, ndp=6):
    lat = clean_float(lat)
    lon = clean_float(lon)
    if lat is None or lon is None:
        return None
    return f"{lat:.{ndp}f}_{lon:.{ndp}f}"


def key_from_filename(fn, ndp=6):
    if not isinstance(fn, str):
        return None
    m = re.search(r"([0-9.]+)_([0-9.]+)", fn)
    if not m:
        return None
    return make_key(m.group(1), m.group(2), ndp)


def find_image(base_dir, tpl, key, year):
    p = Path(base_dir) / tpl.format(y=year) / f"{key}_{year}.png"
    return p if p.exists() else None


# =========================================================
# LOAD DATA (ONCE)
# =========================================================
@st.cache_data(show_spinner=False)
def load_initial_data():
    pred = pd.read_csv(PRED_CSV)
    gt = pd.read_csv(GT_CSV)

    if "lat_lon" in pred.columns:
        pred["key"] = pred["lat_lon"].astype(str)
    else:
        pred["key"] = pred.apply(lambda r: make_key(r["lat"], r["lon"]), axis=1)

    gt["key"] = gt["filename"].apply(key_from_filename)

    df = gt.merge(pred, on="key", how="left")

    df["use_gt_for_fcb_to_zigzag_Category"] = False
    df["reviewed"] = False
    df["manual_transition_year"] = np.nan

    return df, gt


# =========================================================
# APP
# =========================================================
st.set_page_config(layout="wide")
st.title("Kiln Transition Review Dashboard — CORRECT")


# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    pred_col = st.selectbox(
        "Model transition year column",
        [
            "type_transition_year_after",
            "type_transition_year_before",
            "shape_transition_year_after",
            "shape_transition_year_before",
        ],
    )
    auto_jump = st.checkbox("Auto-jump to next unreviewed", value=True)


# =========================================================
# PERSISTENT STATE
# =========================================================
if "review_df" not in st.session_state:
    review_df, gt_df = load_initial_data()
    st.session_state.review_df = review_df
    st.session_state.gt_df = gt_df

review = st.session_state.review_df
review["model_pred_transition_year"] = review[pred_col]


# =========================================================
# PROGRESS
# =========================================================
total = len(review)
done = int(review["reviewed"].sum())
st.progress(done / total if total else 0)
st.caption(f"Reviewed {done} / {total}")


# =========================================================
# AUTO-SELECT NEXT UNREVIEWED
# =========================================================
unreviewed_keys = review.loc[~review["reviewed"], "key"].dropna().unique().tolist()

if "selected_key" not in st.session_state:
    st.session_state.selected_key = unreviewed_keys[0] if unreviewed_keys else None

if st.session_state.selected_key not in unreviewed_keys and unreviewed_keys:
    st.session_state.selected_key = unreviewed_keys[0]


# =========================================================
# SELECT LOCATION
# =========================================================
st.subheader("Inspect Location")

selected = st.selectbox(
    "Location",
    review["key"].dropna().unique().tolist(),
    index=review["key"].dropna().tolist().index(st.session_state.selected_key),
)

st.session_state.selected_key = selected
row = review.loc[review["key"] == selected].iloc[0]

# =========================================================
# IMAGES
# =========================================================
st.subheader("Yearly Images (2014–2025)")

cols = st.columns(len(YEARS))
for i, y in enumerate(YEARS):
    with cols[i]:
        st.write(y)
        p = find_image(BASE_DATA_DIR, FOLDER_TPL, selected, y)
        if p:
            st_image_compat(Image.open(p))
        else:
            st.caption("missing")
# =========================================================
# QUICK DECISION
# =========================================================
b1, b2, b3 = st.columns(3)

if b1.button("Use GT"):
    review.loc[review["key"] == selected, ["use_gt_for_fcb_to_zigzag_Category", "reviewed"]] = [True, True]

if b2.button("Use Model"):
    review.loc[review["key"] == selected, ["use_gt_for_fcb_to_zigzag_Category", "reviewed"]] = [False, True]

if b3.button("Skip"):
    pass


# =========================================================
# MANUAL OVERRIDE (GT ≠ MODEL)
# =========================================================
gt_year = row["fcb_to_zigzag_Category"]
model_year = row["model_pred_transition_year"]

if not pd.isna(gt_year) and not pd.isna(model_year) and gt_year != model_year:
    st.subheader("Manual override (GT and Model both incorrect)")

    manual_year = st.number_input(
        "Enter correct transition year",
        min_value=1990,
        max_value=2026,
        step=1,
        value=int(gt_year),
        key=f"manual_{selected}",
    )

    if st.button("Accept manual year"):
        review.loc[
            review["key"] == selected,
            ["manual_transition_year", "reviewed"]
        ] = [manual_year, True]


# =========================================================
# AUTO-JUMP
# =========================================================
if auto_jump and review.loc[review["key"] == selected, "reviewed"].iloc[0]:
    remaining = review.loc[~review["reviewed"], "key"].dropna().unique().tolist()
    if remaining:
        st.session_state.selected_key = remaining[0]
        st.rerun()


# =========================================================
# INFO PANEL
# =========================================================
st.code(
    f"""
Key: {row.key}
GT year: {row.fcb_to_zigzag_Category}
Model year: {row.model_pred_transition_year}
Manual year: {row.manual_transition_year}
Reviewed: {row.reviewed}
"""
)





# =========================================================
# SAVE FINAL CSV
# =========================================================
st.subheader("Save Final Evaluation CSV")

def compute_final_year(r):
    if not pd.isna(r["manual_transition_year"]):
        return r["manual_transition_year"]
    if r["use_gt_for_fcb_to_zigzag_Category"]:
        return r["fcb_to_zigzag_Category"]
    return r["model_pred_transition_year"]

if st.button("Save new CSV"):
    out = st.session_state.gt_df.copy()
    out["key"] = out["filename"].apply(key_from_filename)

    out["fcb_to_zigzag_Category"] = out["key"].map(
        dict(zip(review["key"], review.apply(compute_final_year, axis=1)))
    )

    out["manual_transition_year"] = out["key"].map(
        dict(zip(review["key"], review["manual_transition_year"]))
    )

    out["reviewed"] = out["key"].map(dict(zip(review["key"], review["reviewed"])))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(GT_CSV).with_name(f"FINAL_fcb_to_zigzag_{ts}.csv")

    out.drop(columns=["key"]).to_csv(out_path, index=False)
    st.success(f"Saved: {out_path}")
