#!/usr/bin/env python3
"""
Streamlit app for building Lucknow ground truth dataset
by verifying and correcting Gemini model predictions
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image
import json

# =========================
# CONFIG
# =========================
YEARS = [2014, 2016, 2018, 2020, 2022, 2024]
DATA_DIR = Path("/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/temporal-analysis/data_lucknow")
FOLDER_TPL = "lucknow_airshed_y_{y}_z_17_buf_25m"
MODEL_PREDICTIONS_CSV = Path("/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/temporal-analysis/gemini-3m-pro_kiln_lucknow_only.csv")
GT_CSV = Path("/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/temporal-analysis/lucknow_ground_truth.csv")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_predictions():
    """Load model predictions CSV"""
    df = pd.read_csv(MODEL_PREDICTIONS_CSV)
    return df

@st.cache_data
def load_or_create_gt():
    """Load existing GT or create new one"""
    if GT_CSV.exists():
        return pd.read_csv(GT_CSV)
    else:
        # Create new GT CSV with required columns
        df = pd.DataFrame(columns=[
            'lat_lon', 'lat', 'lon',
            'presence', 'appearance_year',
            'type_transition_year_before', 'type_transition_year_after',
            'verified_by', 'notes'
        ])
        return df

def get_image_path(lat_lon: str, year: int) -> Path:
    """Get image path for a location and year"""
    folder = DATA_DIR / FOLDER_TPL.format(y=year)
    lat, lon = lat_lon.split('_')
    img_path = folder / f"{lat}_{lon}_{year}.png"
    return img_path

def load_images_for_location(lat_lon: str):
    """Load all year images for a location"""
    images = {}
    for year in YEARS:
        img_path = get_image_path(lat_lon, year)
        if img_path.exists():
            images[year] = Image.open(img_path)
        else:
            images[year] = None
    return images

def save_gt_entry(lat_lon: str, lat: float, lon: float,
                  presence: bool, appearance_year: int,
                  type_transition_year_before: int,
                  type_transition_year_after: int,
                  notes: str = ""):
    """Save or update GT entry"""
    gt_df = load_or_create_gt()

    # Check if entry exists
    if lat_lon in gt_df['lat_lon'].values:
        # Update existing entry
        idx = gt_df[gt_df['lat_lon'] == lat_lon].index[0]
        gt_df.loc[idx, 'presence'] = presence
        gt_df.loc[idx, 'appearance_year'] = appearance_year
        gt_df.loc[idx, 'type_transition_year_before'] = type_transition_year_before
        gt_df.loc[idx, 'type_transition_year_after'] = type_transition_year_after
        gt_df.loc[idx, 'notes'] = notes
        gt_df.loc[idx, 'verified_by'] = 'manual'
    else:
        # Add new entry
        new_row = {
            'lat_lon': lat_lon,
            'lat': lat,
            'lon': lon,
            'presence': presence,
            'appearance_year': appearance_year,
            'type_transition_year_before': type_transition_year_before,
            'type_transition_year_after': type_transition_year_after,
            'verified_by': 'manual',
            'notes': notes
        }
        gt_df = pd.concat([gt_df, pd.DataFrame([new_row])], ignore_index=True)

    # Save to CSV
    GT_CSV.parent.mkdir(parents=True, exist_ok=True)
    gt_df.to_csv(GT_CSV, index=False)

    # Clear cache to reload
    load_or_create_gt.clear()

    return True

def accept_model_prediction(row):
    """Accept model prediction as GT"""
    return save_gt_entry(
        lat_lon=row['lat_lon'],
        lat=row['lat'],
        lon=row['lon'],
        presence=bool(row['presence']),
        appearance_year=int(row['appearance_year']) if pd.notna(row['appearance_year']) else 0,
        type_transition_year_before=int(row['type_transition_year_before']) if pd.notna(row['type_transition_year_before']) else 0,
        type_transition_year_after=int(row['type_transition_year_after']) if pd.notna(row['type_transition_year_after']) else 0,
        notes="Accepted from model prediction"
    )

# =========================
# STREAMLIT APP
# =========================
def main():
    st.set_page_config(page_title="Lucknow GT Builder", layout="wide")

    st.title("🏭 Lucknow Ground Truth Dataset Builder")
    st.markdown("Verify and correct Gemini model predictions to build ground truth dataset")

    # Load data
    predictions_df = load_predictions()
    gt_df = load_or_create_gt()

    # Sidebar - Progress and Navigation
    st.sidebar.header("📊 Progress")
    total_locations = len(predictions_df)
    verified_locations = len(gt_df)
    st.sidebar.metric("Verified", f"{verified_locations}/{total_locations}")
    st.sidebar.progress(verified_locations / total_locations if total_locations > 0 else 0)

    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Navigation")

    # Filter options
    show_mode = st.sidebar.radio(
        "Show:",
        ["Unverified only", "All locations", "Verified only"]
    )

    # Get locations to show
    verified_locs = set(gt_df['lat_lon'].values) if len(gt_df) > 0 else set()

    if show_mode == "Unverified only":
        display_df = predictions_df[~predictions_df['lat_lon'].isin(verified_locs)]
    elif show_mode == "Verified only":
        display_df = predictions_df[predictions_df['lat_lon'].isin(verified_locs)]
    else:
        display_df = predictions_df

    if len(display_df) == 0:
        st.warning(f"No locations to show in '{show_mode}' mode")
        return

    # Location selector
    location_idx = st.sidebar.number_input(
        "Location Index",
        min_value=0,
        max_value=len(display_df)-1,
        value=0,
        step=1
    )

    # Get current location
    current_row = display_df.iloc[location_idx]
    lat_lon = current_row['lat_lon']

    # Show if already verified
    is_verified = lat_lon in verified_locs
    if is_verified:
        st.sidebar.success("✅ This location is already verified")
        # Show GT values
        gt_row = gt_df[gt_df['lat_lon'] == lat_lon].iloc[0]
        st.sidebar.markdown("**Ground Truth:**")
        st.sidebar.write(f"Presence: {gt_row['presence']}")
        st.sidebar.write(f"Appearance Year: {gt_row['appearance_year']}")
        st.sidebar.write(f"Type Transition: {gt_row['type_transition_year_before']} → {gt_row['type_transition_year_after']}")
    else:
        st.sidebar.info("⏳ Not verified yet")

    st.sidebar.markdown("---")

    # Quick navigation buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        prev_disabled = location_idx == 0
        if st.button("⬅️ Previous", disabled=prev_disabled):
            st.session_state.location_idx = location_idx - 1
            st.rerun()
    with col2:
        next_disabled = location_idx >= len(display_df) - 1
        if st.button("Next ➡️", disabled=next_disabled):
            st.session_state.location_idx = location_idx + 1
            st.rerun()

    # Keyboard shortcuts hint
    st.sidebar.markdown("💡 **Quick Actions:**")
    st.sidebar.markdown("- Press 'A' to accept prediction")
    st.sidebar.markdown("- Use ← → for navigation")

    # Main content
    st.header(f"📍 Location: {lat_lon}")
    st.markdown(f"**Lat:** {current_row['lat']:.6f} | **Lon:** {current_row['lon']:.6f}")

    # Display images
    st.subheader("🖼️ Satellite Images Across Years")

    images = load_images_for_location(lat_lon)

    # Display images in columns
    cols = st.columns(len(YEARS))
    for i, year in enumerate(YEARS):
        with cols[i]:
            st.markdown(f"**{year}**")
            if images[year] is not None:
                st.image(images[year], use_column_width=True)
            else:
                st.error("Image not found")

    st.markdown("---")

    # Model Predictions
    st.subheader("🤖 Model Predictions (Gemini 3 Pro)")

    pred_col1, pred_col2, pred_col3, pred_col4 = st.columns(4)

    with pred_col1:
        st.metric("Presence", "✅ Yes" if current_row['presence'] else "❌ No")
    with pred_col2:
        st.metric("Appearance Year",
                  current_row['appearance_year'] if current_row['appearance_year'] != 0 else "N/A")
    with pred_col3:
        st.metric("Type Transition Before",
                  current_row['type_transition_year_before'] if current_row['type_transition_year_before'] != 0 else "N/A")
    with pred_col4:
        st.metric("Type Transition After",
                  current_row['type_transition_year_after'] if current_row['type_transition_year_after'] != 0 else "N/A")

    # Show additional model info
    with st.expander("📄 View Full Model Prediction Details"):
        st.write(f"**Appearance Type:** {current_row.get('appearance_type', 'N/A')}")
        st.write(f"**Type Transition Note:** {current_row.get('type_transition_note', 'N/A')}")
        st.write(f"**Shape Transition:** {current_row.get('shape_transition_year_before', 0)} → {current_row.get('shape_transition_year_after', 0)}")
        st.write(f"**Shape Transition Note:** {current_row.get('shape_transition_note', 'N/A')}")
        st.write(f"**Demolished:** {current_row.get('demolished', False)}")
        st.write(f"**Demolished Year:** {current_row.get('demolished_year', 0)}")
        st.write(f"**Negative Sample:** {current_row.get('negative_sample', False)}")
        st.write(f"**Confidence:** {current_row.get('confidence', 'N/A')}")
        st.write(f"**Monitoring Note:** {current_row.get('monitoring_note_one_line', 'N/A')}")

    st.markdown("---")

    # Manual Entry / Correction
    st.subheader("✏️ Ground Truth Entry")

    # Option to accept model prediction - LARGE BUTTON
    col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 2])

    with col_btn1:
        if st.button("✅ ACCEPT & NEXT", type="primary", key="accept_next"):
            if accept_model_prediction(current_row):
                st.session_state.location_idx = min(location_idx + 1, len(display_df) - 1)
                st.success("✅ Accepted!")
                st.rerun()

    with col_btn2:
        if st.button("❌ REJECT (Skip)", key="reject_skip"):
            st.session_state.location_idx = min(location_idx + 1, len(display_df) - 1)
            st.info("⏭️ Skipped")
            st.rerun()

    with col_btn3:
        if st.button("✏️ MODIFY BELOW", key="modify"):
            st.info("👇 Edit values below and click Save")

    # Manual entry form - collapsible
    with st.expander("✏️ Manual Entry / Correction Form", expanded=False):
        st.markdown("Use this only if you need to modify the prediction")
        with st.form(key=f"gt_form_{lat_lon}"):
            form_col1, form_col2, form_col3, form_col4 = st.columns(4)

            with form_col1:
                presence_input = st.selectbox(
                    "Presence*",
                    options=[True, False],
                    index=0 if current_row['presence'] else 1,
                    format_func=lambda x: "✅ Yes" if x else "❌ No"
                )

            with form_col2:
                appearance_year_options = [0] + YEARS
                try:
                    appearance_year_val = int(current_row['appearance_year']) if pd.notna(current_row['appearance_year']) else 0
                    appearance_year_idx = appearance_year_options.index(appearance_year_val) if appearance_year_val in appearance_year_options else 0
                except:
                    appearance_year_idx = 0

                appearance_year_input = st.selectbox(
                    "Appearance Year*",
                    options=appearance_year_options,
                    index=appearance_year_idx,
                    format_func=lambda x: "N/A" if x == 0 else str(x)
                )

            with form_col3:
                type_before_options = [0] + YEARS
                try:
                    type_before_val = int(current_row['type_transition_year_before']) if pd.notna(current_row['type_transition_year_before']) else 0
                    type_before_idx = type_before_options.index(type_before_val) if type_before_val in type_before_options else 0
                except:
                    type_before_idx = 0

                type_transition_before_input = st.selectbox(
                    "Type Transition Before",
                    options=type_before_options,
                    index=type_before_idx,
                    format_func=lambda x: "N/A" if x == 0 else str(x)
                )

            with form_col4:
                type_after_options = [0] + YEARS
                try:
                    type_after_val = int(current_row['type_transition_year_after']) if pd.notna(current_row['type_transition_year_after']) else 0
                    type_after_idx = type_after_options.index(type_after_val) if type_after_val in type_after_options else 0
                except:
                    type_after_idx = 0

                type_transition_after_input = st.selectbox(
                    "Type Transition After",
                    options=type_after_options,
                    index=type_after_idx,
                    format_func=lambda x: "N/A" if x == 0 else str(x)
                )

            notes_input = st.text_area(
                "Notes (optional)",
                placeholder="Add any observations or notes about this location...",
                height=100
            )

            col_save1, col_save2 = st.columns([1, 1])
            with col_save1:
                submit_button = st.form_submit_button("💾 Save Ground Truth", type="primary")
            with col_save2:
                save_and_next = st.form_submit_button("💾 Save & Next")

            if submit_button or save_and_next:
                # Validate
                if presence_input and appearance_year_input == 0:
                    st.error("⚠️ If presence is True, appearance year must be set!")
                elif type_transition_before_input != 0 and type_transition_after_input == 0:
                    st.error("⚠️ If type transition before is set, after must also be set!")
                elif type_transition_before_input == 0 and type_transition_after_input != 0:
                    st.error("⚠️ If type transition after is set, before must also be set!")
                else:
                    # Save
                    if save_gt_entry(
                        lat_lon=lat_lon,
                        lat=current_row['lat'],
                        lon=current_row['lon'],
                        presence=presence_input,
                        appearance_year=appearance_year_input,
                        type_transition_year_before=type_transition_before_input,
                        type_transition_year_after=type_transition_after_input,
                        notes=notes_input
                    ):
                        st.success(f"✅ Ground truth saved for {lat_lon}!")
                        if save_and_next:
                            st.session_state.location_idx = min(location_idx + 1, len(display_df) - 1)
                        st.rerun()

    # Download GT CSV
    st.sidebar.markdown("---")
    st.sidebar.header("💾 Download")
    if GT_CSV.exists():
        with open(GT_CSV, 'rb') as f:
            st.sidebar.download_button(
                label="📥 Download GT CSV",
                data=f,
                file_name="lucknow_ground_truth.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
