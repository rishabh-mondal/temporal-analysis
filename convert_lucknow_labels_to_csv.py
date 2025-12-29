#!/usr/bin/env python3
"""
Convert Lucknow label files to CSV format similar to qwen3_30b_bbox_pixels_2025.csv

Input: Label files in YOLO format (class x_center y_center width height) - normalized coordinates
Output: CSV with columns: lat, lon, lat_lon, bbox_json (pixel coordinates)

Assumes image size: 256x256 pixels
"""

import os
import json
import pandas as pd
from pathlib import Path

# Paths
labels_dir = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/temporal-analysis/data_lucknow/lucknow_airshed_y_2025_z_17_buf_25m_labels_center"
output_csv = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/temporal-analysis/lucknow_labels_bbox_pixels_2025.csv"

# Image dimensions
IMG_WIDTH = 256
IMG_HEIGHT = 256

def yolo_to_pixels(x_center, y_center, width, height, img_w, img_h):
    """
    Convert YOLO normalized coordinates to pixel coordinates [x_min, y_min, x_max, y_max]

    YOLO format: x_center, y_center, width, height (all normalized 0-1)
    Pixel format: x_min, y_min, x_max, y_max (absolute pixel values)
    """
    # Convert from normalized to pixel coordinates
    x_center_px = x_center * img_w
    y_center_px = y_center * img_h
    width_px = width * img_w
    height_px = height * img_h

    # Calculate bounding box corners
    x_min = x_center_px - (width_px / 2)
    y_min = y_center_px - (height_px / 2)
    x_max = x_center_px + (width_px / 2)
    y_max = y_center_px + (height_px / 2)

    return [x_min, y_min, x_max, y_max]

print("Processing Lucknow label files...")
print(f"Labels directory: {labels_dir}")

# Process all label files
data = []

for label_file in sorted(Path(labels_dir).glob("*.txt")):
    filename = label_file.stem  # e.g., "26.601117_81.061004_2025"

    # Extract lat, lon from filename
    parts = filename.replace("_2025", "").split("_")
    if len(parts) != 2:
        print(f"Warning: Skipping invalid filename: {label_file.name}")
        continue

    lat = float(parts[0])
    lon = float(parts[1])
    lat_lon = f"{lat:.6f}_{lon:.6f}"

    # Read bounding boxes from file
    bboxes = []

    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # Convert to pixel coordinates
            bbox_pixels = yolo_to_pixels(x_center, y_center, width, height, IMG_WIDTH, IMG_HEIGHT)
            bboxes.append(bbox_pixels)

    # If no bounding boxes, add a zero bbox
    if not bboxes:
        bboxes = [[0.0, 0.0, 0.0, 0.0]]

    # Create record
    data.append({
        'lat': lat,
        'lon': lon,
        'lat_lon': lat_lon,
        'bbox_json': json.dumps(bboxes)
    })

print(f"\nProcessed {len(data)} locations")

# Create DataFrame
df = pd.DataFrame(data)

# Sort by lat, lon
df = df.sort_values(['lat', 'lon']).reset_index(drop=True)

# Save to CSV
df.to_csv(output_csv, index=False)

print(f"\nSaved to: {output_csv}")
print(f"Shape: {df.shape}")

# Show statistics
total_locations = len(df)
locations_with_kilns = df[df['bbox_json'] != '[[0.0, 0.0, 0.0, 0.0]]'].shape[0]
locations_without_kilns = total_locations - locations_with_kilns

print(f"\nStatistics:")
print(f"  Total locations: {total_locations}")
print(f"  Locations with kilns: {locations_with_kilns} ({locations_with_kilns/total_locations*100:.2f}%)")
print(f"  Locations without kilns: {locations_without_kilns} ({locations_without_kilns/total_locations*100:.2f}%)")

# Show sample
print(f"\nSample records:")
print(df.head(10))
