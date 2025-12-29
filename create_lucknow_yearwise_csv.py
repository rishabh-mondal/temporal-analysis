#!/usr/bin/env python3
"""
Create a year-wise CSV for Lucknow predictions similar to yolo_prediction_by_year.csv
"""
import csv
from pathlib import Path
from collections import defaultdict

# Configuration
BASE_DIR = Path("/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/temporal-analysis")
INFERENCE_DIR = BASE_DIR / "runs/inference_lucknow"
OUTPUT_CSV = BASE_DIR / "yolo_prediction_lucknow_by_year.csv"

YEARS = [2014, 2016, 2018, 2020, 2022, 2024]
CLASS_NAMES = {
    0: "FCBK",
    1: "Zigzag"
}

def get_prediction_for_image(label_file: Path) -> str:
    """
    Read YOLO label file and return the prediction.
    If file doesn't exist or is empty -> "negetive"
    If class 0 -> "FCBK"
    If class 1 -> "Zigzag"
    """
    if not label_file.exists():
        return "negetive"

    content = label_file.read_text().strip()
    if not content:
        return "negetive"

    # Parse first line (in case of multiple detections, take the first one)
    lines = content.strip().split('\n')
    first_line = lines[0].strip()

    if not first_line:
        return "negetive"

    parts = first_line.split()
    class_id = int(parts[0])

    return CLASS_NAMES.get(class_id, "negetive")


def extract_base_filename(image_path: Path, year: int) -> str:
    """
    Extract the base filename without year suffix.
    E.g., "26.601117_81.061004_2014.jpg" -> "26.601117_81.061004.jpg"
    """
    stem = image_path.stem  # "26.601117_81.061004_2014"
    # Remove the year suffix
    if stem.endswith(f"_{year}"):
        stem = stem[:-5]  # Remove "_YYYY"
    return f"{stem}.jpg"


def main():
    print("Creating year-wise CSV for Lucknow predictions...")

    # Collect all unique image base filenames across all years
    all_images = set()

    for year in YEARS:
        year_dir = INFERENCE_DIR / f"lucknow_{year}"
        if year_dir.exists():
            image_files = list(year_dir.glob("*.jpg")) + list(year_dir.glob("*.png"))
            for img in image_files:
                base_name = extract_base_filename(img, year)
                all_images.add(base_name)
            print(f"  Year {year}: {len(image_files)} images found")

    print(f"\nTotal unique images: {len(all_images)}")

    # Sort images for consistent output
    sorted_images = sorted(all_images)

    # Build predictions dictionary
    predictions = {}

    for base_name in sorted_images:
        predictions[base_name] = {}

        for year in YEARS:
            year_dir = INFERENCE_DIR / f"lucknow_{year}"
            labels_dir = year_dir / "labels"

            # Reconstruct the filename with year suffix
            stem = base_name.replace('.jpg', '').replace('.png', '')
            label_file = labels_dir / f"{stem}_{year}.txt"

            prediction = get_prediction_for_image(label_file)
            predictions[base_name][year] = prediction

    # Write CSV
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        header = ['filename'] + [str(year) for year in YEARS]
        writer.writerow(header)

        # Write data rows
        for base_name in sorted_images:
            row = [base_name]
            for year in YEARS:
                row.append(predictions[base_name].get(year, "negetive"))
            writer.writerow(row)

    print(f"\nCSV created successfully: {OUTPUT_CSV}")
    print(f"Total rows: {len(sorted_images)}")

    # Print some statistics
    stats = defaultdict(lambda: defaultdict(int))
    for base_name in sorted_images:
        for year in YEARS:
            pred = predictions[base_name][year]
            stats[year][pred] += 1

    print("\nPrediction statistics by year:")
    for year in YEARS:
        print(f"\n{year}:")
        for pred_type in ["negetive", "FCBK", "Zigzag"]:
            count = stats[year][pred_type]
            print(f"  {pred_type}: {count}")


if __name__ == "__main__":
    main()
