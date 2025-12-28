import os

def create_center_filtered_labels(original_labels_dir, images_dir):
    """
    Create label files keeping only the bounding box closest to the center.
    For images with multiple bboxes, keep only the one with center nearest to (0.5, 0.5).

    Parameters:
    - original_labels_dir: Directory with original label files
    - images_dir: Directory with images (to create output dir name)
    """
    # Create new labels directory
    labels_dir = images_dir.replace('_buf_25m', '_buf_25m_labels_center')
    os.makedirs(labels_dir, exist_ok=True)

    # Get all label files
    label_files = [f for f in os.listdir(original_labels_dir) if f.endswith('.txt')]

    labels_created = 0
    multi_bbox_filtered = 0

    for label_file in label_files:
        label_path = os.path.join(original_labels_dir, label_file)

        # Read all bboxes
        bboxes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # Calculate distance from image center (0.5, 0.5)
                    distance = ((x_center - 0.5)**2 + (y_center - 0.5)**2)**0.5

                    bboxes.append({
                        'class_id': class_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height,
                        'distance': distance,
                        'line': line.strip()
                    })

        if bboxes:
            # Find bbox closest to center
            closest_bbox = min(bboxes, key=lambda b: b['distance'])

            if len(bboxes) > 1:
                multi_bbox_filtered += 1

            # Write only the closest bbox
            output_path = os.path.join(labels_dir, label_file)
            with open(output_path, 'w') as f:
                f.write(f"{closest_bbox['class_id']} {closest_bbox['x_center']:.6f} {closest_bbox['y_center']:.6f} {closest_bbox['width']:.6f} {closest_bbox['height']:.6f}")

            labels_created += 1

    return labels_dir, labels_created, multi_bbox_filtered

# Main execution
years = [2014, 2016, 2018, 2020, 2022, 2024, 2025]
base_dir = '/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/temporal-analysis/data_lucknow'

print("Creating center-filtered labels (keeping only bbox closest to center)")
print("=" * 80)

total_labels = 0
total_filtered = 0

for year in years:
    images_dir = os.path.join(base_dir, f'lucknow_airshed_y_{year}_z_17_buf_25m')
    original_labels_dir = os.path.join(base_dir, f'lucknow_airshed_y_{year}_z_17_buf_25m_labels')

    if not os.path.exists(original_labels_dir):
        print(f"Year {year}: Original labels not found")
        continue

    labels_dir, labels_count, multi_bbox_filtered = create_center_filtered_labels(
        original_labels_dir,
        images_dir
    )

    total_labels += labels_count
    total_filtered += multi_bbox_filtered

    print(f"✓ Year {year}:")
    print(f"    Total labels: {labels_count}")
    print(f"    Images with multiple bboxes (filtered): {multi_bbox_filtered}")
    print(f"    Images with single bbox (unchanged): {labels_count - multi_bbox_filtered}")
    print(f"    Saved to: {labels_dir}")
    print()

print("=" * 80)
print(f"Summary:")
print(f"  Total labels created: {total_labels}")
print(f"  Images with multiple bboxes filtered: {total_filtered} ({100*total_filtered/total_labels if total_labels > 0 else 0:.1f}%)")
print(f"  Images with single bbox (unchanged): {total_labels - total_filtered} ({100*(total_labels-total_filtered)/total_labels if total_labels > 0 else 0:.1f}%)")
print()

# Show example of filtering
print("\nExample of Filtering Process:")
print("-" * 80)
sample_year = 2025
original_labels_dir = os.path.join(base_dir, f'lucknow_airshed_y_{sample_year}_z_17_buf_25m_labels')
center_labels_dir = os.path.join(base_dir, f'lucknow_airshed_y_{sample_year}_z_17_buf_25m_labels_center')

if os.path.exists(original_labels_dir) and os.path.exists(center_labels_dir):
    # Find a file with multiple bboxes
    multi_bbox_file = None
    for label_file in os.listdir(original_labels_dir):
        if label_file.endswith('.txt'):
            label_path = os.path.join(original_labels_dir, label_file)
            with open(label_path, 'r') as f:
                lines = [line for line in f if line.strip()]
                if len(lines) > 1:
                    multi_bbox_file = label_file
                    break

    if multi_bbox_file:
        print(f"\nFile: {multi_bbox_file}")
        print("\nOriginal labels (multiple bboxes):")
        with open(os.path.join(original_labels_dir, multi_bbox_file), 'r') as f:
            for i, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) >= 5:
                    x, y = float(parts[1]), float(parts[2])
                    dist = ((x - 0.5)**2 + (y - 0.5)**2)**0.5
                    print(f"  Bbox {i}: class={parts[0]}, center=({x:.4f}, {y:.4f}), distance_from_center={dist:.4f}")

        print("\nCenter-filtered label (only closest bbox):")
        with open(os.path.join(center_labels_dir, multi_bbox_file), 'r') as f:
            line = f.read().strip()
            parts = line.split()
            if len(parts) >= 5:
                x, y = float(parts[1]), float(parts[2])
                dist = ((x - 0.5)**2 + (y - 0.5)**2)**0.5
                print(f"  Bbox: class={parts[0]}, center=({x:.4f}, {y:.4f}), distance_from_center={dist:.4f}")
                print(f"  → Kept this bbox because it's closest to image center (0.5, 0.5)")

print("\n" + "=" * 80)
print("✓ Center-filtered labels created successfully!")
print("  Run the notebook to visualize the results.")
