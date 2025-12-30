import os
import shutil
import json

def point_in_polygon(x, y, polygon):
    """
    Ray-casting algorithm to check if a point is inside a polygon.
    polygon should be a list of (x, y) tuples.
    """
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

# Read the Lucknow airshed GeoJSON
geojson_path = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/regions/shapes/lucknow_airshed.geojson"

with open(geojson_path, 'r') as f:
    geojson_data = json.load(f)

# Get the first feature's geometry coordinates
coordinates = geojson_data['features'][0]['geometry']['coordinates'][0]
polygon = [(lon, lat) for lon, lat in coordinates]

print(f"Lucknow airshed polygon with {len(polygon)} vertices loaded")

# Base directories
data_all_dir = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/temporal-analysis/data_all"
data_lucknow_dir = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/temporal-analysis/data_lucknow"

# Years to process
years = [2014, 2016, 2018, 2020, 2022, 2024, 2025]

# Create data_lucknow directory if it doesn't exist
os.makedirs(data_lucknow_dir, exist_ok=True)

print(f"Processing images for Lucknow airshed across {len(years)} years...")
print("-" * 80)

total_copied = 0

for year in years:
    # Source and destination directories
    source_dir = os.path.join(data_all_dir, f"all_kilns_y_{year}_z_17_buf_25m")
    dest_dir = os.path.join(data_lucknow_dir, f"lucknow_airshed_y_{year}_z_17_buf_25m")

    if not os.path.exists(source_dir):
        print(f"⚠️  Year {year}: Source directory not found - {source_dir}")
        continue

    # Create destination directory
    os.makedirs(dest_dir, exist_ok=True)

    # Get all image files
    image_files = [f for f in os.listdir(source_dir) if f.endswith('.png')]

    copied_count = 0

    for img_file in image_files:
        try:
            # Parse filename: lat_lon_year.png
            parts = img_file.replace('.png', '').split('_')
            lat = float(parts[0])
            lon = float(parts[1])

            # Check if point is within the Lucknow airshed
            if point_in_polygon(lon, lat, polygon):
                # Copy the image
                src_path = os.path.join(source_dir, img_file)
                dst_path = os.path.join(dest_dir, img_file)
                shutil.copy2(src_path, dst_path)
                copied_count += 1
        except Exception as e:
            print(f"    Error processing {img_file}: {e}")

    total_copied += copied_count
    print(f"✓ Year {year}: Copied {copied_count:4d} images")

print("-" * 80)
print(f"✓ Total images copied: {total_copied}")
print(f"✓ Data saved to: {data_lucknow_dir}")
