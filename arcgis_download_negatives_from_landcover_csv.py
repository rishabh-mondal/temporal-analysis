#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import threading
import requests
import pandas as pd
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# =========================================================
# INPUT: Delhi airshed EAS land cover CSV (lat_lon.png, land_cover)
# =========================================================
NEG_CENTER_CSV = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/brick_kilns_neurips_2025/notebooks/delhi_airshed_eas_land_cover_above_98.csv"

# =========================================================
# CONFIG (same as before)
# =========================================================
YEAR_TO_TIMEID = {
    2014: 10,
    2016: 6984,
    2018: 2168,
    2019: 4756,
    2020: 18289,
    2021: 9812,
    # 2022: 10321,
    2022: 25982,
    2024: 41468,
    2025: 34007,
}

year = 2022
zoom = 17
tile_size = 256
buffer_m = 25
pad_tiles = 0
max_workers = 16
limit = None  # None = all rows

region = "delhi_airshed_negatives_eas98"
out_dir = f"data_neg/{region}_y_{year}_z_{zoom}_buf_{buffer_m}m"
os.makedirs(out_dir, exist_ok=True)

# Wayback tile URL
time_id = YEAR_TO_TIMEID[year]
url_tpl = (
    "https://wayback.maptiles.arcgis.com/arcgis/rest/services/"
    "world_imagery/wmts/1.0.0/default028mm/mapserver/tile/"
    f"{time_id}" + "/{z}/{y}/{x}"
)

# =========================================================
# HTTP session
# =========================================================
def make_session():
    s = requests.Session()
    retries = Retry(total=4, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries, pool_connections=200, pool_maxsize=200)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    return s

session = make_session()

# =========================================================
# WebMercator helpers
# =========================================================
def lonlat_to_global_pixel(lon, lat, z, tile_size=256):
    n = 2.0 ** z
    x = (lon + 180.0) / 360.0 * n * tile_size
    lat_rad = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n * tile_size
    return x, y

def meters_per_pixel(lat_deg, z):
    return 156543.03392804097 * math.cos(math.radians(lat_deg)) / (2 ** z)

# =========================================================
# Tile cache
# =========================================================
tile_cache = {}
tile_lock = threading.Lock()

def fetch_tile(z, x, y):
    key = (time_id, z, x, y)
    with tile_lock:
        img = tile_cache.get(key)
    if img is not None:
        return img

    url = url_tpl.format(z=z, x=x, y=y)
    r = session.get(url, timeout=20)
    if r.status_code != 200 or not r.content:
        return None

    try:
        img = Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        return None

    with tile_lock:
        tile_cache[key] = img
    return img

def stitch_tiles(x_min, x_max, y_min, y_max, z):
    w = (x_max - x_min + 1) * tile_size
    h = (y_max - y_min + 1) * tile_size
    stitched = Image.new("RGB", (w, h))
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            img = fetch_tile(z, x, y)
            if img is None:
                continue
            stitched.paste(img, ((x - x_min) * tile_size, (y - y_min) * tile_size))
    return stitched

def crop_centered(stitched, x_min, y_min, lon, lat, crop_size):
    gx, gy = lonlat_to_global_pixel(lon, lat, zoom, tile_size=tile_size)
    origin_x = x_min * tile_size
    origin_y = y_min * tile_size
    px = int(round(gx - origin_x))
    py = int(round(gy - origin_y))

    W, H = stitched.size
    half = crop_size // 2

    left = max(0, px - half)
    upper = max(0, py - half)
    right = min(W, left + crop_size)
    lower = min(H, upper + crop_size)

    left = max(0, right - crop_size)
    upper = max(0, lower - crop_size)
    right = min(W, left + crop_size)
    lower = min(H, upper + crop_size)

    cropped = stitched.crop((left, upper, right, lower))

    if cropped.size != (crop_size, crop_size):
        padded = Image.new("RGB", (crop_size, crop_size))
        padded.paste(cropped, (0, 0))
        cropped = padded

    return cropped

def safe_name(lat, lon, year, decimals=6):
    return f"{lat:.{decimals}f}_{lon:.{decimals}f}_{year}"

# =========================================================
# Load negative centers (lat/lon from filename column)
# =========================================================
neg_df = pd.read_csv(NEG_CENTER_CSV)
if "filename" not in neg_df.columns:
    raise ValueError("CSV must contain column: filename (example: 28.6179_76.9057.png)")

def parse_latlon_from_filename(fn):
    stem = os.path.splitext(str(fn).strip())[0]
    parts = stem.split("_")
    if len(parts) < 2:
        return (None, None)
    try:
        lat = float(parts[0])
        lon = float(parts[1])
        return (lat, lon)
    except Exception:
        return (None, None)

neg_df[["center_lat", "center_lon"]] = neg_df["filename"].apply(
    lambda x: pd.Series(parse_latlon_from_filename(x))
)

neg_df = neg_df.dropna(subset=["center_lat", "center_lon"]).reset_index(drop=True)

if limit is not None:
    neg_df = neg_df.iloc[:limit].reset_index(drop=True)

print("Total negative centers loaded:", len(neg_df))

# =========================================================
# Download + crop for negative centers
# =========================================================
results = []
results_lock = threading.Lock()

def process_one(i_row):
    i, row = i_row
    lat = float(row["center_lat"])
    lon = float(row["center_lon"])
    land_cover = row.get("land_cover", "")

    try:
        mpp = meters_per_pixel(lat, zoom)
        radius_px = int(math.ceil(buffer_m / mpp))
        crop_size = 2 * radius_px + 1
        crop_size = max(257, min(crop_size, 1025))

        gx, gy = lonlat_to_global_pixel(lon, lat, zoom, tile_size=tile_size)
        center_tile_x = int(gx // tile_size)
        center_tile_y = int(gy // tile_size)

        tiles_radius = int(math.ceil((crop_size / 2) / tile_size)) + pad_tiles
        x_min = center_tile_x - tiles_radius
        x_max = center_tile_x + tiles_radius
        y_min = center_tile_y - tiles_radius
        y_max = center_tile_y + tiles_radius

        stitched = stitch_tiles(x_min, x_max, y_min, y_max, zoom)

        if stitched.size[0] < crop_size or stitched.size[1] < crop_size:
            with results_lock:
                results.append({
                    "idx": i,
                    "center_lat": lat,
                    "center_lon": lon,
                    "year": year,
                    "land_cover": land_cover,
                    "status": "fail_stitched_too_small",
                    "crop_size": crop_size,
                    "stitched_w": stitched.size[0],
                    "stitched_h": stitched.size[1],
                    "path": ""
                })
            return

        cropped = crop_centered(stitched, x_min, y_min, lon, lat, crop_size)

        fname = safe_name(lat, lon, year) + ".png"
        out_path = os.path.join(out_dir, fname)
        cropped.save(out_path)

        # integrity check
        try:
            with Image.open(out_path) as im:
                im.verify()
        except (UnidentifiedImageError, OSError):
            with results_lock:
                results.append({
                    "idx": i,
                    "center_lat": lat,
                    "center_lon": lon,
                    "year": year,
                    "land_cover": land_cover,
                    "status": "fail_corrupt_image",
                    "crop_size": crop_size,
                    "stitched_w": stitched.size[0],
                    "stitched_h": stitched.size[1],
                    "path": out_path
                })
            return

        with results_lock:
            results.append({
                "idx": i,
                "center_lat": lat,
                "center_lon": lon,
                "year": year,
                "land_cover": land_cover,
                "status": "ok",
                "crop_size": crop_size,
                "stitched_w": stitched.size[0],
                "stitched_h": stitched.size[1],
                "path": out_path
            })

    except Exception as e:
        with results_lock:
            results.append({
                "idx": i,
                "center_lat": lat,
                "center_lon": lon,
                "year": year,
                "land_cover": land_cover,
                "status": "fail_exception",
                "error": repr(e),
                "path": ""
            })

rows = list(neg_df.iterrows())

with ThreadPoolExecutor(max_workers=max_workers) as ex:
    list(tqdm(ex.map(process_one, rows), total=len(rows)))

# =========================================================
# Report
# =========================================================
df_res = pd.DataFrame(results)
total_req = len(rows)
ok = int((df_res["status"] == "ok").sum()) if len(df_res) else 0
failed = total_req - ok

print("Total requested:", total_req)
print("Downloaded OK:", ok)
print("Failed:", failed)

df_res.to_csv(os.path.join(out_dir, "download_report_all.csv"), index=False)
df_res[df_res["status"] != "ok"].to_csv(os.path.join(out_dir, "download_report_failed.csv"), index=False)

print("Saved images to:", out_dir)