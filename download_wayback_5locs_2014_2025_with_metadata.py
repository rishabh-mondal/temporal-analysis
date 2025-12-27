#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download Wayback crops for 5 Delhi locations for years 2014..2025 + save popup-style metadata.

What this script saves:
1) PNG crop for each (location, year)
2) metadata JSON for each (location, year) with capture date / resolution / accuracy (if metadata layer exists)
3) one CSV summary: out_dir/summary_5_locs_2014_2025.csv

Hard requirement:
- You MUST fill YEAR_TO_WAYBACK_DATE (publication date shown in Wayback UI for that year).
  Example: if you use version "2022-09-21" for year 2022, set YEAR_TO_WAYBACK_DATE[2022] = "2022-09-21".
- The script will automatically search ArcGIS Online for the corresponding Metadata layer item id.
"""

import os
import math
import json
import threading
from pathlib import Path
from datetime import datetime
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

import requests
import pandas as pd
from PIL import Image

from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# ArcGIS Python API (pip install arcgis)
from arcgis.gis import GIS
from arcgis.geometry import Point


# =========================
# USER CONFIG
# =========================

# 5 locations in Delhi (EDIT THESE)
# format: (name, lat, lon)
LOCATIONS = [
    ("loc1", 28.208668, 77.420208),
    ("loc2", 28.212481, 77.401398),
    ("loc3", 28.613900, 77.209000),
    ("loc4", 28.704100, 77.102500),
    ("loc5", 28.535500, 77.391000),
]

YEARS = [2014, 2016, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

YEAR_TO_TIMEID = {
    2014: 10,
    2016: 6984,
    2018: 2168,
    2019: 4756,
    2020: 18289,
    2021: 9812,
    2022: 10321,
    2023: 25982,
    2024: 41468,
    2025: 34007,
}

# IMPORTANT: YOU MUST FILL THIS (Wayback publication version date used for that year)
# How to fill: open Wayback UI, pick the version you want for that year, copy the date shown (YYYY-MM-DD).
YEAR_TO_WAYBACK_DATE = {
    # 2014: "YYYY-MM-DD",
    # 2016: "YYYY-MM-DD",
    # 2018: "YYYY-MM-DD",
    # 2019: "YYYY-MM-DD",
    # 2020: "YYYY-MM-DD",
    # 2021: "YYYY-MM-DD",
    2022: "2022-09-21",
    # 2023: "YYYY-MM-DD",
    # 2024: "YYYY-MM-DD",
    # 2025: "YYYY-MM-DD",
}

# Crop / tiles
zoom = 17
tile_size = 256
buffer_m = 25
pad_tiles = 1
max_workers = 12

# Output
region = "delhi_5_locations"
out_dir = Path(f"data_exp/{region}_z_{zoom}_buf_{buffer_m}m_2014_2025").resolve()
(out_dir / "images").mkdir(parents=True, exist_ok=True)
(out_dir / "metadata").mkdir(parents=True, exist_ok=True)


# =========================
# HTTP session
# =========================
def make_session():
    s = requests.Session()
    retries = Retry(total=4, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries, pool_connections=200, pool_maxsize=200)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    return s

session = make_session()


# =========================
# WebMercator helpers
# =========================
def lonlat_to_global_pixel(lon, lat, z, tile_size=256):
    n = 2.0 ** z
    x = (lon + 180.0) / 360.0 * n * tile_size
    lat_rad = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n * tile_size
    return x, y

def meters_per_pixel(lat_deg, z):
    return 156543.03392804097 * math.cos(math.radians(lat_deg)) / (2 ** z)

def safe_name(lat, lon, year, decimals=6):
    return f"{lat:.{decimals}f}_{lon:.{decimals}f}_{year}"


# =========================
# Tiles + cache
# =========================
tile_cache = {}
tile_lock = threading.Lock()

def make_url_tpl(time_id: int):
    return (
        "https://wayback.maptiles.arcgis.com/arcgis/rest/services/"
        "world_imagery/wmts/1.0.0/default028mm/mapserver/tile/"
        f"{time_id}" + "/{z}/{y}/{x}"
    )

def fetch_tile(url_tpl, z, x, y):
    key = (url_tpl, z, x, y)
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

def stitch_tiles(url_tpl, x_min, x_max, y_min, y_max, z):
    w = (x_max - x_min + 1) * tile_size
    h = (y_max - y_min + 1) * tile_size
    stitched = Image.new("RGB", (w, h))

    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            img = fetch_tile(url_tpl, z, x, y)
            if img is None:
                continue
            stitched.paste(img, ((x - x_min) * tile_size, (y - y_min) * tile_size))
    return stitched

def crop_centered(stitched, x_min, y_min, lon, lat, crop_size, z):
    gx, gy = lonlat_to_global_pixel(lon, lat, z, tile_size=tile_size)
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


# =========================
# Wayback metadata (popup) via ArcGIS Online "Metadata" layer
# =========================
gis = GIS()  # anonymous

_meta_itemid_cache = {}
_meta_layer_cache = {}
_meta_cache_lock = threading.Lock()

def find_metadata_item_id(wayback_date: str) -> str | None:
    """
    Searches ArcGIS Online for:
      World Imagery (Wayback YYYY-MM-DD) Metadata
    Returns item id if found.
    """
    with _meta_cache_lock:
        if wayback_date in _meta_itemid_cache:
            return _meta_itemid_cache[wayback_date]

    q = f'title:"World Imagery (Wayback {wayback_date}) Metadata"'
    items = gis.content.search(query=q, item_type="Feature Layer", max_items=10)

    # pick best match
    best = None
    for it in items:
        t = (it.title or "").strip()
        if t.lower() == f"world imagery (wayback {wayback_date}) metadata".lower():
            best = it
            break
    if best is None and items:
        best = items[0]

    item_id = best.id if best else None
    with _meta_cache_lock:
        _meta_itemid_cache[wayback_date] = item_id
    return item_id

def get_metadata_layer(wayback_date: str):
    with _meta_cache_lock:
        if wayback_date in _meta_layer_cache:
            return _meta_layer_cache[wayback_date]

    item_id = find_metadata_item_id(wayback_date)
    if not item_id:
        with _meta_cache_lock:
            _meta_layer_cache[wayback_date] = None
        return None

    item = gis.content.get(item_id)
    lyr = item.layers[0] if item and getattr(item, "layers", None) else None

    with _meta_cache_lock:
        _meta_layer_cache[wayback_date] = lyr
    return lyr

def fetch_wayback_popup_metadata(lon: float, lat: float, wayback_date: str) -> dict:
    """
    Returns dict resembling the Wayback popup:
      capture date, provider, resolution, accuracy, etc.
    Field names can vary; we return both normalized keys + raw attributes.
    """
    lyr = get_metadata_layer(wayback_date)
    if lyr is None:
        return {
            "meta_found": False,
            "meta_wayback_date": wayback_date,
            "meta_item_id": None,
            "meta_note": "Metadata layer not found for this wayback_date. Fix YEAR_TO_WAYBACK_DATE mapping.",
        }

    geom = Point({"x": float(lon), "y": float(lat), "spatialReference": {"wkid": 4326}})
    fs = lyr.query(
        geometry=geom,
        geometry_type="esriGeometryPoint",
        in_sr=4326,
        out_fields="*",
        return_geometry=False
    )

    if not fs or len(fs.features) == 0:
        return {
            "meta_found": False,
            "meta_wayback_date": wayback_date,
            "meta_item_id": lyr._url.split("/services/")[-1] if hasattr(lyr, "_url") else None,
            "meta_note": "No metadata feature returned at this point.",
        }

    attrs = fs.features[0].attributes or {}

    # best-effort normalization (varies by layer)
    def pick(*names):
        for n in names:
            if n in attrs and attrs[n] not in (None, ""):
                return attrs[n]
        return ""

    out = {
        "meta_found": True,
        "meta_wayback_date": wayback_date,  # "when it shown" (version date you selected)
        "meta_src_name": pick("SRC_NAME", "SRCNAME", "SOURCE", "VENDOR"),
        "meta_sensor": pick("SENSOR", "SATELLITE", "PLATFORM"),
        "meta_capture_date": pick("SRC_DATE2", "SRC_DATE", "CAPTUREDATE", "ACQ_DATE"),
        "meta_resolution_m": pick("SRC_RES", "RESOLUTION", "GSD"),
        "meta_accuracy_m": pick("SRC_ACC", "ACCURACY"),
        "meta_raw_json": json.dumps(attrs, ensure_ascii=False),
    }
    return out


# =========================
# Core: process one (location, year)
# =========================
results = []
results_lock = threading.Lock()

def process_one(task):
    name, lat, lon, y = task

    time_id = YEAR_TO_TIMEID[y]
    url_tpl = make_url_tpl(time_id)

    # version date ("when shown") for metadata lookup
    wayback_date = YEAR_TO_WAYBACK_DATE.get(y, None)

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

        stitched = stitch_tiles(url_tpl, x_min, x_max, y_min, y_max, zoom)
        cropped = crop_centered(stitched, x_min, y_min, lon, lat, crop_size, zoom)

        base = safe_name(lat, lon, y)
        img_path = out_dir / "images" / f"{name}__{base}.png"
        cropped.save(img_path)

        # metadata (popup)
        meta = {}
        if wayback_date:
            meta = fetch_wayback_popup_metadata(lon, lat, wayback_date)
        else:
            meta = {
                "meta_found": False,
                "meta_wayback_date": "",
                "meta_note": "Missing YEAR_TO_WAYBACK_DATE[year]. Fill it to enable popup metadata.",
            }

        meta_path = out_dir / "metadata" / f"{name}__{base}.json"
        meta_payload = {
            "location_name": name,
            "lat": lat,
            "lon": lon,
            "year": y,
            "time_id": time_id,
            "zoom": zoom,
            "buffer_m": buffer_m,
            "downloaded_utc": datetime.utcnow().isoformat() + "Z",
            **meta,
        }
        meta_path.write_text(json.dumps(meta_payload, indent=2, ensure_ascii=False))

        row = {
            "location_name": name,
            "lat": lat,
            "lon": lon,
            "year": y,
            "time_id": time_id,
            "wayback_date_shown": meta_payload.get("meta_wayback_date", ""),
            "capture_date": meta_payload.get("meta_capture_date", ""),
            "src_name": meta_payload.get("meta_src_name", ""),
            "sensor": meta_payload.get("meta_sensor", ""),
            "resolution_m": meta_payload.get("meta_resolution_m", ""),
            "accuracy_m": meta_payload.get("meta_accuracy_m", ""),
            "meta_found": meta_payload.get("meta_found", False),
            "image_path": str(img_path),
            "meta_path": str(meta_path),
            "status": "ok",
        }

        with results_lock:
            results.append(row)

    except Exception as e:
        with results_lock:
            results.append({
                "location_name": name,
                "lat": lat,
                "lon": lon,
                "year": y,
                "time_id": time_id,
                "status": "fail",
                "error": repr(e),
            })


def main():
    # sanity check for mapping
    missing_dates = [y for y in YEARS if y not in YEAR_TO_WAYBACK_DATE]
    if missing_dates:
        print("WARNING: Missing YEAR_TO_WAYBACK_DATE for years:", missing_dates)
        print("Metadata popup fields will be missing for those years.")

    tasks = []
    for (name, lat, lon) in LOCATIONS:
        for y in YEARS:
            tasks.append((name, float(lat), float(lon), int(y)))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(tqdm(ex.map(process_one, tasks), total=len(tasks)))

    df = pd.DataFrame(results).sort_values(["location_name", "year"])
    out_csv = out_dir / "summary_5_locs_2014_2025.csv"
    df.to_csv(out_csv, index=False)

    print("Saved:", out_csv)
    print("Images:", out_dir / "images")
    print("Metadata JSON:", out_dir / "metadata")


if __name__ == "__main__":
    from tqdm import tqdm
    main()