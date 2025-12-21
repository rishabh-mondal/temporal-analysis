# End to end: build 5 panels (same lat_lon across years), send 7 images per panel to Gemini concurrently,
# parse strict JSON, add consistency + review flags, save CSV.

import os
import asyncio
import json
from pathlib import Path
from PIL import Image
import pandas as pd
from google import genai
from dotenv import load_dotenv
ENV_PATH = "/home/rishabh.mondal/.env"
load_dotenv(ENV_PATH)


# -------------------------
# 0) CONFIG
# -------------------------
YEARS = [2014, 2016, 2018, 2020, 2022, 2024, 2025]

BASE_DIR = Path("/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/temporal-analysis/data")
FOLDER_TPL = "delhi_airshed_y_{y}_z_17_buf_25m"

MODEL_ID = "models/gemini-3-pro-preview"
MAX_CONCURRENCY = 5
N_LOCATIONS = 753

OUT_CSV = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/temporal-analysis/vlm_kiln_change_results_753_loc.csv"

# Use environment variable GEMINI_API_KEY
# export GEMINI_API_KEY="YOUR_KEY"
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in environment")

client = genai.Client(api_key=API_KEY)

# -------------------------
# 1) BUILD PANELS (PIL READY)
# -------------------------
def build_year_maps():
    year_maps = {}
    for y in YEARS:
        folder = BASE_DIR / FOLDER_TPL.format(y=y)
        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")
        mapping = {}
        for f in folder.glob("*.png"):
            # 28.208668_77.420208_2014.png
            parts = f.stem.split("_")
            if len(parts) >= 3:
                key = f"{parts[0]}_{parts[1]}"
                mapping[key] = f
        year_maps[y] = mapping
    return year_maps

def build_panels(n_locations=N_LOCATIONS):
    year_maps = build_year_maps()
    common = set.intersection(*(set(year_maps[y].keys()) for y in YEARS))
    common = sorted(list(common))

    if len(common) < n_locations:
        raise ValueError(f"Need {n_locations} common locations, found {len(common)}")

    selected = common[:n_locations]
    panels = []
    for loc in selected:
        lat, lon = loc.split("_")
        year_to_path = {y: str(year_maps[y][loc]) for y in YEARS}

        # Preload PIL images in consistent order
        pil_list = []
        for y in YEARS:
            pil_list.append(Image.open(year_to_path[y]).convert("RGB"))

        panels.append(
            {
                "lat": float(lat),
                "lon": float(lon),
                "lat_lon": loc,
                "year_to_path": year_to_path,
                "pil_list_ordered": pil_list,
            }
        )
    return panels

# -------------------------
# 2) PROMPT
# -------------------------
PROMPT = f"""
You are analyzing multi year satellite image chips of the SAME location across years {YEARS}.
Detect brick kiln like structures and track changes.

Return STRICT JSON only. No markdown. No extra text.

Definitions:
kiln_present: whether kiln like structure exists in that year
kiln_shape: one of ["circular_oval","oval_round","rectangular_sharp","unknown","none"]
kiln_type: one of ["FCBK","CFCBK","Zigzag","unknown","none"]

Infer:
appearance_year: first year kiln_present becomes true
type transition: (FCBK or CFCBK) -> Zigzag
shape transition: circular_oval or oval_round -> rectangular_sharp
demolished_year: first year after being present where kiln_present becomes false and stays absent thereafter (best effort)
negative_sample: true if kiln_present is false for all years

Output JSON schema:
{{
  "appearance_year": <int or 0>,
  "appearance_type": "<kiln_type at appearance or 'none'>",

  "type_transition_year_before": <int or 0>,
  "type_transition_year_after": <int or 0>,
  "type_transition_note": "<short>",

  "shape_transition_year_before": <int or 0>,
  "shape_transition_year_after": <int or 0>,
  "shape_transition_note": "<short>",

  "demolished": <true/false>,
  "demolished_year": <int or 0>,
  "negative_sample": <true/false>,
  "monitoring_note_one_line": "<one line summary of evolution over years>",
  "confidence": "<low|medium|high>"
}}

Be conservative. If unsure set type or shape to "unknown". If not present set to "none".
"""

# -------------------------
# 3) CONSISTENCY + REVIEW FLAGS
# -------------------------
def _as_bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    return s in {"true", "1", "yes", "present"}

def presence_sequence_from_output(data):
    roi = data.get("roi_state_by_year", {}) or {}
    seq = []
    for y in YEARS:
        st = roi.get(str(y), {}) or {}
        seq.append(_as_bool(st.get("kiln_present", False)))
    return seq

def has_inconsistent_presence(seq):
    # present -> absent -> present
    seen_present = False
    seen_absent_after_present = False
    for v in seq:
        if v and not seen_present:
            seen_present = True
        elif (not v) and seen_present:
            seen_absent_after_present = True
        elif v and seen_absent_after_present:
            return True
    return False

def confidence_to_score(conf):
    c = str(conf).strip().lower()
    if c == "high":
        return 0
    if c == "medium":
        return 1
    if c == "low":
        return 2
    return 1

def review_priority(confidence, inconsistent):
    # higher score = review earlier
    return confidence_to_score(confidence) + (2 if inconsistent else 0)

# -------------------------
# 4) ASYNC INFERENCE
# -------------------------
async def process_one_location(panel, semaphore):
    year_tagged_prompt = (
        "Images are provided in this exact order: "
        + ", ".join(map(str, YEARS))
        + ".\n"
        + PROMPT
    )

    imgs = panel["pil_list_ordered"]

    async with semaphore:
        resp = await client.aio.models.generate_content(
            model=MODEL_ID,
            contents=[*imgs, year_tagged_prompt],
        )

    text = (resp.text or "").strip()

    # strict JSON parse with salvage
    try:
        data = json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(text[start : end + 1])
        else:
            raise ValueError(f"Non JSON output for {panel['lat_lon']}:\n{text[:500]}")

    # attach ids
    data["lat"] = panel["lat"]
    data["lon"] = panel["lon"]
    data["lat_lon"] = panel["lat_lon"]

    # add flags
    seq = presence_sequence_from_output(data)
    inconsistent = has_inconsistent_presence(seq)
    data["presence_seq"] = {str(YEARS[i]): bool(seq[i]) for i in range(len(YEARS))}
    data["inconsistent_presence"] = bool(inconsistent)
    data["review_priority_score"] = int(review_priority(data.get("confidence", "unknown"), inconsistent))

    return data

async def run_async_panels(panels, max_concurrency=MAX_CONCURRENCY):
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = [process_one_location(p, semaphore) for p in panels]
    return await asyncio.gather(*tasks)

# -------------------------
# 5) FLATTEN TO CSV
# -------------------------
def safe_int(x):
    x = pd.to_numeric(x, errors="coerce")
    if pd.isna(x):
        return 0
    return int(x)

def to_row(d):
    return {
        "lat": float(d.get("lat", 0.0) or 0.0),
        "lon": float(d.get("lon", 0.0) or 0.0),
        "lat_lon": d.get("lat_lon", ""),

        "appearance_year": safe_int(d.get("appearance_year", 0)),
        "appearance_type": d.get("appearance_type", "none"),

        "type_transition_year_before": safe_int(d.get("type_transition_year_before", 0)),
        "type_transition_year_after": safe_int(d.get("type_transition_year_after", 0)),
        "type_transition_note": d.get("type_transition_note", ""),

        "shape_transition_year_before": safe_int(d.get("shape_transition_year_before", 0)),
        "shape_transition_year_after": safe_int(d.get("shape_transition_year_after", 0)),
        "shape_transition_note": d.get("shape_transition_note", ""),

        "demolished": bool(d.get("demolished", False)),
        "demolished_year": safe_int(d.get("demolished_year", 0)),
        "negative_sample": bool(d.get("negative_sample", False)),

        "confidence": d.get("confidence", "unknown"),
        "inconsistent_presence": bool(d.get("inconsistent_presence", False)),
        "review_priority_score": safe_int(d.get("review_priority_score", 0)),

        "monitoring_note_one_line": d.get("monitoring_note_one_line", ""),

        "raw_output": json.dumps(d, ensure_ascii=False),
    }

# -------------------------
# 6) MAIN
# -------------------------
# Jupyter: run "results = await main()" in a cell
# Python script: run "asyncio.run(main())"

async def main():
    panels = build_panels(N_LOCATIONS)
    print("Panels ready:", len(panels), "locations")
    print([p["lat_lon"] for p in panels])

    results = await run_async_panels(panels, MAX_CONCURRENCY)

    df = pd.DataFrame([to_row(r) for r in results])
    df = df.sort_values(["review_priority_score"], ascending=False)
    df.to_csv(OUT_CSV, index=False)
    print("Saved:", OUT_CSV)
    return results, df

# If you are in Jupyter, run:
# results, df = await main()

# If you are in a normal .py file, uncomment:
asyncio.run(main())