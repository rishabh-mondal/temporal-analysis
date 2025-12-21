#!/usr/bin/env python3

import shutil
from pathlib import Path
import subprocess
from datetime import datetime

SRC = Path("/home/patel_zeel")
DST = Path("/mnt/nas_ramanujan/zeel_backup")
LOG = DST / "simple_copy.log"

DST.mkdir(parents=True, exist_ok=True)

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def du(path):
    try:
        out = subprocess.check_output(["du", "-sh", str(path)], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except:
        return "N/A"

def df():
    out = subprocess.check_output(["df", "-h", str(DST)])
    return out.decode().strip()

log("==== START ====")
log(f"SRC={SRC}")
log(f"DST={DST}")
log("Disk usage BEFORE:")
log(df())

for item in SRC.iterdir():
    if not item.is_dir():
        continue

    dst_item = DST / item.name

    log(f"---- Copying folder: {item.name} ----")
    log("SRC size BEFORE: " + du(item))
    log("DST disk BEFORE:")
    log(df())

    if dst_item.exists():
        log("Destination exists, skipping copy.")
    else:
        shutil.copytree(item, dst_item)

    log("DST folder size AFTER: " + du(dst_item))
    log("DST disk AFTER:")
    log(df())

log("==== DONE ====")