"""Download EIA-930 six-month BALANCE files (Jul 2015 -> current half-year) and the reference tables.

Usage:  python -I tools/fetch_eia930.py
Output: datasets/eia930-hourly-grid/source/  (re-running skips complete files, so it resumes)
"""
import datetime as dt
import time
import urllib.request
import os
from pathlib import Path

BASE = "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/"
REF = "https://www.eia.gov/electricity/930-content/EIA930_Reference_Tables.xlsx"
# KDP_DATA_ROOT overrides where sources are downloaded and datasets are built (used by the Kaggle kernels)
DATA_ROOT = Path(os.environ.get("KDP_DATA_ROOT") or Path(__file__).resolve().parents[1] / "datasets")
OUT = DATA_ROOT / "eia930-hourly-grid" / "source"
OUT.mkdir(parents=True, exist_ok=True)


def fetch(url, dest, tries=6):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "dataset-refresh-script"})
            with urllib.request.urlopen(req, timeout=600) as r:
                size = int(r.headers.get("Content-Length") or 0)
                if dest.exists() and size and dest.stat().st_size == size:
                    return "skip"
                tmp = dest.with_suffix(dest.suffix + ".part")
                with tmp.open("wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                if size and tmp.stat().st_size != size:
                    raise IOError(f"short read {tmp.stat().st_size} of {size}")
                tmp.replace(dest)
                return f"{dest.stat().st_size / 1e6:.1f} MB"
        except Exception as e:
            print(f"  retry {i + 1} {dest.name}: {e}", flush=True)
            time.sleep(10 * (i + 1))
    raise RuntimeError(url)


today = dt.date.today()
halves = [(2015, "Jul_Dec")] + [(y, h) for y in range(2016, today.year + 1) for h in ("Jan_Jun", "Jul_Dec")]
halves = [(y, h) for y, h in halves if not (y == today.year and h == "Jul_Dec" and today.month < 7)]
for y, h in halves:
    name = f"EIA930_BALANCE_{y}_{h}.csv"
    print(name, fetch(BASE + name, OUT / name), flush=True)
print("EIA930_Reference_Tables.xlsx", fetch(REF, OUT / "EIA930_Reference_Tables.xlsx"), flush=True)
print("DONE", flush=True)
