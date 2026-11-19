"""Download BTS 'Reporting Carrier On-Time Performance' monthly zips (TranStats PREZIP) for a month range.

Usage:  python -I tools/fetch_bts.py 2022-01 2026-08
Output: datasets/us-flight-on-time/source/zips/  (complete files are skipped, so re-running resumes)
"""
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{y}_{m}.zip"
OUT = Path(__file__).resolve().parents[1] / "datasets" / "us-flight-on-time" / "source" / "zips"
OUT.mkdir(parents=True, exist_ok=True)


def months(a, b):
    y, m = map(int, a.split("-")); y2, m2 = map(int, b.split("-"))
    while (y, m) <= (y2, m2):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def fetch(url, dest, tries=6):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "dataset-refresh-script"})
            with urllib.request.urlopen(req, timeout=600) as r:
                size = int(r.headers.get("Content-Length") or 0)
                if dest.exists() and size and dest.stat().st_size == size:
                    return "skip"
                tmp = dest.with_suffix(".part")
                with tmp.open("wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                if size and tmp.stat().st_size != size:
                    raise IOError(f"short read {tmp.stat().st_size}/{size}")
                tmp.replace(dest)
                return f"{size / 1e6:.1f} MB"
        except Exception as e:  # noqa: BLE001
            print(f"  retry {i + 1} {dest.name}: {e}", flush=True)
            time.sleep(10 * (i + 1))
    raise RuntimeError(url)


if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor

    def job(ym):
        y, m = ym
        dest = OUT / f"ontime_{y}_{m:02d}.zip"
        print(dest.name, fetch(BASE.format(y=y, m=m), dest), flush=True)

    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    with ThreadPoolExecutor(workers) as ex:      # a few parallel connections; the server throttles each one
        list(ex.map(job, months(sys.argv[1], sys.argv[2])))
    print("DONE", flush=True)
