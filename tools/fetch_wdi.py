"""Download the World Bank World Development Indicators bulk CSV and unzip it for build_wdi.py.

Usage:  python -I tools/fetch_wdi.py
Output: datasets/world-development-indicators/source/WDI_CSV.zip and source/raw/*.csv
A complete zip is not downloaded again, so re-running only re-extracts.
"""
import datetime
import os
import time
import urllib.request
import zipfile
from pathlib import Path

URL = "https://databank.worldbank.org/data/download/WDI_CSV.zip"
# KDP_DATA_ROOT overrides where sources are downloaded and datasets are built (used by the Kaggle kernels)
DATA_ROOT = Path(os.environ.get("KDP_DATA_ROOT") or Path(__file__).resolve().parents[1] / "datasets")
SRC = DATA_ROOT / "world-development-indicators" / "source"
RAW = SRC / "raw"
NEEDED = {"WDICSV.csv", "WDICountry.csv", "WDISeries.csv", "WDIcountry-series.csv", "WDIfootnote.csv", "WDIseries-time.csv"}


def fetch(url: str, dest: Path, tries: int = 6) -> str:
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


def extract(zip_path: Path, out: Path) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if Path(n).name in NEEDED]
        missing = NEEDED - {Path(n).name for n in names}
        if missing:
            raise RuntimeError(f"WDI zip is missing {sorted(missing)}; the World Bank may have changed the layout")
        for n in names:
            dest = out / Path(n).name
            dest.write_bytes(z.read(n))
            # keep the World Bank's file time: build_wdi.py reports it as the release date
            stamp = datetime.datetime(*z.getinfo(n).date_time).timestamp()
            os.utime(dest, (stamp, stamp))
    return sorted(Path(n).name for n in names)


if __name__ == "__main__":
    SRC.mkdir(parents=True, exist_ok=True)
    print("WDI_CSV.zip", fetch(URL, SRC / "WDI_CSV.zip"), flush=True)
    print("extracted", extract(SRC / "WDI_CSV.zip", RAW), flush=True)
    print("DONE", flush=True)
