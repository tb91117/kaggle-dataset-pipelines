"""Download the SF Street Tree Inventory (data.sf.gov tkzw-k3nq) via the paged SODA API.

Usage:  python -I tools/fetch_sf_trees.py
Output: datasets/san-francisco-trees/source/pages/page_XXXX.csv, then source/street_trees_full.csv
Each page is retried on its own, and pages already on disk are skipped, so re-running resumes.
"""
import csv
import io
import time
import urllib.request
from pathlib import Path

BASE = "https://data.sf.gov/resource/tkzw-k3nq.csv"
PAGE = 25_000
OUT = Path(__file__).resolve().parents[1] / "datasets" / "san-francisco-trees" / "source"
PAGES = OUT / "pages"
PAGES.mkdir(parents=True, exist_ok=True)


def get(url: str, tries: int = 8) -> bytes:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "dataset-refresh-script"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except Exception as e:  # network resets are common on this link
            print(f"  retry {i + 1}: {e}", flush=True)
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"failed: {url}")


total = int(get(f"{BASE.replace('.csv', '.json')}?$select=count(*)").decode().split('"')[-2])
print(f"rows on portal: {total}", flush=True)

n_pages = -(-total // PAGE)
for p in range(n_pages):
    f = PAGES / f"page_{p:04d}.csv"
    if f.exists() and f.stat().st_size > 0:
        continue
    url = f"{BASE}?$limit={PAGE}&$offset={p * PAGE}&$order=treeid"
    data = get(url)
    rows = sum(1 for _ in csv.reader(io.StringIO(data.decode("utf-8")))) - 1
    f.write_bytes(data)
    print(f"page {p + 1}/{n_pages}: {rows} rows", flush=True)

# stitch: keep header from the first page only
full = OUT / "street_trees_full.csv"
with full.open("w", encoding="utf-8", newline="") as out:
    for i, f in enumerate(sorted(PAGES.glob("page_*.csv"))):
        text = f.read_text(encoding="utf-8")
        out.write(text if i == 0 else text.split("\n", 1)[1])
got = sum(1 for _ in csv.reader(full.open(encoding="utf-8"))) - 1
print(f"DONE {got} rows (portal says {total})", flush=True)
