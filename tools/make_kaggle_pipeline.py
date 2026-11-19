"""Assemble a self-contained Kaggle script kernel that fetches, builds and (optionally) publishes a dataset online.

Usage:  python -I tools/make_kaggle_pipeline.py bts [none|private|public]
Output: pipelines/<name>/pipeline.py + kernel-metadata.json (internet on). Push with `kaggle kernels push -p pipelines/<name>`.
The kernel downloads on Kaggle's servers; only this small script leaves your machine.
PUBLISH=none builds and prints a summary; private/public also creates or versions the dataset, using the
Kaggle Secret KAGGLE_API_TOKEN (Notebook > Add-ons > Secrets), then fixes the cover crop.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"

PUBLISH_CODE = r'''
# =============================================================== publish (Kaggle Secret KAGGLE_API_TOKEN)
import subprocess, time as _time
if PUBLISH in ("private", "public"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle>=2.2.5"], check=True)
    def k(*a):
        r = subprocess.run(["kaggle", *a], capture_output=True, text=True)
        print((r.stdout + r.stderr).strip()[-800:]); return r.returncode
    exists = k("datasets", "status", DATASET_REF) == 0
    if exists:
        k("datasets", "version", "-p", str(OUT), "-m", f"Refresh {pd.Timestamp.now():%Y-%m-%d}")
    else:
        k("datasets", "create", "-p", str(OUT), *(["--public"] if PUBLISH == "public" else []))
    for _ in range(90):
        r = subprocess.run(["kaggle", "datasets", "status", DATASET_REF], capture_output=True, text=True)
        if "ready" in r.stdout: break
        _time.sleep(20)
    from PIL import Image
    from kaggle.api.kaggle_api_extended import CroppedImageRectangle, KaggleApi
    _orig = KaggleApi._upload_dataset_image_file
    def _full(self, p, rel, quiet=False):
        up = _orig(self, p, rel, quiet); w, h = Image.open(os.path.join(p, rel)).size; ch = min(h, w // 2); s = min(w, h)
        rects = []
        for t, box in (("cover image", ((w - 2 * ch) // 2, (h - ch) // 2, 2 * ch, ch)), ("thumbnail", ((w - s) // 2, (h - s) // 2, s, s))):
            r = CroppedImageRectangle(); r.title = t; r.left, r.top, r.width, r.height = box; rects.append(r)
        up.crop_rectangles = rects; return up
    KaggleApi._upload_dataset_image_file = _full
    api = KaggleApi(); api.authenticate(); api.dataset_metadata_update(DATASET_REF, str(OUT))
    print("PUBLISHED", DATASET_REF)
else:
    print("PUBLISH = none: built only. Outputs are in", OUT)
'''


def strip_main_guard(src: str) -> str:
    """Turn `if __name__ == "__main__":` blocks into top-level code (dedent one level)."""
    lines, out, inside = src.splitlines(), [], False
    for ln in lines:
        if ln.startswith('if __name__ == "__main__":'):
            inside = True
            continue
        out.append(ln[4:] if inside and ln.startswith("    ") else ln)
    return "\n".join(out)


def bts(publish: str):
    fetch = (TOOLS / "fetch_bts.py").read_text(encoding="utf-8")
    build = (TOOLS / "build_bts.py").read_text(encoding="utf-8")
    covers = (TOOLS / "make_covers.py").read_text(encoding="utf-8")
    fetch = fetch.replace('OUT = Path(__file__).resolve().parents[1] / "datasets" / "us-flight-on-time" / "source" / "zips"',
                          'OUT = Path("/kaggle/temp/us-flight-on-time/source/zips")')
    fetch = strip_main_guard(fetch).replace("sys.argv[1], sys.argv[2]", "MONTH_FROM, MONTH_TO").replace(
        "int(sys.argv[3]) if len(sys.argv) > 3 else 3", "4")
    build = build.replace('BASE = Path(__file__).resolve().parents[1] / "datasets" / "us-flight-on-time"',
                          'BASE = Path("/kaggle/temp/us-flight-on-time")')
    build = build.replace('SRC, OUT = BASE / "source", BASE / "v1"', 'SRC, OUT = BASE / "source", Path("/kaggle/working/out")')
    build = re.sub(r"import runpy.*\nrunpy\.run_path\(.*\)\[\"flights\"\]\(\)", "flights()", build)
    # covers: helpers + flights(), with D pointing at the Kaggle layout
    head = covers.split("# ----------------------------------------------------------------- SF trees")[0]
    head = head.replace('D = Path(__file__).resolve().parents[1] / "datasets"', 'D = None  # set below')
    fl = covers.split("# ----------------------------------------------------------------- flights: the route network")[1].split("JOBS = ")[0]
    fl = fl.replace('out = D / "us-flight-on-time" / "v1"', "out = OUT")
    lookups = '''
# lookups (small) for the build
import urllib.request as _ur
SRC_DIR = Path("/kaggle/temp/us-flight-on-time/source"); SRC_DIR.mkdir(parents=True, exist_ok=True)
for _name, _url in [("ourairports_airports.csv", "https://davidmegginson.github.io/ourairports-data/airports.csv"),
                    ("L_UNIQUE_CARRIERS.csv", "https://www.transtats.bts.gov/Download_Lookup.asp?Y11x72=Y_haVdhR_PNeeVRef"),
                    ("L_AIRPORT_ID.csv", "https://www.transtats.bts.gov/Download_Lookup.asp?Y11x72=Y_NVecbeg_VQ")]:
    _p = SRC_DIR / _name
    if not _p.exists():
        _p.write_bytes(_ur.urlopen(_ur.Request(_url, headers={"User-Agent": "dataset-refresh-script"}), timeout=300).read())
'''
    header = f'''"""US flight on-time dataset: fetch (BTS) -> build -> cover -> publish, run on Kaggle (internet on).
Generated by tools/make_kaggle_pipeline.py. PUBLISH = none | private | public.
"""
import os, sys
from pathlib import Path
PUBLISH = "{publish}"
MONTH_FROM, MONTH_TO = "2022-01", None   # None = latest month BTS has published
DATASET_REF = "teddybrown414/us-flight-delays-and-on-time-performance-2022-2026"
Path("/kaggle/working/out").mkdir(parents=True, exist_ok=True)
if PUBLISH in ("private", "public"):   # fail fast: check the secret before the long download
    from kaggle_secrets import UserSecretsClient
    os.environ["KAGGLE_API_TOKEN"] = UserSecretsClient().get_secret("KAGGLE_API_TOKEN")
    print("secret KAGGLE_API_TOKEN: ok", flush=True)
'''
    latest = '''
# find the latest published month (HEAD requests are tiny)
import urllib.request as _ur, datetime as _dt
def _exists(y, m):
    try:
        r = _ur.urlopen(_ur.Request(BASE.format(y=y, m=m), method="HEAD", headers={"User-Agent": "dataset-refresh-script"}), timeout=60)
        return int(r.headers.get("Content-Length") or 0) > 1_000_000
    except Exception:
        return False
if MONTH_TO is None:
    _y, _m = _dt.date.today().year, _dt.date.today().month
    while not _exists(_y, _m):
        _y, _m = (_y - 1, 12) if _m == 1 else (_y, _m - 1)
    MONTH_TO = f"{_y}-{_m:02d}"
print("months", MONTH_FROM, "->", MONTH_TO, flush=True)
'''
    fetch_main = fetch.split("for y, m in months(")[0] if "for y, m in months(" in fetch else fetch
    fetch_parts = fetch.split("from concurrent.futures import ThreadPoolExecutor")
    src = (header + "\n# =============================================================== fetch\n" + fetch_parts[0] + latest
           + "from concurrent.futures import ThreadPoolExecutor" + fetch_parts[1]
           + lookups + "\n# =============================================================== covers\n" + head
           + "\n# =============================================================== build\n"
           + build.split("def read_month")[0].split('"""', 2)[2]
           + "def read_month" + build.split("def read_month", 1)[1].replace("flights()", "")
           + "\n# =============================================================== cover\n" + fl + "\nflights()\n"
           + "\nprint(open(OUT / 'dataset-metadata.json', encoding='utf-8').read()[:3000])\n" + PUBLISH_CODE)
    d = ROOT / "pipelines" / "us-flight-delays"
    d.mkdir(parents=True, exist_ok=True)
    (d / "pipeline.py").write_text(src, encoding="utf-8")
    meta = {"id": "teddybrown414/us-flight-delays-dataset-pipeline", "title": "US Flight Delays Dataset Pipeline",
            "code_file": "pipeline.py", "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
            "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    compile(src, "pipeline.py", "exec")
    print("wrote", d, len(src), "chars; PUBLISH =", publish)


if __name__ == "__main__":
    {"bts": bts}[sys.argv[1]](sys.argv[2] if len(sys.argv) > 2 else "none")
