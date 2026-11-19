"""Cut tiny, real-data source samples for the end-to-end build tests.

Usage:  python -I tests/fixtures/make_fixtures.py <full datasets root>
Reads the full downloaded sources (as fetched by tools/fetch_*.py) and writes small samples next to this file,
laid out like a KDP_DATA_ROOT, so the build scripts run on them unchanged.
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

SRC = Path(sys.argv[1])
OUT = Path(__file__).resolve().parent / "data"

# EIA-930: 3 balancing authorities x 7 days (2018) + 21 days (2025), plus the reference tables
e_src, e_out = SRC / "eia930-hourly-grid" / "source", OUT / "eia930-hourly-grid" / "source"
e_out.mkdir(parents=True, exist_ok=True)
# two periods, so the sample spans the mid-2024 change in energy categories like the real history does
keep_ba = ["CISO", "ERCO", "PJM"]
for name, n_days in (("EIA930_BALANCE_2018_Jul_Dec.csv", 7), ("EIA930_BALANCE_2025_Jan_Jun.csv", 21)):
    eia = pd.read_csv(e_src / name, dtype=str)
    days = sorted(eia["Data Date"].unique(), key=lambda d: pd.to_datetime(d, format="%m/%d/%Y"))[:n_days]
    eia[eia["Balancing Authority"].isin(keep_ba) & eia["Data Date"].isin(days)].to_csv(e_out / name, index=False)
shutil.copy2(e_src / "EIA930_Reference_Tables.xlsx", e_out / "EIA930_Reference_Tables.xlsx")

# SF trees: 300 trees spread over the inventory
t_src, t_out = SRC / "san-francisco-trees" / "source", OUT / "san-francisco-trees" / "source"
t_out.mkdir(parents=True, exist_ok=True)
trees = pd.read_csv(t_src / "street_trees_full.csv", dtype=str, keep_default_na=False)
trees.iloc[:: max(1, len(trees) // 300)].head(300).to_csv(t_out / "street_trees_full.csv", index=False)

# WDI: 6 economies + the World aggregate, the headline indicators + 2 SIPRI ones (tests the license exclusion)
import re  # noqa: E402

w_raw, w_out = SRC / "world-development-indicators" / "source" / "raw", OUT / "world-development-indicators" / "source" / "raw"
w_out.mkdir(parents=True, exist_ok=True)
build_src = (Path(__file__).resolve().parents[2] / "tools" / "build_wdi.py").read_text(encoding="utf-8")
codes = set(re.findall(r'\("([A-Z]{2}\.[A-Z0-9.]+)", "[a-z0-9_]+"\)', build_src)) | {"MS.MIL.XPND.GD.ZS", "MS.MIL.XPND.CD"}
countries = {"USA", "CHN", "IND", "BRA", "NGA", "DEU", "WLD"}
read = dict(encoding="utf-8-sig", dtype=str, keep_default_na=False)
for name, ckey, skey in (("WDICSV.csv", "Country Code", "Indicator Code"), ("WDICountry.csv", "Country Code", None),
                         ("WDISeries.csv", None, "Series Code"), ("WDIcountry-series.csv", "CountryCode", "SeriesCode"),
                         ("WDIfootnote.csv", "CountryCode", "SeriesCode"), ("WDIseries-time.csv", None, "SeriesCode")):
    df = pd.read_csv(w_raw / name, **read)
    if ckey:
        df = df[df[ckey].isin(countries)]
    if skey:
        df = df[df[skey].isin(codes)]
    df.to_csv(w_out / name, index=False, encoding="utf-8-sig")
