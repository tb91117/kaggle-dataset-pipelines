"""Build the EIA-930 hourly grid dataset from the six-month BALANCE files.

Usage:  python -I tools/build_eia930.py            (after tools/fetch_eia930.py)
Output: datasets/eia930-hourly-grid/v1/  (Parquet + CSV + dataset-metadata.json; cover via make_covers.py)
Nothing is uploaded; pushing is a separate, explicit step.
"""
import json
import re
import os
from pathlib import Path

import numpy as np
import pandas as pd

# KDP_DATA_ROOT overrides where sources are downloaded and datasets are built (used by the Kaggle kernels)
DATA_ROOT = Path(os.environ.get("KDP_DATA_ROOT") or Path(__file__).resolve().parents[1] / "datasets")
BASE = DATA_ROOT / "eia930-hourly-grid"
SRC, OUT = BASE / "source", BASE / "v1"
OUT.mkdir(parents=True, exist_ok=True)
SLUG, OWNER = "us-electric-grid-hourly-eia-930", "teddybrown414"

FUELS = {
    "Coal": "coal", "Natural Gas": "natural_gas", "Nuclear": "nuclear", "All Petroleum Products": "petroleum",
    "Hydropower Excluding Pumped Storage": "hydro", "Hydropower and Pumped Storage": "hydro_incl_pumped_storage",
    "Pumped Storage": "pumped_storage", "Solar": "solar", "Solar without Integrated Battery Storage": "solar",
    "Solar with Integrated Battery Storage": "solar_with_battery", "Wind": "wind",
    "Wind without Integrated Battery Storage": "wind", "Wind with Integrated Battery Storage": "wind_with_battery",
    "Battery Storage": "battery", "Other Energy Storage": "other_storage", "Unknown Energy Storage": "unknown_storage",
    "Geothermal": "geothermal", "Other Fuel Sources": "other", "Unknown Fuel Sources": "unknown", "Other": "other",
}
MAIN = {"Demand Forecast (MW)": "demand_forecast_mw", "Demand (MW)": "demand_mw", "Net Generation (MW)": "net_generation_mw",
        "Total Interchange (MW)": "total_interchange_mw", "Sum(Valid DIBAs) (MW)": "sum_valid_dibas_mw"}
unknown_cols = set()


def norm(col: str):
    """Return (base_name, variant) with variant in {'', 'imputed', 'adjusted'}, or None for id columns."""
    c = re.sub(r"\s+", " ", col.replace("witho Integrated", "with Integrated")).strip()
    variant = ""
    m = re.match(r"^(.*?) \((Imputed|Adjusted)\)$", c)
    if m:
        c, variant = m.group(1).strip(), m.group(2).lower()
    if c in MAIN:
        return MAIN[c], variant
    m = re.match(r"^Net Generation \(MW\) from (.+)$", c)
    if m:
        f = m.group(1).strip()
        if f not in FUELS:
            unknown_cols.add(col)
        return f"ng_{FUELS.get(f, re.sub(r'[^0-9a-z]+', '_', f.lower()).strip('_'))}_mw", variant
    unknown_cols.add(col)
    return re.sub(r"[^0-9a-z]+", "_", c.lower()).strip("_"), variant


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, thousands=",", low_memory=False)
    out = pd.DataFrame({
        "ba": df["Balancing Authority"].astype(str).str.strip(),
        "region": df["Region"].astype(str).str.strip() if "Region" in df else None,
        "utc_time_end": pd.to_datetime(df["UTC Time at End of Hour"], format="%m/%d/%Y %I:%M:%S %p"),
        "local_time_end": pd.to_datetime(df["Local Time at End of Hour"], format="%m/%d/%Y %I:%M:%S %p"),
    })
    groups = {}
    for col in df.columns:
        if col in ("Balancing Authority", "Region", "UTC Time at End of Hour", "Local Time at End of Hour", "Data Date", "Hour Number"):
            continue
        base, variant = norm(col)
        groups.setdefault(base, {})[variant] = pd.to_numeric(df[col], errors="coerce")
    for base, v in groups.items():
        if "" in v:
            out[base] = v[""].astype("float32")
        if "adjusted" in v:
            out[f"{base}_adjusted"] = v["adjusted"].astype("float32")
        if "imputed" in v:
            out[f"{base}_imputed_flag"] = v["imputed"].notna()
    return out


files = sorted(SRC.glob("EIA930_BALANCE_*.csv"),
               key=lambda p: (int(p.stem.split("_")[2]), p.stem.split("_")[3] != "Jan"))
assert files, "run tools/fetch_eia930.py first"
parts = []
for f in files:
    parts.append(load(f))
    print(f.name, len(parts[-1]), flush=True)
df = pd.concat(parts, ignore_index=True, sort=False)
raw_rows = len(df)
df = df.sort_values(["ba", "utc_time_end"]).drop_duplicates(["ba", "utc_time_end"], keep="last").reset_index(drop=True)
dupes = raw_rows - len(df)
for c in df.columns:
    if c.endswith("_imputed_flag"):
        df[c] = df[c].fillna(False).astype(bool)

# column order: ids, main metrics (raw, adjusted, flag), fuels (raw, adjusted, flag)
def order_key(c):
    if c in ("ba", "region", "utc_time_end", "local_time_end"):
        return (0, ["ba", "region", "utc_time_end", "local_time_end"].index(c), 0)
    base = re.sub(r"_(adjusted|imputed_flag)$", "", c)
    rank = {"": 0, "adjusted": 1, "imputed_flag": 2}[c[len(base) + 1:] if c != base else ""]
    group = 1 if not base.startswith("ng_") else 2
    return (group, list(MAIN.values()).index(base) if base in MAIN.values() else 99, base, rank)

df = df[sorted(df.columns, key=lambda c: tuple(str(x) for x in order_key(c)))]

# outlier flag for adjusted demand: outside [0.5, 1.5] x the median of the same hour on the 14 surrounding days
wide = df.pivot(index="utc_time_end", columns="ba", values="demand_mw_adjusted").asfreq("h")
stack = np.stack([wide.shift(24 * k).values for k in range(-7, 8) if k != 0])
with np.errstate(all="ignore"):
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    ref_med = pd.DataFrame(np.nanmedian(stack, axis=0), index=wide.index, columns=wide.columns)
ratio = (wide / ref_med).stack(future_stack=True).rename("r").reset_index()
df = df.merge(ratio, on=["utc_time_end", "ba"], how="left")
df.insert(df.columns.get_loc("demand_mw_imputed_flag") + 1, "demand_mw_adjusted_outlier_flag",
          ((df.r < 0.5) | (df.r > 1.5)).fillna(False).astype(bool))
df = df.drop(columns="r").sort_values(["ba", "utc_time_end"]).reset_index(drop=True)
n_outliers = int(df.demand_mw_adjusted_outlier_flag.sum())
df["ba"] = df.ba.astype("category"); df["region"] = df.region.astype("category")
df.to_parquet(OUT / "eia930_balance_hourly.parquet", index=False, compression="zstd")

# ---- reference table: balancing authorities
ref = pd.read_excel(SRC / "EIA930_Reference_Tables.xlsx", sheet_name=None)
ba_sheet = next(v for k, v in ref.items() if "BA" in k.upper() and "SUB" not in k.upper())
ba_sheet.columns = [re.sub(r"[^0-9a-z]+", "_", str(c).lower()).strip("_") for c in ba_sheet.columns]
ba_sheet.to_csv(OUT / "balancing_authorities.csv", index=False)

# ---- region + lower-48 hourly totals (sum of BA adjusted values)
val_cols = [c for c in df.columns if c.endswith("_adjusted") and (c.startswith("ng_") or c in ("demand_mw_adjusted", "net_generation_mw_adjusted"))]
agg_src = df[["region", "utc_time_end", "demand_mw_adjusted_outlier_flag"] + val_cols].copy()
agg_src.loc[agg_src.demand_mw_adjusted_outlier_flag, "demand_mw_adjusted"] = np.nan   # leave flagged spikes out of totals
reg = agg_src.groupby(["region", "utc_time_end"], observed=True)[val_cols].sum(min_count=1).reset_index()
us = agg_src.groupby("utc_time_end")[val_cols].sum(min_count=1).reset_index().assign(region="US48")
regional = pd.concat([reg, us], ignore_index=True)
regional.columns = [c.replace("_adjusted", "") for c in regional.columns]
regional.to_parquet(OUT / "region_hourly.parquet", index=False, compression="zstd")

summary = {
    "rows": len(df), "dupes_removed": dupes, "bas": int(df.ba.nunique()), "regions": sorted(map(str, df.region.dropna().unique())),
    "t0": str(df.utc_time_end.min()), "t1": str(df.utc_time_end.max()), "columns": list(df.columns),
    "unknown_source_columns": sorted(unknown_cols), "ba_sheet_columns": list(ba_sheet.columns), "ref_sheets": list(ref),
    "region_rows": len(regional), "outliers": n_outliers,
}
(BASE / "build_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
print(json.dumps({k: v for k, v in summary.items() if k != "columns"}, indent=1, default=str))
print(len(df.columns), "columns")


# ---------------------------------------------------------------- dataset-metadata.json
# fuels with no values at all (e.g. a column EIA added before reporting any data) are left out of the table
first_seen = {c: df.loc[df[c].notna(), "utc_time_end"].min() for c in df.columns
              if c.startswith("ng_") and c.endswith("_mw") and df[c].notna().any()}
fuel_lines = "\n".join(f"| `{c}` | {first_seen[c]:%Y-%m} |" for c in sorted(first_seen, key=lambda c: first_seen[c]))
last_actual = df.loc[df.demand_mw.notna(), "utc_time_end"].max()
last_seen = {c: df.loc[df[c].notna(), "utc_time_end"].max() for c in first_seen}
us48 = regional[regional.region == "US48"].set_index("utc_time_end").demand_mw
peak_t, peak_v = us48.idxmax(), us48.max()
gen_only = int((ba_sheet.generation_only_ba == "Yes").sum())
t0, t1 = df.utc_time_end.min(), last_actual
today = pd.Timestamp.now()
desc = f"""## What this is
Hourly operating data for **every balancing authority (grid operator) in the contiguous United States**, from the U.S. Energy Information Administration's **Form EIA-930** (Hourly Electric Grid Monitor): demand, day-ahead demand forecast, net generation, net generation **by energy source**, and total interchange with neighbouring grids. **{len(df):,} hourly rows** for **{df.ba.nunique()} balancing authorities** in {len(summary['regions'])} regions, **{t0:%Y-%m-%d} to {t1:%Y-%m-%d}** (UTC).

Kaggle's most-voted hourly-load dataset covers one operator (PJM) and stops in 2018. This one covers the whole country to the present, in one tidy table, and is refreshed monthly.

Good for: load forecasting (hourly, with EIA's own day-ahead forecast as a benchmark), renewable-generation and duck-curve analysis, fuel-mix trends (coal, gas, solar, wind, batteries), anomaly detection, and grid-interchange studies.

## Files
| File | Rows | What |
|---|---|---|
| `eia930_balance_hourly.parquet` | {len(df):,} | One row per balancing authority and hour; all metrics below. |
| `region_hourly.parquet` | {len(regional):,} | Hourly totals per EIA region and for the lower 48 (`US48`): demand and generation by source, summed from the BA *adjusted* values (flagged demand outliers excluded). |
| `balancing_authorities.csv` | {len(ba_sheet)} | EIA reference table: BA code, name, time zone, region, generation-only flag, active/retired dates. |

## Columns (hourly file)
- `ba`, `region`: balancing-authority and region codes (see `balancing_authorities.csv`).
- `utc_time_end`, `local_time_end`: **end** of the hour, in UTC and in the BA's local time (EIA convention: the 01:00 row covers 00:00-01:00).
- `demand_forecast_mw`, `demand_mw`, `net_generation_mw`, `total_interchange_mw`, `sum_valid_dibas_mw`: values **as reported** by the BA (MW, hourly average).
- `*_adjusted`: EIA's cleaned series, where missing or implausible reported values are replaced by EIA imputations. **Use these for most analysis.**
- `*_imputed_flag`: True where EIA imputed the adjusted value.
- `demand_mw_adjusted_outlier_flag`: added by this dataset ({n_outliers:,} hours, {n_outliers / df.demand_mw_adjusted.notna().sum():.2%}): adjusted demand below 0.5x or above 1.5x the median of the same hour on the 14 surrounding days. Example: PJM's adjusted demand jumps by about 100 GW for one hour on 2020-07-13.
- `ng_<source>_mw` (+ `_adjusted`, `_imputed_flag`): net generation by energy source. Positive = generation, negative = consumption (e.g. batteries and pumped storage when charging).

Energy-source columns and the first month they have data (EIA added categories over time):

| column | first data |
|---|---|
{fuel_lines}

## Things to know
- **Energy categories changed in mid-2024.** `ng_hydro_incl_pumped_storage_mw` runs until {last_seen['ng_hydro_incl_pumped_storage_mw']:%Y-%m}; after that EIA reports `ng_hydro_mw` and `ng_pumped_storage_mw` separately (and adds battery, geothermal and other storage). For a continuous hydro series, use the combined column before the switch and the sum of the two after it. Before the split, battery and geothermal output sits inside `ng_other_mw`.
- **Raw values contain errors**: {int((df.demand_mw < 0).sum()):,} negative and {int((df.demand_mw > 1.5e5).sum()):,} implausibly large (>150 GW) reported demand values. Adjusted values have no negatives.
- **{gen_only} balancing authorities in EIA's reference table are generation-only** (they own plants but serve no customers), so their demand is empty by design.
- **Rows after {last_actual:%Y-%m-%d %H:%M} UTC contain only the day-ahead forecast**; actual values arrive later (EIA publishes most values the next day, interchange two days later).
- BAs merge, split and retire (see `activation_date` / `retirement_date`), so some series start or end mid-range.
- Lower-48 peak hourly demand in this data: **{peak_v / 1000:,.1f} GW at {peak_t:%Y-%m-%d %H:%M} UTC**.

## Source & license
U.S. Energy Information Administration, Form EIA-930 "Hourly Electric Grid Monitor", six-month BALANCE files (https://www.eia.gov/electricity/gridmonitor/) and EIA-930 reference tables; downloaded {today:%Y-%m-%d}. EIA states: *"U.S. government publications are in the public domain and are not subject to copyright protection. You may use and/or distribute any of our data..."* and asks for an acknowledgment, e.g. **"Source: U.S. Energy Information Administration ({today:%b %Y})"**. Marked CC0 here as the closest Kaggle option to public domain. Not produced or endorsed by EIA. Build scripts: `fetch_eia930.py` / `build_eia930.py` in [kaggle-dataset-pipelines](https://github.com/tb91117/kaggle-dataset-pipelines) (open source).
"""


def fld(name, d, t="number"):
    return {"name": name, "description": d, "type": t}


fields = [fld("ba", "Balancing authority code; join to balancing_authorities.csv (ba_code).", "string"),
          fld("region", "EIA region code (e.g. CAL, TEX, MIDA).", "string"),
          fld("utc_time_end", "End of the hour, UTC.", "datetime"), fld("local_time_end", "End of the hour, BA local time.", "datetime"),
          fld("demand_forecast_mw", "Day-ahead demand forecast as reported (MW)."),
          fld("demand_mw", "Demand as reported (MW)."), fld("demand_mw_adjusted", "Demand after EIA cleaning/imputation (MW)."),
          fld("demand_mw_imputed_flag", "True where EIA imputed the adjusted demand.", "boolean"),
          fld("demand_mw_adjusted_outlier_flag", "True where adjusted demand is <0.5x or >1.5x the same-hour median of the 14 surrounding days (added by this dataset).", "boolean"),
          fld("net_generation_mw", "Net generation as reported (MW)."), fld("net_generation_mw_adjusted", "Net generation after EIA cleaning (MW)."),
          fld("total_interchange_mw", "Net interchange as reported (MW); positive = net export."),
          fld("total_interchange_mw_adjusted", "Net interchange after EIA cleaning (MW)."),
          fld("sum_valid_dibas_mw", "Sum of interchange with directly interconnected BAs (MW).")]
documented = {x["name"] for x in fields}
for c in df.columns:                       # document every remaining column (adjusted / flag variants, fuels)
    if c in documented:
        continue
    base = c.replace("_adjusted", "").replace("_imputed_flag", "")
    what = (f"net generation from {base[3:-3].replace('_', ' ')}" if base.startswith("ng_")
            else base.replace("_mw", "").replace("_", " "))
    if c.endswith("_imputed_flag"):
        fields.append(fld(c, f"True where EIA imputed the adjusted {what}.", "boolean"))
    elif c.endswith("_adjusted"):
        fields.append(fld(c, f"Adjusted {what} after EIA cleaning/imputation (MW)."))
    else:
        fields.append(fld(c, f"{what[0].upper() + what[1:]} as reported (MW); negative = consumption (e.g. charging)."))
fields.sort(key=lambda x: list(df.columns).index(x["name"]))
region_fields = [fld("region", "EIA region code, or US48 for the lower-48 total.", "string"), fld("utc_time_end", "End of the hour, UTC.", "datetime"),
                 fld("demand_mw", "Sum of BA adjusted demand, flagged outliers excluded (MW)."), fld("net_generation_mw", "Sum of BA adjusted net generation (MW).")]
region_fields += [fld(c, f"Sum of BA adjusted net generation from {c[3:-3].replace('_', ' ')} (MW).") for c in regional.columns if c.startswith("ng_")]
meta = {
    "id": f"{OWNER}/{SLUG}",
    "title": f"US Electric Grid Hourly Data (EIA-930) {t0:%Y}-{t1:%Y}",
    "subtitle": "Hourly demand, forecast and generation by fuel for every US grid operator",
    "description": desc,
    "licenses": [{"name": "CC0-1.0"}],
    "keywords": ["energy", "time series analysis", "united states", "electricity"],
    "expectedUpdateFrequency": "monthly",
    "userSpecifiedSources": "U.S. Energy Information Administration, Form EIA-930 Hourly Electric Grid Monitor (https://www.eia.gov/electricity/gridmonitor/). Public domain (U.S. government work).",
    "resources": [
        {"path": "eia930_balance_hourly.parquet", "description": "Hourly data per balancing authority.", "schema": {"fields": fields}},
        {"path": "region_hourly.parquet", "description": "Hourly totals per region and lower 48 (sum of BA adjusted values).", "schema": {"fields": region_fields}},
        {"path": "balancing_authorities.csv", "description": "EIA-930 balancing authority reference table.", "schema": {"fields": [
            fld(c, c.replace("_", " "), "string") for c in ba_sheet.columns]}},
    ],
}
assert 6 <= len(meta["title"]) <= 50 and 20 <= len(meta["subtitle"]) <= 80, (len(meta["title"]), len(meta["subtitle"]))
(OUT / "dataset-metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
print("metadata written:", meta["title"], "|", meta["subtitle"])

import runpy  # noqa: E402  (cover: distinct 2:1 design; push with tools/push_cover.py)
runpy.run_path(str(Path(__file__).with_name("make_covers.py")), run_name="covers")["eia"]()
