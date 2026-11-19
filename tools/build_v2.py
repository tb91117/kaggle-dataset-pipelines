"""Build cleaned v2 data, dataset-metadata.json and cover images for the public datasets.

Usage:  python -I tools/build_v2.py intent|binance|trees
Output: datasets/<slug>/v2/  (data files + dataset-metadata.json + dataset-cover-image.png)
Nothing is uploaded; pushing is a separate, explicit step.
"""
import json
import sys
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# KDP_DATA_ROOT overrides where sources are downloaded and datasets are built (used by the Kaggle kernels)
DATA_ROOT = Path(os.environ.get("KDP_DATA_ROOT") or Path(__file__).resolve().parents[1] / "datasets")
ROOT = DATA_ROOT
OWNER = "teddybrown414"
W, H, DPI = 16, 9, 100  # 1600x900 cover


def out_dir(slug: str) -> Path:
    d = ROOT / slug / "v2"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_meta(slug: str, meta: dict) -> None:
    sub = meta["subtitle"]
    assert 20 <= len(sub) <= 80, f"subtitle length {len(sub)} must be 20-80"
    assert 6 <= len(meta["title"]) <= 50, f"title length {len(meta['title'])} must be 6-50"
    meta = {"id": f"{OWNER}/{slug}", **meta}
    (out_dir(slug) / "dataset-metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def cover_style(ax, fig, title, subtitle):
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors="#94a3b8", labelsize=14)
    fig.text(0.05, 0.90, title, color="white", fontsize=40, fontweight="bold")
    fig.text(0.05, 0.84, subtitle, color="#94a3b8", fontsize=20)


# --------------------------------------------------------------------------- intent
def build_intent():
    slug = "data-for-classifying-intent-of-query"
    src = ROOT / slug / "data" / "data_training_classifier.csv"
    df = pd.read_csv(src)
    df["topic"] = df["topic"].str.strip().str.strip('"')
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    d = out_dir(slug)
    df.to_csv(d / "query_intents.csv", index=False)
    counts = df["topic"].value_counts().sort_values()

    fig, ax = plt.subplots(figsize=(W, H), dpi=DPI)
    cover_style(ax, fig, "Query Intent Routing", "Which tool should an assistant call? 5 intent classes")
    ax.barh(counts.index, counts.values, color=["#38bdf8", "#a78bfa", "#34d399", "#fbbf24", "#f472b6"])
    ax.set_position([0.18, 0.12, 0.75, 0.62])
    ax.tick_params(axis="y", labelsize=22, colors="white")
    ax.set_xlabel("examples", color="#94a3b8", fontsize=16)
    fig.savefig(d / "dataset-cover-image.png", facecolor=fig.get_facecolor())
    plt.close(fig)

    examples = {t: df[df.topic == t].text.head(2).tolist() for t in sorted(df.topic.unique())}
    ex_md = "\n".join(f"| `{t}` | {len(df[df.topic == t])} | {e[0]} / {e[1]} |" for t, e in examples.items())
    write_meta(slug, {
        "title": "Query Intent Routing for AI Assistants",
        "subtitle": "Short user queries labeled by the tool an assistant should call",
        "description": f"""## What this is
A small, clean dataset for **routing user requests to the right tool** in an AI-assistant or agent pipeline: should the request go to an image generator, a web search, a web scraper, a Wikipedia lookup, or plain chat?

It's meant for quick prototyping of intent classifiers (TF-IDF + logistic regression, sentence embeddings + kNN, or zero-shot LLM prompts) and for teaching text classification.

## Contents
`query_intents.csv`: **{len(df)} rows, 2 columns** (`text`, `topic`), 5 classes. `image` has {int((df.topic == 'image').sum())} rows after de-duplication; the other classes have 20 each.

| topic | rows | examples |
|---|---|---|
{ex_md}

## Version 2 changes
- Removed {before - len(df)} exact duplicate rows ({before} → {len(df)}).
- Cleaned the label column: values were stored with a leading space and literal quotes (e.g. ` "image"`); they are now plain strings (`image`).
- Renamed the file to `query_intents.csv`.

## Limitations
- Small and simple: phrasing is template-like, so accuracy on this set will overstate real-world performance. Use it as a starter or seed set, and evaluate on real traffic.
- `scraper` examples are bare URLs, so that class is trivially separable.

## Source & license
Created by the dataset author for an assistant-routing prototype. Released under CC0 1.0 (public domain dedication).
""",
        "licenses": [{"name": "CC0-1.0"}],
        "keywords": ["nlp", "text", "classification"],
        "expectedUpdateFrequency": "never",
        "userSpecifiedSources": "Created by the dataset author (teddybrown414) for an LLM assistant tool-routing prototype.",
        "resources": [{
            "path": "query_intents.csv",
            "description": "One user query per row with the tool/intent it should be routed to.",
            "schema": {"fields": [
                {"name": "text", "description": "The user's request: a question, instruction, or a bare URL.", "type": "string"},
                {"name": "topic", "description": "Intent label: chat, image, scraper, search or wikipedia.", "type": "string"},
            ]},
        }],
    })
    print(f"intent: {before} -> {len(df)} rows; classes {counts.to_dict()}")


# --------------------------------------------------------------------------- binance
def build_binance():
    slug = "binance-ohlc-with-indicators"
    src = ROOT / slug / "data" / "binance_ohlc_with_indicators.csv"
    df = pd.read_csv(src)
    before = len(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    vc = df["timestamp"].value_counts()
    dup_ts, dup_max = int((vc > 1).sum()), int(vc.max())
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    gaps = int((df["timestamp"].diff() > pd.Timedelta(minutes=15)).sum())
    d = out_dir(slug)
    df.to_csv(d / "btcusdt_15m_indicators.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    t0, t1 = df.timestamp.min(), df.timestamp.max()

    fig, ax = plt.subplots(figsize=(W, H), dpi=DPI)
    cover_style(ax, fig, "BTC/USDT 15-Minute Candles", f"Binance spot, {t0:%b %Y} – {t1:%b %Y}, with MACD, RSI, Stochastic, ATR")
    daily = df.set_index("timestamp")["close"].resample("D").last()
    ax.plot(daily.index, daily.values, color="#f59e0b", lw=2.2)
    ax.fill_between(daily.index, daily.values, daily.min(), color="#f59e0b", alpha=0.12)
    ax.set_yscale("log")
    ax.set_position([0.07, 0.10, 0.88, 0.66])
    fig.savefig(d / "dataset-cover-image.png", facecolor=fig.get_facecolor())
    plt.close(fig)

    sig = df["signal"].value_counts().to_dict()
    cols = [
        ("timestamp", "Candle open time (UTC), 15-minute bars.", "datetime"),
        ("open", "Open price in USDT.", "number"),
        ("high", "Highest price in the bar (USDT).", "number"),
        ("low", "Lowest price in the bar (USDT).", "number"),
        ("close", "Close price in USDT.", "number"),
        ("volume", "Traded volume in BTC during the bar.", "number"),
        ("MACD", "MACD line (EMA12 − EMA26 of close).", "number"),
        ("MACD_signal", "MACD signal line (EMA9 of MACD).", "number"),
        ("MACD_hist", "MACD histogram (MACD − signal).", "number"),
        ("RSI", "Relative Strength Index (0–100) of close.", "number"),
        ("stoch_k", "Stochastic oscillator %K (0–100).", "number"),
        ("stoch_d", "Stochastic oscillator %D, smoothed %K (0–100).", "number"),
        ("ATR", "Average True Range in USDT (volatility).", "number"),
        ("signal", "Look-ahead direction label: 1 if the rise to tp is larger than the fall to sl, -1 if the fall is larger, 0 when neither direction dominates (tp and sl are then 0).", "integer"),
        ("tp", "Approximately the highest high over roughly the next 24 hours (~96-100 bars); 0 when signal = 0. Future information.", "number"),
        ("sl", "Approximately the lowest low over roughly the next 24 hours (~96-100 bars); 0 when signal = 0. Future information.", "number"),
    ]
    write_meta(slug, {
        "title": "BTC/USDT 15-min Candles + Indicators 2017-2025",
        "subtitle": f"{len(df):,} Binance spot candles with MACD, RSI, Stochastic, ATR and labels",
        "description": f"""## What this is
Every 15-minute **BTC/USDT** candle on Binance spot from **{t0:%Y-%m-%d}** (the pair's first day of trading) to **{t1:%Y-%m-%d}**, with common technical indicators pre-computed and a long/short label column. Ready for time-series forecasting, feature-engineering practice, and backtesting experiments.

## Contents
`btcusdt_15m_indicators.csv`: **{len(df):,} rows × {df.shape[1]} columns**, sorted by time, one row per candle.
- Price/volume: `open`, `high`, `low`, `close`, `volume`
- Indicators: `MACD`, `MACD_signal`, `MACD_hist`, `RSI`, `stoch_k`, `stoch_d`, `ATR`
- Labels: `signal` (1 long: {sig.get(1, 0):,}, -1 short: {sig.get(-1, 0):,}, 0 none: {sig.get(0, 0):,}), `tp`, `sl`

Indicator columns are empty for the first rows of the series (indicator warm-up).

## ⚠️ Use the labels carefully
`signal`, `tp` and `sl` are built from **future** prices. `tp` and `sl` match the highest high and lowest low over roughly the next 24 hours (about 86% of labelled rows match a 99-bar forward window exactly), and `signal` is always the sign of (rise to `tp`) minus (fall to `sl`); `signal = 0` marks bars where neither direction dominates. Treat these columns as prediction *targets*, not features. Using them (or anything computed from them) as model inputs leaks the future. Split by time, never randomly.

## Version 2 changes
- Removed {before - len(df):,} duplicate rows: {dup_ts:,} candles appeared more than once (up to {dup_max} times), always with identical values ({before:,} → {len(df):,} rows).
- Sorted by timestamp (v1 was partly out of order).
- Renamed the file to `btcusdt_15m_indicators.csv`.
- Remaining time gaps longer than 15 minutes: {gaps} (exchange downtime/maintenance; rows are not forward-filled).

## Ideas
- Forecast next-bar return or volatility; compare GBDTs with sequence models.
- Evaluate indicator-based strategies with walk-forward validation and trading costs.

## Source & license
Candles were downloaded from the Binance public market-data API (symbol `BTCUSDT`, interval `15m`); indicators and labels were computed by the dataset author. Market data © Binance; see the Binance Terms of Use for redistribution terms. Not financial advice.
""",
        "licenses": [{"name": "other"}],
        "keywords": ["finance", "investing", "time series analysis", "currencies and foreign exchange"],
        "expectedUpdateFrequency": "never",
        "userSpecifiedSources": "Binance public market-data API (spot, BTCUSDT, 15m klines). Indicators and labels computed by the dataset author.",
        "resources": [{
            "path": "btcusdt_15m_indicators.csv",
            "description": "BTC/USDT 15-minute candles from Binance spot with technical indicators and long/short labels.",
            "schema": {"fields": [{"name": n, "description": desc, "type": t} for n, desc, t in cols]},
        }],
    })
    print(f"binance: {before} -> {len(df)} rows, {t0} -> {t1}, gaps>15m: {gaps}")


# --------------------------------------------------------------------------- SF trees
def build_trees():
    slug = "san-francisco-trees"
    src = ROOT / slug / "source" / "street_trees_full.csv"
    df = pd.read_csv(src, dtype=str)
    raw_rows = len(df)
    loaded_at = df["data_loaded_at"].dropna().max()
    df = df.drop(columns=["point", "data_loaded_at"]).drop_duplicates("treeid")
    df["planttype"] = df["planttype"].str.strip().str.capitalize()
    sp = df["species"].str.split(" :: ", n=1, expand=True)
    df.insert(df.columns.get_loc("species") + 1, "species_scientific", sp[0].str.strip())
    df.insert(df.columns.get_loc("species") + 2, "species_common", sp[1].str.strip() if 1 in sp else None)
    df.insert(df.columns.get_loc("description") + 1, "address", df["description"].str.replace(r"\s*\|\s*Tree \d+\s*$", "", regex=True))
    for c in ["treeid", "siteorder", "mapdbh", "xcoord", "ycoord", "latitude", "longitude"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["planteddate"] = pd.to_datetime(df["planteddate"], errors="coerce").dt.date
    df["treeid"] = df["treeid"].astype("Int64")
    frac = df["siteorder"].notna() & (df["siteorder"] % 1 != 0)
    n_frac = int(frac.sum())  # e.g. one 0.025026 whose description says "Tree 0"
    df.loc[frac, "siteorder"] = df.loc[frac, "description"].str.extract(r"Tree (\d+)\s*$")[0].astype(float)
    df["siteorder"] = df["siteorder"].astype("Int64")
    df = df.sort_values("treeid").reset_index(drop=True)
    d = out_dir(slug)
    df.to_csv(d / "sf_street_trees.csv", index=False)

    # facts for the description (computed, never hand-typed)
    n = len(df)
    geo = df[["latitude", "longitude"]].notna().all(axis=1)
    top = df["species_common"].value_counts().head(5)
    top_md = "\n".join(f"| {name} | {cnt:,} |" for name, cnt in top.items())
    dbh_rng = df.groupby("dbhrange")["mapdbh"].agg(["min", "max", "count"])
    dbh_md = "; ".join(f"`{k}` = {int(r['min']) if pd.notna(r['min']) else '?'}-{int(r['max']) if pd.notna(r['max']) else '?'} in" for k, r in dbh_rng.iterrows())
    pdates = pd.to_datetime(df["planteddate"])
    as_of = pd.to_datetime(df["data_as_of"]).max()

    fig, ax = plt.subplots(figsize=(W, H), dpi=DPI)
    cover_style(ax, fig, "San Francisco Street Trees", f"{n:,} trees · {df['species_scientific'].nunique()} species labels · San Francisco Public Works")
    g = df[geo & df.latitude.between(37.6, 37.85) & df.longitude.between(-122.53, -122.35)]
    ax.scatter(g.longitude, g.latitude, s=0.5, color="#4ade80", alpha=0.6, linewidths=0)
    ax.set_aspect(1 / 0.79)  # ~1/cos(37.76 deg)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_position([0.05, 0.03, 0.90, 0.78])
    fig.savefig(d / "dataset-cover-image.png", facecolor=fig.get_facecolor())
    plt.close(fig)

    cols = [
        ("treeid", "Street-tree asset unique ID (Public Works).", "integer"),
        ("legalstatus", "Legal status, e.g. DPW Maintained, Significant Tree, Landmark Tree, Section 806 (d), Undocumented.", "string"),
        ("species", "Species as published: 'Scientific name :: Common name'.", "string"),
        ("species_scientific", "Scientific name (part before ' :: ' in species). Added in v2.", "string"),
        ("species_common", "Common name (part after ' :: ' in species). Added in v2.", "string"),
        ("description", "Location description as published, typically 'address | Tree N'.", "string"),
        ("address", "Street address with the ' | Tree N' suffix removed. Added in v2.", "string"),
        ("siteorder", "Order of the tree at the address when several trees share one address.", "integer"),
        ("siteinfo", "Where the tree sits, e.g. 'Sidewalk: Curb side : Cutout'.", "string"),
        ("planttype", "Plant type (Tree).", "string"),
        ("planter", "Agency or party that planted the tree, where known (e.g. Private, DPW, FUF).", "string"),
        ("waterresponsibility", "Party responsible for watering the tree, where known.", "string"),
        ("planteddate", "Date the tree was planted (YYYY-MM-DD); empty for most older trees.", "date"),
        ("mapdbh", "Diameter at breast height, in inches.", "number"),
        ("plotsize", "Tree plot dimensions as recorded (free text, e.g. 'Width 4ft', '3x4').", "string"),
        ("xcoord", "X coordinate, NAD83 / California zone 3 (US feet).", "number"),
        ("ycoord", "Y coordinate, NAD83 / California zone 3 (US feet).", "number"),
        ("latitude", "Latitude, WGS84.", "number"),
        ("longitude", "Longitude, WGS84.", "number"),
        ("supervisordistrict", "Board of Supervisors district (1-11).", "string"),
        ("analysis_neighborhood", "San Francisco Analysis Neighborhood name.", "string"),
        ("dbhrange", "DBH size class used for map symbols (see dataset description for ranges).", "string"),
        ("cnn", "CNN (centerline network number) of the street segment; keep as text (leading zeros).", "string"),
        ("roadwaydescription", "Text description of the street segment (CNN), e.g. 'EDNA ST to FOERSTER ST (400 - 499)'.", "string"),
        ("onstreet", "Street name where the tree is located.", "string"),
        ("crossstreet", "First cross street of the street segment.", "string"),
        ("zip_code", "ZIP code where the tree is located.", "string"),
        ("data_as_of", "Date the record was last updated in the Public Works system.", "date"),
    ]
    assert [c for c, _, _ in cols] == list(df.columns), (list(df.columns), [c for c, _, _ in cols])
    write_meta(slug, {
        "title": "San Francisco Street Tree Inventory 2026",
        "subtitle": "Every street tree in SF: species, size, planting date and location",
        "description": f"""## What this is
The City of San Francisco's official **street tree inventory**: one row per street tree, with species, trunk diameter, planting date, who planted and waters it, and its location (lat/lon, address, street segment, neighborhood, supervisor district, ZIP). Published by **San Francisco Public Works** on DataSF and released into the public domain (PDDL).

Good for urban-forestry and canopy-equity analysis, geospatial mapping and clustering, species-diversity studies, and tabular ML practice (e.g. predicting trunk size from species, age and site).

## Contents
`sf_street_trees.csv`: **{n:,} trees × {df.shape[1]} columns**, {df['species_scientific'].nunique()} distinct species labels, snapshot as of **{as_of:%Y-%m-%d}**.
- {geo.mean():.1%} of trees have coordinates; {df['mapdbh'].notna().mean():.1%} have a trunk diameter; {pdates.notna().mean():.1%} have a planting date ({pdates.min():%Y} to {pdates.max():%Y}).
- `dbhrange` classes: {dbh_md}.

| Most common species | Trees |
|---|---|
{top_md}

## Version 2 changes
- **New source.** v1 was built from the old *Street Tree List* (`uzd4-f6yf`), which the city has since archived. Public Works moved to a new asset system and published a corrected inventory (`tkzw-k3nq`, cleaned in place in September 2026). v2 is that current inventory.
- **Fewer rows, on purpose:** v1 had 197,096 rows; the new inventory keeps only actual trees and excludes empty basins, landscaping areas and potential planting sites, so v2 has {n:,}.
- Real place names instead of opaque codes: v1's `Zip Codes`, `Neighborhoods`, `Police Districts` and similar columns were internal portal region IDs (e.g. "57"), not ZIP codes or names. v2 has actual ZIP codes, neighborhood names and street segments.
- Added `species_scientific`, `species_common` and `address`; `mapdbh` is numeric; `planteddate` is ISO dates; `planttype` capitalisation is fixed; {n_frac} malformed `siteorder` value replaced with the tree number from `description`; dropped `point` (duplicates lat/lon) and the constant `data_loaded_at` column. Every `treeid` is unique ({raw_rows:,} source rows, {n:,} trees).

## Caveats
From the publisher:
- Removed trees are not included. Trees removed on an emergency basis (storm, hazard) may be missing from both this dataset and the [Street Tree Removal Notifications](https://data.sfgov.org/d/qrwx-q4gg) dataset.
- Records are updated when crews interact with a tree, so a row's details can be years old (`data_as_of` shows the last update). Coordinate precision varies.

From profiling this snapshot:
- Only {pdates.notna().mean():.0%} of trees have a planting date, and {(pdates.dt.year >= 1990).sum() / pdates.notna().sum():.0%} of those are from 1990 or later, so "age" analysis is biased toward younger trees.
- `mapdbh` bottoms out at {int(df['mapdbh'].min())} in, and {(df['mapdbh'] == df['mapdbh'].min()).mean():.0%} of trees sit exactly at that value; it likely acts as a minimum/default for small or unmeasured trees.
- Species labels include placeholders (`Tree(s) ::` on {(df['species'] == 'Tree(s) ::').sum():,} trees, `:: To Be Determine`), genus-only entries and a few spelling variants; {df['species'].isna().sum():,} trees have no species. Group or clean labels before diversity counts.

## Provenance & license
Source: [San Francisco Street Tree Inventory](https://data.sf.gov/City-Infrastructure/San-Francisco-Street-Tree-Inventory/tkzw-k3nq), San Francisco Public Works via DataSF, downloaded through the SODA API (portal load of {pd.Timestamp(loaded_at):%Y-%m-%d}). License: [ODC Public Domain Dedication and License (PDDL) 1.0](http://opendatacommons.org/licenses/pddl/1.0/). Data courtesy of San Francisco Public Works. Cleaning: `fetch_sf_trees.py` / `build_v2.py` in [kaggle-dataset-pipelines](https://github.com/tb91117/kaggle-dataset-pipelines) (open source).
""",
        "licenses": [{"name": "PDDL"}],
        "keywords": ["geospatial analysis", "plants", "environment", "cities and urban areas"],
        "expectedUpdateFrequency": "monthly",
        "userSpecifiedSources": "San Francisco Public Works, San Francisco Street Tree Inventory (DataSF dataset tkzw-k3nq), https://data.sf.gov/d/tkzw-k3nq. License: ODC PDDL 1.0.",
        "resources": [{
            "path": "sf_street_trees.csv",
            "description": "One row per San Francisco street tree (current inventory, San Francisco Public Works).",
            "schema": {"fields": [{"name": c, "description": desc, "type": t} for c, desc, t in cols]},
        }],
    })
    print(f"trees: {raw_rows} source rows -> {n} trees; geo {geo.mean():.1%}; as of {as_of:%Y-%m-%d}")


if __name__ == "__main__":
    import runpy
    {"intent": build_intent, "binance": build_binance, "trees": build_trees}[sys.argv[1]]()
    # the per-builder covers above are placeholders; the published covers come from make_covers.py (2:1, distinct designs)
    covers = runpy.run_path(str(Path(__file__).with_name("make_covers.py")), run_name="covers")
    covers[{"binance": "btc"}.get(sys.argv[1], sys.argv[1])]()
