"""Build the World Development Indicators dataset (tidy, documented) from the World Bank bulk CSV.

Usage:  python -I tools/build_wdi.py
Input:  datasets/world-development-indicators/source/raw/  (unzipped WDI_CSV.zip from
        https://databank.worldbank.org/data/download/WDI_CSV.zip)
Output: datasets/world-development-indicators/v1/  (data files + dataset-metadata.json + cover image)
Nothing is uploaded; pushing is a separate, explicit step.
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1] / "datasets" / "world-development-indicators"
RAW, OUT = BASE / "source" / "raw", BASE / "v1"
OUT.mkdir(parents=True, exist_ok=True)
SLUG, OWNER = "world-bank-world-development-indicators", "teddybrown414"

# headline panel: (code, column name)
HEADLINE = [
    ("SP.POP.TOTL", "population"), ("SP.POP.GROW", "population_growth_pct"), ("SP.URB.TOTL.IN.ZS", "urban_pop_pct"),
    ("SP.POP.0014.TO.ZS", "pop_age_0_14_pct"), ("SP.POP.65UP.TO.ZS", "pop_age_65plus_pct"), ("SM.POP.NETM", "net_migration"),
    ("SP.DYN.LE00.IN", "life_expectancy"), ("SP.DYN.TFRT.IN", "fertility_rate"), ("SP.DYN.IMRT.IN", "infant_mortality_per_1000"),
    ("SH.DYN.MORT", "under5_mortality_per_1000"), ("NY.GDP.MKTP.CD", "gdp_usd"), ("NY.GDP.MKTP.KD.ZG", "gdp_growth_pct"),
    ("NY.GDP.PCAP.CD", "gdp_per_capita_usd"), ("NY.GDP.PCAP.PP.KD", "gdp_per_capita_ppp_const2021"),
    ("NY.GNP.PCAP.CD", "gni_per_capita_atlas_usd"), ("FP.CPI.TOTL.ZG", "inflation_cpi_pct"), ("SL.UEM.TOTL.ZS", "unemployment_pct"),
    ("NE.TRD.GNFS.ZS", "trade_pct_gdp"), ("BX.KLT.DINV.WD.GD.ZS", "fdi_inflows_pct_gdp"), ("NV.AGR.TOTL.ZS", "agriculture_pct_gdp"),
    ("NV.IND.TOTL.ZS", "industry_pct_gdp"), ("NV.SRV.TOTL.ZS", "services_pct_gdp"), ("SH.XPD.CHEX.GD.ZS", "health_exp_pct_gdp"),
    ("SH.XPD.CHEX.PC.CD", "health_exp_per_capita_usd"), ("SE.XPD.TOTL.GD.ZS", "education_exp_pct_gdp"),
    ("SE.PRM.ENRR", "school_enroll_primary_gross_pct"), ("SE.SEC.ENRR", "school_enroll_secondary_gross_pct"),
    ("SE.TER.ENRR", "school_enroll_tertiary_gross_pct"), ("SL.TLF.CACT.FE.ZS", "female_labor_participation_pct"),
    ("SG.GEN.PARL.ZS", "women_in_parliament_pct"), ("SI.POV.GINI", "gini_index"), ("SI.POV.DDAY", "poverty_3usd_day_pct"),
    ("EG.ELC.ACCS.ZS", "electricity_access_pct"), ("EG.USE.PCAP.KG.OE", "energy_use_kg_oil_per_capita"),
    ("EG.FEC.RNEW.ZS", "renewable_energy_pct"), ("EN.GHG.CO2.PC.CE.AR5", "co2_t_per_capita"), ("AG.LND.FRST.ZS", "forest_area_pct"),
    ("AG.LND.AGRI.ZS", "agricultural_land_pct"), ("IT.NET.USER.ZS", "internet_users_pct"), ("IT.CEL.SETS.P2", "mobile_subs_per_100"),
]


def snake(s: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", s.strip().lower()).strip("_")


def read(name):
    df = pd.read_csv(RAW / name, encoding="utf-8-sig", dtype=str, keep_default_na=False, na_values=[""])
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]


# ---------------------------------------------------------------- metadata tables
series = read("WDISeries.csv")
series.columns = [snake(c) for c in series.columns]
restricted = series[series.license_type.str.replace(r"[\s-]", "", regex=True).str.upper() != "CCBY4.0"]
excluded = set(restricted.series_code)
series = series[~series.series_code.isin(excluded)].rename(columns={"series_code": "indicator_code", "indicator_name": "indicator_name"})
series.insert(2, "topic_group", series.topic.str.split(":").str[0].str.strip())

countries = read("WDICountry.csv")
countries.columns = [snake(c) for c in countries.columns]
countries.insert(1, "is_aggregate", countries.region.isna())
economies = set(countries.loc[~countries.is_aggregate, "country_code"])

# ---------------------------------------------------------------- values
wide = pd.read_csv(RAW / "WDICSV.csv", encoding="utf-8-sig")
wide = wide.loc[:, ~wide.columns.str.startswith("Unnamed")]
years = [c for c in wide.columns if c.isdigit()]
wide = wide[~wide["Indicator Code"].isin(excluded)]
long = (wide.melt(id_vars=["Country Code", "Indicator Code"], value_vars=years, var_name="year", value_name="value")
            .dropna(subset=["value"]))
long.columns = ["country_code", "indicator_code", "year", "value"]
long["year"] = long.year.astype("int16")
long = long.sort_values(["country_code", "indicator_code", "year"]).reset_index(drop=True)
for c in ("country_code", "indicator_code"):
    long[c] = long[c].astype("category")
long.to_parquet(OUT / "wdi_long.parquet", index=False, compression="zstd")

# notes
fn = read("WDIfootnote.csv")
fn.columns = ["country_code", "indicator_code", "year", "note"]
fn = fn[~fn.indicator_code.isin(excluded)]
fn["year"] = pd.to_numeric(fn.year.str.replace("YR", "", regex=False), errors="coerce").astype("Int16")
fn.to_parquet(OUT / "footnotes.parquet", index=False, compression="zstd")
cs = read("WDIcountry-series.csv"); cs.columns = ["country_code", "indicator_code", "note"]
cs = cs[~cs.indicator_code.isin(excluded)]
cs.to_csv(OUT / "country_indicator_notes.csv", index=False)
st = read("WDIseries-time.csv"); st.columns = ["indicator_code", "year", "note"]
st = st[~st.indicator_code.isin(excluded)]
st["year"] = pd.to_numeric(st.year.str.replace("YR", "", regex=False), errors="coerce").astype("Int16")
st.to_csv(OUT / "indicator_year_notes.csv", index=False)

series.to_csv(OUT / "indicators.csv", index=False)
countries.to_csv(OUT / "countries.csv", index=False)

# headline panel (economies only)
codes = dict(HEADLINE)
assert set(codes) <= set(series.indicator_code), set(codes) - set(series.indicator_code)
h = long[long.indicator_code.isin(codes) & long.country_code.isin(economies)].copy()
h["indicator_code"] = h.indicator_code.astype(str).map(codes)
panel = h.pivot_table(index=["country_code", "year"], columns="indicator_code", values="value", observed=True).reset_index()
meta = countries.set_index("country_code")[["table_name", "region", "income_group"]].rename(columns={"table_name": "country_name"})
panel = panel.join(meta, on="country_code")
panel = panel[["country_code", "country_name", "region", "income_group", "year"] + [c for _, c in HEADLINE if c in panel.columns]]
panel = panel.sort_values(["country_code", "year"])
panel.to_csv(OUT / "wdi_headline_panel.csv", index=False)

# ---------------------------------------------------------------- facts + cover
n_values, n_ind = len(long), long.indicator_code.nunique()
n_econ, n_agg = len(economies), int(countries.is_aggregate.sum())
y0, y1 = int(long.year.min()), int(long.year.max())
latest_full = int(long[long.country_code.isin(economies)].groupby("year").size().loc[lambda s: s > 0.8 * s.max()].index.max())
cov_year = long[long.country_code.isin(economies)].groupby("year").size()

yr = 2023
pt = panel[panel.year == yr].dropna(subset=["gdp_per_capita_ppp_const2021", "life_expectancy", "population"])
fig = plt.figure(figsize=(16, 9), dpi=100)
fig.patch.set_facecolor("#0f172a")
ax = fig.add_axes([0.07, 0.10, 0.88, 0.66]); ax.set_facecolor("#0f172a")
pal = dict(zip(sorted(pt.region.unique()), ["#38bdf8", "#a78bfa", "#34d399", "#fbbf24", "#f472b6", "#fb923c", "#f87171"]))
ax.scatter(pt.gdp_per_capita_ppp_const2021, pt.life_expectancy, s=np.sqrt(pt.population) / 12,
           c=pt.region.map(pal), alpha=0.75, edgecolors="white", linewidths=0.4)
ax.set_xscale("log")
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(colors="#94a3b8", labelsize=13)
ax.set_xlabel(f"GDP per capita, PPP (constant 2021 $), {yr}", color="#94a3b8", fontsize=14)
ax.set_ylabel("Life expectancy (years)", color="#94a3b8", fontsize=14)
fig.text(0.05, 0.90, "World Development Indicators", color="white", fontsize=40, fontweight="bold")
fig.text(0.05, 0.84, f"World Bank · {n_ind:,} indicators · {n_econ} economies · {y0}-{y1} · tidy & documented", color="#94a3b8", fontsize=20)
fig.savefig(OUT / "dataset-cover-image.png", facecolor=fig.get_facecolor()); plt.close(fig)

topics = series.topic_group.value_counts()
last_year = {col: int(panel.loc[panel[col].notna(), "year"].max()) for _, col in HEADLINE}
lagging = ", ".join(f"`{k}` (latest {v})" for k, v in sorted(last_year.items(), key=lambda kv: kv[1]) if v < latest_full) or "none"
topics_md = ", ".join(f"{t} ({n})" for t, n in topics.items())
src_date = pd.Timestamp((RAW / "WDICSV.csv").stat().st_mtime, unit="s").strftime("%Y-%m-%d")
excluded_names = "; ".join(sorted(restricted.indicator_name))

desc = f"""## What this is
The World Bank's **World Development Indicators (WDI)**, the standard cross-country database for development economics: **{n_values:,} values** for **{n_ind:,} indicators**, **{n_econ} economies** plus {n_agg} regional and income aggregates, **{y0}-{y1}**. Population, GDP, health, education, energy, emissions, labour, trade, poverty, gender and more.

Kaggle's most-voted WDI copy was last updated in 2017. This is the World Bank release of **{src_date}**, reshaped into tidy tables with every indicator and country documented, and refreshed monthly.

## Files
| File | Rows | What |
|---|---|---|
| `wdi_long.parquet` | {n_values:,} | **All values**, one row per (country_code, indicator_code, year, value). Empty cells are dropped. |
| `wdi_headline_panel.csv` | {len(panel):,} | **Start here**: {len(HEADLINE)} popular indicators as columns, one row per economy and year, with name, region and income group. Aggregates excluded. |
| `indicators.csv` | {len(series):,} | Every indicator: code, name, topic, unit, short and long definition, source, aggregation method, limitations, license. |
| `countries.csv` | {len(countries):,} | Every economy and aggregate: names, ISO-2 code, region, income group, lending category, currency, national-accounts notes, `is_aggregate`. |
| `footnotes.parquet` | {len(fn):,} | Notes on individual values (country, indicator, year). |
| `country_indicator_notes.csv` | {len(cs):,} | Source notes for a country-indicator pair. |
| `indicator_year_notes.csv` | {len(st):,} | Notes for an indicator in a given year. |

Indicators by topic: {topics_md}.

## Quick start
```python
import pandas as pd
panel = pd.read_csv("/kaggle/input/{SLUG}/wdi_headline_panel.csv")
long = pd.read_parquet("/kaggle/input/{SLUG}/wdi_long.parquet")
ind = pd.read_csv("/kaggle/input/{SLUG}/indicators.csv")
gdp = long[long.indicator_code == "NY.GDP.PCAP.CD"].pivot(index="year", columns="country_code", values="value")
```

## Things to know
- **Aggregates are mixed in** with countries in `wdi_long.parquet` (World, regions, income groups). Filter with `countries.is_aggregate == False` before ranking or averaging countries.
- **Recent years are incomplete**: values per year for economies peak around {int(cov_year.idxmax())} and fall off after {latest_full}, because many series are published with a lag. The last year ({y1}) has {cov_year.loc[y1]:,} values versus {cov_year.max():,} at the peak.
- Headline indicators that stop earlier than {latest_full}: {lagging}.
- Coverage varies a lot by indicator; check `indicators.csv` and count non-missing values before choosing one.
- Values are revised in place by the World Bank, so numbers can change between versions of this dataset.

## Differences from the World Bank file
- Reshaped from wide (one column per year) to long; empty cells dropped; codes stored as categories in Parquet.
- Added `topic_group` (first part of `topic`) to indicators and `is_aggregate` to countries; column names in snake_case.
- **Excluded {len(excluded)} indicators published under separate SIPRI terms** rather than CC BY 4.0, so the whole dataset is CC BY 4.0: {excluded_names}.

## Source & license
World Bank, World Development Indicators, bulk download https://databank.worldbank.org/data/download/WDI_CSV.zip (release of {src_date}); catalog: https://datacatalog.worldbank.org/search/dataset/0037712. License: [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). Please cite *World Bank, World Development Indicators* when you use the data. This dataset is not produced or endorsed by the World Bank. Build script: `build_wdi.py` by the dataset maintainer.
"""

fields_long = [
    {"name": "country_code", "description": "ISO-3-style economy or aggregate code; join to countries.csv.", "type": "string"},
    {"name": "indicator_code", "description": "WDI series code (e.g. NY.GDP.PCAP.CD); join to indicators.csv.", "type": "string"},
    {"name": "year", "description": "Calendar year of the value.", "type": "integer"},
    {"name": "value", "description": "Indicator value, in the unit given in indicators.csv.", "type": "number"},
]
ind_names = series.set_index("indicator_code").indicator_name
fields_panel = ([{"name": "country_code", "description": "Economy code; join to countries.csv.", "type": "string"},
                 {"name": "country_name", "description": "Economy name (World Bank table name).", "type": "string"},
                 {"name": "region", "description": "World Bank region.", "type": "string"},
                 {"name": "income_group", "description": "World Bank income group (current classification).", "type": "string"},
                 {"name": "year", "description": "Calendar year.", "type": "integer"}]
                + [{"name": col, "description": f"{ind_names[code]} [{code}]", "type": "number"} for code, col in HEADLINE])
IND_DOC = {
    "indicator_code": "WDI series code (e.g. NY.GDP.PCAP.CD).", "topic": "Full topic path, e.g. 'Economic Policy & Debt: National accounts'.",
    "topic_group": "Top-level topic (first part of topic). Added by this dataset.", "indicator_name": "Indicator name.",
    "short_definition": "Short definition.", "long_definition": "Full definition.", "unit_of_measure": "Unit, where given.",
    "periodicity": "How often the series is observed (e.g. Annual).", "base_period": "Base year for constant-price or index series.",
    "aggregation_method": "How regional/income aggregates are computed (e.g. weighted average, sum).",
    "limitations_and_exceptions": "Known limitations of the series.", "source": "Original data source.",
    "license_type": "License of the series (all CC BY 4.0 here).",
}
CTRY_DOC = {
    "country_code": "Economy or aggregate code.", "is_aggregate": "True for regions, income groups and other aggregates. Added by this dataset.",
    "short_name": "Short name.", "table_name": "Name used in World Bank tables.", "long_name": "Official long name.",
    "2_alpha_code": "ISO 3166 alpha-2 style code.", "currency_unit": "National currency.", "special_notes": "World Bank notes on the economy.",
    "region": "World Bank region (empty for aggregates).", "income_group": "World Bank income group (current classification).",
    "wb_2_code": "World Bank 2-letter code.", "lending_category": "IDA / IBRD / Blend lending category.",
    "other_groups": "Other groupings (e.g. Euro area, HIPC).",
}
meta = {
    "id": f"{OWNER}/{SLUG}",
    "title": f"World Development Indicators {y0}-{y1}",
    "subtitle": f"World Bank WDI: {n_ind:,} indicators for {n_econ} economies, tidy and documented",
    "description": desc,
    "licenses": [{"name": "Attribution 4.0 International (CC BY 4.0)"}],  # create accepts "CC-BY-4.0"; metadata --update needs this full name
    "keywords": ["economics", "demographics", "global", "health", "social science"],
    "expectedUpdateFrequency": "monthly",
    "userSpecifiedSources": "World Bank, World Development Indicators (https://datacatalog.worldbank.org/search/dataset/0037712), CC BY 4.0.",
    "resources": [
        {"path": "wdi_long.parquet", "description": "All WDI values in long format.", "schema": {"fields": fields_long}},
        {"path": "wdi_headline_panel.csv", "description": "Popular indicators as columns, one row per economy-year.", "schema": {"fields": fields_panel}},
        {"path": "indicators.csv", "description": "Indicator definitions and metadata.", "schema": {"fields": [
            {"name": c, "description": IND_DOC.get(c, c.replace("_", " ").capitalize() + " (World Bank series metadata)."), "type": "string"}
            for c in series.columns]}},
        {"path": "countries.csv", "description": "Economies and aggregates with classifications.", "schema": {"fields": [
            {"name": c, "description": CTRY_DOC.get(c, c.replace("_", " ").capitalize() + " (World Bank country metadata)."),
             "type": "boolean" if c == "is_aggregate" else "string"} for c in countries.columns]}},
        {"path": "footnotes.parquet", "description": "Notes attached to individual values.", "schema": {"fields": [
            {"name": "country_code", "description": "Economy code.", "type": "string"},
            {"name": "indicator_code", "description": "Series code.", "type": "string"},
            {"name": "year", "description": "Year the note applies to.", "type": "integer"},
            {"name": "note", "description": "Footnote text.", "type": "string"}]}},
        {"path": "country_indicator_notes.csv", "description": "Source notes for a country-indicator pair.", "schema": {"fields": [
            {"name": "country_code", "description": "Economy code.", "type": "string"},
            {"name": "indicator_code", "description": "Series code.", "type": "string"},
            {"name": "note", "description": "Note on how this country's series is sourced or estimated.", "type": "string"}]}},
        {"path": "indicator_year_notes.csv", "description": "Notes for an indicator in a given year.", "schema": {"fields": [
            {"name": "indicator_code", "description": "Series code.", "type": "string"},
            {"name": "year", "description": "Year the note applies to.", "type": "integer"},
            {"name": "note", "description": "Note on the series in that year.", "type": "string"}]}},
    ],
}
assert 6 <= len(meta["title"]) <= 50 and 20 <= len(meta["subtitle"]) <= 80, (len(meta["title"]), len(meta["subtitle"]))
(OUT / "dataset-metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"values {n_values:,}; indicators {n_ind}; economies {n_econ}; aggregates {n_agg}; years {y0}-{y1}; panel {panel.shape}; "
      f"excluded {sorted(excluded)}; latest>80% year {latest_full}; src {src_date}")

# the cover drawn above is a placeholder; the published cover comes from make_covers.py (2:1, distinct design)
import runpy  # noqa: E402
runpy.run_path(str(Path(__file__).with_name("make_covers.py")), run_name="covers")["wdi"]()
