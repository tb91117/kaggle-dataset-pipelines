"""Build the US flight on-time dataset from BTS 'Reporting Carrier On-Time Performance' monthly zips.

Usage:  python -I tools/build_bts.py         (after tools/fetch_bts.py)
Output: datasets/us-flight-on-time/v1/  flights_<year>.parquet, airports.csv, carriers.csv, dataset-metadata.json
"""
import io
import json
import zipfile
import os
from pathlib import Path

import pandas as pd

# KDP_DATA_ROOT overrides where sources are downloaded and datasets are built (used by the Kaggle kernels)
DATA_ROOT = Path(os.environ.get("KDP_DATA_ROOT") or Path(__file__).resolve().parents[1] / "datasets")
BASE = DATA_ROOT / "us-flight-on-time"
SRC, OUT = BASE / "source", BASE / "v1"
OUT.mkdir(parents=True, exist_ok=True)
SLUG, OWNER = "us-flight-delays-and-on-time-performance-2022-2026", "teddybrown414"
SUB = "out/"   # the dataset is made from the pipeline notebook output, which keeps the out/ folder

# source column -> (output name, dtype, description)
KEEP = {
    "FlightDate": ("flight_date", "date", "Scheduled flight date (local, origin)."),
    "Reporting_Airline": ("airline", "category", "Reporting airline code (BTS unique carrier code); join to carriers.csv."),
    "Tail_Number": ("tail_number", "string", "Aircraft tail number (registration)."),
    "Flight_Number_Reporting_Airline": ("flight_number", "Int32", "Flight number."),
    "OriginAirportID": ("origin_airport_id", "Int32", "BTS airport ID of the origin (stable across code changes); join to airports.csv."),
    "Origin": ("origin", "category", "Origin airport code (IATA/FAA)."),
    "OriginCityName": ("origin_city", "category", "Origin city and state."),
    "OriginState": ("origin_state", "category", "Origin state (2-letter)."),
    "DestAirportID": ("dest_airport_id", "Int32", "BTS airport ID of the destination; join to airports.csv."),
    "Dest": ("dest", "category", "Destination airport code (IATA/FAA)."),
    "DestCityName": ("dest_city", "category", "Destination city and state."),
    "DestState": ("dest_state", "category", "Destination state (2-letter)."),
    "CRSDepTime": ("sched_dep_hhmm", "Int16", "Scheduled departure time, local, hhmm."),
    "DepTime": ("dep_hhmm", "Int16", "Actual departure time, local, hhmm (2400 = midnight)."),
    "DepDelay": ("dep_delay_min", "float32", "Departure delay in minutes (negative = early)."),
    "DepDel15": ("dep_delayed_15", "float32", "1 if the departure was delayed 15+ minutes."),
    "TaxiOut": ("taxi_out_min", "float32", "Taxi-out time in minutes."),
    "WheelsOff": ("wheels_off_hhmm", "Int16", "Wheels-off time, local, hhmm."),
    "WheelsOn": ("wheels_on_hhmm", "Int16", "Wheels-on time, local, hhmm."),
    "TaxiIn": ("taxi_in_min", "float32", "Taxi-in time in minutes."),
    "CRSArrTime": ("sched_arr_hhmm", "Int16", "Scheduled arrival time, local, hhmm."),
    "ArrTime": ("arr_hhmm", "Int16", "Actual arrival time, local, hhmm."),
    "ArrDelay": ("arr_delay_min", "float32", "Arrival delay in minutes (negative = early)."),
    "ArrDel15": ("arr_delayed_15", "float32", "1 if the arrival was delayed 15+ minutes (the official 'late' definition)."),
    "Cancelled": ("cancelled", "bool", "True if the flight was cancelled."),
    "CancellationCode": ("cancellation_code", "category", "Cancellation reason: A = carrier, B = weather, C = National Air System, D = security."),
    "Diverted": ("diverted", "bool", "True if the flight was diverted."),
    "CRSElapsedTime": ("sched_elapsed_min", "float32", "Scheduled gate-to-gate time in minutes."),
    "ActualElapsedTime": ("actual_elapsed_min", "float32", "Actual gate-to-gate time in minutes."),
    "AirTime": ("air_time_min", "float32", "Time in the air in minutes."),
    "Distance": ("distance_mi", "float32", "Distance between airports in miles."),
    "CarrierDelay": ("delay_carrier_min", "float32", "Minutes of arrival delay attributed to the carrier (reported when arrival delay >= 15)."),
    "WeatherDelay": ("delay_weather_min", "float32", "Minutes attributed to extreme weather."),
    "NASDelay": ("delay_nas_min", "float32", "Minutes attributed to the National Airspace System (non-extreme weather, traffic, ATC)."),
    "SecurityDelay": ("delay_security_min", "float32", "Minutes attributed to security."),
    "LateAircraftDelay": ("delay_late_aircraft_min", "float32", "Minutes attributed to a late-arriving previous flight with the same aircraft."),
}


def read_month(z: Path) -> pd.DataFrame:
    with zipfile.ZipFile(z) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(name) as f:
            df = pd.read_csv(io.TextIOWrapper(f, encoding="latin-1"), usecols=list(KEEP), low_memory=False)
    df = df.rename(columns={k: v[0] for k, v in KEEP.items()})
    for src, (col, dt, _) in KEEP.items():
        if dt == "date":
            df[col] = pd.to_datetime(df[col]).dt.date
        elif dt == "bool":
            df[col] = df[col].fillna(0).astype(float).astype(bool)
        elif dt in ("Int16", "Int32"):
            df[col] = pd.to_numeric(df[col], errors="coerce").round().astype(dt)
        elif dt == "float32":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
        elif dt == "string":
            df[col] = df[col].astype("string")
    df["cancellation_code"] = df["cancellation_code"].where(df["cancellation_code"].notna() & (df["cancellation_code"].astype(str).str.strip() != ""))
    return df


zips = sorted((SRC / "zips").glob("ontime_*.zip"))
assert zips, "run tools/fetch_bts.py first"
stats, by_year = [], {}
for z in zips:
    y = int(z.stem.split("_")[1])
    df = read_month(z)
    by_year.setdefault(y, []).append(df)
    stats.append({"month": z.stem[7:], "rows": len(df)})
    print(z.name, len(df), flush=True)

files, totals = {}, {}
for y, parts in sorted(by_year.items()):
    df = pd.concat(parts, ignore_index=True)
    for c in ("airline", "origin", "origin_city", "origin_state", "dest", "dest_city", "dest_state", "cancellation_code"):
        df[c] = df[c].astype("category")
    df = df.sort_values(["flight_date", "airline", "flight_number", "origin"]).reset_index(drop=True)
    name = f"flights_{y}.parquet"
    df.to_parquet(OUT / name, index=False, compression="zstd")
    files[name] = len(df)
    totals[y] = {"rows": len(df), "cancel_rate": float(df.cancelled.mean()), "late_rate": float(df.arr_delayed_15.mean()),
                 "months": int(pd.to_datetime(df.flight_date).dt.month.nunique())}
    print(name, len(df), flush=True)

# lookups: airports (BTS IDs + OurAirports coordinates), carriers
ids = pd.read_csv(SRC / "L_AIRPORT_ID.csv", dtype={"Code": "Int32"}, encoding="latin-1").rename(columns={"Code": "airport_id", "Description": "bts_description"})
used = set()
for y in by_year:
    f = pd.read_parquet(OUT / f"flights_{y}.parquet", columns=["origin_airport_id", "dest_airport_id", "origin", "dest"])
    used |= set(zip(f.origin_airport_id, f.origin)) | set(zip(f.dest_airport_id, f.dest))
codes = pd.DataFrame(sorted(used), columns=["airport_id", "code"]).drop_duplicates("airport_id", keep="last")
oa = pd.read_csv(SRC / "ourairports_airports.csv", usecols=["iata_code", "name", "latitude_deg", "longitude_deg", "elevation_ft",
                                                             "iso_country", "iso_region", "municipality", "type"])
oa = oa[oa.iata_code.notna()].sort_values("type").drop_duplicates("iata_code")
airports = (codes.merge(ids, on="airport_id", how="left")
                 .merge(oa.rename(columns={"iata_code": "code", "name": "airport_name"}), on="code", how="left"))
airports.to_csv(OUT / "airports.csv", index=False)
carriers = pd.read_csv(SRC / "L_UNIQUE_CARRIERS.csv", encoding="latin-1").rename(columns={"Code": "airline", "Description": "airline_name"})
used_air = set()
for y in by_year:
    used_air |= set(pd.read_parquet(OUT / f"flights_{y}.parquet", columns=["airline"]).airline.astype(str))
carriers = carriers[carriers.airline.isin(used_air)]
carriers.to_csv(OUT / "carriers.csv", index=False)

summary = {"files": files, "totals": totals, "months": stats, "airports": len(airports),
           "airports_with_coords": int(airports.latitude_deg.notna().sum()), "carriers": len(carriers)}
(BASE / "build_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
print(json.dumps({k: v for k, v in summary.items() if k != "months"}, indent=1, default=str))

# ---------------------------------------------------------------- dataset-metadata.json (every number computed)
n_total = sum(files.values())
m0, m1 = stats[0]["month"].replace("_", "-"), stats[-1]["month"].replace("_", "-")
year_rows = "\n".join(f"| `flights_{y}.parquet` | {t['rows']:,} | {t['months']} | {t['late_rate']:.1%} | {t['cancel_rate']:.2%} |"
                      for y, t in sorted(totals.items()))
cols_md = "\n".join(f"| `{v[0]}` | {v[2]} |" for v in KEEP.values())
desc = f"""## What this is
Every domestic flight operated by the large US airlines, flight by flight, from the U.S. Bureau of Transportation Statistics (BTS) **Reporting Carrier On-Time Performance** data: scheduled and actual times, departure and arrival delays, **delay causes** (carrier, weather, air-traffic system, security, late aircraft), cancellations with reasons, diversions, taxi times, distance, and the aircraft tail number. **{n_total:,} flights, {m0} to {m1}**, one Parquet file per year, plus airport (with coordinates) and airline lookup tables.

Kaggle's most-voted flight-delay dataset covers 2015 only. This one runs to the latest month BTS has published and is refreshed monthly.

Good for: delay and cancellation prediction (classification and regression), delay-propagation studies via tail numbers, airport and airline benchmarking, network and graph analysis, and large-scale tabular practice.

## Files
All files are in the `out/` folder.

| File | Flights | Months | Arrived 15+ min late | Cancelled |
|---|---|---|---|---|
{year_rows}

Also included: `airports.csv` ({len(airports)} airports: BTS airport ID, code, BTS name, and the OurAirports name, coordinates, elevation and region, where matched) and `carriers.csv` ({len(carriers)} airlines: code and name).

## Quick start
```python
import pandas as pd
f = pd.read_parquet("/kaggle/input/{SLUG}/{SUB}flights_{max(totals)}.parquet")
airports = pd.read_csv("/kaggle/input/{SLUG}/{SUB}airports.csv")
late_by_airline = f.groupby("airline", observed=True).arr_delayed_15.mean().sort_values()
```

## Columns (flights files)
| column | description |
|---|---|
{cols_md}

## Things to know
- **Times are local** to each airport, in hhmm integers (e.g. 1305 = 1:05 pm; 2400 = midnight). Arrivals can be on the next calendar day: compare `sched_dep_hhmm` and `sched_arr_hhmm` with care.
- **Delay causes are filled only for flights that arrived 15+ minutes late.** They are empty (not zero) otherwise.
- **Cancelled and diverted flights have no arrival delay.** Exclude them, or model them separately, when predicting delays.
- The **"late" rate** above is `arr_delayed_15` averaged over flights with a recorded arrival, the official BTS definition.
- Only the airlines that must report (those above BTS's revenue threshold) are included, so regional flying is covered only where a reporting carrier operates it under its own code. Codeshares are reported by the operating carrier.
- `dest_state` and `origin_state` are where the airport is (e.g. DCA "Washington, DC" is in VA).

## Source & license
U.S. Department of Transportation, Bureau of Transportation Statistics, "Reporting Carrier On-Time Performance (1987-present)", TranStats PREZIP monthly files (https://www.transtats.bts.gov/), with BTS lookup tables `L_UNIQUE_CARRIERS` and `L_AIRPORT_ID`. U.S. federal government work, public domain (17 U.S.C. 105). DOT's own Kaggle release of this data uses CC0. Airport coordinates from [OurAirports](https://ourairports.com/data/), whose data is released to the public domain. Not produced or endorsed by BTS. Build scripts: `fetch_bts.py` / `build_bts.py` in [kaggle-dataset-pipelines](https://github.com/tb91117/kaggle-dataset-pipelines) (open source).
"""


def fld(name, d, t):
    return {"name": name, "description": d, "type": t}


tmap = {"date": "datetime", "category": "string", "string": "string", "Int16": "integer", "Int32": "integer",
        "float32": "number", "bool": "boolean"}
flight_fields = [fld(v[0], v[2], tmap[v[1]]) for v in KEEP.values()]
air_doc = {"airport_id": "BTS airport ID (join to origin_airport_id / dest_airport_id).", "code": "Airport code used in the flights files.",
           "bts_description": "BTS airport description ('City, ST: Airport name').", "airport_name": "Airport name (OurAirports).",
           "latitude_deg": "Latitude (OurAirports).", "longitude_deg": "Longitude (OurAirports).", "elevation_ft": "Elevation in feet (OurAirports).",
           "iso_country": "ISO country code.", "iso_region": "ISO region (e.g. US-CA).", "municipality": "City (OurAirports).",
           "type": "Airport size class (OurAirports)."}
meta = {
    "id": f"{OWNER}/{SLUG}",
    "title": f"US Flight Delays & On-Time Performance {m0[:4]}-{m1[:4]}",
    "subtitle": "US airline flights: delays and causes, cancellations, taxi times, tail numbers",
    "description": desc,
    "licenses": [{"name": "CC0-1.0"}],
    "keywords": ["aviation", "transportation", "united states", "tabular", "time series analysis"],
    "expectedUpdateFrequency": "monthly",
    "userSpecifiedSources": "U.S. DOT Bureau of Transportation Statistics, Reporting Carrier On-Time Performance (TranStats). Public domain. Airport coordinates: OurAirports (public domain).",
    "resources": [{"path": SUB + n, "description": f"Flights in {n[8:12]}.", "schema": {"fields": flight_fields}} for n in files]
    + [{"path": SUB + "airports.csv", "description": "Airports in the flights files, with coordinates.",
        "schema": {"fields": [fld(c, air_doc.get(c, c), "string") for c in airports.columns]}},
       {"path": SUB + "carriers.csv", "description": "Airline codes and names.",
        "schema": {"fields": [fld("airline", "Airline code used in the flights files.", "string"), fld("airline_name", "Airline name (BTS).", "string")]}}],
}
assert 6 <= len(meta["title"]) <= 50 and 20 <= len(meta["subtitle"]) <= 80, (len(meta["title"]), len(meta["subtitle"]))
(OUT / "dataset-metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
print("metadata written:", meta["title"], "|", meta["subtitle"])

import runpy  # noqa: E402  (cover: distinct 2:1 design; push with push_cover.py)
runpy.run_path(str(Path(__file__).with_name("make_covers.py")), run_name="covers")["flights"]()
