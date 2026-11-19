"""Dataset cover images, one distinct design per dataset (1600x800, 2:1, no title text: Kaggle shows the title).

Usage:  python -I tools/make_covers.py [trees|wdi|btc|intent ...]   (default: all)
Writes dataset-cover-image.png into each dataset's build folder; pushing is a separate step.
"""
import sys
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle

# KDP_DATA_ROOT overrides where sources are downloaded and datasets are built (used by the Kaggle kernels)
DATA_ROOT = Path(os.environ.get("KDP_DATA_ROOT") or Path(__file__).resolve().parents[1] / "datasets")
D = DATA_ROOT
W, H, DPI = 16, 8, 100  # Kaggle cover is 2:1; the thumbnail is the centred square


def canvas(bg):
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    fig.patch.set_facecolor(bg)
    return fig


def save(fig, path):
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", path)


# ----------------------------------------------------------------- SF trees: paper map
def trees():
    out = D / "san-francisco-trees" / "v2"
    df = pd.read_csv(out / "sf_street_trees.csv", usecols=["latitude", "longitude"]).dropna()
    df = df[df.latitude.between(37.70, 37.815) & df.longitude.between(-122.52, -122.355)]
    fig = canvas("#f3efe4")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor("#f3efe4"); ax.axis("off")
    ax.scatter(df.longitude, df.latitude, s=0.7, color="#2f6b3a", alpha=0.42, linewidths=0)
    k = 1 / np.cos(np.radians(37.76))
    cx, cy = df.longitude.mean(), df.latitude.median()
    half_h = (df.latitude.max() - df.latitude.min()) / 2 * 1.06
    half_w = half_h * (W / H) / k
    ax.set_xlim(cx - half_w, cx + half_w); ax.set_ylim(cy - half_h, cy + half_h); ax.set_aspect(k)
    fig.text(0.975, 0.035, "Every mapped street tree · San Francisco", ha="right", color="#6b6656", fontsize=13, family="serif", style="italic")
    save(fig, out / "dataset-cover-image.png")


# ----------------------------------------------------------------- WDI: life-expectancy stripes
def wdi():
    out = D / "world-development-indicators" / "v1"
    p = pd.read_csv(out / "wdi_headline_panel.csv", usecols=["country_code", "year", "life_expectancy"])
    m = p.pivot(index="country_code", columns="year", values="life_expectancy").loc[:, 1960:2023]
    m = m[m.notna().sum(axis=1) > 50].interpolate(axis=1, limit_direction="both")
    m = m.loc[m[2023].sort_values().index]
    cmap = LinearSegmentedColormap.from_list("le", ["#f6efe6", "#e9c9a0", "#9fb8a8", "#3f7f86", "#1d3f5e", "#0f2238"])
    fig = canvas("#f6efe6")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(m.values, aspect="auto", cmap=cmap, vmin=30, vmax=85, interpolation="nearest")
    fig.text(0.985, 0.03, f"Life expectancy 1960 → 2023 · one row per economy ({len(m)})", ha="right",
             color="#f6efe6", fontsize=13, family="serif", style="italic")
    save(fig, out / "dataset-cover-image.png")


# ----------------------------------------------------------------- BTC: candlesticks, most volatile day
def btc():
    out = D / "binance-ohlc-with-indicators" / "v2"
    df = pd.read_csv(out / "btcusdt_15m_indicators.csv", usecols=["timestamp", "open", "high", "low", "close", "volume"],
                     parse_dates=["timestamp"])
    day = df.assign(d=df.timestamp.dt.floor("D"), rng=(df.high - df.low) / df.close).groupby("d").rng.sum().idxmax()
    w = df[(df.timestamp >= day - pd.Timedelta(hours=18)) & (df.timestamp < day + pd.Timedelta(hours=30))].reset_index(drop=True)
    up, down, bg = "#26a69a", "#ef5350", "#0b0e11"
    fig = canvas(bg)
    ax = fig.add_axes([0.02, 0.24, 0.92, 0.72]); axv = fig.add_axes([0.02, 0.04, 0.92, 0.18], sharex=ax)
    for a in (ax, axv):
        a.set_facecolor(bg)
        for s in a.spines.values(): s.set_visible(False)
        a.tick_params(colors="#5d6673", labelsize=11, length=0)
        a.grid(color="#1c2230", lw=0.8); a.set_axisbelow(True)
    for i, r in w.iterrows():
        c = up if r.close >= r.open else down
        ax.vlines(i, r.low, r.high, color=c, lw=1.1)
        ax.add_patch(Rectangle((i - 0.35, min(r.open, r.close)), 0.7, max(abs(r.close - r.open), 1e-6), color=c, lw=0))
        axv.bar(i, r.volume, width=0.7, color=c, alpha=0.45)
    ax.yaxis.tick_right(); axv.set_yticks([]); ax.set_xlim(-1, len(w))
    ticks = [i for i, t in enumerate(w.timestamp) if t.minute == 0 and t.hour % 6 == 0]
    axv.set_xticks(ticks, [w.timestamp[i].strftime("%H:%M") if w.timestamp[i].hour else w.timestamp[i].strftime("%d %b %Y") for i in ticks])
    plt.setp(ax.get_xticklabels(), visible=False)
    fig.text(0.02, 0.965, "BTCUSDT · 15m · Binance spot", color="#8a94a6", fontsize=13, family="monospace", va="top")
    save(fig, out / "dataset-cover-image.png")


# ----------------------------------------------------------------- intent: routed queries, centred column
def intent():
    out = D / "data-for-classifying-intent-of-query" / "v2"
    df = pd.read_csv(out / "query_intents.csv")
    colors = {"chat": "#2a78d6", "image": "#eb6834", "search": "#1baf7a", "wikipedia": "#6250d6", "scraper": "#52514e"}
    per = {"chat": 2, "image": 2, "search": 1, "wikipedia": 1, "scraper": 1}
    picks = pd.concat([df[df.topic == t].sample(n, random_state=5) for t, n in per.items()]).sample(frac=1, random_state=4)
    picks["text"] = picks.text.where(picks.text.str.len() <= 36, picks.text.str[:34].str.rstrip() + "…")
    fig = canvas("#fafaf7")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    n = len(picks); gap = 0.98; top = H / 2 + (n - 1) * gap / 2
    for k, (_, r) in enumerate(picks.iterrows()):
        y = top - k * gap
        ax.add_patch(FancyBboxPatch((4.15, y - 0.3), 5.25, 0.6, boxstyle="round,pad=0.02,rounding_size=0.28",
                                    facecolor="white", edgecolor="#d9d8d2", lw=1.2))
        ax.text(4.38, y, r.text, ha="left", va="center", color="#1f1f1c", fontsize=15)
        ax.annotate("", xy=(10.05, y), xytext=(9.55, y), arrowprops=dict(arrowstyle="-|>", color="#9a988f", lw=1.4))
        ax.add_patch(FancyBboxPatch((10.15, y - 0.24), 1.7, 0.48, boxstyle="round,pad=0.02,rounding_size=0.22", color=colors[r.topic], lw=0))
        ax.text(11.0, y, r.topic, ha="center", va="center", color="white", fontsize=13, family="monospace", weight="bold")
    save(fig, out / "dataset-cover-image.png")


# ----------------------------------------------------------------- EIA-930: load calendar (hour x day heatmap)
def eia():
    out = D / "eia930-hourly-grid" / "v1"
    r = pd.read_parquet(out / "region_hourly.parquet", columns=["region", "utc_time_end", "demand_mw"])
    us = r[r.region == "US48"].set_index("utc_time_end").demand_mw.sort_index()
    us.index = us.index - pd.Timedelta(hours=6)          # roughly US Central time, so rows read as local hours
    us = us["2019-01-01":us.dropna().index.max().floor("D") - pd.Timedelta(days=1)]
    grid = us.to_frame("mw").assign(day=us.index.floor("D"), hour=us.index.hour).pivot_table(index="hour", columns="day", values="mw")
    grid = grid.interpolate(axis=1, limit_direction="both")
    cmap = LinearSegmentedColormap.from_list("load", ["#fff4e0", "#fdd49e", "#fc8d59", "#d7301f", "#7f0000", "#3b0000"])
    fig = canvas("#fff8ec")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    lo, hi = np.nanpercentile(grid.values, [1, 99.5])
    ax.imshow(grid.values[::-1], aspect="auto", cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
    fig.text(0.985, 0.03, f"US lower-48 electricity demand · each column a day, {grid.columns.min():%Y}-{grid.columns.max():%Y} · each row an hour",
             ha="right", color="#5a1a00", fontsize=12, family="serif", style="italic",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff8ec", edgecolor="none", alpha=0.85))
    save(fig, out / "dataset-cover-image.png")


# ----------------------------------------------------------------- flights: the route network
def flights():
    out = D / "us-flight-on-time" / "v1"
    latest = sorted(out.glob("flights_*.parquet"))
    if latest:
        latest = latest[-1]
        f = pd.read_parquet(latest, columns=["origin", "dest"])
        routes = f.groupby(["origin", "dest"], observed=True).size().rename("n").reset_index()
        aux = Path("/kaggle/working/aux") if str(out).startswith("/kaggle") else out.parent / "aux"
        aux.mkdir(parents=True, exist_ok=True)
        routes.assign(year=latest.stem[-4:]).to_csv(aux / "routes.csv", index=False)   # small; lets the cover be re-rendered offline
    else:                                                    # offline re-render from aux/routes.csv
        routes = pd.read_csv(out.parent / "aux" / "routes.csv")
        latest = Path(f"flights_{routes.year.iloc[0]}")
    ap = pd.read_csv(out / "airports.csv", usecols=["code", "latitude_deg", "longitude_deg"]).dropna().drop_duplicates("code").set_index("code")
    routes = routes.join(ap, on="origin").join(ap.add_suffix("_d"), on="dest").dropna()
    routes = routes[routes.longitude_deg.between(-125, -66) & routes.longitude_deg_d.between(-125, -66)
                    & routes.latitude_deg.between(24, 50) & routes.latitude_deg_d.between(24, 50)].sort_values("n")
    fig = canvas("#f3f6fb")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor("#f3f6fb"); ax.axis("off")
    w = routes.n / routes.n.max()
    from matplotlib.collections import LineCollection
    segs = np.stack([routes[["longitude_deg", "latitude_deg"]].values, routes[["longitude_deg_d", "latitude_deg_d"]].values], axis=1)
    cols = np.zeros((len(routes), 4)); cols[:, :3] = (0.11, 0.30, 0.85); cols[:, 3] = 0.012 + 0.6 * w
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.2 + 1.6 * w))
    k = 1 / np.cos(np.radians(37.5))
    cx, cy, half_h = -95.5, 37.3, 13.6
    ax.set_xlim(cx - half_h * 2 * k, cx + half_h * 2 * k); ax.set_ylim(cy - half_h, cy + half_h); ax.set_aspect(k)  # 2:1 on screen
    fig.text(0.985, 0.03, f"{len(routes):,} US airline routes flown in {latest.stem[-4:]}, weighted by flights",
             ha="right", color="#334155", fontsize=12, family="serif", style="italic")
    save(fig, out / "dataset-cover-image.png")


JOBS = {"trees": trees, "wdi": wdi, "btc": btc, "intent": intent, "eia": eia, "flights": flights}
if __name__ == "__main__":
    for name in sys.argv[1:] or JOBS:
        JOBS[name]()
