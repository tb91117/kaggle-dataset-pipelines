"""Generate the example notebooks + kernel-metadata.json for the public datasets.

Usage: python -I tools/make_notebooks.py
Output: notebooks/<slug>/<slug>.ipynb and kernel-metadata.json (public; approved 2026-11-18).
"""
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1] / "notebooks"
OWNER = "teddybrown414"

FIND_FILE = '''import glob, os

def find_file(name):
    """Locate a dataset file on Kaggle (/kaggle/input) or locally (../../datasets)."""
    for pattern in (f"/kaggle/input/**/{name}", f"../../datasets/**/{name}"):
        hits = glob.glob(pattern, recursive=True)
        if hits:
            return hits[0]
    raise FileNotFoundError(name)
'''


def build(slug, title, dataset, cells, private=False):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("md:") else nbf.v4.new_code_cell(c) for c in cells]
    d = ROOT / slug
    d.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, d / f"{slug}.ipynb")
    meta = {
        "id": f"{OWNER}/{slug}", "title": title, "code_file": f"{slug}.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": private,
        "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": [dataset], "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote", d)


# ============================================================================ intent
intent_cells = [
    """md:# Query Intent Routing: Rules vs. TF-IDF Baselines

When an AI assistant receives a message, something has to decide **which tool to call**: generate an image, search the web, scrape a URL, look something up on Wikipedia, or just chat. This notebook builds and compares two cheap routers on the [Query Intent Routing for AI Assistants](https://www.kaggle.com/datasets/teddybrown414/data-for-classifying-intent-of-query) dataset:

1. a hand-written **keyword/regex router** (zero training data), and
2. a **TF-IDF + logistic regression** classifier evaluated with repeated stratified cross-validation.

The dataset is small (91 queries), so the goal is a sound evaluation and an honest picture of what works, not a leaderboard number.""",
    FIND_FILE + '''
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(find_file("query_intents.csv"))
print(df.shape)
df.sample(8, random_state=0)''',
    '''counts = df["topic"].value_counts()
ax = counts.sort_values().plot.barh(color="#38bdf8", figsize=(7, 3), title="Examples per intent")
ax.set_xlabel("rows"); plt.tight_layout(); plt.show()
df.assign(n_words=df.text.str.split().str.len()).groupby("topic").n_words.describe()[["mean", "min", "max"]]''',
    """md:**What stands out:** the classes are nearly balanced (`image` has 11 rows, the others 20). `chat`, `image` and `search` queries average 7–9 words, `wikipedia` queries are 2–3 words that start with "Wikipedia", and `scraper` rows are bare URLs. A good router should get `scraper` and `wikipedia` right with a single rule each.

## 1. A keyword router (no training)
Before reaching for a model, write down the obvious rules. This is the baseline any model has to beat.""",
    '''URL = re.compile(r"^(https?://|www\\.)\\S+$", re.I)
RULES = [
    ("scraper",   lambda t: bool(URL.match(t.strip()))),
    ("wikipedia", lambda t: t.lower().startswith("wikipedia") or "wikipedia" in t.lower()),
    ("image",     lambda t: bool(re.search(r"\\b(image|draw|drawing|paint|painting|illustration|picture|portrait|sketch|graphic)", t, re.I))),
    ("search",    lambda t: bool(re.search(r"\\b(search|look for|find|latest|news)\\b", t, re.I))),
]

def rule_router(text):
    for label, rule in RULES:
        if rule(text):
            return label
    return "chat"  # fallback: just talk

df["rule_pred"] = df.text.map(rule_router)
print(f"Rule router accuracy: {(df.rule_pred == df.topic).mean():.3f}")
pd.crosstab(df.topic, df.rule_pred, rownames=["true"], colnames=["predicted"])''',
    '''df.loc[df.rule_pred != df.topic, ["text", "topic", "rule_pred"]]''',
    """md:## 2. TF-IDF + logistic regression
Word n-grams capture verbs like *draw* or *search*; character n-grams add robustness to typos and URL fragments. With only 91 rows a single train/test split would be very noisy, so we use **5-fold stratified CV repeated 10 times** and report the mean and spread.""",
    '''from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, cross_val_predict, StratifiedKFold
from sklearn.pipeline import make_pipeline, make_union
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

model = make_pipeline(
    make_union(
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
    ),
    LogisticRegression(C=10, max_iter=2000),
)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
scores = cross_validate(model, df.text, df.topic, cv=cv, scoring=["accuracy", "f1_macro"])
for m in ("accuracy", "f1_macro"):
    s = scores[f"test_{m}"]
    print(f"{m:9s} {s.mean():.3f} ± {s.std():.3f}")''',
    '''pred = cross_val_predict(model, df.text, df.topic, cv=StratifiedKFold(5, shuffle=True, random_state=0))
print(classification_report(df.topic, pred, digits=3))
ConfusionMatrixDisplay.from_predictions(df.topic, pred, cmap="Blues", xticks_rotation=45)
plt.title("TF-IDF + LR, out-of-fold predictions"); plt.tight_layout(); plt.show()''',
    '''df["model_pred"] = pred
df.loc[df.model_pred != df.topic, ["text", "topic", "model_pred"]]''',
    """md:## 3. Use it
Fit on all rows and route some new messages, including a few phrased differently from the training data.""",
    '''model.fit(df.text, df.topic)
new = [
    "paint me a watercolor of Prague at night",
    "what's the weather like on Mars?",
    "https://www.kaggle.com/competitions",
    "Wikipedia Ada Lovelace",
    "find the newest papers on retrieval augmented generation",
    "tell me something that will cheer me up",
]
proba = model.predict_proba(new)
pd.DataFrame({"query": new, "model": model.predict(new), "confidence": proba.max(1).round(2),
              "rules": [rule_router(t) for t in new]})''',
    """md:## Takeaways
- **The rules win on paper (≈99% vs ≈95%), but that comparison flatters them.** The regexes were written after reading all 91 rows, so their score is effectively a training score, while the model's score is out-of-fold. Always build this baseline before training anything, but evaluate it on data you didn't look at.
- **TF-IDF + LR is weakest on `image`** (only 11 examples; several "draw/portrait" queries were routed to `chat`). It does give a confidence score, which is useful in production: send low-confidence messages to a fallback (ask a clarifying question, or let an LLM decide). Note the 0.50 confidence on the unseen "paint me a watercolor" query.
- **Scores on this dataset overstate real-world accuracy.** Real users write messier, longer, multi-intent messages. Treat this as a seed set: log real traffic, label the router's mistakes, and grow the data.

**Next steps to try:** sentence-embedding + kNN routing, zero-shot routing with a small instruction-tuned LLM, and a "multi-intent" variant where one message needs two tools.""",
]

# ============================================================================ btc
btc_cells = [
    """md:# BTC/USDT 15-Minute Candles: EDA, Label Anatomy, and a Leak-Free Baseline

This notebook explores the [BTC/USDT 15-min Candles + Indicators 2017-2025](https://www.kaggle.com/datasets/teddybrown414/binance-ohlc-with-indicators) dataset (≈260k Binance spot candles) and covers three things:

1. **EDA**: price history, return distribution, and when during the day volatility happens.
2. **Label anatomy**: what the `signal`, `tp` and `sl` columns actually encode, and why they must never be used as features.
3. **A leak-free baseline**: predicting the direction of the next hour with walk-forward validation, plus a demonstration of how a random split inflates scores.""",
    FIND_FILE + '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(find_file("btcusdt_15m_indicators.csv"), parse_dates=["timestamp"])
print(df.shape, df.timestamp.min(), "->", df.timestamp.max())
df.head()''',
    """md:## 1. EDA
### Price history (log scale)""",
    '''daily = df.set_index("timestamp").close.resample("D").last()
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(daily.index, daily, color="#f59e0b"); ax.set_yscale("log")
ax.set_title("BTC/USDT daily close (log scale)"); ax.set_ylabel("USDT"); plt.tight_layout(); plt.show()''',
    """md:### Returns are fat-tailed
15-minute log returns have far more extreme moves than a normal distribution with the same standard deviation would predict.""",
    '''r = np.log(df.close).diff().dropna()
sd = r.std()
tail = pd.DataFrame({
    "threshold": ["> 3 sd", "> 5 sd", "> 10 sd"],
    "observed share": [(r.abs() > k * sd).mean() for k in (3, 5, 10)],
    "normal would give": [2.7e-3, 5.7e-7, 1.5e-23],
})
print(f"std of 15-min log return: {sd:.4%}   excess kurtosis: {r.kurt():.1f}")
tail''',
    """md:### When does BTC move? Volatility by UTC hour and weekday""",
    '''vol = df.assign(absret=np.log(df.close).diff().abs(), hour=df.timestamp.dt.hour, wd=df.timestamp.dt.day_name().str[:3])
pivot = vol.pivot_table(index="wd", columns="hour", values="absret", aggfunc="mean")
pivot = pivot.loc[["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]]
fig, ax = plt.subplots(figsize=(13, 3.5))
im = ax.imshow(pivot.values * 1e4, aspect="auto", cmap="magma")
ax.set_yticks(range(7), pivot.index); ax.set_xticks(range(24)); ax.set_xlabel("hour (UTC)")
ax.set_title("Mean |15-min return| in basis points"); fig.colorbar(im); plt.tight_layout(); plt.show()''',
    """md:Volatility peaks at **14:00–16:00 UTC** (US session open), spikes again at **00:00 UTC** (daily candle rollover), is quietest at **04:00–06:00 UTC**, and weekends are about **20% calmer** than weekdays: useful features for any model.

## 2. Label anatomy: what are `signal`, `tp` and `sl`?
The columns look like trade labels. Let's test what they encode by comparing them with the **future** high/low over the next N bars.""",
    '''labelled = df[df.signal != 0]
rows = []
for n in (24, 48, 96, 99, 144):
    fut_high = df.high[::-1].rolling(n, min_periods=1).max()[::-1].shift(-1)
    fut_low = df.low[::-1].rolling(n, min_periods=1).min()[::-1].shift(-1)
    rows.append({"window (bars)": n, "hours": n / 4,
                 "tp == future max high": np.isclose(labelled.tp, fut_high[labelled.index]).mean(),
                 "sl == future min low": np.isclose(labelled.sl, fut_low[labelled.index]).mean()})
pd.DataFrame(rows).round(3)''',
    '''up, down = labelled.tp - labelled.close, labelled.close - labelled.sl
print("signal == sign(rise to tp - fall to sl):", (np.sign(up - down) == labelled.signal).mean())
print(df.signal.value_counts().rename({1: "long (1)", -1: "short (-1)", 0: "none (0)"}))''',
    """md:**Conclusion:** `tp`/`sl` are (approximately) the highest high and lowest low over the **next ~24 hours**, and `signal` simply says which of the two moves was bigger. `signal = 0` marks bars where neither side dominated. All three columns contain future information: they are **targets**, and using them as inputs would leak the answer. Below we drop them from the feature set.

## 3. A leak-free baseline: will the next hour close higher?
Target: sign of the log return over the next 4 bars (1 hour). Features use only information available at the bar's close.""",
    '''from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, KFold

d = df.copy()
logc = np.log(d.close)
for k in (1, 4, 16, 96):
    d[f"ret_{k}"] = logc.diff(k)
d["vol_16"] = logc.diff().rolling(16).std()
d["atr_pct"] = d.ATR / d.close
d["range_pct"] = (d.high - d.low) / d.close
d["vol_z"] = (d.volume - d.volume.rolling(96).mean()) / d.volume.rolling(96).std()
d["hour"], d["weekday"] = d.timestamp.dt.hour, d.timestamp.dt.day_of_week
d["target"] = (logc.shift(-4) - logc > 0).astype(int)
d = d.iloc[200:-4].dropna()   # drop warm-up and the last rows without a future

FEATURES = ["ret_1", "ret_4", "ret_16", "ret_96", "vol_16", "atr_pct", "range_pct", "vol_z",
            "RSI", "MACD_hist", "stoch_k", "stoch_d", "hour", "weekday"]
X, y = d[FEATURES], d.target
print(X.shape, "positive rate:", y.mean().round(3))''',
    '''def evaluate(splitter, X, y, gap=0):
    aucs = []
    for tr, te in splitter.split(X):
        if gap:
            tr = tr[tr < te.min() - gap]          # purge bars whose target overlaps the test window
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, random_state=0)
        clf.fit(X.iloc[tr], y.iloc[tr])
        aucs.append(roc_auc_score(y.iloc[te], clf.predict_proba(X.iloc[te])[:, 1]))
    return np.array(aucs)

wf = evaluate(TimeSeriesSplit(n_splits=5), X, y, gap=96)
print("Walk-forward AUC per fold:", wf.round(3), "mean", wf.mean().round(3))''',
    """md:An AUC around 0.56 is the honest answer: there is a small, consistent edge (every fold is above 0.54), but next-hour direction is still close to a coin flip, and an edge this size is easily eaten by fees and slippage. Now compare with two **mistakes** people often make.""",
    '''rnd = evaluate(KFold(5, shuffle=True, random_state=0), X, y)
leaky = evaluate(TimeSeriesSplit(n_splits=5), d[FEATURES + ["signal"]], y, gap=96)
pd.DataFrame({
    "setup": ["walk-forward (correct)", "random K-fold split", "walk-forward + `signal` as a feature"],
    "mean AUC": [wf.mean(), rnd.mean(), leaky.mean()],
}).round(3)''',
    """md:- **Random K-fold** mixes past and future bars; overlapping targets and autocorrelated features let the model peek, so the score is optimistic (here modestly; on noisier targets the gap can be much larger).
- **Using `signal` as a feature** leaks tomorrow's price range into today's prediction, so the score looks great and would collapse in live trading.

## Takeaways
- BTC 15-minute returns are fat-tailed and volatility is strongly seasonal by hour and weekday.
- `signal`/`tp`/`sl` are **look-ahead targets** (≈24h future high/low); great for supervised learning *as labels*, never as inputs.
- Use time-ordered validation with a purge gap. With honest validation, simple technical features carry only a small next-hour signal (AUC ≈ 0.56). Better targets to try: volatility forecasting (much more predictable) or the dataset's own `signal` label as a 24-hour target.

*Not financial advice.*""",
]

# ============================================================================ SF trees
trees_cells = [
    """md:# San Francisco's Street Trees: Diversity, Planting History, and a Data Trap

The [San Francisco Street Tree Inventory 2026](https://www.kaggle.com/datasets/teddybrown414/san-francisco-trees) lists every street tree the city's Public Works department tracks: about 145k trees with species, location, planting date and trunk diameter. This notebook asks three questions:

1. **Is the urban forest diverse enough?** Foresters use the *10-20-30 rule* (no more than 10% of trees from one species, 20% from one genus, 30% from one family) as a guard against pests and diseases wiping out a whole street. We check it citywide and by neighborhood.
2. **Who planted the trees, and when?**
3. **Can we measure how fast trees grow?** This is where the data hides a trap worth knowing about before you model anything.""",
    FIND_FILE + '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(find_file("sf_street_trees.csv"),
                 dtype={"cnn": str, "zip_code": str, "supervisordistrict": str},
                 parse_dates=["planteddate", "data_as_of"])
print(f"{len(df):,} trees x {df.shape[1]} columns")
df[["species_common", "address", "analysis_neighborhood", "planteddate", "mapdbh", "planter"]].sample(5, random_state=1)''',
    """md:Some species labels are placeholders (`Tree(s)`, `To Be Determine`) or missing. We drop those for species-level analysis and derive the **genus** from the first word of the scientific name.""",
    '''placeholder = (df.species.isna() | df.species_scientific.isin(["Tree(s)", ""])
               | df.species.str.contains("To Be Determine", na=False))
known = df[~placeholder].copy()
known["genus"] = known.species_scientific.str.split().str[0].str.capitalize()
print(f"{len(known):,} trees with a usable species label ({placeholder.sum():,} dropped)")''',
    """md:## 1. Where the trees are
Each dot is a tree, colored by the eight most common species (internet is off, so no basemap: the street grid draws itself).""",
    '''top8 = known.species_common.value_counts().head(8).index
geo = known.dropna(subset=["latitude", "longitude"])
geo = geo[geo.latitude.between(37.70, 37.82) & geo.longitude.between(-122.52, -122.35)]
fig, ax = plt.subplots(figsize=(11, 10))
ax.scatter(geo.longitude, geo.latitude, s=0.2, color="#d4d4d4", linewidths=0)
colors = [c for i, c in enumerate(plt.cm.tab10.colors) if i != 7]  # skip grey
for name, col in zip(top8, colors):
    g = geo[geo.species_common == name]
    ax.scatter(g.longitude, g.latitude, s=0.6, color=col, linewidths=0, label=f"{name} ({len(g):,})")
ax.set_aspect(1 / np.cos(np.radians(37.76))); ax.set_xticks([]); ax.set_yticks([])
ax.legend(markerscale=12, loc="lower left", fontsize=9, frameon=False)
ax.set_title("San Francisco street trees, top 8 species highlighted"); plt.tight_layout(); plt.show()''',
    """md:## 2. The 10-20-30 rule, citywide
(We check species and genus. The family level would need an external taxonomy table, which this notebook doesn't use.)""",
    '''sp = known.species_scientific.value_counts(normalize=True)
ge = known.genus.value_counts(normalize=True)
print(f"largest species share: {sp.index[0]} {sp.iloc[0]:.1%}   (rule: <= 10%)")
print(f"largest genus share:   {ge.index[0]} {ge.iloc[0]:.1%}   (rule: <= 20%)")
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, s, lim, name in ((axes[0], sp, 0.10, "species"), (axes[1], ge, 0.20, "genus")):
    s.head(12)[::-1].plot.barh(ax=ax, color="#4ade80")
    ax.axvline(lim, color="crimson", ls="--", label=f"{lim:.0%} limit")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_title(f"Top 12 {name} by share of street trees"); ax.legend()
plt.tight_layout(); plt.show()''',
    """md:Citywide, San Francisco passes comfortably: no species is anywhere near 10% and no genus near 20%. But the rule is meant for the scale at which pests spread, and **a street or a neighborhood can be a monoculture even when the city isn't.**

## 3. The same rule, neighborhood by neighborhood""",
    '''def top_share(s):
    return s.value_counts(normalize=True).iloc[0]

def effective_species(s):
    p = s.value_counts(normalize=True)
    return float(np.exp(-(p * np.log(p)).sum()))   # exp(Shannon entropy)

nb = known.groupby("analysis_neighborhood").agg(
    trees=("treeid", "size"),
    top_species=("species_common", lambda s: s.value_counts().index[0]),
    top_share=("species_scientific", top_share),
    effective_species=("species_scientific", effective_species),
)
big = nb[nb.trees >= 500].sort_values("top_share")
print(f"neighborhoods with one species above 10%: {(nb.top_share > 0.10).sum()} of {len(nb)}"
      f"  ({(big.top_share > 0.10).sum()} of {len(big)} with 500+ trees)")
fig, ax = plt.subplots(figsize=(10, 10))
ax.barh(big.index, big.top_share, color=np.where(big.top_share > 0.10, "#f97316", "#4ade80"))
for y, (share, name) in enumerate(zip(big.top_share, big.top_species)):
    ax.text(share + 0.003, y, name, va="center", fontsize=8)
ax.axvline(0.10, color="crimson", ls="--")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.set_title("Share of the most common species (neighborhoods with 500+ street trees)")
plt.tight_layout(); plt.show()''',
    '''nb.sort_values("effective_species")[["trees", "top_species", "top_share", "effective_species"]].round(2).query("trees >= 500").iloc[[0, 1, 2, -3, -2, -1]]''',
    """md:*Effective number of species* is exp(Shannon entropy): the number of equally common species that would give the same diversity. It is a fairer comparison than a raw species count, which grows with the number of trees.

## 4. Who planted the trees, and when?
Only about a quarter of trees have a planting date, mostly from 1990 onward, so read this as *recorded* planting activity rather than the full history.""",
    '''dated = df.dropna(subset=["planteddate"]).assign(year=lambda d: d.planteddate.dt.year)
dated = dated[dated.year.between(1990, 2026)]
groups = ["Private", "DPW", "FUF"]
dated["who"] = dated.planter.where(dated.planter.isin(groups), "Other").where(dated.planter.notna(), "Unknown")
by_year = dated.pivot_table(index="year", columns="who", values="treeid", aggfunc="size", fill_value=0)
ax = by_year[groups + ["Other", "Unknown"]].plot.area(figsize=(12, 4), alpha=0.85,
                                                    color=["#94a3b8", "#38bdf8", "#4ade80", "#fbbf24", "#e5e7eb"])
ax.set_title("Street trees by recorded planting year and planter"); ax.set_ylabel("trees")
plt.tight_layout(); plt.show()
era = pd.cut(dated.year, [1989, 2014, 2020, 2026], labels=["1990-2014", "2015-2020", "2021-2026"])
pd.crosstab(era, dated.who, normalize="index")[groups + ["Other", "Unknown"]].round(3)''',
    """md:`FUF` is Friends of the Urban Forest, the non-profit that runs volunteer planting days; `DPW` is Public Works itself; `Private` means a property owner planted with a permit.

**Who plants has changed.** From 1990 to 2014 about three quarters of dated trees were planted by property owners; since 2021 Friends of the Urban Forest has planted most of them. The 2017-2018 bump is largely trees with no recorded planter, and the 2008 spike was mostly Public Works.

## 5. The data trap: trunk diameter isn't re-measured
A natural next step is a growth curve: trunk diameter (`mapdbh`) against age. Let's look at it first.""",
    '''known["age"] = (df.data_as_of.max() - known.planteddate).dt.days / 365.25
aged = known.dropna(subset=["age", "mapdbh"]).query("age >= 0")
buckets = pd.cut(aged.age, [0, 2, 5, 10, 15, 20, 30, 40], include_lowest=True)
trap = aged.groupby(buckets, observed=True).agg(trees=("mapdbh", "size"),
                                              share_exactly_3in=("mapdbh", lambda s: (s == 3).mean()),
                                              median_dbh=("mapdbh", "median"))
trap.round(2)''',
    """md:Almost every tree under 10 years old is recorded at **exactly 3 inches**, the smallest value in the data. Young street trees do grow, so this is the diameter entered at planting (or a default), not a fresh measurement. Records are only updated when crews work on a tree. A model trained on `mapdbh` would learn "young trees don't grow", which is an artefact of how the data is collected.

What we *can* do is restrict to trees old enough to have been re-measured and look at the species that grow fastest there.""",
    '''mature = aged[(aged.age >= 20) & (aged.mapdbh > 3)]
top = mature.species_common.value_counts()
top = top[top >= 150].index
growth = (mature[mature.species_common.isin(top)]
          .assign(inches_per_year=lambda d: d.mapdbh / d.age)
          .groupby("species_common").inches_per_year.agg(["median", "size"])
          .sort_values("median"))
ax = growth["median"].plot.barh(figsize=(8, 0.35 * len(growth) + 1), color="#4ade80")
ax.set_xlabel("median trunk diameter gained per year (inches)")
ax.set_title("Trees planted 20+ years ago with a re-measured trunk (>3 in), species with 150+ trees")
plt.tight_layout(); plt.show()
growth.round(2)''',
    """md:Treat these rates as rough: they average over decades, and only trees whose diameter was updated are included, which biases toward trees that got attention from crews.

## Takeaways
- **Citywide, the street forest is diverse**: the most common species (London plane) is about 6% of trees, and no genus passes 8%, well within the 10-20-30 rule.
- **Locally it is not.** In 19 of the 34 neighborhoods with 500+ street trees, one species exceeds 10%. Downtown is the extreme: London plane is about a quarter of all street trees in the Tenderloin and Financial District/South Beach, and the New Zealand Christmas tree leads in Seacliff and the western neighborhoods. That is the scale where a pest outbreak would actually hurt.
- **Planting has shifted from property owners to Friends of the Urban Forest** over the last few years.
- **Check how a column is collected before modeling it.** `mapdbh` is effectively frozen at planting for young trees, so naive growth models, age-from-size models, or "canopy growth" maps built on it would be wrong.

**Ideas to try next:** join the [Street Tree Removal Notifications](https://data.sfgov.org/d/qrwx-q4gg) data to study which species get removed, add census tract income to look at canopy equity, or cluster streets by species mix.""",
]

# ============================================================================ WDI
wdi_cells = [
    """md:# 60 Years of Development in 5 Charts (World Bank WDI)

A tour of the [World Development Indicators 1960-2025](https://www.kaggle.com/datasets/teddybrown414/world-bank-world-development-indicators) dataset: how to load it without falling into its traps, and five questions it answers well.

1. **The Preston curve**: richer countries live longer, but how much has the curve moved?
2. **The fertility transition** by income group.
3. **Catch-up growth**: are poorer countries growing faster since 1990?
4. **Decoupling**: which economies grew while cutting CO₂ per person?
5. **The digital divide**: internet adoption by income group.

Every number in the text is computed in the cell above it.""",
    FIND_FILE + '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

panel = pd.read_csv(find_file("wdi_headline_panel.csv"))      # 40 popular indicators, economies only
long = pd.read_parquet(find_file("wdi_long.parquet"))          # everything, incl. aggregates
indicators = pd.read_csv(find_file("indicators.csv"))
countries = pd.read_csv(find_file("countries.csv"))
print(f"panel {panel.shape}, long {len(long):,} values, {len(indicators):,} indicators, {len(countries)} economies + aggregates")
panel.query("year == 2023").sample(3, random_state=0).iloc[:, :12]''',
    """md:## 0. Three things to know before you start
**Aggregates live next to countries** in the long file. `World`, regions and income groups are rows like any country, so averaging or ranking without filtering mixes them in.""",
    '''agg = set(countries.loc[countries.is_aggregate, "country_code"])
print(len(agg), "aggregates, e.g.:", countries.loc[countries.is_aggregate, "table_name"].head(6).tolist())
pop = long[(long.indicator_code == "SP.POP.TOTL") & (long.year == 2023)]
print(f"sum of 'population' over all rows: {pop.value.sum() / 1e9:,.0f} bn   "
      f"(over economies only: {pop[~pop.country_code.isin(agg)].value.sum() / 1e9:.2f} bn)")''',
    """md:**Recent years are thin.** Many series arrive with a lag of one to three years, so "latest year" comparisons need care.""",
    '''econ = long[~long.country_code.isin(agg)]
per_year = econ.groupby("year").size()
ax = per_year.plot(figsize=(11, 3), color="#38bdf8", title="Number of values per year (economies only)")
ax.set_ylabel("values"); plt.tight_layout(); plt.show()
print(f"peak {per_year.idxmax()} ({per_year.max():,}), 2023: {per_year[2023]:,}, 2025: {per_year[2025]:,}")''',
    """md:**Find indicators by searching the metadata**, not by guessing codes:""",
    '''def find(text, n=8):
    m = indicators[indicators.indicator_name.str.contains(text, case=False, na=False)]
    counts = long[long.indicator_code.isin(m.indicator_code)].groupby("indicator_code", observed=True).size()
    return (m.set_index("indicator_code")[["indicator_name", "topic_group"]]
             .join(counts.rename("n_values")).sort_values("n_values", ascending=False).head(n))

find("renewable")''',
    """md:## 1. The Preston curve has shifted up
In 1975 Samuel Preston showed that life expectancy rises steeply with income in poor countries and flattens in rich ones. We fit life expectancy against log GDP per capita (PPP) in 1990 and 2023.""",
    '''def preston(year):
    d = panel[panel.year == year].dropna(subset=["gdp_per_capita_ppp_const2021", "life_expectancy", "population"])
    slope, intercept = np.polyfit(np.log(d.gdp_per_capita_ppp_const2021), d.life_expectancy, 1)
    return d, slope, intercept

fig, ax = plt.subplots(figsize=(11, 6))
rows = []
for year, col in ((1990, "#94a3b8"), (2023, "#16a34a")):
    d, b, a = preston(year)
    ax.scatter(d.gdp_per_capita_ppp_const2021, d.life_expectancy, s=np.sqrt(d.population) / 15,
               color=col, alpha=0.45, edgecolors="white", linewidths=0.3, label=f"{year} ({len(d)} economies)")
    xs = np.geomspace(700, 150_000, 100)
    ax.plot(xs, a + b * np.log(xs), color=col, lw=2.5)
    rows.append({"year": year, "life exp. at $5k": a + b * np.log(5_000), "life exp. at $50k": a + b * np.log(50_000),
                 "years per doubling of income": b * np.log(2)})
ax.set_xscale("log"); ax.set_xlabel("GDP per capita, PPP (constant 2021 international $)"); ax.set_ylabel("life expectancy (years)")
ax.legend(markerscale=0.25); ax.set_title("Life expectancy vs income (bubble = population)"); plt.tight_layout(); plt.show()
pd.DataFrame(rows).set_index("year").round(1)''',
    """md:At the same income, people live markedly longer than in 1990, and the gain is largest for poorer countries, so the curve has become flatter: income matters a bit less than it used to. Cheaper vaccines, oral rehydration, malaria control and HIV treatment are the usual explanations: health technology spread faster than income.

## 2. The fertility transition
Income-group aggregates are already in the long file (`LIC`, `LMC`, `UMC`, `HIC`, `WLD`), so no averaging of countries is needed.""",
    '''groups = {"LIC": "Low income", "LMC": "Lower middle", "UMC": "Upper middle", "HIC": "High income", "WLD": "World"}
colors = {"LIC": "#dc2626", "LMC": "#f97316", "UMC": "#eab308", "HIC": "#2563eb", "WLD": "black"}

def by_group(code):
    return (long[(long.indicator_code == code) & long.country_code.isin(groups)]
            .pivot(index="year", columns="country_code", values="value"))

tfr = by_group("SP.DYN.TFRT.IN")
ax = tfr.rename(columns=groups).plot(figsize=(11, 4.5), color=[colors[c] for c in tfr.columns], lw=2)
ax.axhline(2.1, color="grey", ls="--", lw=1); ax.text(1961, 2.2, "replacement level ≈ 2.1", color="grey")
ax.set_ylabel("births per woman"); ax.set_title("Fertility rate by World Bank income group"); plt.tight_layout(); plt.show()
below = {y: int((panel[panel.year == y].fertility_rate.dropna() < 2.1).sum()) for y in (1990, 2023)}
print("economies below replacement:", below)
growth23 = by_group("SP.POP.GROW").loc[2023].rename(groups).round(2)
print("population growth 2023 (%/yr):", growth23.to_dict())
tfr.loc[[1960, 1990, 2023]].rename(columns=groups).round(2)''',
    """md:Upper-middle-income countries (China, Brazil, Mexico, Thailand, Türkiye...) now have *lower* fertility than high-income countries had in 1990, and more than half of all economies are below replacement. Low-income countries are still well above 4 births per woman, and that is where population is still growing fast (see the 2023 growth rates printed above).

## 3. Catch-up growth since 1990?
Growth theory predicts *convergence*: poorer countries should grow faster. We compare average annual growth of real GDP per capita (PPP) from 1990 to 2023 with the 1990 starting level.""",
    '''g = panel[panel.year.isin([1990, 2023])].pivot(index="country_code", columns="year", values="gdp_per_capita_ppp_const2021").dropna()
g["growth"] = (g[2023] / g[1990]) ** (1 / 33) - 1
g["start"] = pd.qcut(g[1990], 4, labels=["poorest 25%", "second", "third", "richest 25%"])
names = countries.set_index("country_code").table_name
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.scatter(g[1990], g.growth * 100, color="#6366f1", alpha=0.7)
for code in g.growth.nlargest(5).index.tolist() + g.growth.nsmallest(4).index.tolist():
    ax.annotate(names.get(code, code), (g.loc[code, 1990], g.loc[code, "growth"] * 100), fontsize=8, xytext=(3, 3), textcoords="offset points")
ax.axhline(0, color="grey", lw=1); ax.set_xscale("log")
ax.set_xlabel("GDP per capita in 1990, PPP"); ax.set_ylabel("average annual growth 1990-2023 (%)")
ax.set_title(f"Growth vs starting income ({len(g)} economies)"); plt.tight_layout(); plt.show()
print(f"correlation of log(1990 level) with growth: {np.corrcoef(np.log(g[1990]), g.growth)[0, 1]:.2f}")
shrank = g.growth < 0
print(f"{shrank.sum()} economies are poorer than in 1990, {(shrank & (g.start == 'poorest 25%')).sum()} of them from the poorest quarter")
print("fastest:", ", ".join(f"{names.get(c, c)} {v:.1%}" for c, v in g.growth.nlargest(6).items()))
(g.groupby("start", observed=True).growth.agg(["median", "size"]).assign(median=lambda d: (d["median"] * 100).round(2))
  .rename(columns={"median": "median growth %/yr", "size": "economies"}))''',
    """md:There is some catch-up, but it is weak: the correlation is only mildly negative and the middle two quartiles did best. The fastest growers mix East Asian success stories (China, Viet Nam, Myanmar) with oil booms (Equatorial Guinea, Guyana) and a post-war rebound (Bosnia and Herzegovina), while 18 economies, 7 of them from the poorest quarter, are poorer today than in 1990 (printed above). Economists call this the difference between *absolute* convergence (rarely seen) and *conditional* convergence (poorer countries catch up only when institutions, education and investment are in place).

## 4. Decoupling: richer *and* lower CO₂ per person
For each economy we compare 2005 and 2023: did real GDP per capita go up while CO₂ emissions per capita went down?""",
    '''def wide(col):
    return panel.pivot(index="country_code", columns="year", values=col)

y0, y1 = 2005, 2023
gdp, co2 = wide("gdp_per_capita_ppp_const2021"), wide("co2_t_per_capita")
d = pd.DataFrame({"gdp_change": gdp[y1] / gdp[y0] - 1, "co2_change": co2[y1] / co2[y0] - 1,
                  "population": wide("population")[y1]}).dropna()
d["decoupled"] = (d.gdp_change > 0) & (d.co2_change < 0)
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(d.gdp_change * 100, d.co2_change * 100, s=np.sqrt(d.population) / 15,
           c=np.where(d.decoupled, "#16a34a", "#94a3b8"), alpha=0.6, edgecolors="white", linewidths=0.3)
ax.axhline(0, color="black", lw=0.8); ax.axvline(0, color="black", lw=0.8)
offsets = {"USA": (6, 6), "DEU": (6, -12), "GBR": (6, -8), "FRA": (-28, -4), "JPN": (6, 4)}
for code in ["USA", "GBR", "DEU", "CHN", "IND", "FRA", "BRA", "IDN", "JPN"]:
    if code in d.index:
        ax.annotate(code, (d.loc[code, "gdp_change"] * 100, d.loc[code, "co2_change"] * 100), fontsize=9,
                    xytext=offsets.get(code, (5, 3)), textcoords="offset points")
ax.set_xlabel(f"change in GDP per capita {y0}-{y1} (%)"); ax.set_ylabel(f"change in CO₂ per capita {y0}-{y1} (%)")
ax.set_xlim(-60, 330); ax.set_ylim(-80, 260)
ax.set_title("Green = grew richer while cutting CO₂ per person"); plt.tight_layout(); plt.show()
print(f"{d.decoupled.sum()} of {len(d)} economies decoupled")
d[d.decoupled & (d.population > 2e7)].sort_values("co2_change").head(10).assign(
    country=lambda x: x.index.map(names)).set_index("country")[["gdp_change", "co2_change"]].mul(100).round(0)''',
    """md:Caveats: these are **territorial** emissions (where CO₂ is emitted, not where goods are consumed), so offshoring production can flatter rich countries; some cuts reflect recessions or war (Ukraine's GDP barely grew); and the chart is per person, so total emissions can still rise with population. Still, decoupling is clearly common, not a curiosity.

## 5. The digital divide""",
    '''net = by_group("IT.NET.USER.ZS").loc[1990:]
ax = net.rename(columns=groups).plot(figsize=(11, 4.5), color=[colors[c] for c in net.columns], lw=2)
ax.set_ylabel("% of population using the internet"); ax.set_title("Internet users by income group"); plt.tight_layout(); plt.show()
latest = net.dropna(how="all").index.max()
net.loc[[2005, 2015, latest]].rename(columns=groups).round(1)''',
    """md:Upper-middle-income countries followed the high-income S-curve roughly a decade later (about half online in 2015, where high-income countries were in 2005), and lower-middle-income countries are now catching up fast. Low-income countries are still at the start of the curve, with fewer than a quarter of people online. These aggregates run to the latest year, unlike many series in section 0.

## Takeaways
- **Load carefully**: filter out aggregates (`countries.is_aggregate`), expect thin recent years, and search `indicators.csv` instead of guessing codes.
- **Health outran income**: at the same income, life expectancy is much higher than in 1990, especially for poorer countries.
- **Fertility fell everywhere except the poorest countries**, and over half of economies are now below replacement.
- **Catch-up growth is real but weak and uneven.**
- **Dozens of economies grew richer while cutting CO₂ per person**, with the caveats above.

**Ideas for your own analysis:** the 1,492 indicators include debt, inequality, education outcomes, gender gaps and infrastructure; the `footnotes` file tells you which values are estimates. Try forecasting a series, clustering countries by development profile, or testing a growth theory with panel regressions.""",
]

# ============================================================================ EIA-930
eia_cells = [
    """md:# Can You Beat the Grid Operator's Forecast? US Electricity, Hour by Hour

Every US grid operator publishes a **day-ahead demand forecast** next to its actual demand, and EIA collects both in [US Electric Grid Hourly Data (EIA-930)](https://www.kaggle.com/datasets/teddybrown414/us-electric-grid-hourly-eia-930). That makes this dataset a rare thing: a forecasting problem with a professional benchmark built in.

1. **How much electricity does the US use, and when?**
2. **How the fuel mix has changed since 2018** (coal, gas, solar, wind).
3. **California's duck curve**, getting deeper.
4. **The forecasting challenge**: a weather-free model vs. the operators' own forecasts, and the data trap that makes naive comparisons misleading.

Every number in the text is computed in the cell above it.""",
    FIND_FILE + '''
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.simplefilter("ignore", RuntimeWarning)
cols = ["ba", "region", "utc_time_end", "local_time_end", "demand_forecast_mw", "demand_mw_adjusted",
        "demand_mw_adjusted_outlier_flag", "ng_solar_mw_adjusted", "ng_wind_mw_adjusted"]
hourly = pd.read_parquet(find_file("eia930_balance_hourly.parquet"), columns=cols)
region = pd.read_parquet(find_file("region_hourly.parquet"))
bas = pd.read_csv(find_file("balancing_authorities.csv"))
print(f"hourly: {len(hourly):,} rows, {hourly.ba.nunique()} BAs, {hourly.utc_time_end.min()} -> {hourly.utc_time_end.max()}")
us = region[region.region == "US48"].set_index("utc_time_end").sort_index()''',
    """md:## 1. How much, and when?""",
    '''full_years = us.demand_mw.resample("YE").agg(["mean", "count"])
full_years = full_years[full_years["count"] > 8700]
yearly_gw = (full_years["mean"] / 1000).rename(lambda t: t.year)
y0, y1 = yearly_gw.index.min(), yearly_gw.index.max()
print(f"average lower-48 demand: {yearly_gw.iloc[0]:.0f} GW in {y0} -> {yearly_gw.iloc[-1]:.0f} GW in {y1} "
      f"({yearly_gw.iloc[-1] / yearly_gw.iloc[0] - 1:+.1%})")
print(f"peak hour: {us.demand_mw.max() / 1000:.1f} GW at {us.demand_mw.idxmax()} UTC")

local = us.demand_mw.copy(); local.index = local.index - pd.Timedelta(hours=6)   # ~US Central time
prof = local.to_frame("mw").assign(hour=local.index.hour, season=local.index.month.map(
    {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer",
     9: "Autumn", 10: "Autumn", 11: "Autumn"}))
prof = prof.pivot_table(index="hour", columns="season", values="mw") / 1000
fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
yearly_gw.plot.bar(ax=axes[0], color="#38bdf8"); axes[0].set_xlabel(""); axes[0].set_ylabel("GW (average)"); axes[0].set_title("Average lower-48 demand by year")
axes[0].set_ylim(yearly_gw.min() * 0.9, yearly_gw.max() * 1.03)
prof[["Winter", "Spring", "Summer", "Autumn"]].plot(ax=axes[1], color=["#2a78d6", "#1baf7a", "#eb6834", "#6250d6"], lw=2)
axes[1].set_xlabel("hour of day (approx. US Central time)"); axes[1].set_ylabel("GW"); axes[1].set_title("Average daily profile by season")
plt.tight_layout(); plt.show()''',
    """md:Demand was flat in the late 2010s, dipped in 2020 (the pandemic) and has risen since; summer afternoons (air conditioning) set the peaks. Winter has two humps: morning and evening.

## 2. The fuel mix since 2018
Generation by source starts in July 2018. In mid-2024 EIA split hydro into conventional hydro and pumped storage, so we recombine them.""",
    '''gen = us.loc["2018-07-01":].copy()
hydro = gen[[c for c in ["ng_hydro_incl_pumped_storage_mw", "ng_hydro_mw", "ng_pumped_storage_mw"] if c in gen]].sum(axis=1, min_count=1)
mix = pd.DataFrame({"Coal": gen.ng_coal_mw, "Natural gas": gen.ng_natural_gas_mw, "Nuclear": gen.ng_nuclear_mw,
                    "Hydro": hydro, "Wind": gen[["ng_wind_mw", "ng_wind_with_battery_mw"]].sum(axis=1, min_count=1),
                    "Solar": gen[["ng_solar_mw", "ng_solar_with_battery_mw"]].sum(axis=1, min_count=1)})
monthly = mix.resample("MS").sum()
share = monthly.div(gen.net_generation_mw.resample("MS").sum(), axis=0) * 100
share = share.iloc[:-1]                                    # drop the current, incomplete month
colors = ["#52514e", "#eb6834", "#6250d6", "#2a78d6", "#1baf7a", "#eda100"]
ax = share.plot(figsize=(13, 4.5), color=colors, lw=2)
ax.set_ylabel("% of net generation"); ax.set_title("Lower-48 generation share by source (monthly)"); plt.tight_layout(); plt.show()
first, last = share.loc["2019"].mean(), share.iloc[-12:].mean()
pd.DataFrame({"2019 average %": first, "last 12 months %": last, "change (pts)": last - first}).round(1)''',
    """md:Coal's share has shrunk while solar and wind have grown; natural gas is the backbone throughout. Monthly shares are seasonal: solar's share peaks in late spring (May), wind's in spring (April), and gas's in July-August, when demand is highest.

## 3. California's duck curve
*Net load* is demand minus solar and wind: what the rest of the fleet must supply. As solar grows, midday net load sinks and the evening ramp gets steeper (the "duck").""",
    '''ca = hourly[hourly.ba == "CISO"].copy()
ca["net_load"] = ca.demand_mw_adjusted - ca.ng_solar_mw_adjusted.fillna(0) - ca.ng_wind_mw_adjusted.fillna(0)
ca["hour"], ca["year"], ca["month"] = ca.local_time_end.dt.hour, ca.local_time_end.dt.year, ca.local_time_end.dt.month
april = ca[(ca.month == 4) & ca.year.isin([2019, 2022, int(ca.loc[ca.month == 4, "year"].max())])]
curves = april.pivot_table(index="hour", columns="year", values="net_load") / 1000
ax = curves.plot(figsize=(11, 4.5), lw=2.5, color=["#94a3b8", "#38bdf8", "#1d4ed8"])
ax.set_xlabel("hour of day (Pacific time)"); ax.set_ylabel("GW"); ax.set_title("CAISO net load in April (demand - solar - wind)")
plt.tight_layout(); plt.show()
ramp = (curves.loc[17:20].max() - curves.loc[10:15].min()).round(1)
print("midday minimum (GW):", curves.loc[10:15].min().round(1).to_dict())
print("evening ramp, midday min -> evening max (GW):", ramp.to_dict())''',
    """md:Between 2019 and 2022 the belly of the duck sank by about 4 GW and the evening ramp grew by about 4 GW: on spring middays, solar covers most of California's demand, and the rest of the fleet must ramp up by well over 10 GW within a few hours as the sun sets. The most recent April is *not* deeper than 2022 (printed above), so don't extrapolate the trend blindly: higher demand, curtailment of excess solar, and year-to-year weather all move this curve.

## 4. Can a model beat the operator's day-ahead forecast?
**Setup.** For the 15 largest balancing authorities, train on 2019-2024 and test on 2025 onward. The target is EIA's adjusted demand (flagged outliers removed). To be fair to a *day-ahead* forecast, the model may only use demand that is at least 48 hours old, plus calendar features. No weather data, which is the operators' biggest advantage.""",
    '''from sklearn.ensemble import HistGradientBoostingRegressor

def mape(actual, pred):
    m = actual.notna() & pred.notna() & (actual > 0)
    return float((abs(actual[m] - pred[m]) / actual[m]).mean() * 100)

recent = hourly[hourly.utc_time_end >= "2019-01-01"]
top = recent.groupby("ba", observed=True).demand_mw_adjusted.mean().nlargest(15).index.astype(str)
rows, series = [], {}
for ba in top:
    d = recent[recent.ba == ba].set_index("utc_time_end").sort_index().asfreq("h")
    y = d.demand_mw_adjusted.where(~d.demand_mw_adjusted_outlier_flag.fillna(False).astype(bool))
    fc = d.demand_forecast_mw.where(d.demand_forecast_mw > 0)
    # bias correction known in advance: trailing 28-day median of actual/forecast, at least 48 h old
    ratio = (y / fc).shift(48).rolling(24 * 28, min_periods=24 * 7).median()
    X = pd.DataFrame({f"lag{L}": y.shift(L) for L in (48, 72, 96, 120, 144, 168, 336)}, index=d.index)
    X["mean_last_week"] = y.shift(48).rolling(168, min_periods=100).mean()
    lt = d.local_time_end.ffill()
    X["hour"], X["dow"], X["doy"] = lt.dt.hour, lt.dt.day_of_week, lt.dt.day_of_year
    Xh = X.assign(forecast=fc, forecast_ratio=ratio)          # hybrid: model + the operator's forecast
    train = (X.index < "2025-01-01") & y.notna()
    test = (X.index >= "2025-01-01") & y.notna() & (X.index < y.dropna().index.max().floor("D"))
    fit = lambda A: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(A[train], y[train])
    p_model = pd.Series(fit(X).predict(X[test]), index=X.index[test])
    p_hybrid = pd.Series(fit(Xh).predict(Xh[test]), index=X.index[test])
    yt = y[test]
    rows.append({"ba": ba, "mean GW": yt.mean() / 1000,
                 "operator forecast": mape(yt, fc[test]),
                 "operator forecast, bias-corrected": mape(yt, (fc * ratio)[test]),
                 "model (no weather)": mape(yt, p_model),
                 "hybrid model + forecast": mape(yt, p_hybrid),
                 "forecast bias %": float(((fc[test] - yt) / yt).mean() * 100)})
    series[ba] = (yt, fc[test])
res = pd.DataFrame(rows).set_index("ba").round(2)
res''',
    '''summary = res.drop(columns=["mean GW", "forecast bias %"]).median().round(2)
print("median MAPE across the 15 BAs (%):", summary.to_dict())
print("bias correction helps in", int((res["operator forecast, bias-corrected"] < res["operator forecast"]).sum()), "of 15")
print("hybrid beats the raw forecast in", int((res["hybrid model + forecast"] < res["operator forecast"]).sum()), "of 15")
print("weather-free model beats the raw forecast in", int((res["model (no weather)"] < res["operator forecast"]).sum()), "of 15:",
      res.index[res["model (no weather)"] < res["operator forecast"]].tolist())''',
    """md:### The trap: some "forecasts" aren't forecasting the same thing
At first sight, a weather-free model beats several operators' forecasts. Look at the **bias** column before believing that. Some forecast series sit consistently far from actual demand. That is a sign the forecast covers a different footprint or definition than the reported demand, not that the operator can't forecast. A trailing bias correction, which uses only information available a day ahead, fixes most of it.""",
    '''worst = res["forecast bias %"].abs().nlargest(3).index.tolist()
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=False)
for ax, ba in zip(axes, worst):
    yt, f = series[ba]
    err = ((f - yt) / yt * 100).resample("W").mean()
    ax.plot(err.index, err, color="#eb6834"); ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"{ba}: weekly mean forecast error (%)"); ax.tick_params(axis="x", rotation=45)
plt.tight_layout(); plt.show()
res.loc[worst, ["operator forecast", "operator forecast, bias-corrected", "forecast bias %"]]''',
    """md:A constant offset (a level shift that never goes away) or a sudden step on one date are signatures of a definitional or reporting change, worth checking against the operator's documentation before any model comparison. Two more cautions from the table:
- The **hybrid model is not safe after a regime change**: where the forecast's behaviour changed after 2024, a model trained on older data keeps applying the old relationship, while the simple trailing correction adapts.
- A forecast that is *too* good (well under 1% error after correction) deserves suspicion too: some operators may revise their "day-ahead" values after the fact.

## Takeaways
- Lower-48 demand is growing again, peaking on summer afternoons; the peak hour in this data is shown in section 1.
- Coal is shrinking, solar and wind are growing, and California's spring net load now has a deep midday belly and a steep evening ramp (the duck curve).
- **Operators' day-ahead forecasts are hard to beat without weather data**: a lag-and-calendar model is roughly twice as wrong. Feeding the operator forecast into a model (hybrid) helps for most large operators.
- **Before ranking forecasts, check bias over time.** Large constant offsets and step changes usually mean the series aren't comparable.

**Try next:** add weather (e.g. NOAA temperatures for each BA's largest city), forecast net load instead of demand, or predict the evening ramp in CAISO.""",
]

# ============================================================================ flights
flights_cells = [
    """md:# Why Is My Flight Late? 32 Million US Flights, 2022-2026

This notebook explores the [US Flight Delays & On-Time Performance 2022-2026](https://www.kaggle.com/datasets/teddybrown414/us-flight-delays-and-on-time-performance-2022-2026) dataset: every domestic flight reported to the US Bureau of Transportation Statistics, with delays, their causes, cancellations and diversions.

1. How on-time performance moved month by month
2. What causes the delay minutes
3. Why the evening flight is the risky one
4. Airlines and airports
5. **Predicting a late arrival without cheating**: which columns leak the answer""",
    FIND_FILE + '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

files = sorted(glob.glob("/kaggle/input/**/flights_*.parquet", recursive=True)) or \\
        sorted(glob.glob("../../datasets/**/flights_*.parquet", recursive=True))
COLS = ["flight_date", "airline", "origin", "dest", "sched_dep_hhmm", "dep_delay_min", "arr_delay_min",
        "arr_delayed_15", "cancelled", "cancellation_code", "diverted", "distance_mi",
        "delay_carrier_min", "delay_weather_min", "delay_nas_min", "delay_security_min", "delay_late_aircraft_min"]
parts = [pd.read_parquet(f, columns=COLS) for f in files]
for c in ["airline", "origin", "dest", "cancellation_code"]:   # same categories in every year, so concat stays compact
    cats = pd.api.types.union_categoricals([p[c] for p in parts]).categories
    for p in parts:
        p[c] = p[c].cat.set_categories(cats)
df = pd.concat(parts, ignore_index=True); del parts
df["flight_date"] = pd.to_datetime(df.flight_date)
carriers = pd.read_csv(find_file("carriers.csv"))
airports = pd.read_csv(find_file("airports.csv"))
print(f"{len(df):,} flights, {df.flight_date.min():%Y-%m} to {df.flight_date.max():%Y-%m}, "
      f"{df.memory_usage(deep=True).sum() / 1e9:.1f} GB in memory")
df.head()''',
    """md:## 1. On-time performance, month by month
A flight is officially **late** when it arrives 15+ minutes after schedule (`arr_delayed_15`; empty for cancelled and diverted flights).""",
    '''m = df.groupby(df.flight_date.dt.to_period("M")).agg(
    flights=("cancelled", "size"), late=("arr_delayed_15", "mean"), cancelled=("cancelled", "mean"))
fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
a1.plot(m.index.to_timestamp(), m.late * 100, color="#eb6834", lw=2); a1.set_ylabel("% arriving 15+ min late")
a2.bar(m.index.to_timestamp(), m.cancelled * 100, width=25, color="#475569"); a2.set_ylabel("% cancelled")
a1.set_title("US domestic flights: late arrivals and cancellations by month"); plt.tight_layout(); plt.show()
print("worst months for cancellations:\\n", (m.cancelled.nlargest(4) * 100).round(2).to_string())
print("worst months for late arrivals:\\n", (m.late.nlargest(4) * 100).round(1).to_string())
print("best months for late arrivals:\\n", (m.late.nsmallest(4) * 100).round(1).to_string())''',
    """md:**What stands out:** two winters dominate cancellations: **January 2022** (6.2% of flights cancelled, during the Omicron wave and winter storms) and **December 2022** (5.5%, Winter Storm Elliott and the Southwest Airlines meltdown). January 2026 is third at 4.7%. Late arrivals follow the summer instead: **July** was the worst month of 2023, 2024, 2025 and 2026 (28-30% late), and **autumn** (September to November) is the best time to fly (13-16% late).

## 2. What causes the delay minutes?
For every flight that arrives 15+ minutes late, airlines split the delay into five causes. Summing the minutes shows where the lateness comes from.""",
    '''CAUSES = {"delay_late_aircraft_min": "late-arriving aircraft", "delay_carrier_min": "airline (crew, maintenance...)",
          "delay_nas_min": "air traffic system", "delay_weather_min": "extreme weather", "delay_security_min": "security"}
by_year = df.groupby(df.flight_date.dt.year)[list(CAUSES)].sum().rename(columns=CAUSES)
share = by_year.div(by_year.sum(axis=1), axis=0) * 100
ax = share.plot.barh(stacked=True, figsize=(11, 3.6), color=["#eb6834", "#f59e0b", "#38bdf8", "#1d4ed8", "#94a3b8"])
ax.set_xlabel("% of delay minutes"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=5, frameon=False)
ax.set_title("Where the delay minutes come from"); ax.invert_yaxis(); plt.tight_layout(); plt.show()
share.round(1).to_csv("delay_cause_share_by_year.csv")
share.round(1)''',
    '''cc = df[df.cancelled].cancellation_code.value_counts(normalize=True).mul(100).round(1)
cc.index = cc.index.map({"A": "airline", "B": "weather", "C": "air traffic system", "D": "security"})
print("cancellation reasons (% of cancellations):\\n", cc.to_string())''',
    """md:**What stands out:** the most common reason a flight is late is that **its plane was already late**. Late-arriving aircraft account for about 40% of delay minutes every year. The airlines' own share (crew, maintenance, cleaning, fuelling) fell from 39.8% in 2022 to about 33% in 2025-2026, while the air traffic system's share rose from 16.8% to about 21%. Extreme weather causes only about 6% of delay minutes, yet **57% of cancellations**. Ordinary bad weather is counted under "air traffic system", so weather's real share of delays is larger than 6%.

## 3. Delays snowball through the day""",
    '''ok = df[~df.cancelled & ~df.diverted]
hour = (ok.sched_dep_hhmm // 100).clip(0, 23)
h = ok.groupby(hour).agg(flights=("arr_delayed_15", "size"), late=("arr_delayed_15", "mean"),
                         late_aircraft_min=("delay_late_aircraft_min", "mean"))
fig, ax = plt.subplots(figsize=(11, 4))
ax.bar(h.index, h.late * 100, color="#38bdf8"); ax.set_xlabel("scheduled departure hour (local)")
ax.set_ylabel("% arriving 15+ min late"); ax.set_title("The later you leave, the more likely you arrive late")
ax2 = ax.twinx(); ax2.plot(h.index, h.late_aircraft_min, color="#eb6834", lw=2, marker="o")
ax2.set_ylabel("late-aircraft minutes per late flight", color="#eb6834"); plt.tight_layout(); plt.show()
h.assign(late=(h.late * 100).round(1), late_aircraft_min=h.late_aircraft_min.round(1)).to_csv("late_rate_by_hour.csv")
busy = h[h.flights > 100_000]
print("lowest late rate:", busy.late.idxmin(), "h", round(busy.late.min() * 100, 1), "%   highest:",
      busy.late.idxmax(), "h", round(busy.late.max() * 100, 1), "%  (hours with 100k+ flights)")''',
    """md:**What stands out:** a flight scheduled to leave at 5-6 am arrives late 9-10% of the time. By 7-8 pm it is **31%, more than three times as often**. Delays snowball: an evening flight's plane has already flown several legs that day, and the minutes blamed on a late-arriving aircraft (orange line, among late flights) more than double between 5 am and 7 pm. After 9 pm the rate eases as traffic thins. If you can choose, fly early.

## 4. Airlines and airports (latest 12 months)""",
    '''last = df[df.flight_date > df.flight_date.max() - pd.DateOffset(months=12)]
air = (last.groupby("airline", observed=True)
           .agg(flights=("cancelled", "size"), late=("arr_delayed_15", "mean"), cancelled=("cancelled", "mean"),
                mean_arr_delay=("arr_delay_min", "mean"))
           .join(carriers.set_index("airline").airline_name).sort_values("late"))
air[["late", "cancelled"]] *= 100
air = air.round({"late": 1, "cancelled": 2, "mean_arr_delay": 1}); air.to_csv("airlines_last_12_months.csv")
air''',
    '''big = last.groupby("origin", observed=True).agg(flights=("cancelled", "size"), late=("arr_delayed_15", "mean")).nlargest(30, "flights")
big = big.join(airports.drop_duplicates("code").set_index("code").municipality).sort_values("late")
fig, ax = plt.subplots(figsize=(11, 7))
ax.barh([f"{c} ({m})" for c, m in zip(big.index, big.municipality)], big.late * 100, color="#475569")
ax.axvline(last.arr_delayed_15.mean() * 100, color="#eb6834", ls="--", label="all airports")
ax.set_xlabel("% of departures arriving 15+ min late"); ax.set_title("The 30 busiest origin airports, latest 12 months")
ax.legend(); plt.tight_layout(); plt.show()
big.assign(late=(big.late * 100).round(1)).to_csv("busiest_airports_last_12_months.csv")
print("12-month window:", last.flight_date.min().date(), "to", last.flight_date.max().date(), "| all-airport late rate:", round(last.arr_delayed_15.mean() * 100, 1))''',
    """md:**What stands out (September 2025 to August 2026):** Delta has the lowest late rate of the large airlines (18.9%), and JetBlue (28.7%) and Spirit (28.0%) the highest. Late and cancelled are different stories: Southwest is mid-table for lateness (24.3%) but cancels only 0.92% of flights, and Allegiant cancels the fewest (0.68%). Regional airlines PSA (4.8%) and Republic (4.6%) cancel the most, along with Spirit (4.5%). Among the 30 busiest airports, **Salt Lake City** is the most punctual origin (15.8% late), and **Fort Lauderdale, Chicago O'Hare and Washington National** are the least (28-30%).

## 5. Predicting a late arrival without cheating
A common Kaggle task is to predict `arr_delayed_15`. Most columns in this table are **only known after the flight**: departure delay, taxi times, actual times and the delay-cause minutes. A model that uses them looks brilliant and is useless for a traveller booking a ticket.

**Setup.** Operated flights only. Train on a sample of 2024, test on 2025 (a different year, so seasonality and schedule changes are honest). Features known at booking time: month, weekday, scheduled hour, airline, origin, destination, distance.""",
    '''from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ok = df[~df.cancelled & ~df.diverted & df.arr_delayed_15.notna()]
def top_codes(s, k=250):            # HistGradientBoosting allows at most 255 categories per feature
    keep = s.value_counts().index[:k - 1]
    return pd.Series(pd.Categorical(s.astype(str), categories=keep.astype(str)).codes, index=s.index).replace(-1, k - 1)
X = pd.DataFrame({"month": ok.flight_date.dt.month, "dow": ok.flight_date.dt.dayofweek,
                  "hour": (ok.sched_dep_hhmm // 100).clip(0, 23), "distance": ok.distance_mi,
                  "airline": top_codes(ok.airline), "origin": top_codes(ok.origin), "dest": top_codes(ok.dest)})
y, year = ok.arr_delayed_15.astype(int), ok.flight_date.dt.year
rng = np.random.default_rng(0)
tr = np.flatnonzero(year == 2024); tr = rng.choice(tr, min(len(tr), 3_000_000), replace=False)
te = np.flatnonzero(year == 2025); te = rng.choice(te, min(len(te), 1_000_000), replace=False)
def fit_auc(X):
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, categorical_features=[c in ("airline", "origin", "dest") for c in X.columns], random_state=0)
    m.fit(X.iloc[tr], y.iloc[tr])
    return roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1])

auc_honest = fit_auc(X)
auc_leaky = fit_auc(X.assign(dep_delay=ok.dep_delay_min))
print(f"base rate (2025 test): {y.iloc[te].mean():.1%}")
print(f"AUC, booking-time features only : {auc_honest:.3f}")
print(f"AUC, + departure delay (leak)   : {auc_leaky:.3f}")''',
    """md:**What stands out:** booking-time features alone reach an AUC of **0.655**. That beats chance but is far from a reliable prediction (22.2% of test flights were late). Adding departure delay lifts the AUC to **0.933**, and the whole jump comes from that single column, which is only known after the plane leaves: a flight that departs 40 minutes late will arrive late. If a model on this table scores above 0.9, check that no after-the-fact column is among its features (`dep_delay_min`, `dep_delayed_15`, taxi times, actual times, the `delay_*` causes, `cancellation_code`).

## Takeaways
- About 1 in 5 US flights arrives 15+ minutes late. July is the worst month and autumn the best, while cancellations spike in bad winters.
- **Late aircraft are the top cause of delay minutes (about 40%)**, and weather drives most cancellations.
- **Fly early**: 5-6 am departures are late 9-10% of the time, and 7-8 pm departures 31%.
- Airlines and airports differ a lot, and being late and cancelling are separate problems (Southwest, Allegiant).
- **Prediction trap:** most columns are only known after the flight. Honest booking-time models land around AUC 0.65, and anything near 0.93 is probably leaking.

**Try next:** join hourly weather at the origin, add the aircraft's previous flight (via `tail_number`) as a real-time feature, or predict cancellations around big storms.""",
]

if __name__ == "__main__":
    build("why-is-my-flight-late-32m-us-flights-2022-2026", "Why Is My Flight Late? 32M US Flights 2022-2026",
          f"{OWNER}/us-flight-delays-and-on-time-performance-2022-2026", flights_cells)
    build("can-you-beat-the-grid-operator-s-forecast-eia-930", "Can You Beat the Grid Operator's Forecast? EIA-930",
          f"{OWNER}/us-electric-grid-hourly-eia-930", eia_cells)
    build("60-years-of-development-in-5-charts-world-bank", "60 Years of Development in 5 Charts (World Bank)",
          f"{OWNER}/world-bank-world-development-indicators", wdi_cells)
    build("sf-street-trees-diversity-planting-a-data-trap", "SF Street Trees: Diversity, Planting & a Data Trap",
          f"{OWNER}/san-francisco-trees", trees_cells)
    build("query-intent-routing-rules-vs-tf-idf", "Query Intent Routing: Rules vs TF-IDF",
          f"{OWNER}/data-for-classifying-intent-of-query", intent_cells)
    build("btc-15-min-eda-label-anatomy-leak-free-baseline", "BTC 15-min: EDA, Label Anatomy, Leak-Free Baseline",
          f"{OWNER}/binance-ohlc-with-indicators", btc_cells)
