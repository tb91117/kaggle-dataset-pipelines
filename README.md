# kaggle-dataset-pipelines

Reproducible fetch → clean → document → publish pipelines for public open-data datasets on Kaggle.

Each pipeline downloads an official public source, builds tidy Parquet/CSV files with typed columns, writes a full
`dataset-metadata.json` (title, subtitle, description, per-column descriptions, license, keywords) and renders a cover
image. Re-running a pipeline refreshes the dataset, so the Kaggle version stays current with the source.

## Datasets built with this repo

| Dataset on Kaggle | Source | Pipeline |
|---|---|---|
| [US Flight Delays & On-Time Performance 2022–2026](https://www.kaggle.com/datasets/teddybrown414/us-flight-delays-and-on-time-performance-2022-2026) | BTS *Reporting Carrier On-Time Performance* | `fetch_bts.py` → `build_bts.py` |
| [US Electric Grid Hourly Data (EIA-930)](https://www.kaggle.com/datasets/teddybrown414/us-electric-grid-hourly-eia-930) | EIA Hourly Electric Grid Monitor, six-month BALANCE files | `fetch_eia930.py` → `build_eia930.py` |
| [World Development Indicators 1960–2025](https://www.kaggle.com/datasets/teddybrown414/world-bank-world-development-indicators) | World Bank WDI bulk CSV | `fetch_wdi.py` → `build_wdi.py` |
| [San Francisco Street Tree Inventory](https://www.kaggle.com/datasets/teddybrown414/san-francisco-trees) | DataSF `tkzw-k3nq` (SODA API) | `fetch_sf_trees.py` → `build_v2.py trees` |

## Quick start

```bash
pip install -e .
python -I tools/fetch_eia930.py      # downloads into datasets/eia930-hourly-grid/source/ (resumable)
python -I tools/build_eia930.py      # writes datasets/eia930-hourly-grid/v1/ (data + metadata + cover)
```

Nothing is uploaded by the build scripts. Publishing is a separate, explicit step:

```bash
kaggle datasets version -p datasets/eia930-hourly-grid/v1 -m "Refresh"
python -I tools/push_cover.py datasets/eia930-hourly-grid/v1 <owner>/<slug>   # cover with a correct crop
```

### Run on Kaggle instead of locally

Some sources are large (the BTS files are several GB). `make_kaggle_pipeline.py` assembles fetch + build + cover
into one self-contained Kaggle script kernel, so the download happens on Kaggle's servers:

```bash
python -I tools/make_kaggle_pipeline.py wdi none     # bts | eia930 | wdi | trees; none = build only
kaggle kernels push -p pipelines/world-bank-wdi-dataset-pipeline
```

The kernel's output folder is the finished dataset: create or update the Kaggle dataset from it
(dataset page → New version → Notebook output). For `eia930`, `wdi` and `trees` the repo's scripts are embedded
verbatim and run as files with `KDP_DATA_ROOT` pointing at Kaggle's scratch disk, so a kernel run and a local run
produce the same files.

With `private` or `public`, the kernel reads a Kaggle Secret named `KAGGLE_API_TOKEN` (Notebook → Add-ons → Secrets).

## Tools

| Script | What it does |
|---|---|
| `fetch_*.py` | Resumable downloaders with retries; complete files are skipped on re-run. |
| `KDP_DATA_ROOT` | Environment variable: where sources are downloaded and datasets are built (default `./datasets`). |
| `build_*.py` | Clean, type and document a source into a Kaggle-ready folder. |
| `make_covers.py` | One distinct 2:1 cover design per dataset (no title text; Kaggle shows the title). |
| `push_cover.py` | Uploads metadata + cover with a crop that covers the whole image (the kaggle CLI crops the top-left corner). |
| `make_kaggle_pipeline.py` | Turns a fetch + build pair into a single Kaggle script kernel. |
| `make_notebooks.py` | Generates the example notebooks that accompany each dataset. |
| `profile_csv.py` | Prints a compact column profile of every CSV under a dataset folder. |

## Data licenses

The code in this repo is MIT-licensed. The data keeps its source license: BTS and EIA data are US government works
(public domain), World Bank WDI is CC BY 4.0, and DataSF publishes under ODC-PDDL. Check each source before reuse.
