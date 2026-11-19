# kaggle-dataset-pipelines

[![CI](https://github.com/tb91117/kaggle-dataset-pipelines/actions/workflows/ci.yml/badge.svg)](https://github.com/tb91117/kaggle-dataset-pipelines/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Lint your Kaggle dataset before you upload it**, plus the open-data pipelines behind
[four public Kaggle datasets](#datasets-built-with-this-repo).

## kaggle-dataset-lint

Kaggle accepts a `dataset-metadata.json` that is subtly wrong and then shows a broken dataset page. Four examples;
all but the cover crop were found in published datasets:

- **Column descriptions on the wrong columns.** Kaggle matches schema fields to columns *by position*, so two swapped
  fields shift the descriptions without any error.
- **Text that disappears.** Kaggle's renderer drops unknown HTML tags, so `ng_<source>_mw` in a description shows as
  `ng__mw`.
- **A cover that shows only its corner.** The Kaggle CLI always crops the top-left 560x280 pixels, whatever the image
  size.
- **Garbled text from Windows.** The CLI reads `dataset-metadata.json` with the system encoding, so on most Windows
  machines a `×` in a UTF-8 description reaches Kaggle as two other characters. Set `PYTHONUTF8=1` before uploading.

```bash
pip install "git+https://github.com/tb91117/kaggle-dataset-pipelines"
kaggle-dataset-lint path/to/my-dataset     # the folder you pass to `kaggle datasets create -p`
```

No dependencies: it is a single standard-library file, so you can also just download
[`src/kaggle_dataset_lint.py`](src/kaggle_dataset_lint.py) and run `python kaggle_dataset_lint.py my-dataset`.
Install `pyarrow` too if you want Parquet columns checked.

```text
my-dataset/dataset-metadata.json: error: [title-length] title is 51 characters; Kaggle allows 6-50
my-dataset/dataset-metadata.json: warning: [html-stripped] description contains <unit>: Kaggle's renderer drops unknown HTML tags, so the text disappears (`ng_<source>_mw` shows as `ng__mw`). Write {unit} or &lt;unit&gt;
my-dataset/dataset-metadata.json: warning: [cover-cropped] dataset-cover-image.png is 1600x800, but the Kaggle CLI always crops the top-left 560x280 pixels for the header and a 280x280 square at x=140 for the thumbnail. Resize it to 560x280, or set the cover in the web UI
my-dataset/dataset-metadata.json: error: [schema-mismatch] readings.csv: schema fields must list every column in file order, or Kaggle attaches the descriptions to the wrong columns (first difference at position 1: schema has 'station', file has 'time')
my-dataset/dataset-metadata.json: warning: [file-undeclared] no resource entry (so no file or column descriptions) for: notes.txt
my-dataset/dataset-metadata.json: warning: [subfolder-skipped] raw/ is skipped by `kaggle datasets create/version` unless you pass --dir-mode zip or tar
2 error(s), 4 warning(s) in 1 dataset(s)
```

### What it checks

| Level | Checks |
|---|---|
| **Error**: Kaggle rejects it, or the page comes out wrong | title 6-50 characters, subtitle 20-80, exactly one license, `id` is `owner/slug` (no `INSERT_..._HERE` placeholders), valid `expectedUpdateFrequency`, every declared file exists and is declared once, schema fields list every column in file order (CSV, TSV, Parquet), cover is .png/.jpg/.jpeg/.webp and at least 560x280 |
| **Warning**: silent problems and incomplete pages | `<word>` placeholders Kaggle strips, covers the CLI crops to a corner, subfolders `kaggle datasets create/version` skips, non-ASCII text the CLI would garble because it reads the file with the system encoding (cp1252 on most Windows machines), files and columns without descriptions, missing subtitle, description, tags, sources, update frequency or cover image, slugs `kaggle datasets create` rejects |
| **Note**: worth a look, never fails | license names that aren't in Kaggle's documented ids, column types outside Kaggle's best-effort list |

The rules come from the Kaggle CLI source and its
[dataset metadata docs](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets_metadata.md). It also reads the
`{"info": ...}` layout that `kaggle datasets metadata` downloads.

### Options

- `--strict`: exit 1 on warnings as well as errors.
- `--no-files`: check only the metadata, not data files or the cover.
- `--ignore cover-missing,keywords-missing`: skip findings by code.
- `--format github`: print GitHub Actions annotations.

### In CI or pre-commit

```yaml
# GitHub Actions
- run: pip install "git+https://github.com/tb91117/kaggle-dataset-pipelines@v0.2.1"
- run: kaggle-dataset-lint --format github data/my-dataset
```

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/tb91117/kaggle-dataset-pipelines
    rev: v0.2.1
    hooks:
      - id: kaggle-dataset-lint
```

## Open-data pipelines

Each pipeline downloads an official public source, builds tidy Parquet/CSV files with typed columns, writes a full
`dataset-metadata.json` (title, subtitle, description, per-column descriptions, license, keywords) and renders a cover
image. Re-running a pipeline refreshes the dataset, so the Kaggle version stays current with the source. Every build is
tested on small real-data samples and must pass the linter.

### Datasets built with this repo

| Dataset on Kaggle | Source | Pipeline |
|---|---|---|
| [US Flight Delays & On-Time Performance 2022–2026](https://www.kaggle.com/datasets/teddybrown414/us-flight-delays-and-on-time-performance-2022-2026) | BTS *Reporting Carrier On-Time Performance* | `fetch_bts.py` → `build_bts.py` |
| [US Electric Grid Hourly Data (EIA-930)](https://www.kaggle.com/datasets/teddybrown414/us-electric-grid-hourly-eia-930) | EIA Hourly Electric Grid Monitor, six-month BALANCE files | `fetch_eia930.py` → `build_eia930.py` |
| [World Development Indicators 1960–2025](https://www.kaggle.com/datasets/teddybrown414/world-bank-world-development-indicators) | World Bank WDI bulk CSV | `fetch_wdi.py` → `build_wdi.py` |
| [San Francisco Street Tree Inventory](https://www.kaggle.com/datasets/teddybrown414/san-francisco-trees) | DataSF `tkzw-k3nq` (SODA API) | `fetch_sf_trees.py` → `build_v2.py trees` |

### Quick start

```bash
pip install -e ".[pipelines]"
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

### Tools

| Script | What it does |
|---|---|
| `src/kaggle_dataset_lint.py` | The linter above (`kaggle-dataset-lint`). |
| `fetch_*.py` | Resumable downloaders with retries; complete files are skipped on re-run. |
| `KDP_DATA_ROOT` | Environment variable: where sources are downloaded and datasets are built (default `./datasets`). |
| `build_*.py` | Clean, type and document a source into a Kaggle-ready folder. |
| `make_covers.py` | One distinct 2:1 cover design per dataset (no title text; Kaggle shows the title). |
| `push_cover.py` | Uploads metadata + cover with a crop that covers the whole image (the kaggle CLI crops the top-left corner). |
| `make_kaggle_pipeline.py` | Turns a fetch + build pair into a single Kaggle script kernel. |
| `make_notebooks.py` | Generates the example notebooks that accompany each dataset. |
| `profile_csv.py` | Prints a compact column profile of every CSV under a dataset folder. |

## Licenses

The code in this repo is MIT-licensed. The data keeps its source license: BTS and EIA data are US government works
(public domain), World Bank WDI is CC BY 4.0, and DataSF publishes under ODC-PDDL. Check each source before reuse.
