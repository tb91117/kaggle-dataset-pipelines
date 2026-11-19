"""End-to-end builds on tiny real-data samples (tests/fixtures/data), checked with the repo's own linter."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from kaggle_dataset_lint import lint

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "data"

BUILDS = {
    "eia930": (["build_eia930.py"], "eia930-hourly-grid/v1"),
    "wdi": (["build_wdi.py"], "world-development-indicators/v1"),
    "trees": (["build_v2.py", "trees"], "san-francisco-trees/v2"),
}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    shutil.copytree(FIXTURES, root, dirs_exist_ok=True)
    env = {**os.environ, "KDP_DATA_ROOT": str(root), "MPLBACKEND": "Agg"}
    for script, _ in BUILDS.values():
        subprocess.run([sys.executable, "-I", str(ROOT / "tools" / script[0]), *script[1:]], check=True, env=env, cwd=ROOT)
    return root


@pytest.mark.parametrize("name", BUILDS)
def test_build_passes_the_linter(built, name):
    out = built / BUILDS[name][1]
    # covers are 1600x800 on purpose: tools/push_cover.py uploads them with a full-image crop
    problems = [f for f in lint(out) if f.level != "note" and f.code != "cover-cropped"]
    assert problems == [], "\n".join(f"{f.code}: {f.message}" for f in problems)
    raw = (out / "dataset-metadata.json").read_bytes()
    assert raw.isascii(), "write the metadata with ensure_ascii=True: the Kaggle CLI may not read it as UTF-8"
    meta = json.loads(raw)
    assert len(meta["description"]) > 500 and meta["resources"]


def test_wdi_drops_indicators_not_under_cc_by(built):
    import pandas as pd

    long = pd.read_parquet(built / "world-development-indicators/v1/wdi_long.parquet")
    assert not long.indicator_code.astype(str).str.startswith("MS.MIL").any()
    assert len(long) > 1000
