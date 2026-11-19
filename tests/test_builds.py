"""End-to-end builds on tiny real-data samples (tests/fixtures/data), checked against Kaggle's metadata rules."""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "data"

BUILDS = {
    "eia930": (["build_eia930.py"], "eia930-hourly-grid/v1"),
    "wdi": (["build_wdi.py"], "world-development-indicators/v1"),
    "trees": (["build_v2.py", "trees"], "san-francisco-trees/v2"),
}


def columns(path: Path) -> list[str]:
    if path.suffix == ".parquet":
        return pq.read_schema(path).names
    with path.open(encoding="utf-8") as f:
        return f.readline().rstrip("\r\n").split(",")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    shutil.copytree(FIXTURES, root, dirs_exist_ok=True)
    env = {**os.environ, "KDP_DATA_ROOT": str(root), "MPLBACKEND": "Agg"}
    for script, _ in BUILDS.values():
        subprocess.run([sys.executable, "-I", str(ROOT / "tools" / script[0]), *script[1:]], check=True, env=env, cwd=ROOT)
    return root


@pytest.mark.parametrize("name", BUILDS)
def test_build_is_kaggle_ready(built, name):
    out = built / BUILDS[name][1]
    meta = json.loads((out / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"].count("/") == 1
    assert 6 <= len(meta["title"]) <= 50, meta["title"]
    assert 20 <= len(meta["subtitle"]) <= 80, meta["subtitle"]
    assert meta["licenses"] and meta["keywords"] and len(meta["description"]) > 500
    assert not re.findall(r"<[a-z_]+>", meta["description"]), "Kaggle strips <word> as an HTML tag"
    assert (out / "dataset-cover-image.png").stat().st_size > 10_000
    assert meta["resources"], "no files declared"
    for res in meta["resources"]:
        path = out / res["path"]
        assert path.exists(), f"declared but not built: {res['path']}"
        fields = res.get("schema", {}).get("fields", [])
        if fields:
            documented = [f["name"] for f in fields]
            assert documented == columns(path), f"{res['path']}: schema fields don't match the file's columns"
            assert all(f.get("description") for f in fields), f"{res['path']}: undocumented column"


def test_wdi_drops_indicators_not_under_cc_by(built):
    import pandas as pd

    long = pd.read_parquet(built / "world-development-indicators/v1/wdi_long.parquet")
    assert not long.indicator_code.astype(str).str.startswith("MS.MIL").any()
    assert len(long) > 1000
