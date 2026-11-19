"""Smoke tests that need no network and no source data."""
import json
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def test_every_tool_compiles():
    for path in sorted(TOOLS.glob("*.py")):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_strip_main_guard_dedents_the_guarded_block():
    ns = runpy.run_path(str(TOOLS / "make_kaggle_pipeline.py"), run_name="lib")
    src = 'x = 1\nif __name__ == "__main__":\n    run(x)\n    print(x)\n'
    assert ns["strip_main_guard"](src) == "x = 1\nrun(x)\nprint(x)"


def test_bts_kernel_is_generated_and_compiles(tmp_path):
    subprocess.run([sys.executable, "-I", str(TOOLS / "make_kaggle_pipeline.py"), "bts", "none"], check=True, cwd=ROOT)
    out = ROOT / "pipelines" / "us-flight-delays"
    src = (out / "pipeline.py").read_text(encoding="utf-8")
    compile(src, "pipeline.py", "exec")
    assert 'PUBLISH = "none"' in src
    meta = json.loads((out / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["enable_internet"] is True and meta["is_private"] is True


def _kernel_source(name):
    ns = runpy.run_path(str(TOOLS / "make_kaggle_pipeline.py"), run_name="lib")
    d = ns["embedded"](name, "none")
    return d, (d / "pipeline.py").read_text(encoding="utf-8")


def test_embedded_kernels_carry_the_repo_tools_verbatim():
    for name in ("eia930", "wdi", "trees"):
        d, src = _kernel_source(name)
        compile(src, "pipeline.py", "exec")
        sources = {}
        exec(src.split("SOURCES = ", 1)[1].split("\nTOOLS, DATA, OUT", 1)[0].join(["sources.update(", ")"]), {"sources": sources})
        assert "make_covers.py" in sources
        for tool, code in sources.items():
            assert code == (TOOLS / tool).read_text(encoding="utf-8"), tool
        meta = json.loads((d / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert meta["id"].endswith("/" + d.name) and meta["is_private"] is True


def test_embedded_runner_copies_the_build_to_the_output_root(tmp_path):
    _, src = _kernel_source("wdi")
    temp, working = tmp_path / "temp", tmp_path / "working"
    built = temp / "datasets" / "world-development-indicators" / "v1"
    built.mkdir(parents=True)
    working.mkdir()
    (built / "dataset-metadata.json").write_text(json.dumps({"id": "someone/some-dataset"}), encoding="utf-8")
    (built / "data.csv").write_text("a\n1\n", encoding="utf-8")
    src = src.replace("/kaggle/temp", temp.as_posix()).replace("/kaggle/working", working.as_posix())
    src = src.replace("STEPS = [", "STEPS = []\n_unused = [", 1)  # no network in tests
    ns = {"__name__": "kernel"}
    exec(compile(src, "pipeline.py", "exec"), ns)
    assert ns["DATASET_REF"] == "someone/some-dataset"
    assert sorted(p.name for p in working.iterdir()) == ["data.csv", "dataset-metadata.json"]
    assert (temp / "kdp_tools" / "build_wdi.py").exists()


def test_wdi_extract_keeps_the_release_file_time(tmp_path):
    import datetime
    import zipfile

    ns = runpy.run_path(str(TOOLS / "fetch_wdi.py"), run_name="lib")
    zpath = tmp_path / "WDI_CSV.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for name in ns["NEEDED"]:
            z.writestr(zipfile.ZipInfo(name, date_time=(2026, 11, 16, 9, 30, 0)), "x\n")
    out = tmp_path / "raw"
    assert ns["extract"](zpath, out) == sorted(ns["NEEDED"])
    stamp = datetime.datetime.fromtimestamp((out / "WDICSV.csv").stat().st_mtime)
    assert stamp.strftime("%Y-%m-%d") == "2026-11-16"
