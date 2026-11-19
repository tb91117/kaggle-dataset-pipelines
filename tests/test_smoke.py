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
