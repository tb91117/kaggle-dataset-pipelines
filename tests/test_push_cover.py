"""tools/push_cover.py: centred crops, and an upload copy that can't be garbled by the CLI's file encoding."""
import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def push_cover():
    pytest.importorskip("kaggle")
    pytest.importorskip("PIL")
    return runpy.run_path(str(ROOT / "tools" / "push_cover.py"), run_name="lib")


@pytest.mark.parametrize("size, cover, thumbnail", [
    ((1600, 800), (0, 0, 1600, 800), (400, 0, 800, 800)),     # 2:1: the whole image
    ((1000, 800), (0, 150, 1000, 500), (100, 0, 800, 800)),   # taller than 2:1
    ((2000, 500), (500, 0, 1000, 500), (750, 0, 500, 500)),   # wider than 2:1: centred, not left-aligned
])
def test_crops_are_the_largest_centred_rectangles(push_cover, size, cover, thumbnail):
    rects = push_cover["crop_rectangles"](*size)
    assert [(r.title, (r.left, r.top, r.width, r.height)) for r in rects] == [("cover image", cover), ("thumbnail", thumbnail)]


def test_upload_copy_escapes_non_ascii_and_keeps_the_cover(push_cover, tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    src.mkdir()
    dest.mkdir()
    meta = {"id": "someone/trees", "title": "Street Trees", "description": "144,606 trees " + chr(0xD7) + " 28 columns"}
    (src / "dataset-metadata.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    (src / "dataset-cover-image.png").write_bytes(b"png")
    push_cover["ascii_copy"](src, dest)
    raw = (dest / "dataset-metadata.json").read_bytes()
    assert raw.isascii() and json.loads(raw) == meta
    assert (dest / "dataset-cover-image.png").read_bytes() == b"png"
