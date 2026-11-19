"""Upload a dataset's metadata + cover image with a crop that covers the whole image.

The kaggle CLI (2.2.5) hard-codes the cover crop to the top-left 560x280 pixels and the thumbnail to a
280x280 square at x=140, so any larger image shows only a zoomed-in corner. This wrapper replaces those
rectangles with the largest centred 2:1 rectangle (the whole image for a 2:1 cover) and the largest centred square.

The CLI also reads dataset-metadata.json with the system's default encoding, which garbles non-ASCII text on
Windows (or fails the upload). So the upload goes from a temporary copy with every non-ASCII character escaped.

Usage:  python -I tools/push_cover.py <dataset-folder> <owner/slug>
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

from PIL import Image

from kaggle.api.kaggle_api_extended import CroppedImageRectangle, KaggleApi

COVER_FILES = [f"dataset-cover-image{ext}" for ext in (".png", ".jpg", ".jpeg", ".webp")]
_orig = KaggleApi._upload_dataset_image_file


def crop_rectangles(w: int, h: int) -> list[CroppedImageRectangle]:
    """The largest centred 2:1 rectangle (cover) and the largest centred square (thumbnail)."""
    ch = min(h, w // 2)
    cover = CroppedImageRectangle(); cover.title = "cover image"
    cover.left, cover.top, cover.width, cover.height = (w - 2 * ch) // 2, (h - ch) // 2, 2 * ch, ch
    side = min(w, h)
    thumb = CroppedImageRectangle(); thumb.title = "thumbnail"
    thumb.left, thumb.top, thumb.width, thumb.height = (w - side) // 2, (h - side) // 2, side, side
    return [cover, thumb]


def _full_crop(self, metadata_file_path, relative_image_file_path, quiet=False):
    upload = _orig(self, metadata_file_path, relative_image_file_path, quiet)
    with Image.open(Path(metadata_file_path) / relative_image_file_path) as img:
        upload.crop_rectangles = crop_rectangles(*img.size)
    for r in upload.crop_rectangles:
        print(f"crop: {r.title} {r.width}x{r.height}@({r.left},{r.top})")
    return upload


def ascii_copy(folder: Path, dest: Path) -> Path:
    """Copy the metadata (non-ASCII characters escaped) and the cover image into dest."""
    meta = json.loads((folder / "dataset-metadata.json").read_text(encoding="utf-8-sig"))
    (dest / "dataset-metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="ascii")
    images = [meta["image"]] if meta.get("image") else [n for n in COVER_FILES if (folder / n).is_file()]
    for rel in images:
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(folder / rel, dest / rel)
    return dest


KaggleApi._upload_dataset_image_file = _full_crop

if __name__ == "__main__":
    folder, ref = Path(sys.argv[1]), sys.argv[2]
    api = KaggleApi(); api.authenticate()
    with tempfile.TemporaryDirectory() as tmp:
        api.dataset_metadata_update(ref, str(ascii_copy(folder, Path(tmp))))
    print("metadata + cover updated:", ref)
