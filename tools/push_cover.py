"""Upload a dataset's metadata + cover image with a crop that covers the whole image.

The kaggle CLI (2.2.5) hard-codes the cover crop to the top-left 560x280 pixels and the thumbnail to a
280x280 square at x=140, so any larger image shows only a zoomed-in corner. This wrapper replaces those
rectangles with: cover = the full image (expects a 2:1 image), thumbnail = the centred square.

Usage:  python -I tools/push_cover.py <dataset-folder> <owner/slug>
"""
import sys

from PIL import Image

from kaggle.api.kaggle_api_extended import CroppedImageRectangle, KaggleApi

_orig = KaggleApi._upload_dataset_image_file


def _full_crop(self, metadata_file_path, relative_image_file_path, quiet=False):
    upload = _orig(self, metadata_file_path, relative_image_file_path, quiet)
    import os
    w, h = Image.open(os.path.join(metadata_file_path, relative_image_file_path)).size
    ch = min(h, w // 2)
    cover = CroppedImageRectangle(); cover.title = "cover image"
    cover.left, cover.top, cover.width, cover.height = 0, (h - ch) // 2, ch * 2, ch
    side = min(w, h)
    thumb = CroppedImageRectangle(); thumb.title = "thumbnail"
    thumb.left, thumb.top, thumb.width, thumb.height = (w - side) // 2, (h - side) // 2, side, side
    upload.crop_rectangles = [cover, thumb]
    print(f"crop: cover {cover.width}x{cover.height}@({cover.left},{cover.top}), thumbnail {side}x{side}@({thumb.left},{thumb.top})")
    return upload


KaggleApi._upload_dataset_image_file = _full_crop

if __name__ == "__main__":
    folder, ref = sys.argv[1], sys.argv[2]
    api = KaggleApi(); api.authenticate()
    api.dataset_metadata_update(ref, folder)
    print("metadata + cover updated:", ref)
