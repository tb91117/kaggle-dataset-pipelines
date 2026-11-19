"""Lint a Kaggle dataset folder (dataset-metadata.json + files) before you upload it.

Three kinds of checks:
- rules the Kaggle CLI and API enforce: title, subtitle and slug lengths, exactly one license, declared files exist;
- things that go wrong silently: schema fields out of order (Kaggle matches column descriptions by position),
  <word> placeholders stripped from descriptions, covers larger than 560x280 cropped to their top-left corner,
  subfolders skipped by `kaggle datasets create/version`;
- fields that make a dataset page complete: subtitle, description, tags, cover image, sources, update frequency,
  file and column descriptions.

Usage:
    kaggle-dataset-lint DATASET_FOLDER_OR_METADATA_FILE [...] [--strict] [--no-files] [--ignore CODE,...]
                        [--format text|github]
Exit code 1 if there is an error (or, with --strict, a warning). Standard library only; Parquet columns are checked
when pyarrow is installed.
"""
from __future__ import annotations

import argparse
import codecs
import csv
import json
import locale
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

__version__ = "0.2.1"

METADATA_FILES = ("dataset-metadata.json", "datapackage.json")  # the CLI accepts both, in this order
COVER_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
COVER_FILES = tuple("dataset-cover-image" + ext for ext in COVER_EXTENSIONS)
NOT_DATA = set(METADATA_FILES) | set(COVER_FILES) | {"kernel-metadata.json", "model-metadata.json",
                                                     "model-instance-metadata.json"}
HEADER_CROP = (560, 280)  # the CLI's fixed crop rectangles (docs/datasets_metadata.md, "Images")

# from Kaggle's docs/datasets_metadata.md
LICENSES = {
    "CC0-1.0", "CC-BY-SA-3.0", "CC-BY-SA-4.0", "CC-BY-NC-SA-4.0", "GPL-2.0", "ODbL-1.0", "DbCL-1.0",
    "copyright-authors", "other", "unknown", "CC-BY-4.0", "CC-BY-NC-4.0", "PDDL", "CC-BY-3.0", "CC-BY-3.0-IGO",
    "US-Government-Works", "CC-BY-NC-SA-3.0-IGO", "CDLA-Permissive-1.0", "CDLA-Sharing-1.0", "CC-BY-ND-4.0",
    "CC-BY-NC-ND-4.0", "ODC-BY-1.0", "LGPL-3.0", "AGPL-3.0", "FDL-1.3", "EU-ODP-Legal-Notice", "apache-2.0",
    "GPL-3.0",
}
FIELD_TYPES = {  # the docs call this list best-effort; "number" is used in the docs' own example
    "number", "string", "boolean", "numeric", "datetime", "id", "uuid", "latitude", "longitude", "coordinates", "country",
    "province", "postalcode", "address", "email", "url", "integer", "decimal", "city",
}
FREQUENCIES = {"not specified", "never", "annually", "quarterly", "monthly", "weekly", "daily", "hourly"}

# Kaggle's markdown renderer drops HTML tags it doesn't allow, so `ng_<source>_mw` renders as `ng__mw`.
# These formatting tags are commonly allowed in markdown and are not flagged.
FORMATTING_TAGS = {
    "a", "b", "i", "u", "s", "em", "strong", "br", "p", "img", "code", "pre", "sub", "sup", "small", "hr",
    "table", "thead", "tbody", "tr", "td", "th", "ul", "ol", "li", "h1", "h2", "h3", "h4", "h5", "h6",
    "blockquote", "details", "summary", "div", "span", "kbd", "del", "ins", "mark",
}
TAG = re.compile(r"<([A-Za-z][\w-]*)>")


@dataclass(frozen=True)
class Finding:
    level: str  # "error" | "warning" | "note"
    code: str
    message: str
    path: str


class _Report:
    def __init__(self, path: Path):
        self.path, self.findings = str(path), []

    def error(self, code, msg):
        self.findings.append(Finding("error", code, msg, self.path))

    def warning(self, code, msg):
        self.findings.append(Finding("warning", code, msg, self.path))

    def note(self, code, msg):
        self.findings.append(Finding("note", code, msg, self.path))


# ----------------------------------------------------------------------------------------------- image size
def image_size(path: Path) -> tuple[int, int] | None:
    """Width and height of a PNG, JPEG or WebP file, read from its header; None if unrecognised."""
    try:
        return _image_size(path)
    except (OSError, struct.error):
        return None


def _image_size(path: Path) -> tuple[int, int] | None:
    with path.open("rb") as f:
        head = f.read(32)
        if head.startswith(b"\x89PNG\r\n\x1a\n") and head[12:16] == b"IHDR":
            return struct.unpack(">II", head[16:24])
        if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
            chunk = head[12:16]
            if chunk == b"VP8 ":
                w, h = struct.unpack("<HH", head[26:30])
                return w & 0x3FFF, h & 0x3FFF
            if chunk == b"VP8L":
                b = head[21:25]
                return 1 + (((b[1] & 0x3F) << 8) | b[0]), 1 + (((b[3] & 0xF) << 10) | (b[2] << 2) | ((b[1] & 0xC0) >> 6))
            if chunk == b"VP8X":
                return 1 + int.from_bytes(head[24:27], "little"), 1 + int.from_bytes(head[27:30], "little")
            return None
        if head[:2] == b"\xff\xd8":
            f.seek(2)
            while True:
                byte = f.read(1)
                while byte and byte != b"\xff":
                    byte = f.read(1)
                while byte == b"\xff":
                    byte = f.read(1)
                if not byte:
                    return None
                marker = byte[0]
                if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                    continue
                size = struct.unpack(">H", f.read(2))[0]
                if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                    h, w = struct.unpack(">xHH", f.read(5))
                    return w, h
                f.seek(size - 2, 1)
    return None


# ----------------------------------------------------------------------------------------------- columns
def file_columns(path: Path) -> list[str] | None:
    """Column names of a CSV/TSV (header row) or Parquet file; None if the format isn't supported or unreadable."""
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        with path.open(newline="", encoding="utf-8-sig", errors="replace") as f:
            return next(csv.reader(f, delimiter="\t" if suffix == ".tsv" else ","), [])
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError:
            return None
        return list(pq.read_schema(path).names)
    return None


# ----------------------------------------------------------------------------------------------- checks
def _text_tags(r: _Report, field: str, text: str):
    tags = sorted({m for m in TAG.findall(text) if m.lower() not in FORMATTING_TAGS})
    if tags:
        shown = ", ".join(f"<{t}>" for t in tags[:5])
        r.warning("html-stripped", f"{field} contains {shown}: Kaggle's renderer drops unknown HTML tags, so the text "
                                   f"disappears (`ng_<source>_mw` shows as `ng__mw`). Write {{{tags[0]}}} or &lt;{tags[0]}&gt;")


def _check_metadata(r: _Report, meta: dict):
    ref = meta.get("id")
    if not ref and not meta.get("id_no"):
        r.error("id-missing", 'no "id" ("owner/dataset-slug") or "id_no"')
    elif ref:
        parts = str(ref).split("/")
        if len(parts) != 2 or not all(parts):
            r.error("id-format", f'"id" must be "owner/dataset-slug", got "{ref}"')
        elif parts[1] == "INSERT_SLUG_HERE":
            r.error("id-placeholder", '"id" still has the INSERT_SLUG_HERE placeholder from `kaggle datasets init`')
        elif not 6 <= len(parts[1]) <= 50:
            r.warning("slug-length", f'dataset slug "{parts[1]}" is {len(parts[1])} characters; '
                                     "`kaggle datasets create` rejects slugs outside 6-50")

    title = meta.get("title")
    if not title:
        r.error("title-missing", 'no "title" (required to create a dataset)')
    elif title == "INSERT_TITLE_HERE":
        r.error("title-placeholder", '"title" still has the INSERT_TITLE_HERE placeholder')
    elif not 6 <= len(title) <= 50:
        r.error("title-length", f"title is {len(title)} characters; Kaggle allows 6-50")

    subtitle = meta.get("subtitle")
    if not subtitle:
        r.warning("subtitle-missing", "no subtitle; the dataset page and search results show it under the title")
    elif not 20 <= len(subtitle) <= 80:
        r.error("subtitle-length", f"subtitle is {len(subtitle)} characters; Kaggle allows 20-80")

    licenses = meta.get("licenses")
    if not isinstance(licenses, list) or len(licenses) != 1:
        n = len(licenses) if isinstance(licenses, list) else 0
        r.error("license-count", f'"licenses" must have exactly one entry, found {n}')
    else:
        name = (licenses[0] or {}).get("name") if isinstance(licenses[0], dict) else None
        if not name:
            r.error("license-count", 'the license entry has no "name"')
        elif name not in LICENSES:
            r.note("license-unknown", f'license "{name}" is not one of the ids in Kaggle\'s docs (e.g. CC0-1.0, '
                                      "CC-BY-4.0, PDDL, other); Kaggle also accepts its display names, so check the spelling")

    desc = meta.get("description") or ""
    if not desc.strip():
        r.warning("description-missing", "no description")
    else:
        if len(desc) < 200:
            r.warning("description-short", f"description is only {len(desc)} characters; say what the data is, "
                                           "where it comes from and how it was cleaned")
        _text_tags(r, "description", desc)

    if not meta.get("keywords"):
        r.warning("keywords-missing", "no keywords (tags); they must match existing Kaggle tags, unknown ones are "
                                      "silently dropped")

    sources = meta.get("userSpecifiedSources")
    if not sources:
        r.warning("sources-missing", 'no "userSpecifiedSources": say where the data comes from')
    else:
        _text_tags(r, "userSpecifiedSources", sources)

    freq = meta.get("expectedUpdateFrequency")
    if freq is None:
        r.warning("frequency-missing", 'no "expectedUpdateFrequency" (e.g. never, monthly)')
    elif freq not in FREQUENCIES:
        r.error("frequency-invalid", f'"expectedUpdateFrequency" is "{freq}"; allowed: '
                                     + ", ".join(sorted(FREQUENCIES)))


def _check_cover(r: _Report, folder: Path, meta: dict):
    if meta.get("image"):
        cover = folder / meta["image"]
        if cover.suffix.lower() not in COVER_EXTENSIONS:
            r.error("cover-format", f'"image" must be .png, .jpg, .jpeg or .webp: {meta["image"]}')
            return
        if not cover.is_file():
            r.error("cover-not-found", f'"image" file not found: {meta["image"]}')
            return
    else:
        cover = next((folder / n for n in COVER_FILES if (folder / n).is_file()), None)
        if cover is None:
            r.warning("cover-missing", "no dataset-cover-image.png/.jpg/.jpeg/.webp next to the metadata "
                                       "(fine if you set the cover in the web UI)")
            return
    size = image_size(cover)
    if size is None:
        r.note("cover-unreadable", f"could not read the size of {cover.name}")
        return
    w, h = size
    if w < HEADER_CROP[0] or h < HEADER_CROP[1]:
        r.error("cover-too-small", f"{cover.name} is {w}x{h}; Kaggle needs at least 560x280")
    elif (w, h) != HEADER_CROP:
        r.warning("cover-cropped", f"{cover.name} is {w}x{h}, but the Kaggle CLI always crops the top-left 560x280 "
                                   "pixels for the header and a 280x280 square at x=140 for the thumbnail. Resize it "
                                   "to 560x280, or set the cover in the web UI")


def _check_resources(r: _Report, folder: Path, meta: dict, check_files: bool):
    resources = meta.get("resources") or []
    declared = set()
    for res in resources:
        path = res.get("path")
        if not path:
            r.error("resource-path", 'a resource has no "path"')
            continue
        if path in declared:
            r.error("resource-duplicate", f"{path} is declared more than once")
        declared.add(path)
        if not res.get("description"):
            r.warning("resource-no-description", f"{path} has no description")
        fields = (res.get("schema") or {}).get("fields") or []
        names = [f.get("name") for f in fields]
        undocumented = [n for n, f in zip(names, fields) if not (f.get("description") or f.get("title"))]
        if undocumented:
            shown = ", ".join(map(str, undocumented[:5])) + (" ..." if len(undocumented) > 5 else "")
            r.warning("field-no-description", f"{path}: {len(undocumented)} of {len(fields)} columns have no "
                                              f"description ({shown})")
        bad_types = sorted({f["type"] for f in fields if f.get("type") and f["type"] not in FIELD_TYPES})
        if bad_types:
            r.note("field-type-unknown", f"{path}: column type(s) not in Kaggle's best-effort list: {', '.join(bad_types)}")
        if not check_files:
            continue
        full = folder / path
        if not full.is_file():
            r.error("resource-missing-file", f"{path} is declared but not in the folder")
            continue
        if fields:
            cols = file_columns(full)
            if cols is None and full.suffix.lower() == ".parquet":
                r.note("parquet-skipped", f"{path}: install pyarrow to check its columns")
            elif cols is not None and cols != names:
                first = next((i for i, (a, b) in enumerate(zip(names, cols)) if a != b), min(len(names), len(cols)))
                missing = [c for c in cols if c not in names]
                extra = [n for n in names if n not in cols]
                detail = (f"first difference at position {first + 1}: schema has "
                          f"{names[first] if first < len(names) else 'nothing'!r}, file has "
                          f"{cols[first] if first < len(cols) else 'nothing'!r}")
                if missing:
                    detail += f"; not in schema: {', '.join(missing[:5])}"
                if extra:
                    detail += f"; not in file: {', '.join(map(str, extra[:5]))}"
                r.error("schema-mismatch", f"{path}: schema fields must list every column in file order, or Kaggle "
                                           f"attaches the descriptions to the wrong columns ({detail})")
    if not check_files:
        return
    undeclared, subfolders = [], []
    for item in sorted(folder.iterdir()):
        if item.name.startswith(".") or item.name in NOT_DATA:
            continue
        if item.is_dir():
            subfolders.append(item.name)
        elif item.name not in declared:
            undeclared.append(item.name)
    if undeclared:
        shown = ", ".join(undeclared[:5]) + (f" and {len(undeclared) - 5} more" if len(undeclared) > 5 else "")
        r.warning("file-undeclared", f"no resource entry (so no file or column descriptions) for: {shown}")
    for sub in subfolders:
        r.warning("subfolder-skipped", f"{sub}/ is skipped by `kaggle datasets create/version` unless you pass "
                                       "--dir-mode zip or tar")


def _check_encoding(r: _Report, raw: bytes):
    """The Kaggle CLI opens dataset-metadata.json with the system default encoding, not UTF-8."""
    if raw.isascii():
        return
    encoding = locale.getpreferredencoding(False)
    try:
        if codecs.lookup(encoding).name == "utf-8":
            return
    except LookupError:
        pass
    chars = sorted({c for c in raw.decode("utf-8", "replace") if ord(c) > 127})
    shown = ", ".join(f"U+{ord(c):04X}" for c in chars[:5]) + (" ..." if len(chars) > 5 else "")
    r.warning("upload-encoding", f"the file has non-ASCII characters ({shown}) but the Kaggle CLI reads it with this "
                                 f"system's default encoding ({encoding}), so Kaggle shows them garbled or the upload "
                                 "fails. Upload with PYTHONUTF8=1 set, or save the JSON with \\u escapes "
                                 "(json.dump(..., ensure_ascii=True))")


def lint(target: str | Path, check_files: bool = True) -> list[Finding]:
    """Lint a dataset folder, or the metadata file inside one. Returns every finding."""
    target = Path(target)
    if target.is_dir():
        folder = target
        meta_path = next((target / n for n in METADATA_FILES if (target / n).is_file()), target / METADATA_FILES[0])
    else:
        folder, meta_path = target.parent, target
    r = _Report(meta_path)
    if not meta_path.is_file():
        r.error("metadata-missing", "no dataset-metadata.json (create one with `kaggle datasets init -p FOLDER`)")
        return r.findings
    raw = meta_path.read_bytes()
    try:
        meta = json.loads(raw.decode("utf-8-sig"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        r.error("metadata-invalid", f"not valid JSON: {e}")
        return r.findings
    _check_encoding(r, raw)
    if not isinstance(meta, dict):
        r.error("metadata-invalid", "the metadata must be a JSON object")
        return r.findings
    meta = meta.get("info", meta) if "info" in meta and "title" not in meta else meta  # `kaggle datasets metadata` layout
    _check_metadata(r, meta)
    if check_files:
        _check_cover(r, folder, meta)
    _check_resources(r, folder, meta, check_files)
    return r.findings


# ----------------------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="kaggle-dataset-lint", description="Lint Kaggle dataset folders before upload.")
    p.add_argument("targets", nargs="+", help="dataset folders or dataset-metadata.json files")
    p.add_argument("--strict", action="store_true", help="exit 1 on warnings too")
    p.add_argument("--no-files", action="store_true", help="check the metadata only, not the data files or cover")
    p.add_argument("--ignore", default="", help="comma-separated finding codes to skip, e.g. cover-missing")
    p.add_argument("--format", choices=("text", "github"), default="text",
                   help="github prints GitHub Actions annotations")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = p.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # findings quote user text; never crash on a console that can't show it
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass

    ignore ={c.strip() for c in args.ignore.split(",") if c.strip()}
    findings = [f for t in args.targets for f in lint(t, check_files=not args.no_files) if f.code not in ignore]
    def esc(s, prop=False):  # GitHub workflow-command escaping
        s = s.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        return s.replace(":", "%3A").replace(",", "%2C") if prop else s

    for f in findings:
        if args.format == "github":
            kind = {"error": "error", "warning": "warning"}.get(f.level, "notice")
            print(f"::{kind} file={esc(f.path, True)},title={esc(f.code, True)}::{esc(f.message)}")
        else:
            print(f"{f.path}: {f.level}: [{f.code}] {f.message}")
    errors = sum(f.level == "error" for f in findings)
    warnings = sum(f.level == "warning" for f in findings)
    if args.format == "text":
        print(f"{errors} error(s), {warnings} warning(s) in {len(args.targets)} dataset(s)")
    return 1 if errors or (args.strict and warnings) else 0


if __name__ == "__main__":
    sys.exit(main())
