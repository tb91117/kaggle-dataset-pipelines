"""Tests for kaggle_dataset_lint: one complete dataset that must be clean, then each rule on a small change to it."""
import json
import struct
import zlib

import pytest

from kaggle_dataset_lint import image_size, lint, main

DESCRIPTION = "## What this is\n" + "Hourly readings from a weather station, cleaned and documented. " * 5


def png(path, w, h):
    """A valid 1-bit grey PNG of the given size (header-accurate, tiny on disk)."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    rows = b"".join(b"\x00" + b"\x00" * ((w + 7) // 8) for _ in range(h))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 1, 0, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def make_dataset(tmp_path, **overrides):
    folder = tmp_path / "ds"
    folder.mkdir()
    (folder / "readings.csv").write_text("station,time,temp_c\nA,2026-01-01 00:00,3.5\n", encoding="utf-8")
    png(folder / "dataset-cover-image.png", 560, 280)
    meta = {
        "id": "someone/weather-station-readings",
        "title": "Weather Station Readings",
        "subtitle": "Hourly temperature from one station, cleaned",
        "description": DESCRIPTION,
        "licenses": [{"name": "CC0-1.0"}],
        "keywords": ["weather and climate"],
        "expectedUpdateFrequency": "monthly",
        "userSpecifiedSources": "National weather service open data",
        "resources": [{
            "path": "readings.csv",
            "description": "One row per station and hour.",
            "schema": {"fields": [
                {"name": "station", "description": "Station code.", "type": "string"},
                {"name": "time", "description": "Hour (UTC).", "type": "datetime"},
                {"name": "temp_c", "description": "Temperature in Celsius.", "type": "number"},
            ]},
        }],
    }
    for key, value in overrides.items():
        if value is None:
            meta.pop(key, None)
        else:
            meta[key] = value
    (folder / "dataset-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return folder


def codes(findings, level=None):
    return sorted(f.code for f in findings if level is None or f.level == level)


def test_complete_dataset_is_clean(tmp_path):
    assert lint(make_dataset(tmp_path)) == []


@pytest.mark.parametrize("overrides, code", [
    ({"title": "Short"}, "title-length"),
    ({"title": "x" * 51}, "title-length"),
    ({"title": None}, "title-missing"),
    ({"title": "INSERT_TITLE_HERE"}, "title-placeholder"),
    ({"subtitle": "Too short"}, "subtitle-length"),
    ({"subtitle": "x" * 81}, "subtitle-length"),
    ({"licenses": []}, "license-count"),
    ({"licenses": [{"name": "CC0-1.0"}, {"name": "PDDL"}]}, "license-count"),
    ({"id": "no-owner-slug"}, "id-format"),
    ({"id": "someone/INSERT_SLUG_HERE"}, "id-placeholder"),
    ({"id": None}, "id-missing"),
    ({"expectedUpdateFrequency": "sometimes"}, "frequency-invalid"),
])
def test_rules_the_kaggle_cli_enforces_are_errors(tmp_path, overrides, code):
    assert code in codes(lint(make_dataset(tmp_path, **overrides)), "error")


@pytest.mark.parametrize("overrides, code", [
    ({"subtitle": None}, "subtitle-missing"),
    ({"description": None}, "description-missing"),
    ({"description": "Tiny."}, "description-short"),
    ({"keywords": []}, "keywords-missing"),
    ({"userSpecifiedSources": None}, "sources-missing"),
    ({"expectedUpdateFrequency": None}, "frequency-missing"),
    ({"id": "someone/abc"}, "slug-length"),
])
def test_incomplete_pages_are_warnings(tmp_path, overrides, code):
    assert codes(lint(make_dataset(tmp_path, **overrides))) == [code]


def test_placeholder_tags_are_flagged_but_formatting_tags_are_not(tmp_path):
    desc = DESCRIPTION + "\nColumns `ng_<source>_mw` per source.<br>Use <b>bold</b> freely.\n"
    findings = lint(make_dataset(tmp_path, description=desc))
    assert codes(findings) == ["html-stripped"]
    assert "<source>" in findings[0].message and "<b>" not in findings[0].message


def test_schema_out_of_file_order_is_an_error(tmp_path):
    folder = make_dataset(tmp_path)
    meta = json.loads((folder / "dataset-metadata.json").read_text(encoding="utf-8"))
    fields = meta["resources"][0]["schema"]["fields"]
    fields[1], fields[2] = fields[2], fields[1]
    (folder / "dataset-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    findings = lint(folder)
    assert codes(findings) == ["schema-mismatch"]
    assert "position 2" in findings[0].message


def test_schema_missing_a_column_is_an_error(tmp_path):
    folder = make_dataset(tmp_path)
    (folder / "readings.csv").write_text("station,time,temp_c,humidity\nA,2026-01-01,3.5,80\n", encoding="utf-8")
    findings = lint(folder)
    assert codes(findings) == ["schema-mismatch"] and "humidity" in findings[0].message


def test_missing_and_undescribed_columns_and_files(tmp_path):
    folder = make_dataset(tmp_path)
    meta = json.loads((folder / "dataset-metadata.json").read_text(encoding="utf-8"))
    meta["resources"][0]["schema"]["fields"][0]["description"] = ""
    meta["resources"].append({"path": "missing.csv", "description": "Not built."})
    meta["resources"].append({"path": "readings.csv", "description": "Twice."})
    (folder / "dataset-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    (folder / "notes.txt").write_text("extra", encoding="utf-8")
    (folder / "raw").mkdir()
    assert codes(lint(folder)) == ["field-no-description", "file-undeclared", "resource-duplicate",
                                   "resource-missing-file", "subfolder-skipped"]


@pytest.mark.parametrize("size, code, level", [((1600, 800), "cover-cropped", "warning"),
                                               ((400, 200), "cover-too-small", "error")])
def test_cover_size(tmp_path, size, code, level):
    folder = make_dataset(tmp_path)
    png(folder / "dataset-cover-image.png", *size)
    findings = lint(folder)
    assert [(f.code, f.level) for f in findings] == [(code, level)]


def test_missing_cover_and_no_files_mode(tmp_path):
    folder = make_dataset(tmp_path)
    (folder / "dataset-cover-image.png").unlink()
    (folder / "readings.csv").unlink()
    assert codes(lint(folder)) == ["cover-missing", "resource-missing-file"]
    assert lint(folder, check_files=False) == []


def test_downloaded_metadata_layout_is_understood(tmp_path):
    folder = make_dataset(tmp_path)
    meta = json.loads((folder / "dataset-metadata.json").read_text(encoding="utf-8"))
    (folder / "dataset-metadata.json").write_text(json.dumps({"info": meta}), encoding="utf-8")
    assert lint(folder) == []


def test_invalid_json_and_missing_metadata(tmp_path):
    assert codes(lint(tmp_path)) == ["metadata-missing"]
    (tmp_path / "dataset-metadata.json").write_text("{not json", encoding="utf-8")
    assert codes(lint(tmp_path)) == ["metadata-invalid"]


def test_image_size_reads_png_jpeg_and_webp_headers(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image, features

    img = Image.new("RGB", (731, 365), "white")
    cases = {"a.png": {}, "b.jpg": {}, "c.jpg": {"progressive": True}}
    if features.check("webp"):
        cases.update({"d.webp": {}, "e.webp": {"lossless": True}})
    for name, kw in cases.items():
        img.save(tmp_path / name, **kw)
        assert image_size(tmp_path / name) == (731, 365), name
    if features.check("webp"):
        Image.new("RGBA", (731, 365), (0, 0, 0, 0)).save(tmp_path / "f.webp")  # alpha -> extended (VP8X) header
        assert image_size(tmp_path / "f.webp") == (731, 365)
    (tmp_path / "broken.jpg").write_bytes(b"\xff\xd8\xff\xe0\x00")
    assert image_size(tmp_path / "broken.jpg") is None


def test_parquet_columns_are_checked(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    folder = make_dataset(tmp_path)
    (folder / "readings.csv").unlink()
    pq.write_table(pa.table({"station": ["A"], "temp_c": [3.5], "time": ["x"]}), folder / "readings.parquet")
    meta = json.loads((folder / "dataset-metadata.json").read_text(encoding="utf-8"))
    meta["resources"][0]["path"] = "readings.parquet"
    (folder / "dataset-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    assert codes(lint(folder)) == ["schema-mismatch"]


def test_cli_exit_codes_ignore_strict_and_github_format(tmp_path, capsys):
    clean = make_dataset(tmp_path)
    assert main([str(clean)]) == 0
    meta = json.loads((clean / "dataset-metadata.json").read_text(encoding="utf-8"))
    meta.pop("keywords")
    (clean / "dataset-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    assert main([str(clean)]) == 0                       # warnings alone pass
    assert main([str(clean), "--strict"]) == 1           # unless --strict
    assert main([str(clean), "--strict", "--ignore", "keywords-missing"]) == 0
    capsys.readouterr()
    meta["title"] = "Tiny"
    (clean / "dataset-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    assert main([str(clean / "dataset-metadata.json"), "--format", "github"]) == 1
    out = capsys.readouterr().out
    assert out.startswith("::error file=") and "title=title-length::" in out
    assert ":" not in out.split("file=", 1)[1].split(",title=", 1)[0]  # path is escaped for the annotation


def test_github_annotations_use_repo_relative_forward_slash_paths(tmp_path, capsys, monkeypatch):
    make_dataset(tmp_path, title="Tiny")
    monkeypatch.chdir(tmp_path)
    main(["./ds", "--format", "github"])
    out = capsys.readouterr().out
    assert "::error file=ds/dataset-metadata.json,title=title-length::" in out  # GitHub matches this to the file


@pytest.mark.parametrize("encoding, escaped, expected", [
    ("cp1252", False, ["upload-encoding"]),  # Windows: the CLI would garble the text
    ("cp1252", True, []),                    # JSON escapes are plain ASCII, safe everywhere
    ("UTF-8", False, []),                    # Linux and macOS, or PYTHONUTF8=1
])
def test_non_ascii_metadata_on_a_non_utf8_system(tmp_path, monkeypatch, encoding, escaped, expected):
    import locale

    folder = make_dataset(tmp_path)
    meta = json.loads((folder / "dataset-metadata.json").read_text(encoding="utf-8"))
    meta["description"] += " 144,606 trees " + chr(0xD7) + " 28 columns."
    (folder / "dataset-metadata.json").write_text(json.dumps(meta, ensure_ascii=escaped), encoding="utf-8")
    monkeypatch.setattr(locale, "getpreferredencoding", lambda do_setlocale=True: encoding)
    findings = lint(folder)
    assert codes(findings) == expected
    if expected:
        assert "U+00D7" in findings[0].message and "cp1252" in findings[0].message
