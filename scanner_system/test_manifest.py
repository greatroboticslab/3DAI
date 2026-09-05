"""Unit tests for the Excel/CSV manifest workflow.

Runs with no hardware, no Mongo and no Excel install beyond openpyxl: every
test works on a temp file. Covers cell parsing, row validation, the template
round-trip in both formats, and the results write-back that makes an
interrupted batch resumable.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scanner_system import manifest


# ── Cell parsing ────────────────────────────────────────────────────────────

def test_parse_channels_variants():
    assert manifest.parse_channels("1,2,3") == [1, 2, 3]
    assert manifest.parse_channels("1 2 3") == [1, 2, 3]
    assert manifest.parse_channels("2;4") == [2, 4]
    assert manifest.parse_channels("3") == [3]
    # openpyxl hands back floats for numeric cells: "1.0" must mean channel 1.
    assert manifest.parse_channels("1.0") == [1]
    # duplicates collapse, order preserved
    assert manifest.parse_channels("2,2,1") == [2, 1]


def test_parse_channels_blank_uses_defaults():
    for blank in ("", "   ", None, "nan"):
        assert manifest.parse_channels(blank) == manifest.DEFAULT_LASER_CHANNELS


def test_parse_channels_rejects_bad_input():
    for bad in ("0", "5", "9"):
        try:
            manifest.parse_channels(bad)
        except manifest.ManifestError as exc:
            assert "not one of" in str(exc)
        else:
            raise AssertionError(f"{bad!r} should have been rejected")

    try:
        manifest.parse_channels("red")
    except manifest.ManifestError as exc:
        assert "not a number" in str(exc)
    else:
        raise AssertionError("non-numeric channel should have been rejected")


def test_clean_handles_excel_artifacts():
    assert manifest._clean(None) == ""
    assert manifest._clean("  oak  ") == "oak"
    assert manifest._clean("nan") == ""      # pandas-style empty cell
    assert manifest._clean(3.0) == "3.0"


# ── Row validation ──────────────────────────────────────────────────────────

def test_validate_row_requires_label():
    try:
        manifest.validate_row({"label": "   "}, 7)
    except manifest.ManifestError as exc:
        assert "row 7" in str(exc) and "label is required" in str(exc)
    else:
        raise AssertionError("blank label should have been rejected")


def test_validate_row_rejects_bad_mode():
    try:
        manifest.validate_row({"label": "x", "mode": "turbo"}, 3)
    except manifest.ManifestError as exc:
        assert "row 3" in str(exc) and "turbo" in str(exc)
    else:
        raise AssertionError("bad mode should have been rejected")


def test_validate_row_defaults_and_normalizes():
    row = manifest.validate_row({"label": "  oak plank #3 ", "material_class": " wood "}, 2)
    assert row["label"] == "oak plank #3"
    assert row["material_class"] == "wood"
    assert row["material_subclass"] is None      # blank becomes None, not ""
    assert row["mode"] == "full"                 # default
    assert row["laser_channels"] == manifest.DEFAULT_LASER_CHANNELS
    assert row["sample_id"] is None              # blank means "mint one"
    assert row["row_number"] == 2


def test_validate_row_keeps_explicit_sample_id():
    row = manifest.validate_row({"label": "x", "sample_id": "abc-123"}, 2)
    assert row["sample_id"] == "abc-123"


# ── Template / read round-trip ──────────────────────────────────────────────

def _tmp(suffix):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    os.unlink(path)
    return path


def test_xlsx_template_round_trip():
    path = _tmp(".xlsx")
    try:
        manifest.write_template(path)
        assert os.path.isfile(path)
        rows = manifest.read_manifest(path)
        # The template ships exactly one clearly-marked example row.
        assert len(rows) == 1
        assert "EXAMPLE" in rows[0]["label"]
        assert rows[0]["row_number"] == 2       # header is row 1
        parsed = manifest.validate_row(rows[0], rows[0]["row_number"])
        assert parsed["material_class"] == "wood"
        assert parsed["laser_channels"] == [1, 2, 3, 4]   # blank = all four
        assert parsed["surface"] == "matte" and parsed["transparency"] == "opaque"
    finally:
        os.path.isfile(path) and os.unlink(path)


def test_csv_template_round_trip():
    path = _tmp(".csv")
    try:
        manifest.write_template(path)
        rows = manifest.read_manifest(path)
        assert rows == []                        # CSV template is headers only
    finally:
        os.path.isfile(path) and os.unlink(path)


def test_read_rejects_unknown_extension():
    try:
        manifest.read_manifest("samples.txt")
    except manifest.ManifestError as exc:
        assert "unsupported" in str(exc)
    else:
        raise AssertionError("unknown extension should have been rejected")


def test_read_rejects_manifest_without_label_column():
    path = _tmp(".csv")
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("sample_id,notes\n1,hello\n")
        try:
            manifest.read_manifest(path)
        except manifest.ManifestError as exc:
            assert "no 'label' column" in str(exc)
        else:
            raise AssertionError("missing label column should have been rejected")
    finally:
        os.path.isfile(path) and os.unlink(path)


def test_blank_rows_are_skipped():
    path = _tmp(".csv")
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("label,mode\n")
            fh.write("oak,full\n")
            fh.write(",\n")                       # spacer row a tech left behind
            fh.write("pine,full\n")
        rows = manifest.read_manifest(path)
        assert [r["label"] for r in rows] == ["oak", "pine"]
        # row_number still tracks the real spreadsheet line, skipping the blank
        assert [r["row_number"] for r in rows] == [2, 4]
    finally:
        os.path.isfile(path) and os.unlink(path)


# ── Write-back ──────────────────────────────────────────────────────────────

_RESULTS = {"scan_id": "scan-1", "status": "complete", "artifact_count": 7,
            "failure_detail": ""}


def test_xlsx_write_back_targets_the_right_row():
    path = _tmp(".xlsx")
    try:
        manifest.write_template(path)
        from openpyxl import load_workbook
        wb = load_workbook(path)
        ws = wb.active
        second = {c: "" for c in manifest.ALL_COLUMNS}
        second.update({"label": "second sample", "mode": "full"})
        ws.append([second[c] for c in manifest.ALL_COLUMNS])   # row 3
        wb.save(path)

        manifest.write_results(path, 3, _RESULTS)

        rows = manifest.read_manifest(path)
        by_label = {r["label"]: r for r in rows}
        assert by_label["second sample"]["scan_id"] == "scan-1"
        assert by_label["second sample"]["status"] == "complete"
        # the example row on line 2 must be untouched
        example = [r for r in rows if "EXAMPLE" in r["label"]][0]
        assert not manifest._clean(example.get("scan_id"))
    finally:
        os.path.isfile(path) and os.unlink(path)


def test_csv_write_back_round_trip():
    path = _tmp(".csv")
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(",".join(manifest.ALL_COLUMNS) + "\n")
            fh.write("oak" + "," * (len(manifest.ALL_COLUMNS) - 1) + "\n")   # label is column 1
        manifest.write_results(path, 2, _RESULTS)
        rows = manifest.read_manifest(path)
        assert rows[0]["scan_id"] == "scan-1"
        assert rows[0]["status"] == "complete"
        assert rows[0]["label"] == "oak"
    finally:
        os.path.isfile(path) and os.unlink(path)


def test_write_back_ignores_unknown_result_keys():
    """A result key with no column must not crash or shift other columns."""
    path = _tmp(".csv")
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("label,scan_id,status\n")
            fh.write("oak,,\n")
        manifest.write_results(path, 2, {"scan_id": "s1", "no_such_column": "x"})
        rows = manifest.read_manifest(path)
        assert rows[0]["scan_id"] == "s1"
        assert rows[0]["label"] == "oak"
    finally:
        os.path.isfile(path) and os.unlink(path)


# ── Result summarization ────────────────────────────────────────────────────

def test_summarize_flattens_a_scan_package():
    pkg = {
        "_id": "scan-9",
        "status": "partial",
        "started_at": "2026-08-26T10:00:00Z",
        "completed_at": "2026-08-26T10:01:00Z",
        "results": {
            "kinect": {"status": "ok", "detail": ""},
            "projector": {"status": "ok", "detail": ""},
            "laser": {"status": "failed", "detail": "no laser frames captured"},
        },
        "artifacts": {"kinect": [{}], "projector": [{}, {}]},
    }
    out = manifest._summarize(pkg)
    assert out["scan_id"] == "scan-9"
    assert out["status"] == "partial"
    assert out["laser_status"] == "failed"
    assert out["kinect_status"] == "ok"
    assert out["artifact_count"] == 3
    assert "no laser frames captured" in out["failure_detail"]


def test_summarize_survives_a_sparse_package():
    """A failed scan can come back with almost nothing; must not KeyError."""
    out = manifest._summarize({"_id": "s", "status": "failed"})
    assert out["scan_id"] == "s"
    assert out["artifact_count"] == 0
    assert out["failure_detail"] == ""
    assert out["kinect_status"] == ""


# ── Multi-angle rows ────────────────────────────────────────────────────────

def test_validate_row_angles():
    assert manifest.validate_row({"label": "x"}, 2)["angles"] == manifest.DEFAULT_ANGLES == 2
    assert manifest.validate_row({"label": "x", "angles": "3"}, 2)["angles"] == 3
    # openpyxl numeric cells arrive as floats
    assert manifest.validate_row({"label": "x", "angles": "3.0"}, 2)["angles"] == 3
    for bad in ("0", "13", "many"):
        try:
            manifest.validate_row({"label": "x", "angles": bad}, 4)
        except manifest.ManifestError as exc:
            assert "row 4" in str(exc)
        else:
            raise AssertionError(f"angles={bad!r} should have been rejected")


def _pose(status, detail="", count=1):
    return {"scan_id": "id-" + status, "status": status, "kinect_status": "ok",
            "projector_status": "ok", "laser_status": "ok",
            "failure_detail": detail, "artifact_count": count,
            "started_at": "t0", "completed_at": "t1"}


def test_aggregate_single_pose_passthrough():
    s = _pose("complete")
    assert manifest._aggregate([s]) == s


def test_aggregate_multi_pose():
    agg = manifest._aggregate([_pose("complete", count=10), _pose("complete", count=10)])
    assert agg["status"] == "complete"
    assert agg["artifact_count"] == 20
    assert agg["scan_id"] == "id-complete; id-complete"

    # one bad pose degrades the whole row so a re-run retries the object
    agg = manifest._aggregate([_pose("complete"), _pose("failed", detail="kinect died")])
    assert agg["status"] == "partial"
    assert "pose 2: kinect died" in agg["failure_detail"]

    agg = manifest._aggregate([_pose("failed", detail="a"), _pose("failed", detail="b")])
    assert agg["status"] == "failed"


def test_validate_row_known_height():
    assert manifest.validate_row({"label": "x"}, 2)["known_height_mm"] is None
    assert manifest.validate_row({"label": "x", "known_height_mm": "12.5"}, 2)["known_height_mm"] == 12.5
    for bad, why in (("0.1", "range"), ("500", "range"), ("thicc", "number")):
        try:
            manifest.validate_row({"label": "x", "known_height_mm": bad}, 5)
        except manifest.ManifestError as exc:
            assert "row 5" in str(exc)
        else:
            raise AssertionError(f"known_height_mm={bad!r} should have been rejected ({why})")


# ── Surface-property ground truth ────────────────────────────────────────────

def test_validate_row_surface_and_transparency():
    row = manifest.validate_row({"label": "x"}, 2)
    assert row["surface"] is None and row["transparency"] is None
    row = manifest.validate_row({"label": "x", "surface": " Glossy ", "transparency": "OPAQUE"}, 2)
    assert row["surface"] == "glossy" and row["transparency"] == "opaque"   # normalized
    for col, bad in (("surface", "shiny"), ("transparency", "clear")):
        try:
            manifest.validate_row({"label": "x", col: bad}, 6)
        except manifest.ManifestError as exc:
            assert "row 6" in str(exc) and bad in str(exc)
        else:
            raise AssertionError(f"{col}={bad!r} should have been rejected")


def test_blank_channels_means_all_four():
    # A collector who leaves laser_channels empty must get the green laser too.
    assert manifest.validate_row({"label": "x"}, 2)["laser_channels"] == [1, 2, 3, 4]
