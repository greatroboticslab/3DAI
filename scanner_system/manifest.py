"""Spreadsheet manifest workflow: enter samples in Excel, scan the whole sheet.

WHY THIS EXISTS
---------------
The Streamlit Capture page costs about eight interactions per sample, three of
them free-text, and it does not clear the Label / Material fields between runs,
so sample #2 can silently be scanned under sample #1's label. There is also no
way to pre-register a batch: sample creation is welded to the Run-capture
button. That is fine for one-off scans and hopeless for building a dataset.

This module makes a spreadsheet the entry surface and work queue:

    1. Generate a template            -> a sheet with validated columns
    2. A tech fills one row per sample (fast, offline, no lab PC needed)
    3. Run the manifest               -> scans every pending row in order
    4. Results are written back       -> scan_id, status, artifact paths

DESIGN NOTE: the spreadsheet is NOT the database. Mongo stays the store of
record; the sheet is the entry and reporting surface over it. A workbook has no
concurrency control and corrupts silently, so it must not be the only copy of
anything.

CRASH SAFETY: results are written back after EVERY row, not at the end. A scan
is about a minute of wall clock, so a twenty-row manifest is a twenty-minute
run; losing all of it to a failure on row 19 would be unacceptable. Re-running a
manifest skips rows already marked complete, so an interrupted run resumes.

IMPORT SAFETY: importing this module touches no hardware and opens no database.
``capture`` is imported lazily inside run_manifest() only when a scan actually
starts, matching the convention in the rest of scanner_system.

USAGE
    # make a blank manifest to hand to a tech
    python -m scanner_system.manifest template samples.xlsx

    # check a filled-in manifest without scanning anything
    python -m scanner_system.manifest validate samples.xlsx

    # scan every pending row
    python -m scanner_system.manifest run samples.xlsx
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Any, Optional

from . import schema


# ── Column contract ────────────────────────────────────────────────────────
#
# ENTRY columns are filled in by a human before scanning. RESULT columns are
# owned by the scanner and overwritten on every run; a human editing them will
# just have their edits replaced.

# Ordered by what the collector must think about. The first block is the
# material science: the labels the model trains on and the ground truth that
# validates what the lasers measure. The second block is capture control and
# bookkeeping; every column there has a sensible default and may be left blank.
ENTRY_COLUMNS = [
    "label",             # REQUIRED. Freeform, e.g. "oak plank #3"
    "material_class",    # ML target, e.g. "wood"
    "material_subclass",  # ML target, e.g. "oak"
    "surface",           # matte / glossy / textured / mixed. Ground truth for the
                         # laser scatter feature (tight halo = glossy, wide = matte).
    "transparency",      # opaque / translucent / transparent. Ground truth for
                         # subsurface scatter (the wide glow under a laser spot).
    "angles",            # poses per object, e.g. 3 = scan at 3 orientations. Blank = 1.
    "known_height_mm",   # caliper height of a FLAT MATTE object as placed for pose 1.
                         # Fill only for flat-topped objects: it becomes a
                         # calibration anchor. Blank = not an anchor.
    "notes",             # freeform, lands on the scan document
    "sample_id",         # blank = mint a new one. Fill it to reuse/pin an id.
    "mode",              # one of schema.CAPTURE_MODES. Blank = "full".
    "laser_channels",    # e.g. "1,2,4". Blank = ALL FOUR.
    "operator",          # who ran it. Bookkeeping only; blank is fine.
]

# Controlled vocabularies for the surface-property labels. Free text here is
# how "Oak", "oak" and "oak " became three material classes; these are
# validated so the ground truth stays groupable.
SURFACE_FINISHES = ("matte", "glossy", "textured", "mixed")
TRANSPARENCIES = ("opaque", "translucent", "transparent")

RESULT_COLUMNS = [
    "scan_id",
    "status",            # complete / partial / failed
    "kinect_status",
    "projector_status",
    "laser_status",
    "failure_detail",
    "artifact_count",
    "started_at",
    "completed_at",
]

ALL_COLUMNS = ENTRY_COLUMNS + RESULT_COLUMNS

# Blank laser_channels means every laser. This used to be [1, 2, 3], which
# silently dropped the green channel for any collector who left the cell
# empty; for material recognition the green flood is the reflectance probe.
DEFAULT_LASER_CHANNELS = [1, 2, 3, 4]
# A row is considered done and is skipped on re-run only when it reached this.
# "partial" and "failed" rows are retried, which is usually what you want after
# fixing whatever was wrong at the bench.
DONE_STATUSES = {"complete"}


class ManifestError(Exception):
    """Raised for a manifest that cannot be used as-is."""


# ── Parsing helpers ────────────────────────────────────────────────────────

def _clean(value: Any) -> str:
    """Normalize a cell to a stripped string. None/NaN become ''."""
    if value is None:
        return ""
    text = str(value).strip()
    # openpyxl hands back floats for numeric cells, so "1" can arrive as "1.0";
    # and pandas-style NaN can arrive as the literal string "nan".
    if text.lower() == "nan":
        return ""
    return text


def parse_channels(text: str) -> list[int]:
    """Parse "1,2,3" (or "1 2 3", or "2") into [1,2,3]. Blank = defaults."""
    text = _clean(text)
    if not text:
        return list(DEFAULT_LASER_CHANNELS)
    raw = text.replace(";", ",").replace(" ", ",")
    out: list[int] = []
    for part in raw.split(","):
        if not part:
            continue
        try:
            ch = int(float(part))
        except ValueError:
            raise ManifestError(f"laser_channels: {part!r} is not a number")
        if ch not in schema.LASER_CHANNELS:
            raise ManifestError(
                f"laser_channels: channel {ch} is not one of {list(schema.LASER_CHANNELS)}"
            )
        if ch not in out:
            out.append(ch)
    return out


def validate_row(row: dict[str, Any], number: int) -> dict[str, Any]:
    """Validate one manifest row and return it normalized.

    Raises ManifestError with the row number, so a tech can find the bad cell.
    """
    label = _clean(row.get("label"))
    if not label:
        raise ManifestError(f"row {number}: label is required")

    mode = _clean(row.get("mode")) or "full"
    if mode not in schema.CAPTURE_MODES:
        raise ManifestError(
            f"row {number}: mode {mode!r} is not one of {list(schema.CAPTURE_MODES)}"
        )

    try:
        channels = parse_channels(row.get("laser_channels"))
    except ManifestError as exc:
        raise ManifestError(f"row {number}: {exc}") from None

    angles_txt = _clean(row.get("angles"))
    if angles_txt:
        try:
            angles = int(float(angles_txt))
        except ValueError:
            raise ManifestError(f"row {number}: angles {angles_txt!r} is not a number")
        if not (1 <= angles <= 12):
            raise ManifestError(f"row {number}: angles must be 1-12, got {angles}")
    else:
        angles = 1

    surface = _clean(row.get("surface")).lower() or None
    if surface and surface not in SURFACE_FINISHES:
        raise ManifestError(
            f"row {number}: surface {surface!r} is not one of {list(SURFACE_FINISHES)}")
    transparency = _clean(row.get("transparency")).lower() or None
    if transparency and transparency not in TRANSPARENCIES:
        raise ManifestError(
            f"row {number}: transparency {transparency!r} is not one of {list(TRANSPARENCIES)}")

    height_txt = _clean(row.get("known_height_mm"))
    if height_txt:
        try:
            known_height_mm = float(height_txt)
        except ValueError:
            raise ManifestError(
                f"row {number}: known_height_mm {height_txt!r} is not a number")
        if not (0.2 <= known_height_mm <= 200.0):
            raise ManifestError(
                f"row {number}: known_height_mm must be 0.2-200, got {known_height_mm}")
    else:
        known_height_mm = None

    return {
        "row_number": number,
        "sample_id": _clean(row.get("sample_id")) or None,
        "label": label,
        "material_class": _clean(row.get("material_class")) or None,
        "material_subclass": _clean(row.get("material_subclass")) or None,
        "surface": surface,
        "transparency": transparency,
        "mode": mode,
        "laser_channels": channels,
        "angles": angles,
        "known_height_mm": known_height_mm,
        "operator": _clean(row.get("operator")) or None,
        "notes": _clean(row.get("notes")),
        "status": _clean(row.get("status")),
    }


# ── Reading ────────────────────────────────────────────────────────────────

def read_manifest(path: str) -> list[dict[str, Any]]:
    """Read a .xlsx or .csv manifest into a list of raw row dicts.

    Rows are returned in sheet order with a 1-based ``row_number`` that matches
    what the tech sees in Excel (header is row 1, so data starts at row 2).
    Entirely blank rows are skipped rather than reported as errors.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xlsm"):
        return _read_xlsx(path)
    if ext == ".csv":
        return _read_csv(path)
    raise ManifestError(f"unsupported manifest type {ext!r}: use .xlsx or .csv")


def _read_xlsx(path: str) -> list[dict[str, Any]]:
    from openpyxl import load_workbook

    wb = load_workbook(path, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        raise ManifestError("manifest is empty (no header row)")

    headers = [_clean(h).lower() for h in rows[0]]
    _check_headers(headers)

    out = []
    for i, values in enumerate(rows[1:], start=2):
        row = {h: v for h, v in zip(headers, values) if h}
        if not any(_clean(v) for v in row.values()):
            continue  # blank spacer row
        row["row_number"] = i
        out.append(row)
    return out


def _read_csv(path: str) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        headers = [_clean(h).lower() for h in (reader.fieldnames or [])]
        _check_headers(headers)
        out = []
        for i, raw in enumerate(reader, start=2):
            row = {_clean(k).lower(): v for k, v in raw.items() if k}
            if not any(_clean(v) for v in row.values()):
                continue
            row["row_number"] = i
            out.append(row)
    return out


def _check_headers(headers: list[str]) -> None:
    if "label" not in headers:
        raise ManifestError(
            "manifest has no 'label' column. Generate a fresh template with:\n"
            "    python -m scanner_system.manifest template <path>"
        )
    unknown = [h for h in headers if h and h not in ALL_COLUMNS]
    if unknown:
        # Not fatal: extra columns are a normal way for a lab to keep its own
        # notes. Say so once rather than failing the run.
        print(f"NOTE: ignoring unrecognized column(s): {', '.join(unknown)}")


# ── Template generation ────────────────────────────────────────────────────

def write_template(path: str, rows: int = 25) -> str:
    """Write a blank manifest with headers, dropdowns and an example row.

    The in-cell dropdown on ``mode`` is the main reason to prefer .xlsx over
    .csv here: free-text material fields in the GUI are exactly how "Oak",
    "oak" and "oak " became three different classes.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(ALL_COLUMNS)
        return path

    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ws = wb.active
    ws.title = "samples"
    ws.append(ALL_COLUMNS)

    entry_fill = PatternFill("solid", fgColor="DDEBF7")   # blue: you fill these
    result_fill = PatternFill("solid", fgColor="EDEDED")  # grey: scanner owns these
    for col, name in enumerate(ALL_COLUMNS, start=1):
        cell = ws.cell(row=1, column=col)
        cell.font = Font(bold=True)
        cell.fill = entry_fill if name in ENTRY_COLUMNS else result_fill
        ws.column_dimensions[get_column_letter(col)].width = max(14, len(name) + 3)
    ws.freeze_panes = "A2"

    # Dropdowns on every controlled-vocabulary column, so a typo cannot reach
    # the capture layer or split a class label into three spellings.
    for column, vocab, err in (
        ("surface", SURFACE_FINISHES, "Pick a surface finish from the list."),
        ("transparency", TRANSPARENCIES, "Pick a transparency from the list."),
        ("mode", schema.CAPTURE_MODES, "Pick a capture mode from the list."),
    ):
        col = get_column_letter(ALL_COLUMNS.index(column) + 1)
        dv = DataValidation(type="list", formula1='"' + ",".join(vocab) + '"',
                            allow_blank=True, showErrorMessage=True)
        dv.error = err
        ws.add_data_validation(dv)
        dv.add(f"{col}2:{col}{rows + 1}")

    # One example row, clearly marked so nobody scans it by accident. Column
    # order matches ENTRY_COLUMNS: label, class, subclass, surface,
    # transparency, angles, known_height_mm, notes, then the blank-ok block.
    ws.append(["EXAMPLE - delete this row", "wood", "oak", "matte", "opaque",
               "3", "12.5", "flat matte block, calipered", "", "", "", ""])
    ws.cell(row=2, column=1).font = Font(italic=True, color="999999")

    wb.save(path)
    return path


# ── Write-back ─────────────────────────────────────────────────────────────

def write_results(path: str, row_number: int, results: dict[str, Any]) -> None:
    """Write RESULT columns back into one row, in place.

    Called after every scan so an interrupted batch keeps everything finished
    so far.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xlsm"):
        writer = _write_results_xlsx
    elif ext == ".csv":
        writer = _write_results_csv
    else:
        raise ManifestError(f"unsupported manifest type {ext!r}")

    _retry_locked(writer, path, row_number, results,
                  what=f"results for row {row_number}")


def _retry_locked(writer, path: str, *args, what: str = "results") -> None:
    """Call writer(path, *args), waiting out a spreadsheet-app file lock.

    A spreadsheet app (LibreOffice, Excel) holding the file open makes the
    save raise PermissionError. The scan itself is already safe in the
    database at this point; what would be lost is the write-back, and with it
    the resume marker. So wait for the collector rather than crash: a person
    is at the bench, and closing a window is the whole fix.
    """
    import time as _time
    for attempt in range(100):          # ~5 minutes at 3 s
        try:
            writer(path, *args)
            return
        except PermissionError:
            if attempt == 0:
                print(f"    ! {os.path.basename(path)} is open in another program "
                      "(LibreOffice/Excel?). Close it there; I'll keep retrying...")
            _time.sleep(3)
    raise ManifestError(
        f"{path} stayed locked for 5 minutes; {what} were not written to the "
        "sheet (the scan is still in the database)")


FEATURES_SHEET = "laser_features"


def write_features(path: str, rows: list[dict[str, Any]]) -> None:
    """Append laser feature rows: a second tab of the workbook, or for a CSV
    manifest a sibling ``<name>_laser_features.csv``.

    These are the measured laser numbers (scatter halo, reflectance colour,
    speckle, infrared equivalents), one row per scan per channel. They sit on
    their own tab so the collection tab stays a work queue.
    """
    if not rows:
        return
    from .laser_features import FEATURE_COLUMNS
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xlsm"):
        _retry_locked(_append_features_xlsx, path, rows, FEATURE_COLUMNS,
                      what="laser features")
    else:
        side = os.path.splitext(path)[0] + "_laser_features.csv"
        _retry_locked(_append_features_csv, side, rows, FEATURE_COLUMNS,
                      what="laser features")


def _append_features_xlsx(path: str, rows, columns) -> None:
    from openpyxl import load_workbook
    from openpyxl.styles import Font

    wb = load_workbook(path)
    if FEATURES_SHEET in wb.sheetnames:
        ws = wb[FEATURES_SHEET]
    else:
        ws = wb.create_sheet(FEATURES_SHEET)
        ws.append(columns)
        for c in ws[1]:
            c.font = Font(bold=True)
        ws.freeze_panes = "A2"
    for r in rows:
        ws.append([r.get(c) for c in columns])
    wb.save(path)


def _append_features_csv(path: str, rows, columns) -> None:
    new = not os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(columns)
        for r in rows:
            w.writerow(["" if r.get(c) is None else r.get(c) for c in columns])


def _write_results_xlsx(path: str, row_number: int, results: dict[str, Any]) -> None:
    from openpyxl import load_workbook

    wb = load_workbook(path)
    ws = wb.active
    headers = [_clean(c.value).lower() for c in ws[1]]
    for key, value in results.items():
        if key not in headers:
            continue
        ws.cell(row=row_number, column=headers.index(key) + 1, value=value)
    wb.save(path)


def _write_results_csv(path: str, row_number: int, results: dict[str, Any]) -> None:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    if not rows:
        return
    headers = [_clean(h).lower() for h in rows[0]]
    idx = row_number - 1
    if idx >= len(rows):
        return
    while len(rows[idx]) < len(headers):
        rows[idx].append("")
    for key, value in results.items():
        if key in headers:
            rows[idx][headers.index(key)] = "" if value is None else str(value)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)


# ── Batch runner ───────────────────────────────────────────────────────────

def _summarize(pkg: dict[str, Any]) -> dict[str, Any]:
    """Flatten a scan package into RESULT columns."""
    results = pkg.get("results") or {}
    artifacts = pkg.get("artifacts") or {}
    details = [
        f"{name}: {r.get('detail')}"
        for name, r in results.items()
        if r.get("status") == "failed" and r.get("detail")
    ]
    return {
        "scan_id": pkg.get("_id", ""),
        "status": pkg.get("status", ""),
        "kinect_status": (results.get("kinect") or {}).get("status", ""),
        "projector_status": (results.get("projector") or {}).get("status", ""),
        "laser_status": (results.get("laser") or {}).get("status", ""),
        "failure_detail": "; ".join(details),
        "artifact_count": sum(len(v) for v in artifacts.values()),
        "started_at": pkg.get("started_at", ""),
        "completed_at": pkg.get("completed_at", ""),
    }


def _aggregate(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold per-pose scan summaries into one manifest row's RESULT columns.

    A multi-angle row is complete only when every pose is; one bad pose makes
    the whole row partial so re-running retries the object rather than leaving
    a silently thin sample in the dataset.
    """
    if len(summaries) == 1:
        return dict(summaries[0])
    statuses = [s["status"] for s in summaries]
    if all(s == "complete" for s in statuses):
        status = "complete"
    elif any(s in ("complete", "partial") for s in statuses):
        status = "partial"
    else:
        status = "failed"
    details = "; ".join(
        f"pose {i}: {s['failure_detail']}"
        for i, s in enumerate(summaries, start=1) if s["failure_detail"]
    )
    return {
        "scan_id": "; ".join(s["scan_id"] for s in summaries),
        "status": status,
        "kinect_status": summaries[-1]["kinect_status"],
        "projector_status": summaries[-1]["projector_status"],
        "laser_status": summaries[-1]["laser_status"],
        "failure_detail": details,
        "artifact_count": sum(s["artifact_count"] for s in summaries),
        "started_at": summaries[0]["started_at"],
        "completed_at": summaries[-1]["completed_at"],
    }


def run_manifest(
    path: str,
    limit: Optional[int] = None,
    rescan: bool = False,
    prompt: bool = False,
    db=None,
) -> dict[str, Any]:
    """Scan every pending row of a manifest, writing results back as we go.

    Parameters
    ----------
    limit   : stop after this many scans (useful for a cautious first run)
    rescan  : also re-scan rows already marked complete
    db      : injected database handle, for testing

    Returns a summary dict. Never raises on a single row's hardware failure:
    run_capture records failures as statuses rather than exceptions, so one bad
    sample does not abandon the rest of the batch.
    """
    # Imported here, not at module scope: importing capture pulls in the
    # hardware path, and reading or validating a manifest must stay inert.
    from . import capture, scanner_db

    raw_rows = read_manifest(path)
    rows = [validate_row(r, r["row_number"]) for r in raw_rows]

    pending = [r for r in rows
               if rescan or r["status"].lower() not in DONE_STATUSES]
    skipped = len(rows) - len(pending)
    if limit is not None:
        pending = pending[:limit]

    print(f"manifest: {len(rows)} rows, {len(pending)} to scan"
          + (f", {skipped} already complete" if skipped else ""))

    done, failed, skipped_live = 0, 0, 0
    for i, row in enumerate(pending, start=1):
        label = row["label"]
        print(f"\n[{i}/{len(pending)}] row {row['row_number']}: {label}")

        if prompt:
            # Collection pacing: a person has to physically place each object.
            # This is the "system prompts the collector for each object" flow;
            # without it every row scans back-to-back and nobody can swap
            # objects between rows.
            answer = input(f">>> Place {label!r} on the stage, then press Enter "
                           "(or type 's' to skip this row): ").strip().lower()
            if answer in ("s", "skip"):
                print("    skipped by collector")
                skipped_live += 1
                continue

        sample_id = row["sample_id"]
        if not sample_id:
            # run_capture does NOT create the sample and does not check that it
            # exists, so an unknown id would silently produce an orphan scan.
            sample_id = scanner_db.create_sample(
                label=label,
                material_class=row["material_class"],
                material_subclass=row["material_subclass"],
                context={"source": "manifest",
                         "manifest_path": os.path.basename(path),
                         # Surface-property ground truth rides the sample so
                         # export can pair it with the laser scatter features.
                         "surface": row["surface"],
                         "transparency": row["transparency"]},
                db=db,
            )
            print(f"    created sample {sample_id}")

        # One scan per pose. Dr. Zhang's collection design is "shining lasers
        # on it from different angles": the object is physically rotated
        # between poses, each pose is its own scan document tagged
        # angle={index, count}, and all of them share the sample_id.
        n_angles = row["angles"]
        summaries = []
        for k in range(1, n_angles + 1):
            if prompt and k > 1:
                input(f">>> Rotate {label!r} to pose {k}/{n_angles}, "
                      "then press Enter: ")
            if n_angles > 1:
                print(f"    pose {k}/{n_angles}...")
            pkg = capture.run_capture(
                sample_id=sample_id,
                mode=row["mode"],
                laser_channels=row["laser_channels"],
                operator=row["operator"],
                angle={"index": k, "count": n_angles} if n_angles > 1 else None,
                notes=row["notes"],
                # The caliper height describes the object as placed for pose 1;
                # a rotated object has a different height, so later poses are
                # dataset-only, never anchors.
                known_height_mm=row["known_height_mm"] if k == 1 else None,
                db=db,
            )
            summaries.append(_summarize(pkg))

            # The measured laser features land on their own tab right away,
            # so the collector can watch materials separate as objects go by.
            if pkg.get("laser_features"):
                from . import laser_features
                sample_doc = scanner_db.get_sample(sample_id, db=db) or {"_id": sample_id}
                write_features(path, laser_features.feature_rows(
                    sample_doc, pkg, schema.LASER_WAVELENGTHS_NM))

        summary = _aggregate(summaries)
        summary["sample_id"] = sample_id
        write_results(path, row["row_number"], summary)

        status = summary["status"]
        print(f"    {status} ({summary['artifact_count']} artifacts)")
        if summary["failure_detail"]:
            print(f"    ! {summary['failure_detail']}")
        if status == "complete":
            done += 1
        else:
            failed += 1

    print(f"\ndone: {done} complete, {failed} partial/failed, "
          f"{skipped + skipped_live} skipped")
    return {"total": len(rows), "scanned": len(pending) - skipped_live,
            "complete": done, "problem": failed, "skipped": skipped + skipped_live}


# ── CLI ────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_t = sub.add_parser("template", help="write a blank manifest")
    p_t.add_argument("path")
    p_t.add_argument("--rows", type=int, default=25,
                     help="how many rows get the mode dropdown (default 25)")

    p_v = sub.add_parser("validate", help="check a manifest without scanning")
    p_v.add_argument("path")

    p_r = sub.add_parser("run", help="scan every pending row")
    p_r.add_argument("path")
    p_r.add_argument("--limit", type=int, default=None)
    p_r.add_argument("--rescan", action="store_true",
                     help="also re-scan rows already marked complete")
    p_r.add_argument("--prompt", action="store_true",
                     help="pause before each row so the collector can place the object")

    args = ap.parse_args(argv)

    try:
        if args.cmd == "template":
            out = write_template(args.path, rows=args.rows)
            print(f"wrote template -> {out}")
            print(f"  fill in: {', '.join(ENTRY_COLUMNS)}")
            return 0

        if args.cmd == "validate":
            raw = read_manifest(args.path)
            rows = [validate_row(r, r["row_number"]) for r in raw]
            pending = [r for r in rows if r["status"].lower() not in DONE_STATUSES]
            print(f"OK: {len(rows)} rows parsed, {len(pending)} pending")
            for r in rows:
                mark = " " if r["status"].lower() in DONE_STATUSES else "*"
                print(f"  {mark} row {r['row_number']:>3}  {r['label'][:32]:<32} "
                      f"{r['mode']:<16} lasers={r['laser_channels']}")
            return 0

        if args.cmd == "run":
            run_manifest(args.path, limit=args.limit, rescan=args.rescan,
                         prompt=args.prompt)
            return 0
    except ManifestError as exc:
        print(f"ERR {exc}")
        return 2

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
