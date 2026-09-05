"""One-command handoff bundle: images + metadata sheet in a single zip.

Hongbo's requirement for the preliminary study is "save, query, and export
images and metadata". The database does save and query; this produces the
export half as something a person can be handed: a zip whose images are
organized by material class and whose metadata is one spreadsheet row per
file, ready for the report or for a training pipeline on another machine.

Layout inside the zip:
    images/<material_class>/<sample-label>__<angle>/<role>.png
    metadata.xlsx    sheet "files":   one row per exported file
                     sheet "summary": counts per class / modality / pose

Array artifacts (.npz fringe stacks, .npy height maps) are excluded by
default: a 200-object collection carries ~6 GB of fringe stacks, which is a
dataset transfer, not a report attachment. --include-arrays adds them.

USAGE
    python -m scanner_system.export_bundle out.zip
    python -m scanner_system.export_bundle out.zip --material-class cardboard
    python -m scanner_system.export_bundle out.zip --include-arrays
"""

from __future__ import annotations

import argparse
import os
import re
import zipfile
from collections import Counter
from typing import Any, Optional

from . import scanner_db

# capture.STORAGE_ROOT without importing capture (keep this module inert and
# fast to import; capture pulls in the hardware-adjacent orchestrator).
STORAGE_ROOT = os.getenv("SCANNER_STORAGE_ROOT", "").strip() or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scanner_data"
)

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def _slug(text: str) -> str:
    """Filesystem-safe fragment of a freeform label."""
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", (text or "").strip()).strip("-")
    return text or "unnamed"


def build_bundle(
    out_zip: str,
    material_class: Optional[str] = None,
    modality: Optional[str] = None,
    include_arrays: bool = False,
    db=None,
) -> dict[str, Any]:
    """Write the bundle and return a summary dict (also printed by the CLI)."""
    rows = scanner_db.export_dataset(material_class=material_class,
                                     modality=modality, db=db)
    labels = {s["_id"]: s.get("label", "") for s in scanner_db.list_samples(db=db)}

    exported, skipped_kind, missing = [], 0, []
    counts: Counter = Counter()
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for row in rows:
            src_rel = row.get("file_path") or ""
            is_image = src_rel.lower().endswith(IMAGE_SUFFIXES)
            if not is_image and not include_arrays:
                skipped_kind += 1
                continue
            src = os.path.join(STORAGE_ROOT, src_rel)
            if not os.path.isfile(src):
                missing.append(src_rel)
                continue

            cls = row.get("material_class") or "unlabeled"
            pose = (f"angle{row['angle_index']}of{row['angle_count']}"
                    if row.get("angle_index") else "angle1of1")
            # The scan-id fragment keeps entries unique when one sample has
            # several scans of the same pose (repeat captures, old test scans);
            # without it those collide into duplicate archive names.
            scan8 = (row.get("scan_id") or "noscan")[:8]
            sample_dir = f"{_slug(labels.get(row['sample_id'], ''))}__{pose}__{scan8}"
            arc = "/".join(["images", _slug(cls), sample_dir,
                            f"{row.get('role') or 'file'}{os.path.splitext(src_rel)[1]}"])
            zf.write(src, arc)

            out_row = dict(row)
            out_row["bundle_path"] = arc
            out_row["label"] = labels.get(row["sample_id"], "")
            exported.append(out_row)
            counts[("class", cls)] += 1
            counts[("modality", row.get("modality") or "?")] += 1
            counts[("pose", pose)] += 1

        # Measured laser features (scatter halo, reflectance, speckle, IR),
        # one row per scan per channel: the numbers a model trains on.
        from . import schema
        from .laser_features import feature_rows
        feats = []
        for sample in scanner_db.list_samples(db=db):
            if material_class and (sample.get("material") or {}).get("class") != material_class:
                continue
            for scan in scanner_db.scans_for_sample(sample["_id"], db=db):
                if scan.get("laser_features"):
                    feats.extend(feature_rows(sample, scan, schema.LASER_WAVELENGTHS_NM))
        zf.writestr("metadata.xlsx", _metadata_xlsx_bytes(exported, counts, missing, feats))

    summary = {
        "files": len(exported),
        "samples": len({r["sample_id"] for r in exported}),
        "scans": len({r["scan_id"] for r in exported if r.get("scan_id")}),
        "skipped_non_image": skipped_kind,
        "missing_on_disk": len(missing),
        "zip": out_zip,
        "zip_bytes": os.path.getsize(out_zip),
    }
    return summary


def _metadata_xlsx_bytes(exported, counts, missing, feature_rows_=None) -> bytes:
    """Build metadata.xlsx in memory: files, summary, and laser_features sheets."""
    import io

    from openpyxl import Workbook
    from openpyxl.styles import Font
    from .laser_features import FEATURE_COLUMNS

    wb = Workbook()

    ws = wb.active
    ws.title = "files"
    headers = ["bundle_path", "label", "material_class", "material_subclass",
               "modality", "role", "wavelength_nm", "angle_index", "angle_count",
               "sample_id", "scan_id", "file_path"]
    ws.append(headers)
    for c in ws[1]:
        c.font = Font(bold=True)
    for row in exported:
        ws.append([row.get(h) for h in headers])
    ws.freeze_panes = "A2"

    s = wb.create_sheet("summary")
    s.append(["what", "value", "count"])
    for c in s[1]:
        c.font = Font(bold=True)
    for (kind, value), n in sorted(counts.items()):
        s.append([kind, value, n])
    if missing:
        s.append([])
        s.append(["missing on disk (registered in DB, file absent)", "", len(missing)])
        for m in missing:
            s.append(["missing", m, ""])

    if feature_rows_:
        lf = wb.create_sheet("laser_features")
        lf.append(FEATURE_COLUMNS)
        for c in lf[1]:
            c.font = Font(bold=True)
        for row in feature_rows_:
            lf.append([row.get(k) for k in FEATURE_COLUMNS])
        lf.freeze_panes = "A2"

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_zip", help="path of the bundle zip to write")
    ap.add_argument("--material-class", default=None,
                    help="only this material class")
    ap.add_argument("--modality", default=None,
                    help="only this modality (laser/kinect/projector/fusion)")
    ap.add_argument("--include-arrays", action="store_true",
                    help="also pack .npz/.npy arrays (large!)")
    args = ap.parse_args(argv)

    summary = build_bundle(args.out_zip, material_class=args.material_class,
                           modality=args.modality,
                           include_arrays=args.include_arrays)
    print(f"bundle: {summary['zip']}  ({summary['zip_bytes']/1e6:.1f} MB)")
    print(f"  {summary['files']} files from {summary['samples']} samples / "
          f"{summary['scans']} scans")
    if summary["skipped_non_image"]:
        print(f"  {summary['skipped_non_image']} array files skipped "
              "(use --include-arrays to pack them)")
    if summary["missing_on_disk"]:
        print(f"  WARNING: {summary['missing_on_disk']} registered files missing "
              "on disk; listed in metadata.xlsx summary sheet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
