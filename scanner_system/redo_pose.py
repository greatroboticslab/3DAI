"""Re-capture one pose of an object that is already in the sheet.

The bench reality: an object slips, someone bumps the table, a pose gets
captured wrong. The manifest runner only knows whole rows, so redoing pose 2
of a two-pose object meant either re-scanning both poses or editing the
database by hand. This does the one pose:

    python -m scanner_system.redo_pose collection.xlsx "green laser glasses" 2

- finds the sample (by its label in the sheet, or a short/full sample id)
- captures that pose again under the SAME sample id, with the row's mode,
  laser channels and notes
- marks the previous scan of that pose ``superseded`` (kept for the record,
  excluded from exports, which only take complete scans)
- rewrites the row's result cells from the scans that now stand, and appends
  the new laser feature rows to the features tab

Place the object in that pose BEFORE running this; it fires immediately.
Safe to run while scan.bat is waiting at its "New object" prompt (the
runner only holds the Kinect and laser board during a capture), but do not
press Enter in the scan window until this finishes.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Optional

from . import manifest, schema, scanner_db


def _find_row(path: str, target: str) -> Optional[dict[str, Any]]:
    """Last sheet row whose label or sample_id matches ``target``."""
    hit = None
    for raw in manifest.read_manifest(path):
        row = manifest.validate_row(raw, raw["row_number"])
        if row["label"].lower() == target.lower() or (
                row["sample_id"] and target and row["sample_id"].startswith(target)):
            hit = row
    return hit


def _standing_scans(sample_id: str, db=None) -> dict[int, dict[str, Any]]:
    """Newest complete/partial scan per pose index for the sample."""
    by_pose: dict[int, dict[str, Any]] = {}
    for scan in sorted(scanner_db.scans_for_sample(sample_id, db=db),
                       key=lambda s: s.get("started_at", "")):
        if scan.get("status") not in ("complete", "partial", "failed"):
            continue
        idx = (scan.get("angle") or {}).get("index") or 1
        by_pose[idx] = scan
    return by_pose


def redo_pose(path: str, target: str, pose: int, dry_run: bool = False, db=None) -> dict[str, Any]:
    row = _find_row(path, target)
    if row is None:
        raise manifest.ManifestError(f"no row in {path} matches {target!r}")
    if not row["sample_id"]:
        raise manifest.ManifestError(
            f"row {row['row_number']} ({row['label']}) has no sample id yet; scan it normally first")
    sample_id = scanner_db.resolve_sample_id(row["sample_id"], db=db)
    n_angles = row["angles"]
    if not 1 <= pose <= n_angles:
        raise manifest.ManifestError(f"pose must be 1..{n_angles} for this row")

    standing = _standing_scans(sample_id, db=db)
    old = standing.get(pose)
    plan = {
        "row_number": row["row_number"], "label": row["label"], "sample_id": sample_id,
        "pose": pose, "of": n_angles, "mode": row["mode"], "laser_channels": row["laser_channels"],
        "replaces": old["_id"] if old else None,
    }
    if dry_run:
        return plan

    from . import capture
    pkg = capture.run_capture(
        sample_id=sample_id, mode=row["mode"], laser_channels=row["laser_channels"],
        operator=row["operator"],
        angle={"index": pose, "count": n_angles} if n_angles > 1 else None,
        notes=(row["notes"] or "") + f" [redo of pose {pose}]",
        known_height_mm=row["known_height_mm"] if pose == 1 else None,
        db=db,
    )
    new_id = pkg.get("_id", "")
    if old and old["_id"] != new_id:
        scanner_db.set_scan_meta(old["_id"], {
            "status": "superseded",
            "notes": ((old.get("notes") or "") + f"; superseded by redo {new_id}").strip("; "),
        }, db=db)

    if pkg.get("laser_features"):
        from . import laser_features
        sample_doc = scanner_db.get_sample(sample_id, db=db) or {"_id": sample_id}
        manifest.write_features(path, laser_features.feature_rows(
            sample_doc, pkg, schema.LASER_WAVELENGTHS_NM))

    # The row's result cells describe all standing poses, in pose order.
    standing = _standing_scans(sample_id, db=db)
    summaries = [manifest._summarize(scanner_db.scan_package(standing[k]["_id"], db=db))
                 for k in sorted(standing) if k <= n_angles]
    summary = manifest._aggregate(summaries)
    summary["sample_id"] = manifest._short(sample_id)
    manifest.write_results(path, row["row_number"], summary)
    plan.update({"new_scan": new_id, "status": summary["status"],
                 "artifact_count": summary["artifact_count"],
                 "failure_detail": summary["failure_detail"]})
    return plan


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sheet")
    ap.add_argument("object", help="label as typed in the sheet, or a sample id")
    ap.add_argument("pose", type=int)
    ap.add_argument("--dry-run", action="store_true", help="show what would be redone")
    args = ap.parse_args(argv)
    try:
        out = redo_pose(args.sheet, args.object, args.pose, dry_run=args.dry_run)
    except (manifest.ManifestError, KeyError) as exc:
        print(f"ERR {exc}")
        return 2
    for k, v in out.items():
        print(f"  {k}: {v}")
    return 0 if out.get("status", "complete") == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
