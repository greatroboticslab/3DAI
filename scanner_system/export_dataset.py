"""Export the collection to a browsable folder for the GitHub repo.

Hongbo wants to open the repo and see the data: images sorted by material
and one spreadsheet of metadata. export_bundle makes the same thing as a
zip for handing to someone; this writes it as a folder (default
``dataset/``) that push_data.bat commits and pushes.

Layout:
    dataset/
      README.md               what is in here, generated each run
      metadata.xlsx           sheet "files": one row per image, with labels
                              sheet "summary": counts per class / modality / pose
      laser_features.csv      measured laser numbers, one row per scan per channel
      images/<material_class>/<object>__<pose>__<scan>/<role>.png

Demo, smoke-test and verification samples (and anything with no material
label) are left out: they are pipeline checks, not data. Only complete scans
are exported.

The images are JPEG previews (longest side PREVIEW_MAX_PX, 16-bit infrared
and depth frames stretched to 8-bit): a full scan is ~7 MB of PNG, so 200
objects at two poses would be ~3 GB, which GitHub will not take. The
previews are for looking; the numbers the model trains on are in
laser_features.csv at full precision and the original PNGs, fringe stacks
and height maps stay on the scanner PC (export_bundle --include-arrays).
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from . import scanner_db, schema
from .capture import STORAGE_ROOT
from .export_bundle import IMAGE_SUFFIXES, _metadata_xlsx_bytes, _slug
from .laser_features import FEATURE_COLUMNS, feature_rows
from .laser_zoom import make_laser_zoom

# "materials" was the placeholder class of the July API smoke tests.
EXCLUDED_CLASSES = frozenset({"demo", "test", "verification", "smoke", "materials"})
PREVIEW_MAX_PX = 800
PREVIEW_QUALITY = 85


def is_data_sample(sample: dict) -> bool:
    cls = (sample.get("material") or {}).get("class")
    return bool(cls) and cls.lower() not in EXCLUDED_CLASSES


def write_preview(src: str, dst: str) -> None:
    """Save a JPEG preview of an image; 16-bit frames get a percentile stretch."""
    import numpy as np
    from PIL import Image

    with Image.open(src) as src_im:
        if src_im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
            a = np.asarray(src_im, dtype=np.float32)
            lo, hi = np.percentile(a, (1.0, 99.5))
            a = np.clip((a - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0) * 255.0
            im = Image.fromarray(a.astype(np.uint8))
        elif src_im.mode not in ("RGB", "L"):
            im = src_im.convert("RGB")
        else:
            im = src_im.copy()
    im.thumbnail((PREVIEW_MAX_PX, PREVIEW_MAX_PX))
    im.save(dst, "JPEG", quality=PREVIEW_QUALITY)


def export_folder(out_dir: str, db=None, storage_root: str = STORAGE_ROOT) -> dict[str, Any]:
    samples = {s["_id"]: s for s in scanner_db.list_samples(db=db) if is_data_sample(s)}
    complete_scans = {
        scan["_id"]
        for sid in samples
        for scan in scanner_db.scans_for_sample(sid, db=db)
        if scan.get("status") == "complete"
    }
    rows = [r for r in scanner_db.export_dataset(db=db)
            if r["sample_id"] in samples and r.get("scan_id") in complete_scans]

    images_dir = os.path.join(out_dir, "images")
    if os.path.isdir(images_dir):
        shutil.rmtree(images_dir)     # a re-export must not keep stale files
    os.makedirs(images_dir, exist_ok=True)

    exported, missing = [], []
    scan_out_dirs: dict[str, str] = {}
    counts: Counter = Counter()
    for row in rows:
        src_rel = row.get("file_path") or ""
        if not src_rel.lower().endswith(IMAGE_SUFFIXES):
            continue
        src = os.path.join(storage_root, src_rel)
        if not os.path.isfile(src):
            missing.append(src_rel)
            continue
        cls = row.get("material_class")
        pose = (f"angle{row['angle_index']}of{row['angle_count']}"
                if row.get("angle_index") else "angle1of1")
        scan8 = (row.get("scan_id") or "noscan")[:8]
        label = samples[row["sample_id"]].get("label", "")
        role = (row.get("role") or "file").replace("_png", "")
        rel = "/".join(["images", _slug(cls), f"{_slug(label)}__{pose}__{scan8}",
                        f"{role}.jpg"])
        dst = os.path.join(out_dir, *rel.split("/"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        write_preview(src, dst)

        exported.append(dict(row, bundle_path=rel, label=label))
        scan_out_dirs[row.get("scan_id")] = os.path.dirname(dst)
        counts[("class", cls)] += 1
        counts[("modality", row.get("modality") or "?")] += 1
        counts[("pose", pose)] += 1

    feats = []
    zooms = 0
    for sample in samples.values():
        for scan in scanner_db.scans_for_sample(sample["_id"], db=db):
            if scan["_id"] in complete_scans and scan.get("laser_features"):
                feats.extend(feature_rows(sample, scan, schema.LASER_WAVELENGTHS_NM))
                out_scan = scan_out_dirs.get(scan["_id"])
                if out_scan and make_laser_zoom(
                        os.path.join(storage_root, "scans", scan["_id"], "laser"),
                        scan["laser_features"], os.path.join(out_scan, "laser_zoom.jpg")):
                    zooms += 1

    with open(os.path.join(out_dir, "metadata.xlsx"), "wb") as fh:
        fh.write(_metadata_xlsx_bytes(exported, counts, missing, feats))
    with open(os.path.join(out_dir, "laser_features.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(FEATURE_COLUMNS)
        for r in feats:
            w.writerow(["" if r.get(c) is None else r.get(c) for c in FEATURE_COLUMNS])

    by_class: Counter = Counter()
    for s in samples.values():
        by_class[(s.get("material") or {}).get("class")] += 1
    summary = {
        "objects": len(samples),
        "scans": len({r["scan_id"] for r in exported if r.get("scan_id")}),
        "images": len(exported),
        "laser_zooms": zooms,
        "feature_rows": len(feats),
        "missing_on_disk": len(missing),
        "by_class": dict(sorted(by_class.items())),
        "out_dir": out_dir,
    }
    _write_readme(out_dir, summary)
    return summary


def _write_readme(out_dir: str, s: dict[str, Any]) -> None:
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# 3DAI material dataset export",
        "",
        f"Generated {when} by `python -m scanner_system.export_dataset`.",
        "Collection is in progress; this folder is re-exported and pushed as objects are scanned.",
        "",
        f"- Objects: **{s['objects']}**  (scans: {s['scans']}, images: {s['images']})",
        "- Objects per material class: " + ", ".join(f"{k} {v}" for k, v in s["by_class"].items()),
        "",
        "## What is here",
        "",
        "- `metadata.xlsx` - sheet `files`: one row per image with the object's label,",
        "  material class/subclass, surface finish, transparency, pose, modality and laser",
        "  channel/wavelength. Sheet `summary`: counts.",
        "- `laser_features.csv` - the measured laser material signals, one row per scan per",
        "  laser channel: scatter halo radii (`halo_r50`, `halo_r10`), speckle contrast,",
        "  per-channel reflectance (`add_r/g/b`, `flood`), near-infrared equivalents (`ir_*`),",
        "  red/green reflectance ratios and fringe contrast. Channels: 1 and 2 red 635 nm,",
        "  3 near-infrared 940 nm, 4 green 530 nm.",
        "- `images/<material_class>/<object>__<pose>__<scan>/` - the pictures for one scan:",
        "  `color` plain photo, `laser_dark` lasers off, `laser_ch1..4` each laser on,",
        "  `*_ir` the Kinect infrared sensor for the same frame, `kinect_depth` Kinect depth,",
        "  `fringe_white` projector white light, `height_map` reconstructed height.",
        "- `laser_zoom.jpg` in each scan folder: the readable laser view. One row per laser:",
        "  the raw spot zoomed 3x, the same window with the dark frame subtracted and the",
        "  faint scatter halo stretched up, and the radial intensity profile (log axis).",
        "  Rows marked INVALID are channels that did not fire for that scan.",
        "",
        f"The images here are JPEG previews (longest side {PREVIEW_MAX_PX} px; infrared and",
        "depth frames stretched to 8-bit for viewing). The full-resolution PNGs, fringe",
        "stacks, depth arrays and height maps stay on the scanner PC; ask for a bundle",
        "(`python -m scanner_system.export_bundle out.zip --include-arrays`) for training.",
        "",
    ]
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", nargs="?", default="dataset")
    args = ap.parse_args(argv)
    try:
        s = export_folder(args.out_dir)
    except RuntimeError as exc:
        print(f"ERR {exc}")
        return 2
    print(f"exported {s['objects']} objects, {s['scans']} scans, {s['images']} images, "
          f"{s['feature_rows']} laser feature rows -> {s['out_dir']}")
    if s["missing_on_disk"]:
        print(f"  ! {s['missing_on_disk']} files listed in the database were missing on disk")
    for k, v in s["by_class"].items():
        print(f"    {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
