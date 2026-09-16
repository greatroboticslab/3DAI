"""Roboflow export: labeled classification staging, plus direct upload.

Hongbo: "export to roboflow for labeling ... Torres already has roboflow part
code, you can borrow from 4D AI repo." Findings from that borrow (2026-09-05):

- Torres's code POSTs raw images to
  ``https://api.roboflow.com/dataset/{project_id}/upload`` with an api_key and
  a JSON metadata blob. It never calls any annotation endpoint, so every image
  lands UNLABELED in Roboflow's queue - the label would have to be re-typed by
  hand there, which defeats a labeled collection.
- The ``roboflow`` SDK's ``Project.single_upload`` accepts a class name as
  ``annotation_path`` for classification projects, which is the supported way
  to upload an image WITH its label. It needs a file path; our artifacts are
  files on disk, so that fits exactly.
- No Roboflow credentials have ever been configured on this machine (the 4DAI
  runtime's roboflow_settings directory is empty), and the 3DAI/4DAI clone of
  the upload page has a double-POST bug; the roboticsarmAI copy is the fixed
  one. Borrow semantics, not code.

So this module does two things:

STAGE (default, no network, no credentials):
    Copies exported images into ``<out_dir>/<material_class>/...``. Roboflow's
    web uploader treats folder names as class labels for classification
    projects, so dragging the staged directory in produces a LABELED dataset.
    This works today, with no key, and doubles as a portable classification
    dataset for any other trainer.

UPLOAD (--upload, needs credentials):
    Pushes each staged image with its class label via the roboflow SDK.
    Credentials come from env: SCANNER_ROBOFLOW_API_KEY,
    SCANNER_ROBOFLOW_WORKSPACE, SCANNER_ROBOFLOW_PROJECT. The key is never
    written to disk by this module; Torres's server stores keys as plaintext
    JSON, which is not a pattern worth copying.

USAGE
    python -m scanner_system.export_roboflow stage roboflow_out/
    python -m scanner_system.export_roboflow stage roboflow_out/ --roles laser_ch1_png,laser_ch4_png
    python -m scanner_system.export_roboflow upload roboflow_out/   # after staging
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter
from typing import Any, Optional

from . import scanner_db
from .export_bundle import STORAGE_ROOT, _slug

# Roles worth training on by default. The ambient dark frame carries no
# material information on its own (it is the subtraction reference), and the
# height-map PNG is a colormapped render of currently-stale calibration, so
# both stay out unless explicitly asked for.
DEFAULT_ROLES = (
    "laser_ch1_png", "laser_ch2_png", "laser_ch3_png", "laser_ch4_png",
    # side camera (manual exposure) frames, present once that camera is fitted
    "laser_cam_ch1_png", "laser_cam_ch2_png", "laser_cam_ch3_png", "laser_cam_ch4_png",
    "color_png", "fringe_white_png",
)


def stage(
    out_dir: str,
    roles: Optional[list[str]] = None,
    material_class: Optional[str] = None,
    db=None,
) -> dict[str, Any]:
    """Copy labeled images into folder-per-class layout. Returns a summary."""
    roles = list(roles or DEFAULT_ROLES)
    rows = scanner_db.export_dataset(material_class=material_class, db=db)
    labels = {s["_id"]: s.get("label", "") for s in scanner_db.list_samples(db=db)}

    staged, missing = [], []
    counts: Counter = Counter()
    for row in rows:
        if row.get("role") not in roles:
            continue
        if not row.get("material_class"):
            counts[("unlabeled_skipped", "")] += 1
            continue  # no class = nothing to label it with; do not pollute training
        src = os.path.join(STORAGE_ROOT, row.get("file_path") or "")
        if not os.path.isfile(src):
            missing.append(row.get("file_path"))
            continue

        cls = _slug(row["material_class"])
        pose = (f"a{row['angle_index']}" if row.get("angle_index") else "a1")
        scan8 = (row.get("scan_id") or "noscan")[:8]
        name = (f"{_slug(labels.get(row['sample_id'], ''))}"
                f"__{pose}__{scan8}__{row['role']}{os.path.splitext(src)[1]}")
        dst = os.path.join(out_dir, cls, name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        staged.append({"path": dst, "class": row["material_class"],
                       "sample_id": row["sample_id"], "role": row["role"]})
        counts[("class", cls)] += 1

    return {"staged": len(staged), "classes": len({s['class'] for s in staged}),
            "missing_on_disk": len(missing), "out_dir": out_dir,
            "counts": dict(counts), "files": staged}


def upload(staged_dir: str, split: str = "train") -> dict[str, Any]:
    """Upload a staged folder-per-class tree with labels via the roboflow SDK.

    Reads SCANNER_ROBOFLOW_API_KEY / _WORKSPACE / _PROJECT from env. Never
    persists the key.
    """
    api_key = os.getenv("SCANNER_ROBOFLOW_API_KEY", "").strip()
    workspace = os.getenv("SCANNER_ROBOFLOW_WORKSPACE", "").strip()
    project_id = os.getenv("SCANNER_ROBOFLOW_PROJECT", "").strip()
    if not (api_key and workspace and project_id):
        raise SystemExit(
            "Set SCANNER_ROBOFLOW_API_KEY, SCANNER_ROBOFLOW_WORKSPACE and "
            "SCANNER_ROBOFLOW_PROJECT first (get them from Torres/the Roboflow "
            "project settings). The key is read from env only and never stored."
        )
    try:
        from roboflow import Roboflow
    except ImportError:
        raise SystemExit(
            "The 'roboflow' package is not installed in this environment. "
            "Install the version proven on this machine: pip install roboflow==1.4.0\n"
            "(It is deliberately not bundled: it drags in its own OpenCV, which "
            "can conflict with the scanner venv's headless OpenCV.)"
        )

    project = Roboflow(api_key=api_key).workspace(workspace).project(project_id)
    sent, failed = 0, []
    for cls in sorted(os.listdir(staged_dir)):
        cls_dir = os.path.join(staged_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            path = os.path.join(cls_dir, fname)
            try:
                # For classification projects annotation_path is, per the SDK
                # docs, "a class name, e.g. 'dog'" - the labeled-upload path
                # Torres's raw POST was missing.
                project.single_upload(image_path=path, annotation_path=cls,
                                      split=split)
                sent += 1
            except Exception as exc:  # keep going; report at the end
                failed.append((path, f"{type(exc).__name__}: {exc}"))
    return {"uploaded": sent, "failed": failed}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_s = sub.add_parser("stage", help="write folder-per-class labeled images")
    p_s.add_argument("out_dir")
    p_s.add_argument("--roles", default=None,
                     help=f"comma list (default: {','.join(DEFAULT_ROLES)})")
    p_s.add_argument("--material-class", default=None)

    p_u = sub.add_parser("upload", help="SDK-upload a staged tree with labels")
    p_u.add_argument("staged_dir")
    p_u.add_argument("--split", default="train", choices=("train", "valid", "test"))

    args = ap.parse_args(argv)

    if args.cmd == "stage":
        roles = [r.strip() for r in args.roles.split(",")] if args.roles else None
        s = stage(args.out_dir, roles=roles, material_class=args.material_class)
        print(f"staged {s['staged']} images across {s['classes']} classes "
              f"-> {s['out_dir']}")
        for (kind, val), n in sorted(s["counts"].items()):
            print(f"  {kind:18} {val:20} {n}")
        if s["missing_on_disk"]:
            print(f"  WARNING: {s['missing_on_disk']} registered files missing on disk")
        print("drag the folder into a Roboflow CLASSIFICATION project: folder "
              "names become class labels. Or: export_roboflow upload <dir>")
        return 0

    if args.cmd == "upload":
        r = upload(args.staged_dir, split=args.split)
        print(f"uploaded {r['uploaded']} labeled images")
        for path, why in r["failed"]:
            print(f"  FAILED {path}: {why}")
        return 1 if r["failed"] else 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
