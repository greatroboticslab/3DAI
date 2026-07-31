"""Publish a scanner scan INTO 4DAI, using 4DAI's own public API.

Direction: the scanner drives 4DAI (not the other way round). 4DAI is left
completely unmodified -- we are just a client of its existing endpoints:

    POST /collection/submission        -> creates a 4DAI sample, returns its id
    POST /collection/images/upload     -> attaches an image to that sample

So a scan captured on the scanner (fringe / multispectral laser / kinect) gets
pushed into 4DAI as a sample with its images, viewable in 4DAI's normal UI.
Nothing in 4DAI's codebase changes; we only consume its public API.

Usage:
    python -m scanner_system.push_to_4dai <scanner_scan_id> [category]
Env: FOURDAI_URL (default http://127.0.0.1:8000)
"""

from __future__ import annotations

import os
import sys
from datetime import date

import requests

from scanner_system import scanner_db, capture

FOURDAI_URL = os.getenv("FOURDAI_URL", "http://127.0.0.1:8000").rstrip("/")


def publish_scan(scan_id: str, category: str = "materials",
                 fourdai_url: str = FOURDAI_URL, db=None) -> dict:
    """Push one scanner scan's images into 4DAI. Returns a summary dict.

    Creates a 4DAI sample (submission) carrying the material label + a back
    reference to the scanner sample_id, then uploads each PNG artifact. Never
    modifies 4DAI code -- only calls its public endpoints.
    """
    d = scanner_db.get_db(db)
    pkg = scanner_db.scan_package(scan_id, db=d)
    if pkg is None:
        raise ValueError(f"scan {scan_id} not found")
    sample = scanner_db.get_sample(pkg["sample_id"], db=d) or {}
    material = sample.get("material", {})

    # 1) create the 4DAI sample via its public submission endpoint
    sub = requests.post(f"{fourdai_url}/collection/submission", json={
        "category": category,
        "date": str(date.today()),
        "data": {
            "label": sample.get("label"),
            "material_class": material.get("class"),
            "material_subclass": material.get("subclass"),
            "scanner_sample_id": pkg["sample_id"],   # back-reference / shared key
            "scanner_scan_id": scan_id,
        },
    }, timeout=30)
    sub.raise_for_status()
    fourdai_sid = sub.json()["sample_id"]

    # 2) upload each image artifact to 4DAI (skip non-image data like .npz)
    uploaded = []
    for modality, arts in pkg["artifacts"].items():
        for a in arts:
            fp = str(a.get("file_path", ""))
            if not fp.endswith(".png"):
                continue
            path = os.path.join(capture.STORAGE_ROOT, fp)
            if not os.path.isfile(path):
                continue
            # encode modality + wavelength into the filename so it's identifiable in 4DAI
            ls = a.get("laser_state") or {}
            wl = f"_{ls['wavelength_nm']}nm" if ls.get("wavelength_nm") else ""
            fname = f"{modality}_{a.get('role', 'img')}{wl}.png"
            with open(path, "rb") as f:
                up = requests.post(
                    f"{fourdai_url}/collection/images/upload",
                    files={"file": (fname, f, "image/png")},
                    data={"sample_id": fourdai_sid, "category": category},
                    timeout=30)
            if up.status_code == 200:
                uploaded.append(fname)

    # 3) record the 4DAI id on the scanner sample (cross-reference both ways)
    d["samples"].update_one({"_id": pkg["sample_id"]},
                            {"$set": {"fourdai_sample_id": fourdai_sid}})

    return {
        "scanner_sample_id": pkg["sample_id"],
        "scanner_scan_id": scan_id,
        "fourdai_sample_id": fourdai_sid,
        "category": category,
        "uploaded_images": uploaded,
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m scanner_system.push_to_4dai <scan_id> [category]")
        return 2
    scan_id = sys.argv[1]
    category = sys.argv[2] if len(sys.argv) > 2 else "materials"
    print(f"Publishing scanner scan {scan_id[:8]} into 4DAI ({category})...")
    res = publish_scan(scan_id, category=category)
    print(f"  4DAI sample_id : {res['fourdai_sample_id']}")
    print(f"  uploaded       : {len(res['uploaded_images'])} images")
    for f in res["uploaded_images"]:
        print(f"     - {f}")
    print("Done. The scan is now in 4DAI, viewable in its normal UI. 4DAI unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
