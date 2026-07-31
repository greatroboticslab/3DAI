"""HTTP API for the scanner, so another system (4DAI) can drive it.

Cross-talk model: 4DAI owns the sample identity. It creates a ``sample_id``,
asks the scanner to capture under that id, and reads the multimodal result back
-- everything joined on the shared ``sample_id``. The scanner never invents ids
here; if 4DAI passes one that doesn't exist yet, we create the sample with
exactly that id.

Run it:
    scanner_system/.venv/Scripts/python -m uvicorn scanner_system.api:app \
        --host 0.0.0.0 --port 8600

Endpoints:
    GET  /health
    POST /capture               {sample_id?, label?, material_class?, mode?, laser_channels?}
    GET  /samples/{sample_id}   the sample's scans + artifacts, grouped by modality
    GET  /artifacts/{id}        stream one captured image
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from scanner_system import scanner_db, capture

app = FastAPI(title="Scanner API", version="1.0")


class CaptureRequest(BaseModel):
    sample_id: Optional[str] = None      # 4DAI's shared key; created if new
    label: Optional[str] = None
    material_class: Optional[str] = None
    material_subclass: Optional[str] = None
    mode: str = "full"
    laser_channels: list[int] = [1, 2, 3]
    wait: bool = False                   # True = block until the scan finishes


def _artifact_view(a: dict) -> dict:
    return {
        "artifact_id": a["_id"],
        "modality": a.get("modality"),
        "role": a.get("role"),
        "laser_state": a.get("laser_state"),
        "file_path": a.get("file_path"),
        "download_url": f"/artifacts/{a['_id']}",
    }


@app.get("/health")
def health():
    try:
        scanner_db.get_db()["samples"].find_one({})
        return {"status": "ok", "db": scanner_db.get_db_name()}
    except Exception as exc:
        return {"status": "db_unavailable", "detail": str(exc)}


def _run_capture_bg(sample_id: str, mode: str, laser_channels: list[int]):
    """Run a capture in the background. Errors are swallowed here; the scan's
    own per-instrument status records what happened."""
    try:
        capture.run_capture(sample_id, mode=mode, laser_channels=laser_channels)
    except Exception:
        pass


@app.post("/capture")
def do_capture(req: CaptureRequest, background: BackgroundTasks):
    """Trigger a multimodal capture for a sample and return immediately.

    If ``sample_id`` is given and unknown, the sample is created with that id
    (this is how 4DAI's key flows in). The scan runs in the BACKGROUND so the
    caller (e.g. a 4DAI submission) is not blocked for the ~minute a full scan
    takes; poll GET /samples/{sample_id} for status.
    """
    db = scanner_db.get_db()
    sid = req.sample_id
    if sid:
        if scanner_db.get_sample(sid, db=db) is None:
            scanner_db.create_sample(
                req.label or f"sample {sid[:8]}",
                material_class=req.material_class,
                material_subclass=req.material_subclass,
                sample_id=sid, db=db)
    else:
        sid = scanner_db.create_sample(
            req.label or "scanner sample",
            material_class=req.material_class,
            material_subclass=req.material_subclass, db=db)

    if req.wait:
        pkg = capture.run_capture(sid, mode=req.mode,
                                  laser_channels=req.laser_channels, db=db)
        return {"sample_id": sid, "scan_id": pkg["_id"], "status": pkg["status"],
                "results": {k: v.get("status") for k, v in pkg.get("results", {}).items()},
                "artifacts": {m: len(v) for m, v in pkg["artifacts"].items() if v}}

    background.add_task(_run_capture_bg, sid, req.mode, req.laser_channels)
    return {"sample_id": sid, "status": "started",
            "message": "capture running; poll GET /samples/{sample_id}"}


@app.get("/samples/{sample_id}")
def get_sample(sample_id: str):
    """Return everything the scanner has for a sample: its scans and their
    artifacts grouped by modality. This is what 4DAI reads back."""
    db = scanner_db.get_db()
    sample = scanner_db.get_sample(sample_id, db=db)
    if sample is None:
        raise HTTPException(status_code=404, detail="sample not found on scanner")
    scans = []
    for sc in scanner_db.scans_for_sample(sample_id, db=db):
        pkg = scanner_db.scan_package(sc["_id"], db=db)
        grouped = {m: [_artifact_view(a) for a in v]
                   for m, v in pkg["artifacts"].items() if v}
        scans.append({
            "scan_id": sc["_id"],
            "status": sc.get("status"),
            "mode": sc.get("mode"),
            "results": sc.get("results", {}),
            "artifacts": grouped,
        })
    return {
        "sample_id": sample_id,
        "label": sample.get("label"),
        "material": sample.get("material"),
        "scans": scans,
    }


@app.get("/artifacts/{artifact_id}")
def get_artifact(artifact_id: str):
    db = scanner_db.get_db()
    a = db["artifacts"].find_one({"_id": artifact_id})
    if a is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    path = os.path.join(capture.STORAGE_ROOT, a.get("file_path", ""))
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="artifact file missing on disk")
    return FileResponse(path, media_type=a.get("media_type", "application/octet-stream"))
