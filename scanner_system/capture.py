"""Capture orchestrator: run one multimodal scan of a sample.

Ties the real hardware (ESP32 lasers via relay_controller, Kinect via
capture_tool) to the data layer: fire the selected lasers, capture a Kinect
frame under each, save the files to disk, and register them as artifacts in
MongoDB. Records per-instrument status honestly (ok/skipped/failed), so a
messy/partial setup is captured truthfully rather than silently.

Safe by construction:
- Importing this module touches no hardware.
- The relay/laser control and Kinect are opened only when a capture actually
  runs, and every code path turns lasers/relays OFF when done.
- If the ESP32 or Kinect is absent, the relevant instrument is marked "failed"
  with a plain-language reason; the scan still records what did work.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

from . import scanner_db, schema, hardware

# Where captured files are written. Artifact file_paths are stored relative to
# this, and the GUI reads images from here (SCANNER_STORAGE_ROOT).
STORAGE_ROOT = os.getenv("SCANNER_STORAGE_ROOT", "").strip() or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scanner_data"
)


def _rel(path: str) -> str:
    """Path relative to STORAGE_ROOT, for storing in the artifact document."""
    return os.path.relpath(path, STORAGE_ROOT).replace(os.sep, "/")


def _kinect_grab(out_path: str) -> dict[str, Any]:
    """Capture one Kinect color frame to out_path. Returns {ok, detail, ...}.

    Uses the real Kinect via capture_tool's runtime. Never raises; a missing
    sensor or driver comes back as ok=False with a reason.
    """
    try:
        import numpy as np
        import cv2
    except Exception as exc:
        return {"ok": False, "detail": f"numpy/cv2 unavailable: {exc}"}
    try:
        import capture_tool
        kinect = capture_tool._get_kinect()
    except Exception as exc:
        return {"ok": False, "detail": f"Kinect not available: {exc}"}

    color = None
    deadline = time.time() + 15.0
    while time.time() < deadline and color is None:
        if kinect.has_new_color_frame():
            cf = kinect.get_last_color_frame()
            h, w = kinect.color_frame_desc.Height, kinect.color_frame_desc.Width
            color = cf.reshape((h, w, 4))
        time.sleep(0.03)
    if color is None:
        return {"ok": False, "detail": "Kinect delivered no frame in 15s "
                                       "(check power brick + USB3)."}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, color[:, :, :3])
    return {"ok": True, "detail": "captured",
            "size_bytes": os.path.getsize(out_path)}


def run_capture(
    sample_id: str,
    mode: str = "full",
    laser_channels: Optional[list[int]] = None,
    wavelengths: Optional[dict[int, int]] = None,
    operator: Optional[str] = None,
    port: Optional[str] = None,
    db=None,
) -> dict[str, Any]:
    """Run one scan of ``sample_id`` in ``mode`` and register its artifacts.

    - "full" / "laser_only": for each laser channel, fire it, capture a Kinect
      frame under that illumination, register a ``laser`` artifact (with the
      laser's wavelength). Lasers are turned OFF between channels and at the end.
    - "full" / "kinect_projector" / "kinect_only": also capture a plain Kinect
      frame with lasers off (a ``kinect`` artifact).
    - projector is recorded as attempted/skipped (fringe capture is a separate
      pipeline; wired in later).

    Returns the finished scan package. Never leaves lasers on.
    """
    d = scanner_db.get_db(db)
    laser_channels = laser_channels or []
    wavelengths = wavelengths or {}

    scan_id = scanner_db.start_scan(sample_id, mode=mode, operator=operator, db=d)
    scan_dir = os.path.join(STORAGE_ROOT, "scans", scan_id)

    wants_laser = mode in ("full", "laser_only") and laser_channels
    wants_kinect_plain = mode in ("full", "kinect_projector", "kinect_only")
    wants_projector = mode in ("full", "kinect_projector", "projector_only")

    # ── Laser modality: capture the sample under each laser ─────────────────
    if wants_laser:
        RelayController = hardware._load_relay_controller()
        rc = None
        try:
            if RelayController is None:
                raise RuntimeError("relay controller unavailable (pyserial?)")
            p = port or hardware.likely_esp32_port()
            if p is None:
                raise RuntimeError("no ESP32 serial port found (is it plugged in "
                                   "with a data cable?)")
            rc = RelayController(p)
            if not rc.connect():
                raise RuntimeError(f"no response from ESP32 on {p}")

            any_ok = False
            for ch in laser_channels:
                # fire this laser, capture, then turn it off
                if not rc.set_channel(ch, True):
                    scanner_db.record_instrument(
                        scan_id, f"laser_ch{ch}", "failed",
                        detail=f"could not turn CH{ch} ON", db=d)
                    continue
                time.sleep(0.3)  # settle
                wl = wavelengths.get(ch) or schema.LASER_WAVELENGTHS_NM.get(ch)
                out = os.path.join(scan_dir, "laser", f"las{ch}.png")
                grab = _kinect_grab(out)
                rc.set_channel(ch, False)
                if grab["ok"]:
                    any_ok = True
                    scanner_db.register_artifact(
                        scan_id, sample_id, "laser", f"laser_ch{ch}_png",
                        _rel(out), media_type="image/png",
                        size_bytes=grab.get("size_bytes"),
                        laser_state=schema.build_laser_state(ch, wavelength_nm=wl),
                        db=d)
                else:
                    scanner_db.record_instrument(
                        scan_id, f"laser_ch{ch}", "failed",
                        detail=grab["detail"], db=d)
            scanner_db.record_instrument(
                scan_id, "laser", "ok" if any_ok else "failed",
                detail="" if any_ok else "no laser frames captured", db=d)
        except Exception as exc:
            scanner_db.record_instrument(scan_id, "laser", "failed", detail=str(exc), db=d)
        finally:
            # never leave a laser on
            if rc is not None:
                try:
                    for ch in laser_channels:
                        rc.set_channel(ch, False)
                except Exception:
                    pass
                rc.disconnect()
    elif mode in ("full", "laser_only"):
        scanner_db.record_instrument(scan_id, "laser", "skipped",
                                     detail="no laser channels selected", db=d)

    # ── Plain Kinect frame (lasers off) ─────────────────────────────────────
    if wants_kinect_plain:
        out = os.path.join(scan_dir, "kinect", "color.png")
        grab = _kinect_grab(out)
        if grab["ok"]:
            scanner_db.register_artifact(
                scan_id, sample_id, "kinect", "color_png", _rel(out),
                media_type="image/png", size_bytes=grab.get("size_bytes"), db=d)
            scanner_db.record_instrument(scan_id, "kinect", "ok", db=d)
        else:
            scanner_db.record_instrument(scan_id, "kinect", "failed",
                                         detail=grab["detail"], db=d)

    # ── Projector / fringe (separate pipeline; recorded as skipped for now) ──
    if wants_projector:
        scanner_db.record_instrument(
            scan_id, "projector", "skipped",
            detail="fringe capture pipeline not wired into this demo yet", db=d)

    scanner_db.finish_scan(scan_id, db=d)
    return scanner_db.scan_package(scan_id, db=d)
