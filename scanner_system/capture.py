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


# The Kinect needs the patched C:/KinectEnv interpreter (has pykinect2); this
# code runs in the project venv, which does not. So we shell out to a tiny
# standalone grab script under that interpreter. Override the interpreter path
# with SCANNER_KINECT_PYTHON if it lives elsewhere.
KINECT_PYTHON = os.getenv("SCANNER_KINECT_PYTHON", "").strip() or r"C:\KinectEnv\Scripts\python.exe"
_GRAB_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kinect_grab_once.py")
_PROJECT_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "project_solid.py")


def _start_projector_black():
    """Launch a background process that holds the projector fully black.

    Returns the Popen handle (or None if it couldn't start). Used during the
    laser stage so the lasers are the only light. Import-local so a missing
    display never breaks a capture.
    """
    # Use the KinectEnv interpreter: it has GUI-capable OpenCV (the project venv
    # ships headless OpenCV, which can't open a window).
    import subprocess
    try:
        return subprocess.Popen([KINECT_PYTHON, _PROJECT_SCRIPT, "0", "120"])
    except Exception:
        return None


def _stop_projector(proc):
    if proc is not None:
        try:
            proc.terminate()
        except Exception:
            pass


def _kinect_grab(out_path: str) -> dict[str, Any]:
    """Capture one Kinect color frame to out_path via the KinectEnv interpreter.

    Returns {ok, detail, size_bytes?}. Never raises; a missing interpreter,
    sensor, or driver comes back as ok=False with a plain-language reason.
    """
    import subprocess

    if not os.path.isfile(KINECT_PYTHON):
        return {"ok": False, "detail": f"Kinect interpreter not found at {KINECT_PYTHON} "
                                       "(set SCANNER_KINECT_PYTHON)."}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        proc = subprocess.run(
            [KINECT_PYTHON, _GRAB_SCRIPT, out_path],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "detail": "Kinect grab timed out (30s)."}
    except Exception as exc:
        return {"ok": False, "detail": f"Kinect grab failed to launch: {exc}"}

    tail = (proc.stdout or "").strip().splitlines()[-1:] or [""]
    if proc.returncode == 0 and os.path.isfile(out_path):
        return {"ok": True, "detail": "captured", "size_bytes": os.path.getsize(out_path)}
    return {"ok": False, "detail": tail[0] or (proc.stderr or "").strip()[-200:] or "Kinect grab failed."}


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
        # Projector to BLACK for the whole laser stage so the lasers are the only
        # light on the sample (its normal image otherwise washes the scene out).
        projector = _start_projector_black()
        time.sleep(0.8)  # let the black window come up before firing
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
            _stop_projector(projector)  # restore the projector
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

    # ── Projector / fringe: structured-light 3D stage ───────────────────────
    # Projects a multi-frequency fringe sequence and captures each with the
    # Kinect (both need the KinectEnv interpreter, so it runs there). We register
    # the white-illumination photo and the fringe stack; 3D reconstruction is a
    # separate step and needs current calibration.
    if wants_projector:
        import subprocess
        fringe_dir = os.path.join(scan_dir, "fringe")
        cap_script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "scan_test", "capture_multifreq.py")
        maxval = os.getenv("SCANNER_PROJECTOR_MAXVAL", "180")  # bright room default
        try:
            proc = subprocess.run(
                [KINECT_PYTHON, cap_script, fringe_dir, str(maxval)],
                capture_output=True, text=True, timeout=180)
            white = os.path.join(fringe_dir, "white.png")
            npz = os.path.join(fringe_dir, "scan.npz")
            if proc.returncode == 0 and os.path.isfile(white):
                scanner_db.register_artifact(
                    scan_id, sample_id, "projector", "fringe_white_png",
                    _rel(white), media_type="image/png",
                    size_bytes=os.path.getsize(white), db=d)
                if os.path.isfile(npz):
                    scanner_db.register_artifact(
                        scan_id, sample_id, "projector", "fringe_stack_npz",
                        _rel(npz), media_type="application/x-npz",
                        size_bytes=os.path.getsize(npz), db=d)
                scanner_db.record_instrument(scan_id, "projector", "ok", db=d)
            else:
                detail = (proc.stderr or proc.stdout or "").strip()[-200:] or "fringe capture failed"
                scanner_db.record_instrument(scan_id, "projector", "failed", detail=detail, db=d)
        except subprocess.TimeoutExpired:
            scanner_db.record_instrument(scan_id, "projector", "failed",
                                         detail="fringe capture timed out (180s)", db=d)
        except Exception as exc:
            scanner_db.record_instrument(scan_id, "projector", "failed", detail=str(exc), db=d)

    scanner_db.finish_scan(scan_id, db=d)
    return scanner_db.scan_package(scan_id, db=d)
