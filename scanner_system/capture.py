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


# Scan-zone crop (fractions x0,y0,x1,y1 of the full Kinect frame), applied to
# EVERY png artifact (laser, dark, kinect, projector white) so all images of a
# scan share one window and laser-minus-dark stays pixel-aligned. The fringe
# .npz stays full frame; it feeds reconstruction, not eyes.
#
# The default box was settled 2026-09-05 by the operator after seeing three
# candidates on real scans: fringe-amplitude footprint x 326-1124, y 111-659
# plus a generous pad, extra at the bottom for the keystone's dim far corner.
# The projector throws off-axis, so its zone is a tilted quad in the camera;
# a crop tighter than this amputates scene the operator wants visible.
#
# The default is MEASURED, not guessed, and must be re-measured whenever the
# Kinect or projector moves. Method that actually works: take a fringe
# capture's highest-frequency stack and run
# fpp_tools.temporal_unwrap.footprint_from_amplitude on it - fringes exist
# only under DIRECT projection, so the amplitude footprint is the projector
# zone. Thresholding a projector-on minus projector-off difference does NOT
# work: in a dark room the projector's spill bounces off everything and the
# "lit" box swallows half the frame, and any ambient change between the two
# frames fakes lit area too (both failure modes observed 2026-09-05).
#
# Measured 2026-09-05 after the Kinect was repositioned (and re-checked after
# a projector tilt, which moved it by only ~16 px): fringe footprint
# x 326-1124, y 111-659 in the 1920x1080 frame. The pad on top of that is
# deliberately GENEROUS, extra at the bottom: the projector throws off-axis,
# so its zone is a keystone quad whose dim far corner can fall below what the
# amplitude detector bounds, and a slightly loose presentation crop costs
# nothing while a tight one visibly amputates the scene. The pre-move default
# (0.24, 0.34, 0.52, 0.64) cut off everything left of x=460.
def _scan_roi() -> tuple[float, float, float, float]:
    raw = os.getenv("SCANNER_SCAN_ROI", "").strip()
    if raw:
        try:
            x0, y0, x1, y1 = (float(v) for v in raw.split(","))
            return x0, y0, x1, y1
        except Exception:
            pass
    # GEOMETRY_ID kinect_lowered_20260916: the projected zone in the lowered
    # Kinect's frame, with a small margin (measured 2026-09-16 from a live frame).
    return 0.07, 0.01, 0.64, 0.70


def _crop_to_roi(path: str) -> Optional[int]:
    """Crop a saved image in place to the scan-zone ROI. Returns new size in
    bytes, or None on any failure (leaving the full frame untouched)."""
    try:
        import cv2
        img = cv2.imread(path)
        if img is None:
            return None
        h, w = img.shape[:2]
        x0, y0, x1, y1 = _scan_roi()
        crop = img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
        if crop.size == 0:
            return None
        cv2.imwrite(path, crop)
        return os.path.getsize(path)
    except Exception:
        return None


# The Kinect needs the patched C:/KinectEnv interpreter (has pykinect2); this
# code runs in the project venv, which does not. So we shell out to a tiny
# standalone grab script under that interpreter. Override the interpreter path
# with SCANNER_KINECT_PYTHON if it lives elsewhere.
KINECT_PYTHON = os.getenv("SCANNER_KINECT_PYTHON", "").strip() or r"C:\KinectEnv\Scripts\python.exe"
# Physical rig layout in effect. 2026-09-16: Kinect lowered toward the table
# at Dr. Zhang's request (bigger laser spot in frame); new crop box and fringe
# reference ref20_20260916 go with it.
GEOMETRY_ID = "kinect_lowered_20260916"
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


_LASER_SEQ_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "kinect_laser_sequence.py")


def _laser_sequence(scan_dir: str, channels: list[int], port: str) -> dict[str, Any]:
    """Capture dark + every laser frame in one KinectEnv session.

    Returns {ok, captured: {"dark": path, ch: path}, errors: {ch: reason},
    detail}. Never raises; a missing sensor or board comes back ok=False.
    """
    import subprocess

    out_dir = os.path.join(scan_dir, "laser")
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.isfile(KINECT_PYTHON):
        return {"ok": False, "captured": {}, "errors": {},
                "detail": f"Kinect interpreter not found at {KINECT_PYTHON}"}
    try:
        proc = subprocess.run(
            [KINECT_PYTHON, _LASER_SEQ_SCRIPT, out_dir,
             ",".join(str(c) for c in channels), "--port", port],
            capture_output=True, text=True, timeout=90)
    except subprocess.TimeoutExpired:
        return {"ok": False, "captured": {}, "errors": {},
                "detail": "laser sequence timed out (90s)"}
    except Exception as exc:
        return {"ok": False, "captured": {}, "errors": {},
                "detail": f"laser sequence failed to launch: {exc}"}

    # Keys are the script's own tokens: "dark", "dark_ir", "ch3", "ch3_ir".
    captured: dict[str, str] = {}
    errors: dict[int, str] = {}
    for line in (proc.stdout or "").splitlines():
        parts = line.split(maxsplit=2)
        if len(parts) == 3 and parts[0] == "OK":
            captured[parts[1]] = parts[2]
        elif parts and parts[0] == "ERR":
            # channel-specific ("ERR ch3 ...") or fatal
            tok = parts[1] if len(parts) >= 2 else ""
            if tok.startswith("ch") and tok[2:].isdigit():
                errors[int(tok[2:])] = line[4:]
    # A fatal ERR (no board, no camera) leaves nothing captured.
    if not captured:
        tail = (proc.stdout or proc.stderr or "").strip().splitlines()[-1:] or ["laser sequence produced no frames"]
        return {"ok": False, "captured": {}, "errors": errors, "detail": tail[0]}
    return {"ok": True, "captured": captured, "errors": errors, "detail": ""}


def _reference_dir(repo_root: str) -> str:
    """Empty-stage fringe reference in effect (see _reconstruct_height)."""
    calib_dir = os.path.join(repo_root, "data", "scan_test", "calib_new")
    return os.getenv("SCANNER_FRINGE_REFERENCE",
                     os.path.join(calib_dir, "ref20_20260916"))


def _calibration_file(repo_root: str) -> str:
    calib_dir = os.path.join(repo_root, "data", "scan_test", "calib_new")
    return os.getenv("SCANNER_HEIGHT_CALIB",
                     os.path.join(calib_dir, "calibration_temporal_20260731.txt"))


def _git_commit(repo_root: str) -> Optional[str]:
    """Short commit of the capture code, best effort (provenance only)."""
    import subprocess
    try:
        r = subprocess.run(["git", "-C", repo_root, "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=10, check=False)
        return r.stdout.strip() or None
    except Exception:
        return None


def _reconstruct_height(scan_id, sample_id, fringe_dir, repo_root, d):
    """Robust wrapped-phase 3D reconstruction of a fringe capture -> height map.

    Best-effort post-process: subtracts the canonical empty-stage reference and
    maps delta phase through the current calibration (env SCANNER_FRINGE_REFERENCE
    / SCANNER_HEIGHT_CALIB, defaulting to the 2026-07-31 MAXVAL-20 refit). The
    median object height is metric; the per-pixel map tilts with the projector's
    shallow angle (see calib_new report). Never raises -- capture already stands
    on its own if reconstruction is unavailable.
    """
    import subprocess

    calib_dir = os.path.join(repo_root, "data", "scan_test", "calib_new")
    # ref20_20260916: empty-stage reference captured 2026-09-16 after the
    # Kinect was lowered (GEOMETRY_ID). Earlier references (ref20_20260905,
    # ref20) belong to earlier geometries; a reference from the wrong
    # geometry makes the footprint check fail and the height map garbage.
    # The per-pixel gain map pairs with the July reference and skips itself
    # (footprint mismatch), so reconstruction uses the global curve, which
    # itself needs re-fitting from flat calipered anchors in this geometry.
    ref_dir = os.getenv("SCANNER_FRINGE_REFERENCE",
                        os.path.join(calib_dir, "ref20_20260916"))
    calib_txt = os.getenv("SCANNER_HEIGHT_CALIB",
                          os.path.join(calib_dir, "calibration_temporal_20260731.txt"))
    recon_script = os.path.join(repo_root, "data", "scan_test", "reconstruct_height.py")
    # Every exit below records a "reconstruction" instrument result. This used to
    # return silently and swallow every exception, so a scan could report
    # projector=ok and status=complete while containing no 3D data at all, with
    # the subprocess stderr captured and then discarded. Six scans on 2026-07-31
    # did exactly that (they predated the calibration), and the only way to find
    # out was comparing file timestamps by hand. Reconstruction is still
    # best-effort and still never raises; it is just no longer invisible.
    if not os.path.isfile(os.path.join(ref_dir, "scan.npz")):
        scanner_db.record_instrument(
            scan_id, "reconstruction", "skipped",
            detail=f"no fringe reference at {ref_dir}", db=d)
        return
    if not os.path.isfile(calib_txt):
        scanner_db.record_instrument(
            scan_id, "reconstruction", "skipped",
            detail=f"no height calibration at {calib_txt}", db=d)
        return
    try:
        proc = subprocess.run(
            [KINECT_PYTHON, recon_script, fringe_dir, ref_dir, calib_txt, fringe_dir],
            capture_output=True, text=True, timeout=120)
        hnpy = os.path.join(fringe_dir, "height_mm.npy")
        hpng = os.path.join(fringe_dir, "height_mm.png")
        if proc.returncode != 0 or not os.path.isfile(hnpy):
            # Surface why. stderr is the only explanation that exists.
            why = (proc.stderr or proc.stdout or "").strip().splitlines()
            scanner_db.record_instrument(
                scan_id, "reconstruction", "failed",
                detail=(why[-1][:300] if why
                        else f"reconstruct_height.py exited {proc.returncode}, no height_mm.npy"),
                db=d)
            return
        scanner_db.register_artifact(
            scan_id, sample_id, "fusion", "height_map_npy",
            _rel(hnpy), media_type="application/x-npy",
            size_bytes=os.path.getsize(hnpy), db=d)
        if os.path.isfile(hpng):
            scanner_db.register_artifact(
                scan_id, sample_id, "fusion", "height_map_png",
                _rel(hpng), media_type="image/png",
                size_bytes=os.path.getsize(hpng), db=d)
        scanner_db.record_instrument(scan_id, "reconstruction", "ok", db=d)
    except subprocess.TimeoutExpired:
        scanner_db.record_instrument(
            scan_id, "reconstruction", "failed",
            detail="reconstruct_height.py timed out (120s)", db=d)
    except Exception as exc:
        scanner_db.record_instrument(
            scan_id, "reconstruction", "failed",
            detail=f"{type(exc).__name__}: {exc}"[:300], db=d)


def run_capture(
    sample_id: str,
    mode: str = "full",
    laser_channels: Optional[list[int]] = None,
    wavelengths: Optional[dict[int, int]] = None,
    operator: Optional[str] = None,
    angle: Optional[dict[str, int]] = None,
    notes: str = "",
    known_height_mm: Optional[float] = None,
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

    scan_id = scanner_db.start_scan(sample_id, mode=mode, operator=operator,
                                    angle=angle, notes=notes,
                                    known_height_mm=known_height_mm, db=d)
    scan_dir = os.path.join(STORAGE_ROOT, "scans", scan_id)

    # Provenance: everything that would change the numbers if the rig, the
    # calibration or the code changed under the collection. Without this a
    # later analysis cannot tell scan #40 from scan #140 after a bump.
    _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scanner_db.set_scan_meta(scan_id, {"capture_config": {
        # Bumped whenever the Kinect, projector or lasers physically move:
        # pixel-scale features (halo radii) and the fringe reference are only
        # comparable within one geometry.
        "geometry": os.getenv("SCANNER_GEOMETRY_ID", GEOMETRY_ID),
        "reference_dir": os.path.basename(_reference_dir(_repo)),
        "calibration_file": os.path.basename(_calibration_file(_repo)),
        # The phase-to-height curve was fitted for the July geometry; until
        # flat calipered anchors re-fit it, metric heights are approximate.
        "height_calibration_geometry": "kinect_july_2026",
        "scan_roi": list(_scan_roi()),
        "projector_maxval": int(os.getenv("SCANNER_PROJECTOR_MAXVAL", "20")),
        "wavelengths_nm": {str(k): v for k, v in schema.LASER_WAVELENGTHS_NM.items()},
        "laser_channels": list(laser_channels),
        "single_session_laser": True,
        "code_commit": _git_commit(_repo),
    }}, db=d)

    wants_laser = mode in ("full", "laser_only") and laser_channels
    wants_kinect_plain = mode in ("full", "kinect_projector", "kinect_only")
    wants_projector = mode in ("full", "kinect_projector", "projector_only")

    # Projector to BLACK for the laser stage AND the plain photo. Without the
    # hold the projector shows the Windows desktop between stages, and the
    # 2026-09-15 plain photos all carry the wallpaper projected onto the object.
    projector = None
    if wants_laser or wants_kinect_plain:
        projector = _start_projector_black()
        time.sleep(0.8)  # let the black window come up before firing

    # ── Laser modality: capture the sample under each laser ─────────────────
    if wants_laser:
        try:
            p = port or hardware.likely_esp32_port()
            if p is None:
                raise RuntimeError("no ESP32 serial port found (is it plugged in "
                                   "with a data cable?)")
            # Capture the dark reference AND every laser frame in ONE camera
            # session (kinect_laser_sequence.py). The old path shelled out a
            # fresh process per frame, so each frame re-triggered the Kinect's
            # auto-exposure and the dark/lit frames did not share an exposure
            # state -- a background offset of -3 to -15 leaked into every
            # laser-minus-dark result, worst on the faint channels. Exposure
            # cannot be locked on the Kinect V2 (SDK is read-only), so the fix
            # is to hold the camera open; measured residual drift then ~0.
            res = _laser_sequence(scan_dir, laser_channels, p)
            if not res["ok"]:
                raise RuntimeError(res["detail"])

            captured = res["captured"]      # {"dark": p, "dark_ir": p, "ch3": p, "ch3_ir": p}
            if "dark" in captured:
                dark = captured["dark"]
                dsize = _crop_to_roi(dark) or os.path.getsize(dark)
                scanner_db.register_artifact(
                    scan_id, sample_id, "laser", "laser_dark_png",
                    _rel(dark), media_type="image/png", size_bytes=dsize, db=d)
            # Infrared frames are NOT cropped: the IR sensor is 512x424 with
            # its own optics, and the scan-zone box is in color-frame
            # fractions; applying it here would cut the wrong region.
            if "dark_ir" in captured:
                p_ir = captured["dark_ir"]
                scanner_db.register_artifact(
                    scan_id, sample_id, "laser", "laser_dark_ir_png",
                    _rel(p_ir), media_type="image/png",
                    size_bytes=os.path.getsize(p_ir), db=d)
            # Kinect time-of-flight depth (uint16 mm, 512x424): metric 3D shape
            # independent of the projector calibration. Captured once per scan.
            if "depth" in captured:
                p_d = captured["depth"]
                scanner_db.register_artifact(
                    scan_id, sample_id, "kinect", "kinect_depth_png",
                    _rel(p_d), media_type="image/png",
                    size_bytes=os.path.getsize(p_d),
                    metadata={"units": "mm", "sensor": "kinect_v2_tof"}, db=d)

            any_ok = False
            for ch in laser_channels:
                if f"ch{ch}" not in captured:
                    scanner_db.record_instrument(
                        scan_id, f"laser_ch{ch}", "failed",
                        detail=res["errors"].get(ch, "not captured"), db=d)
                    continue
                any_ok = True
                out = captured[f"ch{ch}"]
                wl = wavelengths.get(ch) or schema.LASER_WAVELENGTHS_NM.get(ch)
                state = schema.build_laser_state(ch, wavelength_nm=wl)
                size = _crop_to_roi(out) or os.path.getsize(out)
                scanner_db.register_artifact(
                    scan_id, sample_id, "laser", f"laser_ch{ch}_png",
                    _rel(out), media_type="image/png", size_bytes=size,
                    laser_state=state, db=d)
                # The IR camera's view of the same lit moment. For CH3 (NIR)
                # this is the frame that actually contains the laser.
                if f"ch{ch}_ir" in captured:
                    p_ir = captured[f"ch{ch}_ir"]
                    scanner_db.register_artifact(
                        scan_id, sample_id, "laser", f"laser_ch{ch}_ir_png",
                        _rel(p_ir), media_type="image/png",
                        size_bytes=os.path.getsize(p_ir), laser_state=state, db=d)
            # Per-channel on/off intensity traces (Dr. Zhang's transient ask).
            if "transient" in captured:
                p_t = captured["transient"]
                scanner_db.register_artifact(
                    scan_id, sample_id, "laser", "laser_transient_npz",
                    _rel(p_t), media_type="application/x-npz",
                    size_bytes=os.path.getsize(p_t),
                    metadata={"note": "crop mean and top-1% intensity vs time across "
                                      "each laser's on and off edge at ~30 fps; "
                                      "resolves relay timing and auto-exposure "
                                      "settling, not sub-frame material physics"},
                    db=d)
            scanner_db.record_instrument(
                scan_id, "laser", "ok" if any_ok else "failed",
                detail="" if any_ok else "no laser frames captured", db=d)

            # The material features ARE the laser data as far as a spreadsheet
            # or a model is concerned: scatter halo, reflectance colour,
            # speckle, and the infrared equivalents. Computed here, once, from
            # the clean single-session frames, and stored on the scan.
            if any_ok:
                try:
                    from . import laser_features
                    feats = laser_features.compute_features(
                        os.path.join(scan_dir, "laser"), channels=laser_channels)
                    if feats:
                        scanner_db.set_laser_features(scan_id, feats, db=d)
                        scanner_db.record_instrument(scan_id, "laser_features", "ok", db=d)
                    else:
                        scanner_db.record_instrument(
                            scan_id, "laser_features", "failed",
                            detail="no dark frame to subtract", db=d)
                except Exception as exc:
                    scanner_db.record_instrument(
                        scan_id, "laser_features", "failed",
                        detail=f"{type(exc).__name__}: {exc}"[:300], db=d)
        except Exception as exc:
            scanner_db.record_instrument(scan_id, "laser", "failed", detail=str(exc), db=d)
    elif mode in ("full", "laser_only"):
        scanner_db.record_instrument(scan_id, "laser", "skipped",
                                     detail="no laser channels selected", db=d)

    # ── Plain Kinect frame (lasers off) ─────────────────────────────────────
    if wants_kinect_plain:
        out = os.path.join(scan_dir, "kinect", "color.png")
        grab = _kinect_grab(out)
        if grab["ok"]:
            size = _crop_to_roi(out) or grab.get("size_bytes")
            scanner_db.register_artifact(
                scan_id, sample_id, "kinect", "color_png", _rel(out),
                media_type="image/png", size_bytes=size, db=d)
            scanner_db.record_instrument(scan_id, "kinect", "ok", db=d)
        else:
            scanner_db.record_instrument(scan_id, "kinect", "failed",
                                         detail=grab["detail"], db=d)
    _stop_projector(projector)  # fringe stage drives the projector itself

    # ── Projector / fringe: structured-light 3D stage ───────────────────────
    # Projects a multi-frequency fringe sequence and captures each with the
    # Kinect (both need the KinectEnv interpreter, so it runs there). We register
    # the white-illumination photo and the fringe stack; 3D reconstruction is a
    # separate step and needs current calibration.
    if wants_projector:
        import subprocess
        fringe_dir = os.path.join(scan_dir, "fringe")
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cap_script = os.path.join(repo_root, "data", "scan_test", "capture_multifreq.py")
        # MAXVAL 20 is the calibrated exposure: higher values overexpose the
        # Kinect and wash the fringe to near-zero contrast (height reads 0).
        maxval = os.getenv("SCANNER_PROJECTOR_MAXVAL", "20")
        try:
            proc = subprocess.run(
                [KINECT_PYTHON, cap_script, fringe_dir, str(maxval)],
                capture_output=True, text=True, timeout=180)
            white = os.path.join(fringe_dir, "white.png")
            npz = os.path.join(fringe_dir, "scan.npz")
            if proc.returncode == 0 and os.path.isfile(white):
                wsize = _crop_to_roi(white) or os.path.getsize(white)
                scanner_db.register_artifact(
                    scan_id, sample_id, "projector", "fringe_white_png",
                    _rel(white), media_type="image/png",
                    size_bytes=wsize, db=d)
                if os.path.isfile(npz):
                    scanner_db.register_artifact(
                        scan_id, sample_id, "projector", "fringe_stack_npz",
                        _rel(npz), media_type="application/x-npz",
                        size_bytes=os.path.getsize(npz), db=d)
                    _reconstruct_height(scan_id, sample_id, fringe_dir, repo_root, d)
                    # Calibration-free material signal from the same stack:
                    # how well the surface holds the fringes, and its albedo.
                    try:
                        from . import fringe_features
                        ff = fringe_features.compute_fringe_features(
                            fringe_dir, _reference_dir(repo_root))
                        if ff:
                            scanner_db.set_scan_meta(scan_id, {"fringe_features": ff}, db=d)
                            scanner_db.record_instrument(scan_id, "fringe_features", "ok", db=d)
                        else:
                            scanner_db.record_instrument(
                                scan_id, "fringe_features", "failed",
                                detail="no reference/stack or no background region", db=d)
                    except Exception as exc:
                        scanner_db.record_instrument(
                            scan_id, "fringe_features", "failed",
                            detail=f"{type(exc).__name__}: {exc}"[:300], db=d)
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
