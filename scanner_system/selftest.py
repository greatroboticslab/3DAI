"""Session-start laser self-test: fire each laser once and confirm it lit.

The single most common failure this week was a laser that silently stopped
firing (a loose CH4 wire, twice), discovered only after several objects had
been scanned with dead-channel data. This fires all four relay lasers once,
with the projector held black, and reads back each visible laser's spot from
the Kinect. If one did not light, it says so before any object is scanned.

Relay channels only (GPIO18/19/25/26). The GPIO21 big-laser path is never
touched. Frames go to a temp folder and are not kept.

    python -m scanner_system.selftest          # asks before firing
    python -m scanner_system.selftest --yes     # fire without asking (scan.bat)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time

from . import qc

NIR_MIN_NM = 750


def run_selftest(channels=(1, 2, 3, 4), port="COM3") -> dict:
    """Fire the lasers on whatever is on the stage; return per-channel result."""
    from . import capture, laser_features
    from .schema import LASER_WAVELENGTHS_NM

    if not os.path.isfile(capture.KINECT_PYTHON):
        return {"ok": False, "detail": f"Kinect interpreter not found at {capture.KINECT_PYTHON}"}

    tmp = tempfile.mkdtemp(prefix="selftest_")
    proj = None
    try:
        proj = capture._start_projector_black()
        time.sleep(1.0)
        proc = subprocess.run(
            [capture.KINECT_PYTHON, capture._LASER_SEQ_SCRIPT, tmp,
             ",".join(str(c) for c in channels), "--port", port],
            capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return {"ok": False, "detail": "laser sequence timed out"}
    finally:
        capture._stop_projector(proj)
    if proc.returncode != 0:
        return {"ok": False, "detail": (proc.stderr or proc.stdout or "").strip()[-200:]}

    feats = laser_features.compute_features(tmp, channels=tuple(channels))
    if not feats:
        return {"ok": False, "detail": "no laser features (dark frame missing?)"}

    results = {}
    all_fired = True
    for ch in channels:
        f = (feats.get("channels") or {}).get(str(ch)) or {}
        wl = LASER_WAVELENGTHS_NM.get(ch)
        core = f.get("core")
        if wl is not None and wl >= NIR_MIN_NM:
            # NIR: report the IR-sensor core, do not pass/fail on it.
            results[ch] = {"nir": True, "ir_core": f.get("ir_core"), "fired": None}
        else:
            fired = core is not None and core >= qc.CORE_FIRED_MIN
            results[ch] = {"nir": False, "core": core, "fired": fired}
            all_fired = all_fired and fired
    return {"ok": all_fired, "channels": results}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true", help="fire without the confirmation prompt")
    ap.add_argument("--port", default="COM3")
    args = ap.parse_args(argv)

    print("Laser self-test: fires each laser once (relay channels only).")
    if not args.yes:
        ans = input(">>> Projector on, lights off. Press Enter to fire, or 's' to skip: ").strip().lower()
        if ans in ("s", "skip"):
            print("  skipped.")
            return 0

    res = run_selftest(port=args.port)
    if not res.get("ok") and "channels" not in res:
        print(f"  [XX] self-test could not run: {res.get('detail')}")
        return 1

    dead = []
    for ch, r in sorted(res["channels"].items()):
        if r["nir"]:
            print(f"  [OK] CH{ch} (near-infrared): IR core {r.get('ir_core')} "
                  "(not spot-checkable; verify with a phone camera if unsure)")
        elif r["fired"]:
            print(f"  [OK] CH{ch}: fired (spot core {r['core']:.0f})")
        else:
            print(f"  [XX] CH{ch}: NO SPOT (core {r.get('core')}) - check the laser wire/connector")
            dead.append(ch)

    if dead:
        print()
        print(f"STOP: laser channel(s) {', '.join('CH'+str(c) for c in dead)} did not light. "
              "Check the wiring, then run scan.bat again.")
        return 1
    print("  all lasers fired.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
