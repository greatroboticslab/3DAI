"""Single-session laser capture: dark + every laser frame from ONE camera open.

WHY THIS EXISTS
---------------
capture.py used to shell out to kinect_grab_once.py once per laser channel, plus
once for the dark frame. Each call opens the Kinect fresh, waits ~1.5 s, and
lets auto-exposure re-converge from scratch, so the frames of one scan do not
share an exposure state. That leaves a background offset between the dark frame
and the laser frames (measured: about -3 to -15 per channel), which contaminates
the laser-minus-dark material signal, worst on the faint channels. Auto-exposure
cannot be locked on the Kinect V2 (the SDK exposes ExposureTime/Gain read-only),
so the fix is not to disable it but to stop re-triggering it: hold the camera
open and capture the whole sequence in one session, where exposure stays put and
the subtraction comes out clean (measured background drift ~0).

Runs under C:/KinectEnv. Fires ONLY relay channels (digital on/off); the GPIO21
PWM laser is never touched. Every exit path drives the fired channel off and
issues SAFE.

USAGE (invoked by capture.py, not by hand):
    kinect_laser_sequence.py <out_dir> <ch,ch,...> [--port COM3]

Writes into <out_dir>: dark.png, las<ch>.png for each channel, plus the
Kinect INFRARED camera's view of the same moments as dark_ir.png and
las<ch>_ir.png (512x424, 16-bit). The IR frames exist because CH3 is a
near-infrared laser: the color camera is filtered against it and records a
smudge, while the IR sensor (near 860 nm) sees it as brightly as the visible
lasers. exposure.json records per-frame color ExposureTime + Gain (metadata;
the drift fix is the single session, not a post-hoc division). Prints one
OK/ERR line per frame so capture.py can register artifacts honestly.

Requires the 2026-09-05 pykinect2 patch that wires up the infrared source
(PyKinectRuntime.py.bak-20260905-infrared is the pre-patch copy).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import cv2
from pykinect2 import PyKinectV2, PyKinectRuntime

sys.path.insert(0, r"C:\Users\Robotics_Lab\3DAI\3DAI\lib_3dai")


def _grab(k):
    """Latest color frame as BGR float, plus (exposure_100ns, gain) from the
    same frame when the settings interface is reachable (None otherwise)."""
    h, w = k.color_frame_desc.Height, k.color_frame_desc.Width
    exp = gain = None
    try:
        fr = k._color_frame_reader.AcquireLatestFrame()
        s = fr.ColorCameraSettings
        exp, gain = int(s.ExposureTime), float(s.Gain)
    except Exception:
        pass
    buf = k.get_last_color_frame().reshape((h, w, 4))[:, :, :3]
    return buf.copy(), exp, gain


def _grab_ir(k, timeout=2.0):
    """Latest infrared frame as uint16 (512x424), or None if none arrives."""
    h, w = k.infrared_frame_desc.Height, k.infrared_frame_desc.Width
    deadline = time.time() + timeout
    while time.time() < deadline:
        if k.has_new_infrared_frame():
            return k.get_last_infrared_frame().reshape((h, w)).astype(np.uint16)
        time.sleep(0.003)
    return None


def _wait_frame(k, timeout=12.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if k.has_new_color_frame():
            return True
        time.sleep(0.003)
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("channels", help="comma list, e.g. 1,2,3,4")
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--settle", type=float, default=0.4,
                    help="seconds between firing a laser and grabbing")
    args = ap.parse_args(argv)

    channels = [int(c) for c in args.channels.split(",") if c.strip()]
    if any(not (1 <= c <= 4) for c in channels):
        print("ERR channels must be 1-4 (relay channels only)")
        return 2
    os.makedirs(args.out_dir, exist_ok=True)

    from relay_controller import RelayController

    rc = None
    for _ in range(4):
        rc = RelayController(args.port)
        if rc.connect():
            break
        rc.disconnect(); rc = None
        time.sleep(1.5)
    if rc is None:
        print("ERR no PONG from relay board")
        return 2

    # One camera open for the whole sequence: this is the fix.
    k = PyKinectRuntime.PyKinectRuntime(
        PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth
        | PyKinectV2.FrameSourceTypes_Infrared)
    exposure_log = {}
    try:
        time.sleep(1.5)
        if not _wait_frame(k):
            print("ERR no color frame in 12s (check power brick + USB3)")
            return 2
        _grab(k)  # discard the first, proves frames flow

        # Dark reference first, all lasers off, same session as the lit frames.
        rc.safe_all()
        time.sleep(args.settle)
        dark, e, g = _grab(k)
        dark_path = os.path.join(args.out_dir, "dark.png")
        cv2.imwrite(dark_path, dark)
        exposure_log["dark"] = {"exposure_100ns": e, "gain": g}
        print(f"OK dark {dark_path}")
        ir = _grab_ir(k)
        if ir is not None:
            ir_path = os.path.join(args.out_dir, "dark_ir.png")
            cv2.imwrite(ir_path, ir)
            print(f"OK dark_ir {ir_path}")

        for ch in channels:
            if not rc.set_channel(ch, True):
                print(f"ERR ch{ch} could not turn ON")
                continue
            time.sleep(args.settle)
            lit, e, g = _grab(k)
            ir = _grab_ir(k)             # same lit moment, infrared sensor
            rc.set_channel(ch, False)
            out = os.path.join(args.out_dir, f"las{ch}.png")
            cv2.imwrite(out, lit)
            exposure_log[f"ch{ch}"] = {"exposure_100ns": e, "gain": g}
            print(f"OK ch{ch} {out}")
            if ir is not None:
                ir_path = os.path.join(args.out_dir, f"las{ch}_ir.png")
                cv2.imwrite(ir_path, ir)
                print(f"OK ch{ch}_ir {ir_path}")
    finally:
        try:
            rc.safe_all()
        finally:
            rc.disconnect()
            k.close()
        with open(os.path.join(args.out_dir, "exposure.json"), "w") as f:
            json.dump(exposure_log, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
