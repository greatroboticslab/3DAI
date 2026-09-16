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


def _grab_depth(k, timeout=2.0):
    """Latest depth frame as uint16 millimetres (512x424), or None."""
    h, w = k.depth_frame_desc.Height, k.depth_frame_desc.Width
    deadline = time.time() + timeout
    while time.time() < deadline:
        if k.has_new_depth_frame():
            return k.get_last_depth_frame().reshape((h, w)).astype(np.uint16)
        time.sleep(0.003)
    return None


def _trace(k, seconds, x0, y0, x1, y1):
    """Per-frame intensity trace over the scan crop for `seconds`.

    Returns (t, mean, top1) arrays: time since call, mean crop intensity, and
    the mean of the brightest 1% of crop pixels (tracks the laser spot without
    knowing where it is yet). This is the on/off transient Dr. Zhang asked
    for, recorded honestly: at ~30 fps it resolves relay timing and the
    camera's own auto-exposure settling, not sub-frame material physics.
    """
    h, w = k.color_frame_desc.Height, k.color_frame_desc.Width
    t0 = time.perf_counter(); ts, means, tops = [], [], []
    while time.perf_counter() - t0 < seconds:
        if k.has_new_color_frame():
            f = k.get_last_color_frame().reshape((h, w, 4))[y0:y1, x0:x1, :3]
            g = f.mean(axis=2)
            ts.append(time.perf_counter() - t0); means.append(float(g.mean()))
            tops.append(float(np.partition(g.ravel(), -max(1, g.size // 100))[-max(1, g.size // 100):].mean()))
        else:
            time.sleep(0.002)
    return np.array(ts), np.array(means, np.float32), np.array(tops, np.float32)


def _wait_frame(k, timeout=12.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if k.has_new_color_frame():
            return True
        time.sleep(0.003)
    return False


class SideCam:
    """Optional second camera (any UVC device: a webcam, or a 4K camera
    behind an HDMI capture stick) grabbed right after each Kinect frame.

    Unlike the Kinect it takes a MANUAL exposure, which is the whole point:
    the laser spot can be exposed so its core does not clip and the
    object stays visible. Exposure is the DirectShow log2-seconds scale
    (-8 = 1/256 s); a capture stick ignores it and uses the camera body's
    own manual setting. Any failure leaves ``ok`` False and the sequence
    carries on with the Kinect alone.
    """

    def __init__(self, index, size=None, exposure=None, gain=None):
        self.ok = False
        self.state = {}
        try:
            self.cap = cv2.VideoCapture(int(index), cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                return
            if size:
                w, h = (int(v) for v in size.lower().split("x"))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            if exposure is not None:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # DSHOW: manual
                self.cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
            if gain is not None:
                self.cap.set(cv2.CAP_PROP_GAIN, float(gain))
            for _ in range(6):            # flush frames taken under old settings
                self.cap.read()
            self.state = {
                "index": int(index),
                "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "auto_exposure": self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
                "exposure": self.cap.get(cv2.CAP_PROP_EXPOSURE),
                "gain": self.cap.get(cv2.CAP_PROP_GAIN),
            }
            self.ok = True
        except Exception as exc:          # a missing camera must not stop a scan
            self.state = {"error": f"{type(exc).__name__}: {exc}"}

    def grab(self):
        if not self.ok:
            return None
        for _ in range(2):                # drop buffered frames from before the laser edge
            self.cap.grab()
        okf, frame = self.cap.read()
        return frame if okf else None

    def release(self):
        if getattr(self, "cap", None) is not None:
            try:
                self.cap.release()
            except Exception:
                pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("channels", help="comma list, e.g. 1,2,3,4")
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--settle", type=float, default=0.4,
                    help="seconds between firing a laser and grabbing")
    ap.add_argument("--edge", type=float, default=1.0,
                    help="seconds of intensity trace to record across each on/off edge")
    ap.add_argument("--roi", default="0.12,0.05,0.65,0.72",
                    help="crop fractions x0,y0,x1,y1 for the transient trace")
    ap.add_argument("--cam", type=int, default=None,
                    help="UVC index of a second camera to grab alongside the Kinect")
    ap.add_argument("--cam-size", default=None, help="e.g. 1920x1080 (default: device default)")
    ap.add_argument("--cam-exposure", type=float, default=None,
                    help="manual exposure, DirectShow log2 seconds (e.g. -8)")
    ap.add_argument("--cam-gain", type=float, default=None)
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
    cam = SideCam(args.cam, args.cam_size, args.cam_exposure, args.cam_gain) if args.cam is not None else None
    if cam is not None:
        exposure_log["cam"] = dict(cam.state, ok=cam.ok)
        print("OK cam" if cam.ok else f"WARN cam {args.cam} not available: {cam.state}")
    try:
        time.sleep(1.5)
        if not _wait_frame(k):
            print("ERR no color frame in 12s (check power brick + USB3)")
            return 2
        _grab(k)  # discard the first, proves frames flow
        H, W = k.color_frame_desc.Height, k.color_frame_desc.Width
        fx0, fy0, fx1, fy1 = (float(v) for v in args.roi.split(","))
        cx0, cy0, cx1, cy1 = int(fx0 * W), int(fy0 * H), int(fx1 * W), int(fy1 * H)
        transient = {}

        # Absolute depth from the Kinect's time-of-flight sensor: metric 3D
        # shape independent of the projector calibration. Lasers do not
        # matter for it; one frame per scan.
        depth = _grab_depth(k)
        if depth is not None:
            dpath = os.path.join(args.out_dir, "depth.png")
            cv2.imwrite(dpath, depth)
            print(f"OK depth {dpath}")

        # Dark reference first, all lasers off, same session as the lit frames.
        rc.safe_all()
        time.sleep(args.settle)
        dark, e, g = _grab(k)
        dark_path = os.path.join(args.out_dir, "dark.png")
        cv2.imwrite(dark_path, dark)
        exposure_log["dark"] = {"exposure_100ns": e, "gain": g}
        print(f"OK dark {dark_path}")
        if cam is not None and cam.ok:
            cf = cam.grab()
            if cf is not None:
                cpath = os.path.join(args.out_dir, "cam_dark.png")
                cv2.imwrite(cpath, cf)
                print(f"OK cam_dark {cpath}")
        ir = _grab_ir(k)
        if ir is not None:
            ir_path = os.path.join(args.out_dir, "dark_ir.png")
            cv2.imwrite(ir_path, ir)
            print(f"OK dark_ir {ir_path}")

        for ch in channels:
            # ON edge: start recording, then fire, so the step is inside the trace.
            t_on = _trace_start = time.perf_counter()
            pre_t, pre_m, pre_p = _trace(k, min(0.25, args.edge / 4), cx0, cy0, cx1, cy1)
            if not rc.set_channel(ch, True):
                print(f"ERR ch{ch} could not turn ON")
                continue
            on_t, on_m, on_p = _trace(k, args.edge, cx0, cy0, cx1, cy1)
            time.sleep(max(0.0, args.settle - 0.1))
            lit, e, g = _grab(k)
            ir = _grab_ir(k)             # same lit moment, infrared sensor
            cf = cam.grab() if (cam is not None and cam.ok) else None
            # OFF edge
            off0 = time.perf_counter()
            rc.set_channel(ch, False)
            off_t, off_m, off_p = _trace(k, args.edge, cx0, cy0, cx1, cy1)
            transient[f"ch{ch}"] = {
                "pre_t": pre_t, "pre_mean": pre_m, "pre_top1": pre_p,
                "on_t": on_t, "on_mean": on_m, "on_top1": on_p,
                "off_t": off_t, "off_mean": off_m, "off_top1": off_p,
            }
            out = os.path.join(args.out_dir, f"las{ch}.png")
            cv2.imwrite(out, lit)
            exposure_log[f"ch{ch}"] = {"exposure_100ns": e, "gain": g}
            print(f"OK ch{ch} {out}")
            if ir is not None:
                ir_path = os.path.join(args.out_dir, f"las{ch}_ir.png")
                cv2.imwrite(ir_path, ir)
                print(f"OK ch{ch}_ir {ir_path}")
            if cf is not None:
                cpath = os.path.join(args.out_dir, f"cam_las{ch}.png")
                cv2.imwrite(cpath, cf)
                print(f"OK cam_ch{ch} {cpath}")
    finally:
        try:
            rc.safe_all()
        finally:
            rc.disconnect()
            k.close()
            if cam is not None:
                cam.release()
        with open(os.path.join(args.out_dir, "exposure.json"), "w") as f:
            json.dump(exposure_log, f, indent=2)
        if transient:
            flat = {f"{ch}_{k}": v for ch, d in transient.items() for k, v in d.items()}
            tpath = os.path.join(args.out_dir, "transient.npz")
            np.savez_compressed(tpath, **flat)
            print(f"OK transient {tpath}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
