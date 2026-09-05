"""Timestamped Kinect burst capture across a laser transition, run UNDER C:/KinectEnv.

WHY THIS EXISTS
---------------
``kinect_grab_once.py`` opens the Kinect, waits 1.5 s for the sensor to spin up,
grabs ONE frame and closes it again. capture.py shells out to it per laser
channel, so a laser capture is a single steady-state frame and every transient
is discarded. Dr. Zhang has twice asked for TEMPORAL / transitional laser data:
the rise, the decay and the settling are the material-discriminative signal.
That is impossible with a process-per-frame design.

This script instead opens the Kinect ONCE and streams frames into memory with a
timestamp per frame, optionally driving a laser relay from the SAME process so
the laser edge and the frame clock share one time base. Output is a frame stack
plus the laser event log, so rise/decay curves can be fit downstream.

SAFETY
------
- Lasers are OFF by default. Nothing is driven unless --laser-channel is given.
- Only RELAY channels are supported (digital on/off). The high-power 455 nm
  PWM path on GPIO21 is deliberately NOT touched here; relays must never be PWM'd.
- Every exit path, including exceptions and Ctrl-C, drives the channel OFF and
  issues SAFE before closing the port.
- Requires pyserial in C:/KinectEnv (installed 2026-08-26).

USAGE
    C:/KinectEnv/Scripts/python.exe kinect_burst_capture.py --out <dir> [options]

    # camera only, no laser, 3 s of frames (safe to run unattended)
    ... --out data/burst_test --duration 3.0

    # fire relay channel 1 from t=1.0 s to t=2.0 s inside a 3 s burst
    ... --out data/burst_ch1 --duration 3.0 \
        --laser-port COM3 --laser-channel 1 --laser-on-at 1.0 --laser-off-at 2.0

OUTPUT (in --out)
    burst.npz     frames (N,h,w,3) uint8 BGR, t_frame (N,) seconds from start,
                  mean_roi (N,) float32 mean ROI intensity per frame
    burst.json    metadata: timings, laser event log, ROI, frame rate achieved
    preview.png   first frame, for a quick sanity look
"""

import argparse
import json
import os
import sys
import threading
import time

import numpy as np
import cv2
from pykinect2 import PyKinectV2, PyKinectRuntime


# Default scan-zone crop as fractions (x0, y0, x1, y1) of the full color frame.
# Matches the fixed central ROI capture.py uses: the sample sits where the
# projector and lasers converge, so this isolates it and keeps the frame stack
# small enough to hold a multi-second burst in RAM.
DEFAULT_ROI = (0.30, 0.25, 0.72, 0.80)

SENSOR_WARMUP_S = 1.5       # same warmup kinect_grab_once.py uses
FIRST_FRAME_TIMEOUT_S = 12.0  # how long to wait for the sensor to start streaming
POLL_SLEEP_S = 0.002        # poll well above the ~30 fps frame rate, do not alias


def _parse_roi(text):
    parts = [float(p) for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be x0,y0,x1,y1 as fractions")
    x0, y0, x1, y1 = parts
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError(f"ROI fractions out of order or out of range: {parts}")
    return x0, y0, x1, y1


class _Laser:
    """In-process relay driver, so laser edges share the frame clock.

    Wraps the project's RelayController rather than re-speaking the serial
    protocol. An earlier version of this class reimplemented PING/SET with a
    single readline() and failed against a perfectly healthy board, because the
    firmware's reply is not always the first line back. RelayController already
    reads the first line on the full timeout and then drains follow-ups, so
    reuse it and keep one source of truth for the protocol.

    Imported by path from lib_3dai rather than as a package, to avoid pulling in
    the projector/Kinect modules in lib_3dai/__init__ while this process already
    owns the sensor.
    """

    def __init__(self, port, channel):
        import sys as _sys
        _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _sys.path.insert(0, os.path.join(_repo, "lib_3dai"))
        from relay_controller import RelayController

        self.channel = channel
        self.events = []
        self._threads = []
        # Opening the port DTR-resets the ESP32, and a run that follows hard
        # on the heels of a previous run's disconnect can catch the board
        # mid-boot: observed as an intermittent no-PONG on an otherwise
        # healthy board. Retry a couple of times before declaring it dead.
        self._rc = RelayController(port)
        for attempt in range(3):
            if self._rc.connect():
                break
            self._rc.disconnect()
            time.sleep(1.0)
        else:
            raise RuntimeError(f"no PONG from relay firmware on {port} after 3 tries")

    def set(self, on, t_rel):
        """Drive the channel WITHOUT blocking the caller.

        The serial round-trip takes ~150 ms. Done inline it stalls the frame
        loop and drops ~5 frames at exactly the laser edge, which are the
        frames the transient lives in (measured: 170 ms timestamp holes at
        both edges of the first live burst). So the command runs on a thread;
        the event is appended immediately and the ack fields are filled in
        when the reply lands. join_pending() waits for the threads.
        """
        event = {
            "action": "ON" if on else "OFF",
            "channel": self.channel,
            "t_scheduled": t_rel,
            "t_sent": time.perf_counter(),
            "t_acked": None,
            "ok": None,
        }
        self.events.append(event)

        def _worker():
            ok = self._rc.set_channel(self.channel, on)
            event["t_acked"] = time.perf_counter()
            event["ok"] = bool(ok)

        th = threading.Thread(target=_worker, daemon=True)
        self._threads.append(th)
        th.start()

    def join_pending(self, timeout=5.0):
        """Wait for in-flight laser commands (call after the burst loop)."""
        for th in self._threads:
            th.join(timeout)

    def shutdown(self):
        """Always drive OFF and SAFE, whatever happened."""
        try:
            self._rc.set_channel(self.channel, False)
            self._rc.safe_all()
        except Exception:
            pass
        finally:
            self._rc.disconnect()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--duration", type=float, default=3.0,
                    help="seconds of frames to record (default 3.0)")
    ap.add_argument("--roi", default=None,
                    help="crop as x0,y0,x1,y1 fractions (default central scan zone)")
    ap.add_argument("--full-frame", action="store_true",
                    help="store full 1920x1080 frames. WARNING ~8 MB/frame, "
                         "a 3 s burst is >2 GB. Use only for short bursts.")
    ap.add_argument("--laser-port", default=None, help="e.g. COM3")
    ap.add_argument("--laser-channel", type=int, default=None,
                    help="relay channel 1-4 to fire. Omitted = no laser at all.")
    ap.add_argument("--laser-on-at", type=float, default=1.0,
                    help="seconds from start to switch the laser ON")
    ap.add_argument("--laser-off-at", type=float, default=2.0,
                    help="seconds from start to switch the laser OFF")
    args = ap.parse_args(argv)

    if args.laser_channel is not None and args.laser_port is None:
        print("ERR --laser-channel requires --laser-port")
        return 2
    if args.laser_channel is not None and not (1 <= args.laser_channel <= 4):
        print("ERR --laser-channel must be 1-4 (relay channels only)")
        return 2
    if args.laser_on_at >= args.laser_off_at:
        print("ERR --laser-on-at must be before --laser-off-at")
        return 2
    if args.laser_channel is not None and args.laser_off_at > args.duration:
        print("ERR --laser-off-at is after --duration; laser would still be on at the end")
        return 2

    os.makedirs(args.out, exist_ok=True)
    roi = None if args.full_frame else _parse_roi(args.roi) if args.roi else DEFAULT_ROI

    # Open Color AND Depth. Color-only was tried and delivered no frames at all
    # on this rig; the working single-grab path opens both, so match it.
    kinect = PyKinectRuntime.PyKinectRuntime(
        PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth
    )
    laser = None
    try:
        time.sleep(SENSOR_WARMUP_S)
        h = kinect.color_frame_desc.Height
        w = kinect.color_frame_desc.Width

        if roi is None:
            x0 = y0 = 0
            x1, y1 = w, h
        else:
            x0, y0 = int(roi[0] * w), int(roi[1] * h)
            x1, y1 = int(roi[2] * w), int(roi[3] * h)
        rh, rw = y1 - y0, x1 - x0

        # Preallocate. Kinect color is ~30 fps; allow headroom so a fast sensor
        # never overruns the buffer mid-burst.
        max_frames = int(args.duration * 40) + 20
        mb = max_frames * rh * rw * 3 / 1e6
        print(f"INFO roi={rw}x{rh} max_frames={max_frames} buffer~{mb:.0f}MB")
        frames = np.zeros((max_frames, rh, rw, 3), dtype=np.uint8)
        t_frame = np.zeros(max_frames, dtype=np.float64)

        if args.laser_channel is not None:
            laser = _Laser(args.laser_port, args.laser_channel)
            print(f"INFO laser relay ch{args.laser_channel} ready on {args.laser_port}")

        # Wait for the sensor to actually start delivering before starting the
        # burst clock. The warmup sleep alone is not enough: the first frame can
        # take seconds, and starting the clock early burns the whole window
        # waiting, so the burst ends with nothing captured.
        first_deadline = time.perf_counter() + FIRST_FRAME_TIMEOUT_S
        while not kinect.has_new_color_frame():
            if time.perf_counter() > first_deadline:
                print(f"ERR no color frame within {FIRST_FRAME_TIMEOUT_S:.0f}s "
                      "(check power brick + USB3)")
                return 2
            time.sleep(POLL_SLEEP_S)
        kinect.get_last_color_frame()   # discard; it only proves frames flow

        # ── the burst loop: keep it tight, do no work that can wait ──────────
        n = 0
        fired_on = fired_off = False
        t0 = time.perf_counter()
        while True:
            t = time.perf_counter() - t0
            if t >= args.duration or n >= max_frames:
                break

            if laser is not None:
                if not fired_on and t >= args.laser_on_at:
                    laser.set(True, args.laser_on_at); fired_on = True
                elif fired_on and not fired_off and t >= args.laser_off_at:
                    laser.set(False, args.laser_off_at); fired_off = True

            if kinect.has_new_color_frame():
                cf = kinect.get_last_color_frame().reshape((h, w, 4))
                frames[n] = cf[y0:y1, x0:x1, :3]      # BGRA -> BGR, cropped
                t_frame[n] = t
                n += 1
            else:
                time.sleep(POLL_SLEEP_S)

        elapsed = time.perf_counter() - t0
    finally:
        if laser is not None:
            laser.join_pending()   # let in-flight commands land before OFF/SAFE
            laser.shutdown()
        kinect.close()

    if n == 0:
        print("ERR no color frames captured (check power brick + USB3)")
        return 2

    frames = frames[:n]
    t_frame = t_frame[:n]
    # Mean ROI intensity per frame: the rise/decay curve, cheap to compute here
    # and the first thing any downstream fit wants.
    mean_roi = frames.reshape(n, -1).mean(axis=1).astype(np.float32)

    fps = n / elapsed if elapsed > 0 else 0.0
    gaps = np.diff(t_frame) if n > 1 else np.array([0.0])

    np.savez_compressed(os.path.join(args.out, "burst.npz"),
                        frames=frames, t_frame=t_frame, mean_roi=mean_roi)
    cv2.imwrite(os.path.join(args.out, "preview.png"), frames[0])

    meta = {
        "n_frames": int(n),
        "duration_requested_s": args.duration,
        "duration_actual_s": round(elapsed, 4),
        "fps_achieved": round(fps, 2),
        "frame_interval_ms": {
            "median": round(float(np.median(gaps)) * 1000, 2),
            "min": round(float(gaps.min()) * 1000, 2),
            "max": round(float(gaps.max()) * 1000, 2),
        },
        "roi_fractions": list(roi) if roi else None,
        "roi_pixels": {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "w": rw, "h": rh},
        "laser": None if laser is None else {
            "channel": args.laser_channel,
            "port": args.laser_port,
            "events": laser.events,
        },
    }
    with open(os.path.join(args.out, "burst.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"OK {n} frames in {elapsed:.2f}s "
          f"({fps:.1f} fps, median gap {meta['frame_interval_ms']['median']}ms) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
