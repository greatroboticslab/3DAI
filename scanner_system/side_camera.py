"""Find and check the optional side camera used for readable laser images.

    python -m scanner_system.side_camera --list
    python -m scanner_system.side_camera --test 1 --exposure -8 --out check.png

--list  prints every video device Windows exposes (index, resolution).
--test  opens one device, applies the manual exposure/gain if given, grabs a
        frame to --out so you can see where it points, and reports whether
        the device actually accepted manual exposure (a fixed-auto webcam
        answers False to every setting; a camera behind an HDMI capture
        stick also answers False but honours the exposure set on its body).

Runs under the Kinect interpreter because that OpenCV build has the
DirectShow backend; this venv's headless build does not.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

_PROBE = r'''
import sys, json, time, cv2
mode = sys.argv[1]
if mode == "list":
    found = []
    for i in range(6):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            found.append({"index": i, "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                          "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})
        cap.release()
    print(json.dumps(found))
else:
    idx, exposure, gain, out = int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    rep = {"opened": cap.isOpened()}
    if cap.isOpened():
        for _ in range(5): cap.read()
        if exposure != "none":
            rep["manual_exposure_accepted"] = bool(cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                                                   and cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure)))
        if gain != "none":
            rep["gain_accepted"] = bool(cap.set(cv2.CAP_PROP_GAIN, float(gain)))
        time.sleep(0.5)
        for _ in range(8): cap.read()
        ok, f = cap.read()
        if ok:
            cv2.imwrite(out, f)
            rep.update({"frame": out, "width": int(f.shape[1]), "height": int(f.shape[0]),
                        "mean_brightness": round(float(f.mean()), 1)})
        rep["reported_exposure"] = cap.get(cv2.CAP_PROP_EXPOSURE)
        cap.release()
    print(json.dumps(rep))
'''


def _kinect_python() -> str:
    from .capture import KINECT_PYTHON
    return KINECT_PYTHON


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--test", type=int, default=None, help="device index to test")
    ap.add_argument("--exposure", default=None, help="DirectShow log2 seconds, e.g. -8")
    ap.add_argument("--gain", default=None)
    ap.add_argument("--out", default="side_camera_check.png")
    args = ap.parse_args(argv)
    py = _kinect_python()
    if not os.path.isfile(py):
        print(f"ERR Kinect interpreter not found at {py}")
        return 2
    if args.list:
        cmd = [py, "-c", _PROBE, "list"]
    elif args.test is not None:
        cmd = [py, "-c", _PROBE, "test", str(args.test), args.exposure or "none",
               args.gain or "none", os.path.abspath(args.out)]
    else:
        ap.print_help()
        return 2
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    lines = [ln for ln in (proc.stdout or "").splitlines() if ln.startswith(("[", "{"))]
    print(lines[-1] if lines else (proc.stderr or "").strip()[-300:] or "no output")
    return 0 if proc.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
