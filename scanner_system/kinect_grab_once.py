"""Standalone single-frame Kinect grab, run UNDER C:/KinectEnv.

The scanner GUI + capture orchestrator run in the project venv, which does not
have pykinect2. Real Kinect capture needs the patched C:/KinectEnv interpreter.
So the orchestrator shells out to THIS script under that interpreter to grab one
color frame and save it, then reads the file back.

Usage (invoked by capture.py, not by hand):
    C:/KinectEnv/Scripts/python.exe kinect_grab_once.py <out_color.png>

Prints "OK <path>" on success or "ERR <reason>" on failure, and exits 0/2.
Self-contained: imports only pykinect2/numpy/cv2 (present in KinectEnv). Does
not import lib_3dai (avoids pulling in projector/relay code).
"""

import os
import sys
import time

import numpy as np
import cv2
from pykinect2 import PyKinectV2, PyKinectRuntime


def main() -> int:
    if len(sys.argv) < 2:
        print("ERR no output path given")
        return 2
    out_path = sys.argv[1]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    kinect = PyKinectRuntime.PyKinectRuntime(
        PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth
    )
    time.sleep(1.5)  # let the sensor start delivering frames

    color = None
    deadline = time.time() + 12.0
    while time.time() < deadline and color is None:
        if kinect.has_new_color_frame():
            cf = kinect.get_last_color_frame()          # flat uint8 BGRA
            h, w = kinect.color_frame_desc.Height, kinect.color_frame_desc.Width
            color = cf.reshape((h, w, 4))
        time.sleep(0.03)

    kinect.close()

    if color is None:
        print("ERR no color frame in 12s (check power brick + USB3)")
        return 2

    cv2.imwrite(out_path, color[:, :, :3])  # BGRA -> BGR
    print(f"OK {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
