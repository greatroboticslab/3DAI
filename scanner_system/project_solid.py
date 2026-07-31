"""Show a solid fullscreen color on the projector (2nd display).

Used by the capture orchestrator to make the projector go BLACK during the laser
multispectral stage, so the lasers are the only light hitting the sample (the
projector's normal image otherwise washes the scene out).

Runs as its own process with its own GUI event loop, so the fullscreen holds
steady even while the orchestrator is busy shelling out to grab Kinect frames.
The orchestrator launches it in the background and terminates it when the stage
is done.

Usage:
    python project_solid.py [value] [seconds]
      value   : 0-255 grayscale (default 0 = black)
      seconds : how long to hold before self-exiting (default 60; the parent
                usually kills it sooner)
"""

import sys
import time

import numpy as np
import cv2

# Projector geometry (matches data/scan_test/capture_multifreq.py).
SECOND_SCREEN_X = 1920
PROJ_W, PROJ_H = 1280, 800


def main() -> int:
    value = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0

    win = "projector"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.moveWindow(win, SECOND_SCREEN_X, 0)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    frame = np.full((PROJ_H, PROJ_W), value, np.uint8)
    deadline = time.time() + seconds
    while time.time() < deadline:
        cv2.imshow(win, frame)
        if cv2.waitKey(50) == 27:  # Esc to bail
            break
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
