"""Project an alignment grid + center bullseye on the projector (2nd display).

Gives a physical aiming target so the lasers can be walked onto a known point
(the projector-zone center = where the sample sits and where all instruments
converge). Grid is on a BLACK background so a red laser dot pops against it, and
you can see the dot and the target at the same time while aiming.

Usage:
    python project_target.py [seconds]
"""

import sys
import time

import numpy as np
import cv2

SECOND_SCREEN_X = 1920
PROJ_W, PROJ_H = 1280, 800
CX, CY = PROJ_W // 2, PROJ_H // 2


def build_target() -> np.ndarray:
    img = np.zeros((PROJ_H, PROJ_W, 3), np.uint8)
    grid = (40, 40, 40)          # dim grey grid, won't wash out the laser
    # grid lines every 80 px
    for x in range(0, PROJ_W, 80):
        cv2.line(img, (x, 0), (x, PROJ_H), grid, 1)
    for y in range(0, PROJ_H, 80):
        cv2.line(img, (0, y), (PROJ_W, y), grid, 1)
    # bright center crosshair
    white = (255, 255, 255)
    cv2.line(img, (CX, CY - 60), (CX, CY + 60), white, 2)
    cv2.line(img, (CX - 60, CY), (CX + 60, CY), white, 2)
    # concentric bullseye rings
    for r in (25, 55, 90):
        cv2.circle(img, (CX, CY), r, white, 2)
    # a bright green center dot as the exact aim point (distinct from red laser)
    cv2.circle(img, (CX, CY), 4, (0, 255, 0), -1)
    return img


def main() -> int:
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    win = "projector"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.moveWindow(win, SECOND_SCREEN_X, 0)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    frame = build_target()
    deadline = time.time() + seconds
    while time.time() < deadline:
        cv2.imshow(win, frame)
        if cv2.waitKey(50) == 27:
            break
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
