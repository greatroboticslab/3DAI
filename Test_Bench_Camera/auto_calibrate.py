# auto_calibrate.py — REPO-NATIVE FULL PIPELINE
import cv2
import numpy as np
import os
import time
import telnetlib
from ftplib import FTP

# === CONFIG ===
CAMERA_IP = "169.254.4.187"
USERNAME = 'admin'
PASSWORD = ''
PROJECTOR_RES = (1280, 800)
SECOND_SCREEN_X = 1920
CAPTURE_DIR = "captures"
PATTERN_DIR = "patterns"
os.makedirs(CAPTURE_DIR, exist_ok=True)

# === PROJECTOR ===
def project(img_path):
    img = cv2.imread(img_path)
    win = "PROJECTOR"
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow(win, SECOND_SCREEN_X, 0)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(win, img)
    cv2.waitKey(100)

# === CAPTURE ===
def capture(prefix):
    print(f"  → Capturing {prefix}...")
    tn = telnetlib.Telnet(CAMERA_IP, 23, timeout=5)
    tn.write(b'admin\r\n\r\nSE8\r\n')
    time.sleep(1.2)
    tn.close()
    ftp = FTP(CAMERA_IP, timeout=5)
    ftp.login('admin', '')
    with open("temp.bmp", "wb") as f:
        ftp.retrbinary("RETR image.bmp", f.write)
    ftp.quit()
    img = cv2.imread("temp.bmp")
    path = f"{CAPTURE_DIR}/cap_{prefix}.jpg"
    cv2.imwrite(path, img)
    os.remove("temp.bmp")
    return path

# === PATTERN LIST (from repo) ===
patterns = ["black", "white"] + [f"fringe_{i}" for i in range(4)] + [f"gray_{i}" for i in range(10)]

# === PROJECT + CAPTURE ===
captured = []
for name in patterns:
    path = f"{PATTERN_DIR}/{name}.png"
    if not os.path.exists(path):
        print(f"Missing {path} — run create_patterns.py first!")
        exit()
    project(path)
    time.sleep(0.8)
    cap_path = capture(name)
    captured.append(cap_path)
    time.sleep(0.3)

cv2.destroyAllWindows()
print(f"Captured {len(captured)} images.")