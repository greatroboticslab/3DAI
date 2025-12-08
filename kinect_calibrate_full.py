# kinect_calibrate_full.py
# Complete Kinect v2 + Projector calibration pipeline
# → Generates patterns on-the-fly using fpp_tools.generate_fringe_patterns()
# → Captures with pykinect2
# → Computes phase with gamma compensation
# → Calibrates Δφ → real height (mm) using known flat plates

import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from glob import glob
from scipy.signal import medfilt2d
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import fpp_tools as fpp

# ========================== CONFIG ==========================
PROJECTOR_RES = (1280, 800)           # Your projector resolution
SECOND_SCREEN_X = 1920                # X offset of your second monitor
CAPTURE_DIR = "captures_kinect"
PATTERN_DIR = "patterns_auto"         # Will be created automatically
os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(PATTERN_DIR, exist_ok=True)

# === CALIBRATION PLATES (in mm) ===
KNOWN_THICKNESSES_MM = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0]   # ← change to your actual gauges

# Kinect color ROI — adjust once by eye (where the projector pattern fully appears)
ROI_Y, ROI_X = slice(100, 980), slice(300, 1620)   # 880×1320 px typical safe zone

# Fringe parameters (tune once for your setup)
NUM_FRINGES = 24          # How many vertical fringes across projector width → ~50–60 px per fringe is ideal
PHASES_DEG = [0, 90, 180, 270]   # Classic 4-step phase shifting (best noise/gamma performance)
GRAY_LEVELS = 10          # For optional flat-fielding / gamma estimation

# =========================================================

# ------------------- Kinect -------------------
kinect = PyKinectRuntime.PyKinectRuntime(FrameSourceTypes_Color)

def capture_kinect(prefix: str):
    print(f"  → Capturing {prefix}...", end="")
    while True:
        if kinect.has_new_color_frame():
            frame = kinect.get_last_color_frame()
            frame = frame.reshape((1080, 1920, 4))[:, :, :3]          # BGRA → BGR
            frame = frame.astype(np.uint8)
            frame_roi = frame[ROI_Y, ROI_X]
            path = f"{CAPTURE_DIR}/cap_{prefix}.jpg"
            cv2.imwrite(path, frame_roi)
            print(" done")
            return path, frame_roi
        time.sleep(0.01)

# ------------------- Projector -------------------
def project_image(img_np_uint8):
    win = "PROJECTOR"
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow(win, SECOND_SCREEN_X, 0)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(win, img_np_uint8)
    cv2.waitKey(100)

# ------------------- Generate All Patterns Once -------------------
print("Generating patterns using fpp_tools.generate_fringe_patterns() ...")
# Black & White
black = np.zeros((PROJECTOR_RES[1], PROJECTOR_RES[0]), dtype=np.uint8)
white = np.full((PROJECTOR_RES[1], PROJECTOR_RES[0]), 255, dtype=np.uint8)
cv2.imwrite(f"{PATTERN_DIR}/black.png", black)
cv2.imwrite(f"{PATTERN_DIR}/white.png", white)

# 4-step sinusoidal fringes
phases_rad = np.deg2rad(PHASES_DEG)
fpp.generate_fringe_patterns(
    Nx=PROJECTOR_RES[1],
    Ny=PROJECTOR_RES[0],
    phases=phases_rad,
    num_fringes=NUM_FRINGES,
    # vertical fringes
    gamma=1.0,
    filebase=f"{PATTERN_DIR}/fringe_"
)

# Optional: uniform gray levels for better gamma estimation
for i, level in enumerate(np.linspace(20, 235, GRAY_LEVELS)):
    gray = np.full((PROJECTOR_RES[1], PROJECTOR_RES[0]), int(level), dtype=np.uint8)
    cv2.imwrite(f"{PATTERN_DIR}/gray_{i:02d}.png", gray)

print(f"Generated {4 + 2 + GRAY_LEVELS} patterns in '{PATTERN_DIR}/'\n")

# ------------------- Pattern list (same order every time) -------------------
patterns = ["black", "white"] + [f"fringe_{i}" for i in range(4)] + [f"gray_{i:02d}" for i in range(GRAY_LEVELS)]

# ========================= MAIN CALIBRATION LOOP =========================
print("=== START CALIBRATION ===\n")
print("Place reference plane (0 mm) and press ENTER")
input()

for idx, h_mm in enumerate(KNOWN_THICKNESSES_MM):
    print(f"\n=== STEP {idx+1}/{len(KNOWN_THICKNESSES_MM)} : {h_mm:.1f} mm ===")
    if h_mm > 0:
        input(f"   → Place {h_mm:.1f} mm gauge block / plate → press ENTER")
    
    height_dir = f"{CAPTURE_DIR}/h{h_mm:05.1f}mm"
    os.makedirs(height_dir, exist_ok=True)

    for pat in patterns:
        img_path = f"{PATTERN_DIR}/{pat}.png"
        img = cv2.imread(img_path)
        project_image(img)
        time.sleep(0.9)  # let everything settle
        path, live = capture_kinect(f"h{h_mm:05.1f}mm_{pat}")
        # Move into height-specific folder
        new_path = os.path.join(height_dir, os.path.basename(path))
        os.rename(path, new_path)

        # Optional live preview
        cv2.imshow("Kinect Live (cropped)", cv2.resize(live, (900, 560)))
        cv2.waitKey(50)

    print(f"Finished {h_mm:.1f} mm")

cv2.destroyAllWindows()
kinect.close()
print("\nCapture phase complete!")

# ========================= PHASE RECONSTRUCTION & CALIBRATION =========================
print("\nReconstructing phase maps...")

ref_phi_flat = None
mean_dphi_list = []

for h_dir in sorted(glob(f"{CAPTURE_DIR}/h*mm")):
    h_val = float(os.path.basename(h_dir)[1:-2].replace("mm", ""))
    print(f"→ Processing {os.path.basename(h_dir)}")

    # Load only the 6 essential images: black, white, fringe_0..3
    fringe_files = [f for f in sorted(glob(f"{h_dir}/*.jpg")) if "fringe_" in f]  # 4
    bw_files = [f for f in sorted(glob(f"{h_dir}/*.jpg")) if "black" in f or "white" in f]
    files = bw_files + fringe_files
    stack = [cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32) for f in files]
    imagestack = np.dstack(stack)

    # Best method in the repo: order=2 → automatically compensates projector gamma!
    phi_img, amp_img, bias_img, gamma_img, deltas = fpp.estimate_deltas_and_phi_lsq(
        imagestack, order=2, eps=1e-4
    )

    # Unwrap + remove tilt plane
    unwrapped = np.unwrap(np.unwrap(phi_img, axis=1), axis=0)
    h, w = unwrapped.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    A = np.c_[X.ravel(), Y.ravel(), np.ones(X.size)]
    coeffs = np.linalg.lstsq(A, unwrapped.ravel(), rcond=None)[0]
    plane = (coeffs[0]*X + coeffs[1]*Y + coeffs[2]).reshape(h, w)
    phi_flat = unwrapped - plane

    # Light median filtering
    phi_flat = medfilt2d(phi_flat, 5)

    if ref_phi_flat is None:
        ref_phi_flat = phi_flat

    delta_phi = phi_flat - ref_phi_flat
    mean_delta_phi = np.mean(delta_phi)

    mean_dphi_list.append(mean_delta_phi)

    plt.figure(figsize=(9,5))
    im = plt.imshow(delta_phi, cmap='rainbow', vmin=-5, vmax=5)
    plt.title(f"Δφ map – {h_val:.1f} mm   (mean = {mean_delta_phi:.3f} rad)")
    plt.colorbar(im)
    plt.show(block=False)

plt.show()
print("\nAll phase maps computed.")

# ========================= FIT CALIBRATION CURVE =========================
heights = np.array(KNOWN_THICKNESSES_MM)
dphi_measured = np.array(mean_dphi_list)

# Use only non-zero heights for fitting
valid = heights > 0.1
poly_coeffs = np.polyfit(dphi_measured[valid], heights[valid], deg=2)

print("\nCALIBRATION RESULT")
print("Δφ → height (mm) polynomial (quadratic fit):")
print(f"   h_mm = {poly_coeffs[0]:.7f}·(Δφ)² + {poly_coeffs[1]:.6f}·Δφ + {poly_coeffs[2]:.6f}")

# Plot
dphi_range = np.linspace(dphi_measured.min()-1, dphi_measured.max()+1, 500)
h_fit = np.polyval(poly_coeffs, dphi_range)

plt.figure(figsize=(8,6))
plt.plot(dphi_measured, heights, 'ro', markersize=8, label='Measured plates')
plt.plot(dphi_range, h_fit, 'b-', linewidth=2, label='Quadratic fit')
plt.xlabel('Mean phase difference Δφ (radians)')
plt.ylabel('Height (mm)')
plt.title('Kinect + Projector – Height Calibration Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Save calibration
header = ("Calibration coefficients [a b c] for:\n"
          "height_mm = a*(Δφ)² + b*Δφ + c\n"
          f"Generated on {time.strftime('%Y-%m-%d %H:%M')}")
np.savetxt("calibration_kinect.txt", poly_coeffs, header=header, fmt="%.10f")
print("\nCalibration saved as 'calibration_kinect.txt'")
print("Use it later with:")
print("   coeffs = np.loadtxt('calibration_kinect.txt')")
print("   height_mm_map = np.polyval(coeffs, delta_phi_map)")

print("\nDone! You now have a fully calibrated Kinect fringe projection system.")