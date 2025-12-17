# test_calibration.py
# Simple real-time test of your Kinect + Projector calibration
# Place any object on the reference plane and watch the height map!

import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from glob import glob
from scipy.signal import medfilt2d
import fpp_tools as fpp
import lib_3dai
import config

# Load your calibration
coeffs = np.loadtxt("calibration_kinect.txt")
print(f"Loaded calibration: a={coeffs[0]:.8f}, b={coeffs[1]:.7f}, c={coeffs[2]:.7f}")


# Required patterns (must exist from calibration)
patterns = lib_3dai.get_patterns(config.PATTERN_DIR)

# ========================== REFERENCE PHASE (from 0 mm) ==========================
print("\n=== CAPTURING REFERENCE (empty plane) ===")
input("Make sure the reference plane is empty and flat. Press ENTER to continue...")

lib_3dai.capture_projections(f"{config.CAPTURE_DIR}/ref")

# Compute reference phase
ref_files = sorted(glob(f"{config.CAPTURE_DIR}/ref/fringe_*.png"))
stack = [cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32) for f in ref_files]
imagestack_ref = np.dstack(stack)

phi_ref, _, _, _ = fpp.estimate_deltas_and_phi_lsq(imagestack_ref, order=1, eps=1e-4)

# Unwrap and remove tilt (same as calibration)
unwrapped_ref = np.unwrap(np.unwrap(phi_ref, axis=1), axis=0)
h, w = unwrapped_ref.shape
X, Y = np.meshgrid(np.arange(w), np.arange(h))
A = np.c_[X.ravel(), Y.ravel(), np.ones(X.size)]
plane_coeffs = np.linalg.lstsq(A, unwrapped_ref.ravel(), rcond=None)[0]
plane_ref = (plane_coeffs[0]*X + plane_coeffs[1]*Y + plane_coeffs[2]).reshape(h, w)
ref_phi_flat = unwrapped_ref - plane_ref
ref_phi_flat = medfilt2d(ref_phi_flat, 5)

print("Reference phase captured and processed.\n")

# ========================== OBJECT SCAN ==========================
print("=== PLACE YOUR OBJECT AND SCAN ===")
input("Place the object on the reference plane. Press ENTER to scan...")

lib_3dai.capture_projections(f"{config.CAPTURE_DIR}/obj")

# Compute object phase
obj_files = sorted(glob(f"{config.CAPTURE_DIR}/obj/fringe_*.png"))
stack = [cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32) for f in obj_files]
imagestack_obj = np.dstack(stack)

phi_obj, _, _, _ = fpp.estimate_deltas_and_phi_lsq(imagestack_obj, order=1, eps=1e-4)

unwrapped_obj = np.unwrap(np.unwrap(phi_obj, axis=1), axis=0)
plane_obj = (plane_coeffs[0]*X + plane_coeffs[1]*Y + plane_coeffs[2]).reshape(h, w)
obj_phi_flat = unwrapped_obj - plane_obj
obj_phi_flat = medfilt2d(obj_phi_flat, 5)

# Height map
delta_phi = obj_phi_flat - ref_phi_flat
height_mm = np.polyval(coeffs, delta_phi)

# Mask low modulation or invalid areas (simple threshold)
amp_obj = np.std(imagestack_obj, axis=2)  # rough amplitude
mask = amp_obj > 15  # adjust if needed
height_mm[~mask] = np.nan

print(f"Height range: {np.nanmin(height_mm):.1f} mm to {np.nanmax(height_mm):.1f} mm")

# Display
plt.figure(figsize=(10, 6))
im = plt.imshow(height_mm, cmap='jet', vmin=0, vmax=10)  # adjust vmax to your expected range
plt.title("Height Map (mm)")
plt.colorbar(im, label="Height (mm)")
plt.axis('off')
plt.show()

# Optional: save as numpy array or PNG
save = input("Save height map as 'height_map.npy'? (y/n): ")
if save.lower() == 'y':
    np.save("height_map.npy", height_mm)
    print("Saved as height_map.npy")
    # Also save a pretty PNG
    norm = plt.Normalize(vmin=0, vmax=30)
    rgba = plt.cm.jet(norm(height_mm))
    rgba[~mask] = 0  # black background
    cv2.imwrite("height_map.png", (rgba[:, :, :3] * 255).astype(np.uint8)[..., ::-1])
    print("Also saved pretty PNG as height_map.png")

print("\nTest complete! Your scanner works!")