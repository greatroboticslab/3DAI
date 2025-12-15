import os
import config
from glob import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt2d
import fpp_tools as fpp
import time


# ========================= PHASE RECONSTRUCTION & CALIBRATION =========================
print("\nReconstructing phase maps...")

ref_phi_flat = None
mean_dphi_list = []

for h_dir in sorted(glob(f"{config.CAPTURE_DIR}/h*mm")):
    h_val = float(os.path.basename(h_dir)[1:-2].replace("mm", ""))
    print(f"→ Processing {os.path.basename(h_dir)}")

    # Load only the 6 essential images: black, white, fringe_0..3
    files = [f for f in sorted(glob(f"{h_dir}/*.png")) if "fringe_" in f]  # 4
    stack = [cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32) for f in files]
    imagestack = np.dstack(stack)

    # Use order=1 to avoid singularity
    phi_img, amp_img, bias_img, deltas = fpp.estimate_deltas_and_phi_lsq(
        imagestack, order=1, eps=1e-4
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
heights = np.array(config.KNOWN_THICKNESSES_MM)
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

# Save calibration — Windows-safe version (no Greek letters in header)
header = ("Calibration coefficients [a b c] for:\n"
          "height_mm = a*(dphi)^2 + b*dphi + c\n"
          "where dphi = phase difference in radians (object phase - reference phase)\n"
          f"Generated on {time.strftime('%Y-%m-%d %H:%M')}")

np.savetxt("calibration_kinect.txt", poly_coeffs,
           header=header,
           fmt="%.10f",
           encoding="utf-8")   # forces UTF-8 → works everywhere

print("\nCalibration saved as 'calibration_kinect.txt'")
print("Use it later with:")
print("   coeffs = np.loadtxt('calibration_kinect.txt')")
print("   height_mm_map = np.polyval(coeffs, delta_phi_map)")

print("\nDone! You now have a fully calibrated Kinect fringe projection system.")