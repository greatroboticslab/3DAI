"""
collect_data.py

Runs a full fringe projection capture session, saves the results in a format
optimised for AI training, and optionally visualises them.

Save format — NumPy .npz archive:
    images    float32  (h, w, N)  — grayscale fringe images
    depths    float32  (h, w, N)  — depth in mm, pixel-aligned with images
    patterns  object              — array of pattern descriptor strings

One .npz file is written per session, named with a UTC timestamp so that
multiple sessions never collide and are trivially sortable.

Usage:
    python collect_data.py                  # capture and save
    python collect_data.py --visualize      # capture, save, and show results
    python collect_data.py --visualize --output-dir ./my_data
"""

import argparse
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

import config
from lib_3dai import capture_projections_single


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_capture(data: dict, output_dir: str = "data") -> Path:
    """
    Saves a capture session dict (from capture_projections_single) as a
    NumPy .npz archive.

    The archive contains three arrays that can be loaded back with
    np.load(..., allow_pickle=True):

        images   float32  (h, w, N)  — grayscale fringe images
        depths   float32  (h, w, N)  — depth in mm
        patterns object              — 1-D array of pattern descriptor strings

    Args:
        data:       Return value of capture_projections_single.
        output_dir: Directory to write into (created if it does not exist).

    Returns:
        Path to the saved file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    filepath = out / f"capture_{timestamp}.npz"

    np.savez_compressed(
        filepath,
        images=data["images"],
        depths=data["depths"],
        patterns=np.array(data["patterns"], dtype=object),
    )

    print(f"\n  Saved → {filepath}")
    print(f"  images : {data['images'].shape}  dtype={data['images'].dtype}")
    print(f"  depths : {data['depths'].shape}  dtype={data['depths'].dtype}")
    print(f"  patterns: {data['patterns']}")

    return filepath


# ---------------------------------------------------------------------------
# Visualise
# ---------------------------------------------------------------------------

def visualize_capture(data: dict) -> None:
    """
    Displays each captured pattern as a side-by-side panel:
        left  — grayscale fringe image
        right — depth map (viridis colormap, invalid pixels masked in grey)

    Args:
        data: Return value of capture_projections_single.
    """
    images   = data["images"]    # (h, w, N)
    depths   = data["depths"]    # (h, w, N)
    patterns = data["patterns"]
    n = images.shape[2]

    fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n))
    fig.suptitle("Fringe Projection Capture", fontsize=14, fontweight="bold")

    # Ensure axes is always 2-D even for a single pattern
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, pat in enumerate(patterns):
        img   = images[:, :, i]
        depth = depths[:, :, i]

        # --- Image panel ---
        axes[i, 0].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[i, 0].set_title(f"Pattern: {pat} — Image")
        axes[i, 0].axis("off")

        # --- Depth panel ---
        # Mask pixels with no depth data (value == 0)
        depth_masked = np.where(depth > 0, depth, np.nan)
        im = axes[i, 1].imshow(depth_masked, cmap="viridis")
        axes[i, 1].set_title(f"Pattern: {pat} — Depth (mm)")
        axes[i, 1].axis("off")
        fig.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04, label="mm")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fringe projection data collector")
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Show image + depth visualisation after capture",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="data",
        help="Directory to save .npz files into (default: ./data)",
    )
    args = parser.parse_args()

    print("=== Starting capture session ===")
    data = capture_projections_single(config.PROJECTOR_RES, config.PHASES_DEG)

    if not data["patterns"]:
        print("No patterns captured successfully — nothing saved.")
        return

    save_capture(data, output_dir=args.output_dir)

    if args.visualize:
        visualize_capture(data)


if __name__ == "__main__":
    main()