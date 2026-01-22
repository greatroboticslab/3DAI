from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import time
import cv2
import numpy as np
import config
from pathlib import Path


# ------------------- Kinect -------------------
kinect = PyKinectRuntime.PyKinectRuntime(FrameSourceTypes_Color)


def capture_kinect(prefix: str, dir_path, roi_y, roi_x):
    """
    Captures multiple frames from Kinect, averages them to reduce noise,
    and saves the averaged result.
    
    Args:
        prefix (str): Prefix for the output filename
        dir_path (str or Path): Directory where images will be saved
        roi_y (slice): y-range slice for ROI (e.g. slice(400, 800))
        roi_x (slice): x-range slice for ROI (e.g. slice(600, 1200))
    
    Returns:
        tuple: (path_to_averaged_image, averaged_frame_np_array)
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    
    num_captures = config.NUMBER_OF_CAPTURES 
    
    print(f"  → Capturing {num_captures} frames for {prefix} ...")
    
    frames = []
    capture_count = 0
    
    while capture_count < num_captures:
        if kinect.has_new_color_frame():
            frame = kinect.get_last_color_frame()
            frame = frame.reshape((1080, 1920, 4))[:, :, :3].astype(np.uint16)  # use uint16 to prevent overflow during sum
            
            frame_roi = frame[roi_y, roi_x]
            frames.append(frame_roi)
            
            # Optional: save individual captures (useful for debugging)
            # temp_path = dir_path / f"{prefix}_frame_{capture_count:03d}.jpg"
            # cv2.imwrite(str(temp_path), frame_roi.astype(np.uint8))
            
            capture_count += 1
            print(f"    captured {capture_count}/{num_captures}", end="\r")
            
        time.sleep(0.1429)  # ~7 fps, to avoid sync with 60fps
    
    print(" " * 50, end="\r")  # clear line
    print("  → Averaging frames...", end="")
    
    if not frames:
        raise RuntimeError("No frames were captured")
    
    # Stack along new axis and take median (axis=0)
    stacked = np.stack(frames, axis=0)                     # shape: (N, h, w, 3)
    median_frame = np.median(stacked, axis=0).astype(np.uint8)
    
    # Final filename
    timestamp = int(time.time() * 1000)
    output_path = dir_path / f"{timestamp}.png"
    
    cv2.imwrite(str(output_path), median_frame)
    
    print(" done")
    print(f"    Saved averaged image: {output_path.name}")
    
    return str(output_path), median_frame


# ────────────────────────────────────────────────
# Example usage:
#
# roi_y = slice(500, 700)
# roi_x = slice(800, 1100)
# path, img = capture_and_average_kinect("white_balance", "captures", roi_y, roi_x)