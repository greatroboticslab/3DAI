from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import time
import numpy as np
import config
import cv2

# ------------------- Kinect -------------------
kinect = PyKinectRuntime.PyKinectRuntime(FrameSourceTypes_Color)

def capture_kinect(roi_y, roi_x, channel_mask=0b111):
    """
    Captures multiple frames from Kinect, averages them to reduce noise,
    filters by color channel, and returns as grayscale.
    
    Args:
        roi_y (slice): y-range slice for ROI (e.g. slice(400, 800))
        roi_x (slice): x-range slice for ROI (e.g. slice(600, 1200))
        channel_mask (int): Binary mask for color channels:
            0b100 (4) = Red only
            0b010 (2) = Green only  
            0b001 (1) = Blue only
            0b111 (7) = All channels (default)
    
    Returns:
        ndarray: Grayscale image as numpy array
    """
    num_captures = config.NUMBER_OF_CAPTURES 
    
    print(f"  → Capturing {num_captures} frames for ...")
    
    frames = []
    capture_count = 0
    
    while capture_count < num_captures:
        if kinect.has_new_color_frame():
            frame = kinect.get_last_color_frame()
            frame = frame.reshape((1080, 1920, 4))[:, :, :3].astype(np.uint16)  # use uint16 to prevent overflow during sum
            
            frame_roi = frame[roi_y, roi_x]
            frames.append(frame_roi)
            
            capture_count += 1
            
        time.sleep(0.1429)  # ~7 fps, to avoid sync with 60fps
    
    
    if not frames:
        raise RuntimeError("No frames were captured")
    
    # Stack along new axis and take median (axis=0)
    stacked = np.stack(frames, axis=0)                     # shape: (N, h, w, 3)
    median_frame = np.median(stacked, axis=0).astype(np.uint8)
    
    # Apply color channel filtering
    if channel_mask != 0b111:
        filtered_frame = median_frame.copy()
        # Assuming BGR format (typical for Kinect/OpenCV)
        if not (channel_mask & 0b001):  # Blue channel
            filtered_frame[..., 0] = 0
        if not (channel_mask & 0b010):  # Green channel
            filtered_frame[..., 1] = 0
        if not (channel_mask & 0b100):  # Red channel
            filtered_frame[..., 2] = 0
    else:
        filtered_frame = median_frame
    
    # Convert to grayscale
    grayscale_frame = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2GRAY)
    
    return grayscale_frame