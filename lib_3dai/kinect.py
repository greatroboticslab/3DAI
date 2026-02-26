from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
from pykinect2 import PyKinectV2
import ctypes
import time
import numpy as np
import config
import cv2

# ------------------- Kinect -------------------
kinect = PyKinectRuntime.PyKinectRuntime(FrameSourceTypes_Color | FrameSourceTypes_Depth)


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

    print(f"  → Capturing {num_captures} frames ...")

    frames = []
    capture_count = 0

    while capture_count < num_captures:
        if kinect.has_new_color_frame():
            frame = kinect.get_last_color_frame()
            frame = frame.reshape((1080, 1920, 4))[:, :, :3].astype(np.uint16)

            frame_roi = frame[roi_y, roi_x]
            frames.append(frame_roi)
            capture_count += 1

        time.sleep(0.1429)

    if not frames:
        raise RuntimeError("No frames were captured")

    stacked = np.stack(frames, axis=0)
    median_frame = np.median(stacked, axis=0).astype(np.uint8)

    if channel_mask != 0b111:
        filtered_frame = median_frame.copy()
        if not (channel_mask & 0b001):
            filtered_frame[..., 0] = 0
        if not (channel_mask & 0b010):
            filtered_frame[..., 1] = 0
        if not (channel_mask & 0b100):
            filtered_frame[..., 2] = 0
    else:
        filtered_frame = median_frame

    grayscale_frame = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2GRAY)
    return grayscale_frame


def grab_single_color():
    """
    Captures a single raw color frame from the Kinect.

    Returns:
        ndarray: uint8 BGR image, shape (1080, 1920, 3)
    """
    while not kinect.has_new_color_frame():
        time.sleep(0.005)

    frame = kinect.get_last_color_frame()
    # Kinect returns BGRA — drop the alpha channel
    return frame.reshape((1080, 1920, 4))[:, :, :3].astype(np.uint8)


def grab_single_depth():
    """
    Captures a single raw depth frame from the Kinect at native resolution.
    No coordinate mapping or cropping is applied.

    Returns:
        ndarray: uint16 depth array, shape (424, 512).
                 Values are in millimetres. 0 = no data.
    """
    while not kinect.has_new_depth_frame():
        time.sleep(0.005)

    depth_frame = kinect.get_last_depth_frame()
    return depth_frame.reshape((424, 512)).astype(np.uint16)


def capture_kinect_single():
    """
    Captures a tightly coupled raw color + depth frame pair.
    Both frames are grabbed back-to-back with no sleeping between them
    to minimise the temporal gap.

    Stale frames are drained first so both grabs block on the next
    live frame rather than returning buffered data.

    Returns:
        tuple:
            color (ndarray): uint8 BGR image,    shape (1080, 1920, 3)
            depth (ndarray): uint16 depth in mm, shape (424, 512)
    """
    # Drain stale buffered frames
    if kinect.has_new_color_frame():
        kinect.get_last_color_frame()
    if kinect.has_new_depth_frame():
        kinect.get_last_depth_frame()

    color = grab_single_color()
    depth = grab_single_depth()

    return color, depth