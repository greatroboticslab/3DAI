from .fringes import generate_pattern
from .projector import project_image
from .kinect import capture_kinect, capture_kinect_single
import time
import config
import numpy as np


def capture_projections(resolution, patterns):
    """
    Original multi-frame averaged capture. Captures multiple Kinect frames
    per pattern and takes the median to reduce noise. No depth data.

    Args:
        resolution: Projector resolution passed to generate_pattern
        patterns: List of pattern descriptors passed to generate_pattern

    Returns:
        ndarray: float32 grayscale image stack, shape (h, w, num_patterns)
    """
    image_stack = []

    for pat in patterns:
        
        img_proj = generate_pattern(resolution, pat, config.NUM_FRINGES)
        
        project_image(img_proj, config.SECOND_SCREEN_X)
        time.sleep(config.SETTLE_TIME)
        
        live_img = capture_kinect(config.ROI_Y, config.ROI_X)
        
        if live_img is None or live_img.size == 0:
            print(f"Capture failed for {pat}")
        else:
            image_stack.append(live_img.astype(np.float32))
        
    print("\n=== Capture and Processing Complete ===")
    
    imagestack = np.dstack(image_stack)
    
    return imagestack


def capture_projections_single(resolution, patterns):
    """
    Single-frame capture with tightly coupled raw color + depth for each pattern.
    No cropping or coordinate mapping is applied — both frames are returned at
    their native Kinect resolutions for post-processing alignment later.

    Color and depth are at different resolutions and fields of view:
        color — (1080, 1920, 3)  uint8  BGR
        depth — (424,  512)      uint16 millimetres, 0 = no data

    Args:
        resolution: Projector resolution passed to generate_pattern
        patterns:   List of pattern descriptors passed to generate_pattern

    Returns:
        dict with keys:
            "colors"   — uint8   array, shape (1080, 1920, 3, N)  BGR
            "depths"   — uint16  array, shape (424,  512,    N)  mm
            "patterns" — list of pattern descriptors that succeeded
    """
    color_stack = []
    depth_stack = []
    captured_patterns = []

    for pat in patterns:

        img_proj = generate_pattern(resolution, pat, config.NUM_FRINGES)

        project_image(img_proj, config.SECOND_SCREEN_X)
        time.sleep(config.SETTLE_TIME)

        color, depth = capture_kinect_single()

        if color is None or color.size == 0 or depth is None or depth.size == 0:
            print(f"  ✗ Capture failed for pattern: {pat}")
            continue

        color_stack.append(color)
        depth_stack.append(depth)
        captured_patterns.append(pat)
        print(f"  ✓ Captured pattern: {pat}")

    print("\n=== Single-Frame Capture Complete ===")

    # Stack along a new last axis so each [:,:,:,i] or [:,:,i] slice is one pattern
    return {
        "colors":   np.stack(color_stack, axis=-1).astype(np.uint8),    # (1080, 1920, 3, N)
        "depths":   np.stack(depth_stack, axis=-1).astype(np.uint16),   # (424,  512,    N)
        "patterns": captured_patterns,
    }