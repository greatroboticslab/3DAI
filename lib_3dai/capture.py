from .fringes import generate_pattern
from .projector import project_image
from .kinect import capture_kinect
import time
import config
import numpy as np


def capture_projections(resolution, patterns):
    
    image_stack = []

    for pat in patterns:
        
        img_proj = generate_pattern(resolution, pat, config.NUM_FRINGES)
        
        # Project the original RGB image
        project_image(img_proj, config.SECOND_SCREEN_X)
        time.sleep(config.SETTLE_TIME)
        
        # Capture from Kinect
        live_img = capture_kinect(config.ROI_Y, config.ROI_X)
        
        if live_img is None or live_img.size == 0:
            print(f"Capture failed for {pat}")
        else:
            # Convert to float32 to match the old loading behavior
            image_stack.append(live_img.astype(np.float32))
        
    print("\n=== Capture and Processing Complete ===")
    
    # Stack along depth axis to match the old cv2.imread behavior
    imagestack = np.dstack(image_stack)
    
    return imagestack