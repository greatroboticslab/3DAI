import os
from glob import glob
from .fringes import get_patterns
from .projector import project_image
from .kinect import capture_kinect
import cv2
import time
import config

def capture_projections(dir):
    os.makedirs(dir, exist_ok=True)
    
    # <<< SAFETY: Remove old captures for this height >>>
    for old_file in glob(f"{dir}/*.png"):
        try:
            os.remove(old_file)
        except:
            pass
    print(f"   Cleared old images in {dir}")

    patterns = get_patterns(config.PATTERN_DIR)

    for pat in patterns:
        img_path = f"{config.PATTERN_DIR}/{pat}"
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Pattern not found: {img_path}")
        
        project_image(img,config.SECOND_SCREEN_X)
        time.sleep(1.0)  # stable projection
        
        # Simple clean name — safe because we deleted old ones
        clean_name = f"{pat}" 
        final_path = os.path.join(dir, clean_name)
        
        _, live_img = capture_kinect(f"temp",dir,config.ROI_Y,config.ROI_X)
        os.rename(glob(f"{dir}/cap_*.jpg")[0], final_path)  # move the temp file