import os
from glob import glob
import config
import lib_3dai
import time
import cv2

print("=== START CAPTURE ===\n")
print("Place reference plane (0 mm) and press ENTER")
input()

for idx, h_mm in enumerate(config.KNOWN_THICKNESSES_MM):
    print(f"\n=== STEP {idx+1}/{len(config.KNOWN_THICKNESSES_MM)} : {h_mm:.1f} mm ===")
    if h_mm > 0:
        input(f"   → Place {h_mm:.1f} mm plate → press ENTER when ready...")
    
    height_dir = f"{config.CAPTURE_DIR}/h{h_mm:05.1f}mm"
    os.makedirs(height_dir, exist_ok=True)
    
    # <<< SAFETY: Remove old captures for this height >>>
    for old_file in glob(f"{height_dir}/*.png"):
        try:
            os.remove(old_file)
        except:
            pass
    print(f"   Cleared old images in {height_dir}")

    patterns = lib_3dai.get_patterns(config.PATTERN_DIR)

    for pat in patterns:
        img_path = f"{config.PATTERN_DIR}/{pat}"
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Pattern not found: {img_path}")
        
        lib_3dai.project_image(img,config.SECOND_SCREEN_X)
        time.sleep(1.0)  # stable projection
        
        # Simple clean name — safe because we deleted old ones
        clean_name = f"{pat}" 
        final_path = os.path.join(height_dir, clean_name)
        
        _, live_img = lib_3dai.capture_kinect(f"h{h_mm:05.1f}mm_{pat}",config.CAPTURE_DIR,config.ROI_Y,config.ROI_X)
        os.rename(glob(f"{config.CAPTURE_DIR}/cap_*.jpg")[0], final_path)  # move the temp file
        
        # cv2.imshow("Kinect Live", cv2.resize(live_img, (900, 560)))
        cv2.waitKey(50)

    print(f"Finished {h_mm:.1f} mm → {len(patterns)} images saved cleanly")

print("\nCapture phase complete!")