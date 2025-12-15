from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import time
import cv2
import numpy as np

# ------------------- Kinect -------------------
kinect = PyKinectRuntime.PyKinectRuntime(FrameSourceTypes_Color)

def capture_kinect(prefix: str,dir,roi_y,roi_x):
    print(f"  → Capturing {prefix}...", end="")
    while True:
        if kinect.has_new_color_frame():
            frame = kinect.get_last_color_frame()
            frame = frame.reshape((1080, 1920, 4))[:, :, :3].astype(np.uint8)
            frame_roi = frame[roi_y, roi_x]
            
            # Temporary unique name in root capture dir
            temp_path = f"{dir}/cap_temp_{int(time.time()*1000)}.jpg"
            cv2.imwrite(temp_path, frame_roi)
            print(" done")
            return temp_path, frame_roi
        time.sleep(0.01)