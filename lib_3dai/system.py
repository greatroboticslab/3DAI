import platform
import cv2
import os 


class SystemManager:
    """
    Manages system-specific setup for camera.

    - Detects the OS (Linux, Windows, MacOS)  
    - Provides a method to determine the correct camera device and OpenCV capture backend
    """
    def __init__(self):
        self.system_name = platform.system()


    def get_camera_device(self):
        if self.system_name == "Linux":
            devices = [f"/dev/{d}" for d in os.listdir("/dev") if d.startswith("video")]
            if not devices:
                print("\n❌ No camera device found under /dev/\n")
                return 0, cv2.CAP_V4L2
            return devices[0], cv2.CAP_V4L2
    
        elif self.system_name == "Windows":
         return 0, cv2.CAP_DSHOW
    
        elif self.system_name == "Darwin":
            return 0, cv2.CAP_AVFOUNDATION
    
        else:
            return 0, cv2.CAP_ANY