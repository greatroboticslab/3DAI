# camera_cognex.py
from __future__ import annotations
import time
import telnetlib
from ftplib import FTP
import cv2
import numpy as np
from camera import Camera

class CameraCognex(Camera):
    def __init__(self, ip: str = "169.254.4.187", username: str = 'admin', password: str = ''):
        self.ip = ip
        self.username = username
        self.password = password
        self.type = 'cognex'
        # Defaults; replace with actual values or Telnet commands if available
        self._exposure = 4850  # From config.json example
        self._gamma = 0.5
        self._gain = 1.0
        print(f'Cognex camera initialized at {ip}')

    @staticmethod
    def get_available_cameras(cameras_num_to_find: int = 1) -> list[Camera]:
        # Since single camera, ignore num_to_find and return one instance
        return [CameraCognex()]

    def get_image(self) -> np.ndarray:
        # Logic from auto_calibrate.py: Trigger capture via Telnet
        try:
            tn = telnetlib.Telnet(self.ip, 23, timeout=5)
            tn.write(b'admin\r\n\r\nSE8\r\n')  # Capture command; adjust if needed
            time.sleep(1.2)  # Wait for capture (tune based on camera speed)
            tn.close()
        except Exception as e:
            raise RuntimeError(f"Telnet capture failed: {e}")

        # Download via FTP
        try:
            ftp = FTP(self.ip, timeout=5)
            ftp.login(self.username, self.password)
            with open("temp.bmp", "wb") as f:
                ftp.retrbinary("RETR image.bmp", f.write)
            ftp.quit()
            img = cv2.imread("temp.bmp")
            if img is None:
                raise RuntimeError("Failed to read captured image")
            # Convert to grayscale if needed (repo often uses gray)
            if len(img.shape) > 2:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img
        except Exception as e:
            raise RuntimeError(f"FTP download failed: {e}")
        finally:
            try:
                os.remove("temp.bmp")
            except:
                pass

    @property
    def exposure(self):
        return self._exposure

    @exposure.setter
    def exposure(self, value):
        # TODO: Implement Telnet command to set exposure (e.g., tn.write(b'SET EXPOSURE {value}\r\n'))
        # If not possible, log warning and use default
        print(f"Warning: Exposure setting not implemented for Cognex; using default {self._exposure}")
        self._exposure = value

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        # TODO: Similar to exposure
        print(f"Warning: Gamma setting not implemented for Cognex; using default {self._gamma}")
        self._gamma = value

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, value):
        # TODO: Similar to exposure
        print(f"Warning: Gain setting not implemented for Cognex; using default {self._gain}")
        self._gain = value