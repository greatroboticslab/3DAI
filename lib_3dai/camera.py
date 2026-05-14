import cv2
import time
from PIL import Image
from pathlib import Path 

class Camera ():
	"""
	 Handles webcam initialization, frame capture, and image storage.
	"""

	VIDEO_WIDTH_LENGTH = (1280,720) # HD resolution

	def __init__(self,system,storage=None):	
		#Webcam
		self.system = system 
		self.img = None
		self.storage_path = storage 
		self.camera_setting, self.camera_backend = self.system.get_camera_device()

		if self.storage_path:
			self.storage_path = Path(self.storage_path)

	def on(self):
		self.cam = cv2.VideoCapture(self.camera_setting,self.camera_backend)
		if not self.cam.isOpened():
			print("ERROR: could not open video source!")
			exit()

	def set_storage(self,storage_path:str):
		self.storage_path = Path(storage_path)

	def take_photo(self):
		"""
        Saves the most recent captured frame to disk using a timestamped filename.
        """
		if not self.storage_path:
			print("CANNOT TAKE PHOTO...CANNOT FIND STORAGE!")
			exit(1) 
		photo_name = self.storage_path / f"photo_{int(time.time())}.png"
		self.img.save(photo_name)
		print(f"✅ Photo Saved: {photo_name}")

	def update_frame(self):
		"""
        Captures a frame from the webcam, converts it to RGB,
        flips it horizontally, and stores it as a PIL Image.
        """
		ret, frame = self.cam.read()
		if ret:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			frame = cv2.flip(frame,1)
			self.img = Image.fromarray(frame)
			return self.img 
		
	def stop(self):
		self.cam.release()