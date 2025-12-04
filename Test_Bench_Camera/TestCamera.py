from pykinect2 import PyKinectV2, PyKinectRuntime
import cv2
import numpy as np

# Initialize Kinect runtime for color frames
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)

print("Starting Kinect RGB test. Press ESC to exit.")

while True:
    # Check if a new color frame is available
    if kinect.has_new_color_frame():
        # Get the last color frame
        frame = kinect.get_last_color_frame()
        
        # Kinect color frames are 1080x1920x4 (BGRA)
        frame = frame.reshape((1080, 1920, 4))
        
        # Drop the alpha channel
        frame = frame[:, :, :3]

        # Convert BGRA to BGR for OpenCV display
        frame = frame.astype(np.uint8)
        
        # Resize to fit screen (optional)
        display_frame = cv2.resize(frame, (960, 540))
        
        # Show the frame
        cv2.imshow("Kinect RGB Feed", display_frame)

    # Exit on ESC key
    if cv2.waitKey(1) == 27:
        break

# Clean up
cv2.destroyAllWindows()
