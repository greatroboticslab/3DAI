# This project is to adapt the fringe projection python scripts from the Structured Light Project written by, Anton Poroykov, Ph.D., associated professor and Nikita Sivov, graduate student, to use a Cognex camera and integrate triggering with a robotic cell that makes decisions based on the point cloud and images generated.

Findings:
The Structured Light Project scripts are hard-coded for two cameras. Options to accomidiate this appear to be to refactor the code to work with a single camera and a projector. It may be possible to use one real camera and one simulated camera to minimize code changes.

To further test the code without hardware, a test setup using two simulated cameras. The code initially included a simulated camera which just copies the projected image. Using two simulated cameras resulted in failure in triangulation calculations as the images were identical. Code was added to add noise to the simulated cameras and to shift the image by 25 pixels in the X axis to corrsipond with a hoizontal displacment of the cameras. A virtual calibration file was created that corrisponded to a 200mm displacment between the cameras. A minor rotational miss alignment was added in the virtual calibration to increase numerical stability.

It was discovered that even with ideal images the scripts would generate very few points with large holes missing. Turning on multicore processing seemed to eliminate that issue. The exact cause was not determined, but it seems likely it is due to either a processing time or iteration limit built into the code or libraries.

An image of the flat plan generated via simulation is located at Testing Results/PointCloudSimulation.png

Additions:
Added a camera_cognex module for connecting to older cognex cameras to download files through FTP and trigger via Telnet. This remains untested.


# 3DAI
