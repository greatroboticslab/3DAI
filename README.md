# This project is to adapt the fringe projection python scripts from the Structured Light Project written by, Anton Poroykov, Ph.D., associated professor and Nikita Sivov, graduate student, to use a Cognex camera and integrate triggering with a robotic cell that makes decisions based on the point cloud and images generated.

# Findings:
The Structured Light Project scripts are hard-coded for two cameras. Options to accomidiate this appear to be to refactor the code to work with a single camera and a projector. It may be possible to use one real camera and one simulated camera to minimize code changes.

To further test the code without hardware, a test setup using two simulated cameras. The code initially included a simulated camera which just copies the projected image. Using two simulated cameras resulted in failure in triangulation calculations as the images were identical. Code was added to add noise to the simulated cameras and to shift the image by 25 pixels in the X axis to corrsipond with a hoizontal displacment of the cameras. A virtual calibration file was created that corrisponded to a 200mm displacment between the cameras. A minor rotational miss alignment was added in the virtual calibration to increase numerical stability.

It was discovered that even with ideal images the scripts would generate very few points with large holes missing. Turning on multicore processing seemed to eliminate that issue. The exact cause was not determined, but it seems likely it is due to either a processing time or iteration limit built into the code or libraries.

!['Testing Results/PointCloudSimulation.png'](https://github.com/greatroboticslab/3DAI/blob/main/Testing%20Results/PointCloudSimulation.png)
 
# Additions:
Added a camera_cognex module for connecting to older cognex cameras to download files through FTP and trigger via Telnet. This remains untested.

# Transition to fpp_tools:

Due to extensive hard-coding assumptions in the original Structured Light Project repository—particularly the requirement for two physical cameras and tightly coupled triangulation logic—it became impractical to adapt the code for a single-camera, projector-based system integrated into a robotic cell. Refactoring the original codebase to decouple camera geometry, triggering, and phase computation would have required significant invasive changes and risked diverging from the upstream implementation.

To address this, the project transitioned to using the FPP Tools repository (https://github.com/nzhagen/fpp_tools
) as the core fringe-processing backend. The fpp_tools library provides well-isolated, camera-agnostic implementations for generating phase-shifted fringe patterns and computing per-pixel wrapped phase (Δφ) from a sequence of captured images. This separation allows image acquisition, hardware triggering, and calibration to be handled independently of the fringe-processing logic, making it better suited for integration with Cognex cameras, Kinect-based prototyping, and robotic decision-making pipelines. The move also simplifies testing, simulation, and future hardware changes by cleanly separating fringe generation and phase computation from system-specific acquisition and triangulation code.

# Phase-to-Height Calibration Explanation (calibrate.py)

In fringe projection, each pixel’s measured wrapped phase φ represents the relative position of the projected fringe pattern as observed by the camera. When an object is introduced at a known height, the geometry between the projector, object surface, and camera causes a phase shift Δφ relative to a reference flat plane. This phase difference is directly related to surface height, but the relationship is nonlinear due to projector–camera geometry, perspective effects, and lens distortion.

This calibration procedure establishes an empirical mapping between mean phase difference Δφ (radians) and physical height (mm). A reference phase map is first captured on a flat surface and flattened by removing its best-fit plane to eliminate tilt and system bias. For each known calibration height, the same flattening is applied, and the per-pixel phase difference Δφ relative to the reference is computed. The mean Δφ over high-contrast pixels is then associated with the known physical height.

A quadratic polynomial is fitted to these (Δφ, height) pairs, yielding a calibration curve that converts measured phase differences directly into physical dimensions. Once calibrated, this mapping can be applied per pixel to transform a Δφ map into a dense height or depth map without requiring explicit triangulation.

# Test Calibration and Height Map Generation

The test_calibration.py script validates the Kinect + projector fringe projection system by converting newly captured fringe images into a metric height map using the previously generated phase-to-height calibration. A reference phase map is first captured on an empty, flat plane and processed by phase unwrapping and plane subtraction to remove global tilt and system bias. This establishes a consistent zero-height baseline aligned with the calibration procedure.

An object is then scanned using the same fringe patterns and processing steps. The per-pixel phase difference Δφ between the object and reference phase maps is computed and converted into physical height (mm) by applying the calibrated polynomial mapping. Low-confidence pixels are masked based on fringe modulation, and the resulting height map is displayed and optionally saved for inspection or downstream use. This script provides a simple end-to-end check that the calibration and phase reconstruction pipeline produces meaningful physical measurements.

# Results:
The figure below shows a scan of a mobile phone captured after system calibration. The reconstructed surface exhibits a residual wave-like error pattern, which is suspected to result from the interaction between the Kinect RGB camera’s rolling shutter and the use of a single-chip DLP projector. The projector displays color sequentially, while the Kinect camera samples different image rows at slightly different times, causing varying color contributions across the captured frame. This temporal mismatch introduces phase artifacts that manifest as periodic wave errors in the reconstructed height map. Because there is no practical mechanism to synchronize the projector’s color sequencing with the Kinect’s image acquisition, the most effective mitigation strategy is to project fringes using a single color channel, thereby reducing temporal color aliasing. 

!['Testing Results/Camera_Scan_Improved_Calibration.png'](https://github.com/greatroboticslab/3DAI/blob/main/Testing%20Results/Camera_Scan_Improved_Calibration.png)

After refractoring the script to allow for an arbitrary number of fringe projection phases, results were significantly improved using 60 fringe projections.



# 3DAI Requirements
Running pykinect2 requirments on Windows 10
1. python version 3.9.13 # setup an environment
2. pip install comtypes==1.4.13
3. pip install numpy
4. pip install pykinect2
5. Manually patch pykinect2
6. {Enviroment}\Lib\site-packages\pykinect2\PyKinectV2.py
7.  Change line 2216 from: assert sizeof(tagSTATSTG) == 72, sizeof(tagSTATSTG)
8.  to: assert sizeof(tagSTATSTG) >= 72, sizeof(tagSTATSTG)
9.  Comment out line 2863: from comtypes import _check_version; _check_version('')
10. {Enviroment}\Lib\site-packages\pykinect2\PyKinectRuntime.py
11. Replace all time.clock() calls with time.perf_counter()

# Running scripts on the Lab Computer

1. Adjust settings in C:\Users\Robotics_Lab\3DAI\3DAI\config.py
2. Open CMD terminal
3. Start the enviroment with the patched pykinect2 library:  c:\kinectEnv\scripts\activate
4. Navigate to local repo location: C:\Users\Robotics_Lab\3DAI\3DAI
5. Generate projector fringes: run: python generate_fringes.py
6. Calibration: run: python capture_patterns.py, follow prompts, place requested gauges in the scanning area when requested. After captureing images, check images in C:\Users\Robotics_Lab\3DAI\3DAI\captures_kinect. If fringes are not fully visible adjust cropping settings in config.py
7. Calibration: run: python calibrate.py
8. Use: run: python test_calibration.py