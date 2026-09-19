# Side camera for readable laser images

Dr. Zhang wants a laser photo where the spot is big and not blown out and
the object is still visible. The Kinect cannot do it (its SDK has no manual
exposure and its auto-exposure always lands on the long setting). A second
camera with a fixed, manual exposure can. The software for it is already in
place; this page is what to do when the camera and its connection arrive.

## What the scanner does with it

During the laser stage the second camera is grabbed right after each Kinect
frame: once with all lasers off (`cam_dark.png`), then under each laser
(`cam_las1..4.png`). They are registered as `laser_cam_*` artifacts,
measured into `cam_*` columns of `laser_features.csv`, and shown as extra
rows in each scan's `laser_zoom.jpg`. If the camera is missing or fails, the
scan continues with the Kinect alone; nothing else changes.

## What the camera must be able to do (measured 2026-09-19)

The laser spot core is at least 7x, probably far more, over an 8-bit
sensor's ceiling at any exposure where the object is still visible, and the
scatter halo around it is 1-2% of the core. Both in one frame needs a
dynamic range of a few thousand to one. Consequences for the purchase:

- **8-bit video (any HDMI capture stick, any webcam) cannot do it.** Manual
  exposure on an 8-bit feed only chooses which part clips: core or halo.
- What works: **12-bit or deeper stills** (RAW / high-bit-depth capture over
  USB tethering or a machine-vision camera such as a Basler/FLIR/Arducam
  USB3 module with 10-12-bit output, ~$100-300), or an 8-bit camera driven
  in an **exposure bracket** (one short frame for the core, one long frame
  for the halo and object, per laser). The bracket is the cheap route and
  the software can be extended to grab two exposures per laser.

## Hooking it up

1. Connect it so Windows sees it as a video device (a UVC camera or
   machine-vision module with manual exposure), or as a tethered stills
   camera if it delivers RAW. Set it to **manual exposure** (fixed shutter,
   fixed ISO/gain, manual white balance). For a 4K consumer camera, an HDMI
   capture stick will let you look at the spot but not measure it; see above.
2. Find its index and check it:

       scanner_system\.venv\Scripts\python.exe -m scanner_system.side_camera --list
       scanner_system\.venv\Scripts\python.exe -m scanner_system.side_camera --test 1 --exposure -8 --out check.png

   Open `check.png`: it should show the table. `manual_exposure_accepted`
   true means the camera itself takes exposure commands; false is normal
   for a capture stick (set exposure on the camera body instead).
3. Aim it at the table so the laser spot sits mid-frame, then fix it in
   place. Like the Kinect, moving it later means the pixel-scale numbers
   from before and after are not comparable.
4. Put its settings in `side_camera.cfg` in this folder (one per line):

       SCANNER_LASER_CAM=1
       SCANNER_LASER_CAM_SIZE=1920x1080
       SCANNER_LASER_CAM_EXPOSURE=-8
       SCANNER_LASER_CAM_GAIN=0

   Leave out `EXPOSURE`/`GAIN` when the camera body controls them.
   `scan.bat` reads this file every time it starts.
5. Pick the exposure once: lights as for collection, a light matte object on
   the table, a red laser on; the spot core should be bright but not pure
   white and the object clearly visible. Roughly: start at -8 (1/256 s) and
   go up (-7, -6) until the core just stops clipping. Then do not touch it.

## Checking it worked

After a scan, `scanner_data\scans\<scan>\laser\` has `cam_dark.png` and
`cam_las1..4.png`, and the scan's `laser_zoom.jpg` in the next export shows
"[side camera, manual exposure]" rows.
