# 3DAI material dataset export

Generated 2026-09-15 18:26 UTC by `python -m scanner_system.export_dataset`.
Collection is in progress; this folder is re-exported and pushed as objects are scanned.

- Objects: **24**  (scans: 59, images: 763)
- Objects per material class: cardboard 2, foam 2, metal 5, paper 2, plastic 13

## What is here

- `metadata.xlsx` - sheet `files`: one row per image with the object's label,
  material class/subclass, surface finish, transparency, pose, modality and laser
  channel/wavelength. Sheet `summary`: counts.
- `laser_features.csv` - the measured laser material signals, one row per scan per
  laser channel: scatter halo radii (`halo_r50`, `halo_r10`), speckle contrast,
  per-channel reflectance (`add_r/g/b`, `flood`), near-infrared equivalents (`ir_*`),
  red/green reflectance ratios and fringe contrast. Channels: 1 and 2 red 635 nm,
  3 near-infrared 940 nm, 4 green 530 nm.
- `images/<material_class>/<object>__<pose>__<scan>/` - the pictures for one scan:
  `color` plain photo, `laser_dark` lasers off, `laser_ch1..4` each laser on,
  `*_ir` the Kinect infrared sensor for the same frame, `kinect_depth` Kinect depth,
  `fringe_white` projector white light, `height_map` reconstructed height.

The images here are JPEG previews (longest side 800 px; infrared and
depth frames stretched to 8-bit for viewing). The full-resolution PNGs, fringe
stacks, depth arrays and height maps stay on the scanner PC; ask for a bundle
(`python -m scanner_system.export_bundle out.zip --include-arrays`) for training.
