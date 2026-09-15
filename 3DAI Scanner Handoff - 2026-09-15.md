# 3DAI Scanner Handoff - 2026-09-15

First real collection session for the material-recognition dataset, plus the
tooling that lets anyone run the next ones.

## Data collected

- 24 objects, 59 complete scans, two poses each, all four lasers.
- Classes: plastic 13, metal 5, cardboard 2, paper 2, foam 2.
- Every scan stores: lasers-off dark frame, one colour frame per laser
  (CH1/CH2 red 635 nm, CH3 near-infrared 940 nm, CH4 green 530 nm), the
  Kinect infrared-sensor frame for each, the laser transient array, a plain
  photo, Kinect depth, the projector fringe stack with height reconstruction
  and fringe contrast, and per-channel features (scatter halo radii, speckle
  contrast, reflectance, IR equivalents) normalized by camera gain.
- Known gap: the first three objects (cardboard piece 01, polished metal 01,
  white breadboard 01) have CH4 flagged `valid = false` (loose wire, fixed at
  the bench). Rescan them when convenient; `scan.bat` will not redo them on
  its own because their rows are marked complete.
- Everything scanned before cardboard piece 01 (July demos, smoke tests) is
  archived out of the live database (`scanner_system/archive.py --list`).

## Where the data is

- GitHub: `dataset/` on `main` (previews by material class, `metadata.xlsx`,
  `laser_features.csv`, README). Updated by `push_data.bat`.
- Live read-only view: https://desktop-oe5h474.tailed15a6.ts.net (Tailscale
  Funnel from the scanner PC; needs the PC on and logged in). Restarted by
  `go_online.bat`, which also runs by itself at every login (shortcut in the
  user's Startup folder). Sleep is disabled while on AC power.
- Full-resolution frames and arrays: `scanner_data/scans/<scan_id>/` on the
  scanner PC; database `scanner` in the `scanner-mongo` Docker container.

## How the next person collects

`HOW_TO_SCAN.md`. Three Desktop shortcuts: SCAN objects, PUSH data to GitHub,
GO ONLINE. No spreadsheet editing or Docker knowledge needed.

## Things to do next

1. Rescan the three CH4-flagged objects (delete their `status` cell in
   `collection.xlsx`, run `scan.bat`).
2. Get flat, matte, caliper-measured objects into the sheet with
   `known_height_mm`: zero anchors passed the flatness gate so far, so the 3D
   height calibration is still the July curve.
3. Check the two labels that may be in the wrong class: "taped black metal"
   (paper) and "taped breadboard" (foam).
4. Red lasers saturate the spot core on light objects (`saturated_core = 1`);
   halo and speckle are measured outside the core and stay valid, but a
   neutral-density filter or lower laser current would recover the core.
5. The pipeline verification and archive tooling means the database is now
   the collection only; keep it that way (use `archive.py` for test scans).
