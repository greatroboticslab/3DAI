# Scanning Bigger Objects

Plain guide to extending the scanner past small flat targets, with the costs of
each option. Written 2026-06-26 after a real test: a tall lab pot scanned at
72% reliable with a 20.5 rad relief span (vs ~0.83 rad / 89% for a good flat
target), i.e. the height "wrapped" and most of the surface was unusable.

Read this before changing the rig. The current calibration is hard-won
(`.claude/PROJECT_MEMORY.md`); any geometry change invalidates it.

## Two separate "too big" problems

There are two limits, and they have different fixes. Know which one you are
hitting.

### 1. Too TALL (vertical range / phase wrapping)

How height is measured: the projector throws striped patterns; height shows up
as how far the stripes shift. But stripes repeat, so a shift of "1.5 stripes"
looks identical to "0.5 stripes." The scanner resolves this by combining a
coarse pattern (does not repeat across the scene) with a fine one (precise).
This is the `FREQS = [1, 6, 24]` list in `data/scan_test/capture_multifreq.py`:
frequency 1 is the coarse, no-repeat pattern; 24 is the fine, precise one.

A tall object pushes the height shift past what frequency 1 can keep
unambiguous. The height "wraps around" and you get nonsense (the 20.5 rad span
from the pot). Fine-resolution does not help; the COARSE end sets the range.

### 2. Too WIDE (footprint)

The projector only lights a fixed rectangle, and the Kinect only looks at a
cropped region. Current usable footprint is about `y377:781 x405:1037` (~405 x
632 px) inside a 1280 x 800 projector frame. Anything outside that lit/cropped
box is simply not measured. Making the measured area physically larger means
projecting onto a bigger area and widening the crop.

## Options, easiest to hardest

### A. Raise the reference plane to the object's base (no recalibration)

Height is measured relative to a saved empty "reference" scan
(`data/scan_test/calib/ref`). If your object is tall but the SURFACE you care
about is flat-ish, put a riser under the object (or under the whole work
surface) so the surface of interest sits back inside the good measuring window,
then re-shoot the reference at that base height. This MOVES the measurement
window up; it does not make it bigger. Cheapest win for "tall but flat top."

Cost: a new reference capture. No calibration change. Does not help genuinely
deep/irregular objects.

### B. Extend vertical range with a coarser pattern (recalibration recommended)

Make the coarse end coarser or add another even-coarser frequency so the
no-repeat range covers more height. Edit `FREQS` in `capture_multifreq.py` and
the matching unwrap order in `reconstruct_multifreq.py` /
`fpp_tools/temporal_unwrap.py` (they must agree on the frequency list and use
lowest-first temporal unwrapping). More frequencies = a few more captured frames
and slightly less fine precision, but real added range.

Cost: code change in BOTH capture and reconstruct, plus a calibration re-check
because the height-to-phase fit assumed the old frequency set. Moderate.

### C. Bigger footprint: zoom/move the rig out, widen ROI, RECALIBRATE

To scan physically WIDER things, increase the projected/measured area: move the
projector and Kinect back (or use a wider throw), widen the Kinect crop
(`ROI_Y, ROI_X` in `config.py`, and the bbox the scan scripts use), then redo
calibration with known-height targets at the new distance. Bigger area = coarser
per-pixel detail.

Cost: physically moving the rig (which the lab notes warn against) AND a full
recalibration. The current calibration (`calibration_temporal.txt`) becomes
invalid the instant the geometry moves. This is the biggest-payoff, biggest-risk
option. Do not do it without Dr. Zhang's sign-off and a plan to recapture all
calibration anchors.

### D. Stitch multiple scans (most work)

Scan the object in overlapping pieces or from several angles and merge the
height maps / point clouds into one. Handles truly large or self-occluding
objects, but it is a whole feature to build (capture poses, alignment, merge).
Only worth it if A-C cannot cover the need.

## Recommended path if recalibrating anyway

Since recalibration is on the table, the highest-value single change is C
(bigger footprint) combined with B (more vertical range), done together so you
only recalibrate once:

1. Decide the target working volume (max width x depth x height you need).
2. Set projector throw / Kinect distance so the lit area covers that width+depth;
   widen `ROI` and the scan bbox to match.
3. Choose a `FREQS` set whose coarse end keeps the full target height
   unambiguous.
4. Recapture the empty reference at the new geometry.
5. Recapture ALL calibration anchors (flat known-height targets spanning the new
   height range; prefer blue-taped matte surfaces per the calibration notes) and
   refit `calibration_temporal.txt`.
6. Validate against a couple of known objects before trusting it.

## Reality check

- Soil in a deep pot is a hard target regardless: dark, rough, crumbly, and
  self-shadowing surfaces give low fringe contrast even within range. For clean
  soil scans, consider a shallow tray instead of a deep pot.
- Every option except A requires recalibration. Budget time for capturing new
  anchors, not just code edits.
