# 3DAI Scanner Handoff - 2026-08-26 - Adain

Work log for 2026-08-26. ESP32 relay/laser hardware was used with explicit
approval throughout, with the user present at the bench. The 455 nm 5.5 W
laser TTL path (GPIO21) was NEVER energized: it remained `LASER UNCONFIGURED`
for the whole session. The Kinect was found unpowered and absent from USB, so
no camera capture was possible.

---

## State BEFORE this day's work

- Branch `feat/4dai-fusion-endpoint` at `b34fa89`, 13 commits ahead of its
  remote; `main` 16 behind.
- **The ESP32 rig had been unreachable since roughly 2026-08-13.** It answered
  `PING` but ignored every other command. A 2026-08-13 diagnostic session had
  concluded the board was fine and left a no-GPIO diagnostic firmware on it.
- The SETM "fire all relays together" firmware fix was written, compiled, and
  **uncommitted** - existing only in the working tree.
- 9 modified files uncommitted, including firmware, relay tooling, capture and GUI.
- Dr. Zhang had stated three priorities: an Excel DB for fast entry/scanning,
  much better 3D scans, and actually useful laser material data.

---

## Part 1 - The ESP32 mystery, solved

### Root cause

A stray `CONFIG` had written **channel 1 to GPIO3**, which is UART0 RX. Dumped
from the chip's NVS and parsed:

```
CH1: pin=3    <-- WRONG. Should be 18. GPIO3 is the serial RECEIVE pin.
CH2: pin=19   CH3: pin=25   CH4: pin=26    (all correct)
```

`loadChannels()` runs `pinMode(pin, OUTPUT)` on whatever NVS supplies, with no
validation, and it runs in `setup()` **before** printing `READY`. So on every
boot the firmware converted its own receive pin into a relay output. Signature:
the board transmits perfectly (`READY` appears; TX is GPIO1) and receives
nothing, in any DTR/RTS state.

**Why reflashing could never have fixed it:** `pio run -t upload` erases only
`0x1000-0x5fff`, `0x8000-0x8fff` and `0xe000-0xffff`. NVS at **`0x9000` is
never touched**. The poison outlived every flash.

**Why the 2026-08-13 diagnosis said "board is fine":** the `ESP32_Diag_NoGPIO`
build reads no NVS and drives no pins, so it is immune. That session did the
correct bisection and its conclusion was right - the silicon was healthy. It
stopped one step short of suspecting NVS.

### Repair sequence (repeatable recipe)

1. Hold IO0, `esptool.py --port COM3 read_flash 0x9000 0x5000 nvs.bin` (evidence)
2. `esptool.py --port COM3 erase_region 0x9000 0x5000`
3. Release IO0, reset
4. `python lib_3dai/provision_relays.py --commit --port COM3`

Step 4 works precisely because the correct pin map lives in a version-controlled
script rather than only on the board. That was a good call by whoever wrote it.

### Verified after repair

- Real `POWERON_RESET` -> `READY` -> `STATUS` answers correctly across a genuine reboot
- `SET 1 ON` - clicked, LED on, held 4 s
- `SETM 1 ON` - identical to SET
- `SETM 1,2,3,4 ON` - **all four clicked and held**; ESP32 did not reset

### Firmware hardened so this cannot recur

`isReservedPin()` rejects GPIO 0 (bootstrap), 1 and 3 (UART0 console), and 6-11
(SPI flash), enforced in two places:

- `CONFIG` and `LASER CONFIG` return `ERR PIN_RESERVED`. **Verified live:**
  PIN 3/1/6/0 all refused, `LASER CONFIG PIN 3` refused, normal pin 27 accepted.
- `loadChannels()` re-validates what it reads from NVS, leaves the channel
  unconfigured, and prints `WARN CH<n> REJECTED_STORED_PIN <p>`. A `CONFIG` guard
  alone would not help a board whose NVS is already poisoned. **This boot-time
  path is logic-only, not yet exercised against a poisoned NVS image.**

A missing pin key now reads as sentinel 255 rather than silently becoming GPIO0.

---

## Part 2 - Dr. Zhang's three priorities

### 1. Excel DB for fast entry and scanning - DONE

`scanner_system/manifest.py` + 18 tests. Verified end to end.

```
python -m scanner_system.manifest template samples.xlsx
python -m scanner_system.manifest validate samples.xlsx
python -m scanner_system.manifest run samples.xlsx
```

One row per sample. `mode` is an in-cell validated dropdown, which is the main
reason to use .xlsx here: free-text material fields in the GUI are exactly how
"Oak", "oak" and "oak " became three distinct classes.

Design decisions worth keeping:
- **The spreadsheet is NOT the database.** Mongo stays the store of record; the
  sheet is the entry and reporting surface. A workbook has no concurrency
  control and corrupts silently.
- **Results written back after every row**, not at the end. A scan is ~1 minute,
  so a 20-row manifest is a 20-minute run; losing it on row 19 is unacceptable.
- **Re-running resumes**: rows marked `complete` are skipped; `partial`/`failed`
  are retried, which is what you want after fixing something at the bench.
- Importing the module touches no hardware and opens no DB.

Friction this removes: the Streamlit Capture page costs ~8 interactions per
sample, 3 of them free-text, and **does not clear Label/Material between runs**,
so sample #2 could silently be scanned under sample #1's label.

Added `openpyxl` (pure Python, no deps, cannot disturb the `streamlit<1.50` pin).

### 2. Much better 3D - mapped, one fix shipped, work identified

**Correction to prior belief: per-pixel tilt correction is ALREADY running.**
`calib_new/gain_map_20260731.npz` holds a `b_map` (411,640) multiplicative gain
field, 83.7% valid; real scans record `"method": "per_pixel_gain_map"`. It
swings -29.6 mm/rad on the left to -11.9 on the right (global b = -24.61), a
2.5x gradient. It halves the tilt. The **remaining** error is an additive
residual, so the right next step is a background plane fit in mm space, not more
gain work. `build_pointcloud.py:55-72` (`_flatten_background`) is a ready-made,
already-validated implementation to lift.

Active calibration is `calib_new/calibration_temporal_20260731.txt`
(`a=0.579 b=-24.613 c=0`, RMSE 0.101 mm). `.claude/PROJECT_MEMORY.md` still
describes the older `a=0.0005 b=-8.008` curve - **that file is stale.**

Sign convention: **a raised object gives NEGATIVE dphi.** Get this backwards and
a tilt correction doubles the gradient.

**Shipped:** reconstruction failures are no longer invisible. Previously
`_reconstruct_height` returned silently on missing calibration, did nothing at
all on a non-zero subprocess exit, and swallowed every exception - while the
projector instrument was recorded `ok` regardless. A scan could report
`projector: ok`, `status: complete`, and contain no 3D data with the explanation
captured and discarded. Every exit path now records a `reconstruction`
instrument result carrying the reason.

### 3. Useful laser material data - built, not yet validated

`scanner_system/kinect_burst_capture.py`.

The blocker was structural, not a tuning problem. `kinect_grab_once.py` opens
the Kinect, sleeps 1.5 s, grabs ONE frame and closes it; `capture.py` shells out
to it per laser channel. That is seconds per frame with a full sensor open/close
each time, so a 33 ms transient cannot be sampled at any price. The
one-frame-per-laser design was not a choice - it was the only thing the
architecture could do.

The new script opens the sensor once and streams frames into a preallocated
buffer with a timestamp each, optionally driving the laser relay **from the same
process** so the laser edge and frame clock share one time base. Saves the frame
stack, timestamps, per-frame mean ROI intensity (the rise/decay curve itself),
and laser send/ack times.

Safety: lasers off unless `--laser-channel` is given; relay channels only (the
GPIO21 PWM path is untouched, and relays must never be PWM'd); every exit path
including Ctrl-C drives the channel off and issues SAFE.

**Taylor installed the GPIO21 5V level shifter** between 2026-08-03 and
2026-08-26, closing the "fix pending" from 2026-07-20. This unlocks
variable-power/PWM laser drive - intensity ramps rather than hard on/off - which
is what temporal material data actually wants. **Still completely untested:**
everything fired this session went through relays on GPIO18/19/25/26, and the
board reported `LASER UNCONFIGURED`.

---

## Honest correction trail (for auditability)

Four claims made during the session that were wrong and were corrected:

1. **"esptool auto-reset fails because the firmware drives GPIO0."** Wrong. The
   failure persists with a fully correct pin map, and the 2026-07-20 notes record
   the same IO0-hold requirement before any corruption existed. It is just this
   board's CP210x auto-reset circuit. Manual IO0 is normal here.
2. **"67% of scans silently produced no 3D."** Real but **historical, not
   ongoing.** All 6 failures were 12:59-14:09 on 2026-07-31; the reference was
   created 14:51 and the calibration 15:05. Every scan after that produced a
   height map. The design flaw was real - establishing this took timestamp
   forensics because the code recorded nothing - but there was no active fire.
3. **"The NVS fix explains the relays working again."** Not established, and
   should not be claimed. Taylor's 5V hardware landed in the same window, so the
   original current-starvation theory may have been correct and fixed in
   hardware. These were probably two independent problems.
4. **"Per-pixel tilt correction is an open item."** Outdated. A gain map has been
   running since 2026-07-31.

---

## State AFTER this day's work

Commits added to `feat/4dai-fusion-endpoint`:

| Commit | What |
|---|---|
| `db8a73c` | SETM multi-relay firmware command (was uncommitted and unprotected) |
| `8c6912e` | Reserved-GPIO guards so a bad pin map cannot brick the board |
| `e96616a` | `--status` no longer reports unknown firmware as an empty table |
| `e59cb03` | Spreadsheet manifest workflow + 18 tests |
| `bca43f0` | Timestamped Kinect burst capture across a laser transition |
| `5104b64` | Height reconstruction wiring + failure visibility |

Hardware state:
- ESP32: 3DAI firmware with SETM and reserved-pin guards; CH1=18, CH2=19,
  CH3=25, CH4=26, all active-HIGH, all SAFE/OFF. Verified across a real reboot.
- Kinect: **absent from USB, unpowered.** Blocks all capture.
- Laser PWM (GPIO21): level shifter installed by Taylor, `LASER UNCONFIGURED`, untested.

Tests: 18 manifest + 9 scanner_db + 2 temporal_unwrap, all passing.

Environments modified (with permission): `pyserial` into `C:/KinectEnv`;
`openpyxl` + `pytest` into `scanner_system/.venv`; `pytest` into `.venv`.
Patched pykinect2/numpy/cv2 versions verified unchanged.

---

## Next steps

**Blocking / do first**

1. **Power the Kinect.** It is absent from USB. `kinect_grab_once.py` fails
   identically to the new burst script, so use it as the "is it just my code?"
   check. Nothing can be captured until this is fixed.
2. **Get `data/` under version control.** `data/` is gitignored, so
   `reconstruct_height.py`, `build_pointcloud.py`, `capture_multifreq.py`, every
   calibration file and the gain map are **untracked and unbacked-up**. The live
   capture path shells out to `data/scan_test/reconstruct_height.py` - the single
   most important file in the 3D pipeline has no version history. This is a
   bigger risk than anything currently on the roadmap.

**Two latent landmines** (harmless today only because the active calib is the
plain 3-float variant)

3. The two PIECEWISE readers branch in **opposite** directions:
   `produce_metric_map.py:118` uses `d >= split`, `build_pointcloud.py:50` uses
   `d <= split`. One is wrong.
4. `reconstruct_height.py:55` reads calibration with a bare `np.loadtxt`, so it
   **cannot parse a PIECEWISE file** and will raise. `calib_newpos/` already
   contains two PIECEWISE files, so this is one env-var change from firing.

**Then**

5. Validate the burst capture once the Kinect is powered (camera-only first,
   then with a relay laser).
6. Bring up Taylor's GPIO21 PWM path with full 455 nm precautions, and ask Zhang
   what temporal resolution he actually needs - Kinect color is ~33 ms, and if
   the transient is faster than that, no software fixes it.
7. Background plane fit for the additive tilt residual, inserted in
   `reconstruct_height.py` after the gain map and before saving (mm space, gated
   on background pixels only, or the object biases the fit).
8. Wire `build_pointcloud.py` into the live path so output is a real colored mesh
   rather than a height PNG. Register as `fusion`/`mesh_ply`. Note
   `push_to_4dai.py:66` skips anything not `.png`, and `height_stats.json` lacks
   the row0/col0 offsets a world-coordinate mesh needs.

---

## LAYMAN'S SUMMARY

The scanner's laser controller had been dead for about two weeks. It turned out
that one bad setting, stored in the chip's own memory, told it to use its
"listening" wire as a light switch. Every time it powered on it effectively
gagged itself: it could still talk, but it could never hear another instruction.
Reinstalling the software could never fix it, because that one setting lived in
a part of memory the installer never erases. We found it, erased it, put the
correct settings back, and taught the firmware to refuse that setting forever.
All four lasers now fire correctly, together.

For Dr. Zhang's three requests: the spreadsheet workflow is finished and working
- a tech can fill in a sheet and the scanner works through it automatically,
saving results as it goes. The laser timing capture is written but cannot be
tested yet, because the Kinect camera is currently unplugged. The 3D work is
mapped out, one real bug is fixed, and the specific improvements are listed
above.

Two things need attention. The camera needs power. And the most important files
in the 3D pipeline are not backed up anywhere - they exist only on that one PC.
