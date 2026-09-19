"""Capture-time quality control: judge a scan the moment it is made.

Every incident this week produced data that looked valid: a laser whose
wire was loose still wrote a feature row, an empty table still scanned, an
auto-exposure that was still moving still saved frames. The pipeline could
not tell any of them from good data, and the only reason they were caught
was a person happening to look. This module looks, automatically, right
after each pose, from the numbers the scan already stored, and returns a
plain-English verdict the runner can act on (redo now, or move on).

It reads only the returned scan package (laser_features, fringe_features,
results, capture_config). No hardware, no database, no files: it cannot
slow a capture or fail it, and it is trivially testable.

THRESHOLDS come from real scans (2026-09-15/16):
  a dead visible laser reads core ~7, a live one 38-105  -> fired floor 15
  auto-exposure drift showed as a dark/lit gain spread of 1.38 (good ~1.0)
  a dark-room dark frame sits at the Kinect's max 66 ms exposure; room
  light or projector leak drops it below that.
Near-infrared (CH3, >=750 nm) is deliberately NOT fired-checked: the
Kinect's own IR illuminator floods that sensor, so a dark NIR frame looks
much like a lit one and any threshold would cry wolf.
"""

from __future__ import annotations

from typing import Any

CORE_FIRED_MIN = 15.0          # visible-laser dark-subtracted spot core
GAIN_SPREAD_MAX = 1.15         # max/min camera gain across a scan's frames
DARK_EXPOSURE_MIN_MS = 60.0    # dark frame should sit near the 66 ms max
SATURATED_CORE_WARN = 0.5      # share of the spot core clipped at 255
OBJECT_MIN_PX = 2000           # fringe-detected raised object region
NIR_MIN_NM = 750


def _ms(state: dict) -> float:
    return float((state or {}).get("exposure_100ns") or 0.0) / 10000.0


def check(pkg: dict[str, Any], requested_channels=None) -> dict[str, Any]:
    """Return {ok, problems, warnings, redo} for a finished scan package.

    ``problems`` are reasons to redo the pose (a real laser did not fire, the
    lights were on, the exposure was still moving, a capture instrument
    failed). ``warnings`` are worth logging but not redoing (saturation, a
    small/absent object, missing fringe signal). ``redo`` is True iff there
    are problems.
    """
    problems: list[str] = []
    warnings: list[str] = []

    lf = pkg.get("laser_features") or {}
    channels = lf.get("channels") or {}
    wavelengths = ((pkg.get("capture_config") or {}).get("wavelengths_nm")) or {}
    requested = set(requested_channels) if requested_channels is not None else {
        int(c) for c in channels}

    # 1. Instruments that outright failed (kinect/projector/laser capture).
    for inst, r in (pkg.get("results") or {}).items():
        if inst.startswith("laser_ch") or inst in (
                "reconstruction", "laser_features", "fringe_features"):
            continue          # per-channel + derived steps handled below / recomputable
        if isinstance(r, dict) and r.get("status") == "failed":
            problems.append(f"{inst} failed: {r.get('detail') or 'no detail'}")

    # 2. Each requested visible laser actually fired.
    for ch in sorted(requested):
        f = channels.get(str(ch))
        if f is None:
            problems.append(f"CH{ch} produced no laser frame")
            continue
        if f.get("valid") is False:
            problems.append(f"CH{ch} flagged invalid: {f.get('invalid_reason') or 'did not fire'}")
            continue
        wl = wavelengths.get(str(ch)) or wavelengths.get(ch)
        if wl is not None and float(wl) >= NIR_MIN_NM:
            continue          # NIR: cannot be fired-checked from these frames
        core = f.get("core")
        if core is not None and core < CORE_FIRED_MIN:
            problems.append(
                f"CH{ch} may not have fired (spot core {core:.0f}, expected >{CORE_FIRED_MIN:.0f}; "
                "check the laser wire/connector)")

    # 3. Camera exposure held steady across the scan (single-session invariant).
    exposure = lf.get("exposure") or {}
    gains = [float(v["gain"]) for v in exposure.values()
             if isinstance(v, dict) and v.get("gain")]
    if len(gains) >= 2 and min(gains) > 0 and max(gains) / min(gains) > GAIN_SPREAD_MAX:
        problems.append(
            f"camera exposure moved during capture (gain {min(gains):.1f}..{max(gains):.1f}); "
            "brightness features are not comparable, redo")

    # 4. Lights off / no projector leak: the dark frame at the exposure max.
    dark_ms = _ms(exposure.get("dark"))
    if dark_ms and dark_ms < DARK_EXPOSURE_MIN_MS:
        problems.append(
            f"dark frame exposure only {dark_ms:.0f} ms (expected ~66); "
            "room lights on or the projector is leaking light")

    # 5. Warnings: saturation, object coverage, missing fringe signal.
    for ch in sorted(requested):
        f = channels.get(str(ch)) or {}
        sat = f.get("saturated_core")
        if sat is not None and sat >= SATURATED_CORE_WARN:
            warnings.append(f"CH{ch} spot core {int(round(sat * 100))}% saturated")
    ff = pkg.get("fringe_features")
    if not ff:
        warnings.append("no fringe/3D signal (projector off, or surface holds no pattern)")
    else:
        opx = ff.get("fringe_object_px")
        if opx is not None and opx < OBJECT_MIN_PX:
            warnings.append(f"little object detected under the pattern ({opx} px); "
                            "is the object in the middle of the table?")

    return {"ok": not problems, "problems": problems, "warnings": warnings,
            "redo": bool(problems)}


def format_verdict(v: dict[str, Any]) -> str:
    """One block of plain text for the operator."""
    lines = []
    if v["problems"]:
        lines.append("  QC PROBLEMS (recommend redo):")
        lines += [f"    - {p}" for p in v["problems"]]
    if v["warnings"]:
        lines.append("  QC notes:")
        lines += [f"    - {w}" for w in v["warnings"]]
    if not lines:
        lines.append("  QC: all checks passed")
    return "\n".join(lines)
