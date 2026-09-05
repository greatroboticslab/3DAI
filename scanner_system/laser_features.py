"""Per-scan laser material features: scatter halo, reflectance, speckle, IR.

WHY THIS EXISTS
---------------
The laser frames are the raw record; these numbers are the material signal
a person reads in a spreadsheet and a model trains on. They were first
computed by hand on cardboard vs polished metal (2026-09-05) and shown to
separate matte from glossy; this module makes them a stored, per-scan product
instead of an ad-hoc analysis.

Inputs are the single-session laser artifacts of one scan (color and infrared
frames, dark references), which is what makes the subtraction clean: dark and
lit frames share one exposure state. All maths is on laser-minus-dark.

FEATURES PER CHANNEL (color camera)
    spot_x, spot_y        spot center, crop-frame pixels (argmax of a blurred
                          laser-minus-dark map; robust to speckle spikes)
    core                  mean added brightness in the central disc (r <= 3 px).
                          Deliberately a disc mean, not the peak pixel: a
                          laser spot is speckly, and normalizing to the
                          brightest pixel made halo widths collapse to 1 px on
                          rough surfaces (observed on cardboard CH1).
    halo_r50, halo_r10    radius (px) where the ring-mean profile falls below
                          50% / 10% of core. Tight = glossy/hard, wide = matte
                          or subsurface-scattering. Capped at 60 px.
    halo_energy_10_40     added light within r<=10 as a fraction of r<=40.
                          Robust companion to r50/r10 (1.0 = pinpoint).
    speckle               std/mean of added light in a 24x24 patch at the spot.
                          Coherent light granulates on rough surfaces.
    add_r, add_g, add_b   mean added R,G,B at the spot: reflectance colour.
    lit_fraction, flood   share of the frame the laser changed; flood = the
                          channel illuminates the whole scene (CH4 green), so
                          it is a REFLECTANCE probe and halo numbers are n/a.
    saturated_core        fraction of core pixels clipped at 255 (trust flag).

FEATURES PER CHANNEL (infrared camera, when the *_ir frames exist)
    ir_core, ir_halo_r50, ir_halo_r10, ir_speckle, ir_saturated_core
    For CH3 (near-infrared) these are the frames that actually contain the
    laser. The IR sensor clips at 65535 on bright spots; ir_saturated_core
    says how much of the core is clipped.

SCAN-LEVEL
    red_green_ratio       ch1 core / ch4 core (colour reflectance fingerprint)
    red2_green_ratio      ch2 core / ch4 core

Pure numpy + PIL. No hardware, no database: compute_features() takes a
directory and returns a dict; callers store it.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import numpy as np

CORE_RADIUS = 3
PATCH = 12                 # half-size of the speckle patch (24x24)
RMAX = 60
LIT_THRESHOLD = 15.0       # "this pixel changed" on the 0-255 scale
FLOOD_FRACTION = 0.25      # more than this share of the frame lit = flood
IR_LIT_THRESHOLD = 2000.0  # on the uint16 scale


def _load_rgb(path: str) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def _load_ir(path: str) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path), dtype=np.float32)


def _box_blur(a: np.ndarray, k: int = 7) -> np.ndarray:
    """Separable box blur via cumulative sums (no scipy dependency)."""
    pad = k // 2
    ap = np.pad(a, pad, mode="edge")
    c = np.cumsum(ap, axis=0); c = np.vstack([np.zeros((1, c.shape[1])), c])
    a1 = (c[k:] - c[:-k]) / k
    c = np.cumsum(a1, axis=1); c = np.hstack([np.zeros((c.shape[0], 1)), c])
    return (c[:, k:] - c[:, :-k]) / k


def _spot_center(mag: np.ndarray) -> tuple[float, float]:
    """Laser spot center = argmax of a blurred magnitude map.

    A percentile-of-brightest-pixels centroid was tried first and failed on
    real scans: 0.5% of a 1018x723 frame is ~3,700 pixels, far more than a
    ~200 px laser spot, so noise elsewhere pulled the centroid off the spot
    and the core read ~0. Blurring with a 7 px box then taking the maximum
    ignores isolated speckle spikes and scattered noise and lands on the
    densest bright region, which is the spot.
    """
    b = _box_blur(np.clip(mag, 0, None), 7)
    cy, cx = np.unravel_index(int(np.argmax(b)), b.shape)
    return float(cy), float(cx)


def _radial(mag: np.ndarray, cy: float, cx: float) -> tuple[float, int, int, float]:
    """(core, r50, r10, energy_10_40) from ring means around (cy, cx)."""
    h, w = mag.shape
    y0, y1 = max(0, int(cy) - RMAX - 1), min(h, int(cy) + RMAX + 2)
    x0, x1 = max(0, int(cx) - RMAX - 1), min(w, int(cx) + RMAX + 2)
    sub = mag[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    rr = np.hypot(yy - cy, xx - cx)

    core_mask = rr <= CORE_RADIUS
    core = float(sub[core_mask].mean()) if core_mask.any() else 0.0
    if core <= 0:
        return 0.0, RMAX, RMAX, 0.0

    r50 = r10 = RMAX
    found50 = found10 = False
    for r in range(CORE_RADIUS + 1, RMAX + 1):
        ring = (rr >= r - 0.5) & (rr < r + 0.5)
        if not ring.any():
            continue
        v = float(sub[ring].mean()) / core
        if not found50 and v < 0.5:
            r50, found50 = r, True
        if not found10 and v < 0.1:
            r10, found10 = r, True
            break

    pos = np.clip(sub, 0, None)
    e10 = float(pos[rr <= 10].sum())
    e40 = float(pos[rr <= 40].sum())
    energy = e10 / e40 if e40 > 0 else 0.0
    return core, int(r50), int(r10), energy


def _speckle(mag: np.ndarray, cy: float, cx: float) -> float:
    y, x = int(cy), int(cx)
    patch = mag[max(0, y - PATCH):y + PATCH, max(0, x - PATCH):x + PATCH]
    m = float(patch.mean()) if patch.size else 0.0
    return float(patch.std() / m) if m > 2.0 else 0.0


def _color_channel(lit_path: str, dark: np.ndarray) -> dict[str, Any]:
    lit = _load_rgb(lit_path)
    diff = lit - dark
    mag = diff.mean(axis=2)
    cy, cx = _spot_center(mag)
    core, r50, r10, energy = _radial(mag, cy, cx)
    lit_frac = float((mag > LIT_THRESHOLD).mean())

    h, w = mag.shape
    yy, xx = np.ogrid[:h, :w]
    core_mask = np.hypot(yy - cy, xx - cx) <= CORE_RADIUS
    # Reflectance colour read at the same core disc as core/halo, so every
    # number in a row describes the same patch of surface.
    add = [float(diff[:, :, i][core_mask].mean()) for i in range(3)] if core_mask.any() else [0.0] * 3
    sat = float((lit.max(axis=2)[core_mask] >= 250).mean()) if core_mask.any() else 0.0

    flood = lit_frac > FLOOD_FRACTION
    # The stage is a flat white matte table, present in every scan. For a
    # flood channel, the laser's brightness on the table far from the spot
    # is a per-scan reference for laser output drift, independent of the
    # object. For spot lasers this is ~0 by construction (they hit the
    # object), so it is reported only for flood channels.
    far = np.hypot(yy - cy, xx - cx) > 150
    flood_bg = float(mag[far].mean()) if (flood and far.any()) else None

    return {
        "flood_background": None if flood_bg is None else round(flood_bg, 2),
        "spot_x": round(cx, 1), "spot_y": round(cy, 1),
        "core": round(core, 2),
        "halo_r50": None if flood else r50,
        "halo_r10": None if flood else r10,
        "halo_energy_10_40": None if flood else round(energy, 3),
        "speckle": round(_speckle(mag, cy, cx), 3),
        "add_r": round(add[0], 1), "add_g": round(add[1], 1), "add_b": round(add[2], 1),
        "lit_fraction": round(lit_frac, 4),
        "flood": bool(flood),
        "saturated_core": round(sat, 2),
    }


def _ir_channel(lit_ir_path: str, dark_ir: np.ndarray) -> dict[str, Any]:
    lit = _load_ir(lit_ir_path)
    mag = lit - dark_ir
    cy, cx = _spot_center(mag)
    core, r50, r10, energy = _radial(mag, cy, cx)
    h, w = mag.shape
    yy, xx = np.ogrid[:h, :w]
    core_mask = np.hypot(yy - cy, xx - cx) <= CORE_RADIUS
    sat = float((lit[core_mask] >= 65000).mean()) if core_mask.any() else 0.0
    return {
        "ir_spot_x": round(cx, 1), "ir_spot_y": round(cy, 1),
        "ir_core": round(core, 0),
        "ir_halo_r50": r50, "ir_halo_r10": r10,
        "ir_halo_energy_10_40": round(energy, 3),
        "ir_speckle": round(_speckle(mag, cy, cx), 3),
        "ir_lit_fraction": round(float((mag > IR_LIT_THRESHOLD).mean()), 4),
        "ir_saturated_core": round(sat, 2),
    }


def compute_features(laser_dir: str, channels=(1, 2, 3, 4)) -> Optional[dict[str, Any]]:
    """Features for one scan's laser directory. None if no dark frame."""
    dark_path = os.path.join(laser_dir, "dark.png")
    if not os.path.isfile(dark_path):
        return None
    # Only single-session captures are subtractable: kinect_laser_sequence.py
    # writes exposure.json, and only there do dark and lit frames share an
    # exposure state. Older scans grabbed each frame in a separate process and
    # carry a -12..-15 background offset that would masquerade as signal.
    if not os.path.isfile(os.path.join(laser_dir, "exposure.json")):
        return None
    dark = _load_rgb(dark_path)
    dark_ir_path = os.path.join(laser_dir, "dark_ir.png")
    dark_ir = _load_ir(dark_ir_path) if os.path.isfile(dark_ir_path) else None

    out: dict[str, Any] = {"channels": {}}
    for ch in channels:
        lit = os.path.join(laser_dir, f"las{ch}.png")
        if not os.path.isfile(lit):
            continue
        feats = _color_channel(lit, dark)
        lit_ir = os.path.join(laser_dir, f"las{ch}_ir.png")
        if dark_ir is not None and os.path.isfile(lit_ir):
            feats.update(_ir_channel(lit_ir, dark_ir))
        out["channels"][str(ch)] = feats

    chs = out["channels"]
    g = chs.get("4", {}).get("core") or 0.0
    out["red_green_ratio"] = round(chs["1"]["core"] / g, 3) if "1" in chs and g > 0 else None
    out["red2_green_ratio"] = round(chs["2"]["core"] / g, 3) if "2" in chs and g > 0 else None

    exp_path = os.path.join(laser_dir, "exposure.json")
    if os.path.isfile(exp_path):
        try:
            with open(exp_path) as fh:
                out["exposure"] = json.load(fh)
        except Exception:
            pass
    return out


# Column order for the spreadsheet tab and the export sheet: one row per
# (scan, channel). Kept here so every writer agrees.
FEATURE_COLUMNS = [
    "label", "material_class", "material_subclass", "surface", "transparency",
    "angle_index", "channel", "wavelength_nm", "ir_laser",
    "core", "halo_r50", "halo_r10", "halo_energy_10_40", "speckle",
    "add_r", "add_g", "add_b", "flood", "flood_background", "saturated_core",
    "ir_core", "ir_halo_r50", "ir_halo_r10", "ir_halo_energy_10_40", "ir_speckle",
    "ir_saturated_core",
    "red_green_ratio", "red2_green_ratio",
    # per-scan projector/fringe features (same value on every channel row)
    "fringe_contrast_object", "fringe_contrast_background", "fringe_contrast_ratio",
    "fringe_albedo_object", "fringe_albedo_background", "fringe_reliable_fraction",
    "fringe_object_px",
    "sample_id", "scan_id",
]


def feature_rows(sample: dict[str, Any], scan: dict[str, Any],
                 wavelengths: dict[int, Optional[int]]) -> list[dict[str, Any]]:
    """Flatten a scan's stored features into FEATURE_COLUMNS rows."""
    feats = scan.get("laser_features") or {}
    fringe = scan.get("fringe_features") or {}
    ctx = sample.get("context") or {}
    mat = sample.get("material") or {}
    rows = []
    for ch_txt, f in (feats.get("channels") or {}).items():
        ch = int(ch_txt)
        wl = wavelengths.get(ch)
        row = {
            "label": sample.get("label"),
            "material_class": mat.get("class"), "material_subclass": mat.get("subclass"),
            "surface": ctx.get("surface"), "transparency": ctx.get("transparency"),
            "angle_index": (scan.get("angle") or {}).get("index"),
            "channel": ch, "wavelength_nm": wl,
            "ir_laser": bool(wl is not None and wl >= 750),
            "red_green_ratio": feats.get("red_green_ratio"),
            "red2_green_ratio": feats.get("red2_green_ratio"),
            "sample_id": sample.get("_id"), "scan_id": scan.get("_id"),
        }
        for k in FEATURE_COLUMNS:
            if k in f:
                row[k] = f[k]
            elif k in fringe:
                row[k] = fringe[k]
        rows.append({k: row.get(k) for k in FEATURE_COLUMNS})
    return rows
