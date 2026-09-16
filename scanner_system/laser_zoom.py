"""A human-readable laser view built from the frames a scan already stores.

Dr. Zhang's request: make the laser images readable, the spot big and
visible. The Kinect cannot be told to expose less (no manual exposure in
the SDK) and the lasers cannot be defocused, so in the raw frame the spot
is a few pixels of clipped white on a bright table. This makes the picture
a person wants to look at from the data we have:

    one row per laser channel, three tiles each:
      lit      the raw frame, zoomed on the spot
      halo     lit minus dark, stretched with a gamma so the faint scatter
               halo around the core becomes visible (the material signal)
      profile  the radial intensity fall-off, core to halo, on a log axis

Zoom window and scale are fixed so tiles are comparable between objects.
The spot centre comes from the stored features (spot_x/spot_y), which are in
the coordinates of the cropped PNGs on disk. Nothing here changes any
number in the dataset; it is a rendering.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

WINDOW = 120          # half-size of the zoom window in source pixels
SCALE = 2             # upscale factor for the tiles (3 made ~1 MB files; 400 scans would be 400 MB)
GAMMA = 0.45          # < 1 lifts the faint halo
CHANNEL_NAMES = {1: "CH1 red 635 nm", 2: "CH2 red 635 nm", 3: "CH3 NIR 940 nm", 4: "CH4 green 530 nm"}


IR_CHANNELS = {3}     # channels the colour camera cannot see (NIR)


def _load(path: str) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def _load_ir_rgb(path: str) -> np.ndarray:
    """16-bit IR frame as a float RGB array on the 0-255 scale (grey)."""
    from PIL import Image
    a = np.asarray(Image.open(path), dtype=np.float32) / 257.0
    return np.repeat(a[..., None], 3, axis=2)


def _window(a: np.ndarray, cy: float, cx: float) -> np.ndarray:
    """Fixed-size window around (cy, cx), padded at the frame edges."""
    h, w = a.shape[:2]
    y0, x0 = int(round(cy)) - WINDOW, int(round(cx)) - WINDOW
    out = np.zeros((2 * WINDOW, 2 * WINDOW) + a.shape[2:], dtype=a.dtype)
    ys, ye = max(0, y0), min(h, y0 + 2 * WINDOW)
    xs, xe = max(0, x0), min(w, x0 + 2 * WINDOW)
    out[ys - y0:ye - y0, xs - x0:xe - x0] = a[ys:ye, xs:xe]
    return out


def _stretch_halo(diff: np.ndarray) -> np.ndarray:
    """Dark-subtracted window -> 8-bit with the halo lifted; colour kept."""
    mag = diff.mean(axis=2)
    top = float(np.percentile(mag, 99.9)) or 1.0
    norm = np.clip(mag / top, 0.0, 1.0) ** GAMMA
    # tint by the laser's own colour balance in the diff, so red reads red
    rgb = np.clip(diff, 0, None)
    tint = rgb.reshape(-1, 3).mean(axis=0)
    tint = tint / (tint.max() or 1.0)
    return (norm[..., None] * tint[None, None, :] * 255.0).astype(np.uint8)


def _profile_tile(diff: np.ndarray, size: int) -> np.ndarray:
    """Radial fall-off of the spot on a log axis, drawn as a small plot."""
    from PIL import Image, ImageDraw
    mag = diff.mean(axis=2)
    cy = cx = WINDOW
    yy, xx = np.ogrid[:mag.shape[0], :mag.shape[1]]
    r = np.hypot(yy - cy, xx - cx).astype(int)
    rmax = WINDOW - 1
    prof = np.array([mag[r == k].mean() if (r == k).any() else 0.0 for k in range(rmax)])
    prof = np.clip(prof, 1e-3, None)
    img = Image.new("RGB", (size, size), (20, 20, 24))
    d = ImageDraw.Draw(img)
    pad = 24
    lo, hi = np.log10(1e-3), np.log10(max(prof.max(), 1.0))
    pts = []
    for k, v in enumerate(prof):
        x = pad + (size - 2 * pad) * k / max(rmax - 1, 1)
        y = size - pad - (size - 2 * pad) * (np.log10(v) - lo) / max(hi - lo, 1e-6)
        pts.append((x, y))
    d.line(pts, fill=(255, 210, 80), width=2)
    d.line([(pad, size - pad), (size - pad, size - pad)], fill=(120, 120, 120))
    d.line([(pad, pad), (pad, size - pad)], fill=(120, 120, 120))
    d.text((pad + 2, pad - 14), "intensity (log)", fill=(180, 180, 180))
    d.text((size - pad - 60, size - pad + 4), f"r = {rmax} px", fill=(180, 180, 180))
    return np.asarray(img)


def make_laser_zoom(laser_dir: str, features: dict[str, Any], out_path: str,
                    channels=(1, 2, 3, 4)) -> Optional[str]:
    """Write the composite for one scan; returns out_path or None if no frames."""
    from PIL import Image, ImageDraw

    dark_path = os.path.join(laser_dir, "dark.png")
    if not os.path.isfile(dark_path):
        return None
    dark = _load(dark_path)
    tile = 2 * WINDOW * SCALE
    label_h = 22
    rows = []
    dark_ir_path = os.path.join(laser_dir, "dark_ir.png")
    for ch in channels:
        lit_path = os.path.join(laser_dir, f"las{ch}.png")
        lit_ir_path = os.path.join(laser_dir, f"las{ch}_ir.png")
        feat = (features.get("channels") or {}).get(str(ch)) or {}
        # The colour camera is filtered against near-infrared. The Kinect's IR
        # sensor can see the NIR spot, but its own illuminator usually
        # saturates it (ir_saturated_core ~1), in which case the IR frames
        # are noise and the row is shown from the colour camera, labelled.
        ir_usable = (ch in IR_CHANNELS and feat.get("ir_spot_x") is not None
                     and float(feat.get("ir_saturated_core") or 1.0) < 0.5
                     and os.path.isfile(lit_ir_path) and os.path.isfile(dark_ir_path))
        if ir_usable:
            lit = _load_ir_rgb(lit_ir_path)
            dark_f = _load_ir_rgb(dark_ir_path)
            cy, cx = float(feat["ir_spot_y"]), float(feat["ir_spot_x"])
            source = "IR sensor"
        else:
            if not os.path.isfile(lit_path) or feat.get("spot_x") is None:
                continue
            lit, dark_f = _load(lit_path), dark
            cy, cx = float(feat["spot_y"]), float(feat["spot_x"])
            source = ("colour camera; NIR not visible here, IR sensor saturated"
                      if ch in IR_CHANNELS else "colour camera")
        lit_w = _window(lit, cy, cx)
        diff_w = lit_w - _window(dark_f, cy, cx)
        tiles = [
            Image.fromarray(np.clip(lit_w, 0, 255).astype(np.uint8)).resize((tile, tile), Image.NEAREST),
            Image.fromarray(_stretch_halo(diff_w)).resize((tile, tile), Image.NEAREST),
            Image.fromarray(_profile_tile(diff_w, tile)),
        ]
        valid = feat.get("valid", True)
        name = (CHANNEL_NAMES.get(ch, f"CH{ch}") + f" [{source}]"
                + ("" if valid else "   (INVALID: laser did not fire)"))
        rows.append((name, tiles, feat))

        # Side camera with manual exposure, when the scan has it: the frame
        # Dr. Zhang asked for, spot core unclipped and the object visible.
        camf = (features.get("cam_channels") or {}).get(str(ch)) or {}
        cam_lit, cam_dark_p = (os.path.join(laser_dir, f"cam_las{ch}.png"),
                               os.path.join(laser_dir, "cam_dark.png"))
        if camf.get("spot_x") is not None and os.path.isfile(cam_lit) and os.path.isfile(cam_dark_p):
            ccy, ccx = float(camf["spot_y"]), float(camf["spot_x"])
            c_lit_w = _window(_load(cam_lit), ccy, ccx)
            c_diff_w = c_lit_w - _window(_load(cam_dark_p), ccy, ccx)
            rows.append((CHANNEL_NAMES.get(ch, f"CH{ch}") + " [side camera, manual exposure]", [
                Image.fromarray(np.clip(c_lit_w, 0, 255).astype(np.uint8)).resize((tile, tile), Image.NEAREST),
                Image.fromarray(_stretch_halo(c_diff_w)).resize((tile, tile), Image.NEAREST),
                Image.fromarray(_profile_tile(c_diff_w, tile)),
            ], camf))
    if not rows:
        return None

    w = 3 * tile
    h = len(rows) * (tile + label_h)
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    d = ImageDraw.Draw(canvas)
    for i, (name, tiles, feat) in enumerate(rows):
        y = i * (tile + label_h)
        info = (f"{name}    core {feat.get('core')}  halo r50 {feat.get('halo_r50')} px  "
                f"r10 {feat.get('halo_r10')} px  speckle {feat.get('speckle')}  "
                f"saturated {feat.get('saturated_core')}")
        d.text((6, y + 4), info, fill=(230, 230, 230))
        for j, t in enumerate(tiles):
            canvas.paste(t, (j * tile, y + label_h))
        for j, cap in enumerate((f"lit (raw, zoom x{SCALE})", "lit minus dark, halo stretched", "radial profile")):
            d.text((j * tile + 6, y + label_h + 4), cap, fill=(255, 255, 255))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path, "JPEG", quality=82)
    return out_path
