"""Per-scan projector/fringe material features: contrast, albedo, coverage.

The structured-light stack is captured for 3D, but it carries a
calibration-free material signal that the height pipeline discards: how
well a surface HOLDS the projected fringes. A glossy surface washes the
pattern out (specular return), a matte surface holds it, a translucent one
blurs it through subsurface scattering. The June/July calibration work saw
exactly this (white matte near the contrast floor, blue tape holding ~87,
glossy cases with high variance). Store it per scan next to the laser
features so a model can use it and so it is never re-derived by hand.

FEATURES (all computed on the highest-frequency stack, object vs background)
    fringe_contrast_object      mean fringe amplitude on the object region
    fringe_contrast_background  mean fringe amplitude on the empty stage
    fringe_contrast_ratio       object / background (surface finish signal,
                                normalised against that scan's own projector
                                brightness and exposure)
    fringe_albedo_object        mean projector illumination reflected by the
                                object: the phase-averaged stack intensity,
                                i.e. brightness under known full illumination
    fringe_reliable_fraction    share of footprint pixels passing the contrast
                                gate (coverage; drops on very dark or very
                                glossy objects)
    fringe_object_px            size of the detected object region

Object/background come from the same delta-phase and contrast gate the
reconstruction uses (raised object => negative dphi on this rig), so the
numbers describe the same pixels the height map does. Pure numpy + fpp_tools;
no hardware, no database.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np


def compute_fringe_features(fringe_dir: str, reference_dir: str) -> Optional[dict[str, Any]]:
    """Features for one scan's fringe capture against the empty-stage reference.

    Returns None when either stack is missing or the frequencies differ.
    """
    from fpp_tools.temporal_unwrap import load_multifrequency_npz, temporal_delta_phase

    if not (os.path.isfile(os.path.join(fringe_dir, "scan.npz"))
            and os.path.isfile(os.path.join(reference_dir, "scan.npz"))):
        return None
    freqs, obj = load_multifrequency_npz(fringe_dir)
    ref_freqs, ref = load_multifrequency_npz(reference_dir)
    if not np.allclose(freqs, ref_freqs):
        return None

    try:
        result = temporal_delta_phase(ref, obj, freqs, method="wrapped")
    except ValueError:
        # No detectable projector footprint (projector off, or the reference
        # is not a real capture). No fringes means no fringe features.
        return None
    d, rel, con = result.delta, result.reliable, result.contrast
    fp = result.footprint

    background = rel & (np.abs(d) < 0.08)
    if background.sum() < 500:
        return None
    noise = float(np.std(d[background]))
    obj_mask = rel & (d < -max(0.1, 5.0 * noise))

    # Phase-averaged intensity of the highest-frequency stack over the same
    # crop = reflected projector illumination (albedo proxy).
    hi = obj[-1][fp.row0:fp.row1, fp.col0:fp.col1].astype(np.float32)
    mean_illum = hi.mean(axis=2)

    def _m(a: np.ndarray, mask: np.ndarray) -> Optional[float]:
        return float(a[mask].mean()) if mask.any() else None

    c_obj, c_bg = _m(con, obj_mask), _m(con, background)
    # Spread of the contrast over the object: texture / mixed finish shows as
    # a wide spread, a uniform surface as a narrow one. Independent of the
    # laser spot and of the clipped core.
    c_obj_std = float(np.std(con[obj_mask])) if obj_mask.sum() > 50 else None
    return {
        "fringe_contrast_object": None if c_obj is None else round(c_obj, 2),
        "fringe_contrast_object_std": None if c_obj_std is None else round(c_obj_std, 2),
        "fringe_contrast_background": None if c_bg is None else round(c_bg, 2),
        "fringe_contrast_ratio": (round(c_obj / c_bg, 3)
                                  if c_obj is not None and c_bg else None),
        "fringe_albedo_object": (None if not obj_mask.any()
                                 else round(float(mean_illum[obj_mask].mean()), 1)),
        "fringe_albedo_background": round(float(mean_illum[background].mean()), 1),
        "fringe_reliable_fraction": round(float(rel.mean()), 4),
        "fringe_object_px": int(obj_mask.sum()),
        "fringe_footprint_rc": [int(fp.row0), int(fp.row1), int(fp.col0), int(fp.col1)],
        "fringe_background_noise_rad": round(noise, 4),
    }
