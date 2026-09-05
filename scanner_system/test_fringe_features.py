"""Unit tests for fringe material features on synthetic stacks (no hardware)."""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scanner_system import fringe_features as ff
from scanner_system import laser_features as lf


def _stack(h, w, freqs, phases, phase_shift=None, amplitude=None):
    """Synthetic multi-frequency fringe stacks like capture_multifreq.py saves.

    phase_shift: (h, w) added phase (raised object => negative), or None.
    amplitude:   (h, w) fringe amplitude multiplier (1 = full contrast).
    """
    _, x = np.indices((h, w))
    # A real capture has a lit projector zone against a dark surround; the
    # footprint detector (Otsu on fringe amplitude) needs that contrast.
    lit = np.zeros((h, w)); lit[10:h - 10, 20:w - 20] = 1.0
    out = {}
    for fi, nf in enumerate(freqs):
        k = 2 * np.pi * nf / w
        frames = []
        for ph in phases:
            base = k * x + ph + (phase_shift if phase_shift is not None else 0)
            amp = 40.0 * lit * (amplitude if amplitude is not None else 1.0)
            frames.append(np.clip(20 + 60 * lit + amp * np.cos(base), 0, 255))
        out[f"gray_{fi}"] = np.stack(frames, axis=-1).astype(np.uint8)
    return out


def _write(dirpath, freqs, stacks):
    np.savez_compressed(os.path.join(dirpath, "scan.npz"),
                        freqs=np.array(freqs, dtype=float), **stacks)


def test_fringe_features_separate_matte_from_glossy_object():
    h, w = 120, 200
    freqs, phases = [1, 6, 24], [2 * np.pi * i / 8 for i in range(8)]
    # temporal_delta_phase reports reference-minus-object, so a POSITIVE
    # synthetic phase shift produces the negative delta this rig records for
    # a raised object (verified 2026-09-05: shift +0.6 -> delta -0.600).
    raised = np.zeros((h, w)); raised[40:80, 70:130] = +0.6     # object region
    with tempfile.TemporaryDirectory() as ref, \
         tempfile.TemporaryDirectory() as matte, \
         tempfile.TemporaryDirectory() as glossy:
        _write(ref, freqs, _stack(h, w, freqs, phases))
        _write(matte, freqs, _stack(h, w, freqs, phases, phase_shift=raised))
        washed = np.ones((h, w)); washed[40:80, 70:130] = 0.25    # fringes washed out
        _write(glossy, freqs, _stack(h, w, freqs, phases, phase_shift=raised, amplitude=washed))

        fm = ff.compute_fringe_features(matte, ref)
        fg = ff.compute_fringe_features(glossy, ref)
        assert fm and fg
        assert fm["fringe_object_px"] > 1000
        # matte object holds contrast like the background; glossy does not
        assert fm["fringe_contrast_ratio"] > 0.8
        assert fg["fringe_contrast_ratio"] < fm["fringe_contrast_ratio"] * 0.6
        assert 0 < fm["fringe_reliable_fraction"] <= 1


def test_fringe_features_none_without_stacks():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        assert ff.compute_fringe_features(a, b) is None


def test_feature_rows_carry_fringe_columns():
    sample = {"_id": "s", "label": "x", "material": {}, "context": {}}
    scan = {"_id": "sc", "laser_features": {"channels": {"1": {"core": 1.0}}},
            "fringe_features": {"fringe_contrast_ratio": 0.42, "fringe_albedo_object": 88.0}}
    row = lf.feature_rows(sample, scan, {1: 635})[0]
    assert row["fringe_contrast_ratio"] == 0.42 and row["fringe_albedo_object"] == 88.0
    assert "fringe_object_px" in row and row["fringe_object_px"] is None
