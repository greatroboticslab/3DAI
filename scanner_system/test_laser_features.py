"""Unit tests for the laser material features: synthetic spots, no hardware."""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scanner_system import laser_features as lf


def _write_png(path, arr):
    from PIL import Image
    Image.fromarray(arr).save(path)


def _gaussian_spot(h, w, cy, cx, sigma, amp):
    yy, xx = np.mgrid[:h, :w]
    return amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))


def _make_scan(tmp, sigma=8.0, amp=120.0, noise=0.0, flood_level=None):
    h, w = 200, 300
    dark = np.full((h, w, 3), 60, np.uint8)
    _write_png(os.path.join(tmp, "dark.png"), dark)
    with open(os.path.join(tmp, "exposure.json"), "w") as fh:
        fh.write("{}")           # single-session marker
    rng = np.random.default_rng(0)
    for ch in (1, 2, 3, 4):
        if flood_level is not None and ch == 4:
            add = np.full((h, w), flood_level, np.float32)
        else:
            add = _gaussian_spot(h, w, 100, 150, sigma, amp).astype(np.float32)
        if noise:
            add = add * (1 + noise * rng.standard_normal(add.shape)).astype(np.float32)
        lit = np.clip(dark.astype(np.float32) + add[..., None], 0, 255).astype(np.uint8)
        _write_png(os.path.join(tmp, f"las{ch}.png"), lit)


def test_gaussian_halo_width_matches_sigma():
    with tempfile.TemporaryDirectory() as tmp:
        _make_scan(tmp, sigma=8.0)
        f = lf.compute_features(tmp)
        ch = f["channels"]["1"]
        # ring mean falls to 50% at r = sigma*sqrt(2 ln 2) ~ 9.4 px
        assert 8 <= ch["halo_r50"] <= 11, ch
        assert 16 <= ch["halo_r10"] <= 20, ch
        assert abs(ch["spot_x"] - 150) < 2 and abs(ch["spot_y"] - 100) < 2
        assert ch["flood"] is False
        assert ch["saturated_core"] == 0.0


def test_tighter_spot_gives_smaller_halo():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        _make_scan(a, sigma=4.0); _make_scan(b, sigma=12.0)
        ra = lf.compute_features(a)["channels"]["1"]["halo_r50"]
        rb = lf.compute_features(b)["channels"]["1"]["halo_r50"]
        assert ra < rb


def test_speckle_rises_with_granular_spot():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        _make_scan(a, noise=0.0); _make_scan(b, noise=0.6)
        sa = lf.compute_features(a)["channels"]["1"]["speckle"]
        sb = lf.compute_features(b)["channels"]["1"]["speckle"]
        assert sb > sa + 0.2


def test_flood_channel_flagged_and_halo_na():
    with tempfile.TemporaryDirectory() as tmp:
        _make_scan(tmp, flood_level=40.0)
        ch4 = lf.compute_features(tmp)["channels"]["4"]
        assert ch4["flood"] is True
        assert ch4["halo_r50"] is None and ch4["halo_r10"] is None


def test_red_green_ratio_and_no_dark_means_none():
    with tempfile.TemporaryDirectory() as tmp:
        _make_scan(tmp)
        f = lf.compute_features(tmp)
        assert f["red_green_ratio"] is not None and abs(f["red_green_ratio"] - 1.0) < 0.05
        os.unlink(os.path.join(tmp, "dark.png"))
        assert lf.compute_features(tmp) is None


def test_feature_rows_flatten_with_labels():
    sample = {"_id": "s1", "label": "blk", "material": {"class": "metal", "subclass": "al"},
              "context": {"surface": "glossy", "transparency": "opaque"}}
    scan = {"_id": "sc1", "angle": {"index": 2, "count": 3},
            "laser_features": {"channels": {"3": {"core": 5.0, "halo_r50": 7, "speckle": 0.4}},
                               "red_green_ratio": 0.5}}
    rows = lf.feature_rows(sample, scan, {3: 940})
    assert len(rows) == 1 and list(rows[0]) == lf.FEATURE_COLUMNS
    r = rows[0]
    assert r["channel"] == 3 and r["wavelength_nm"] == 940 and r["ir_laser"] is True
    assert r["surface"] == "glossy" and r["angle_index"] == 2 and r["halo_r50"] == 7


def test_spot_found_despite_scattered_noise_spikes():
    """Isolated bright pixels far from the spot must not pull the center."""
    with tempfile.TemporaryDirectory() as tmp:
        _make_scan(tmp, sigma=6.0, amp=80.0)
        from PIL import Image
        p = os.path.join(tmp, "las1.png")
        lit = np.asarray(Image.open(p)).copy()
        rng = np.random.default_rng(1)
        ys = rng.integers(0, lit.shape[0], 3000); xs = rng.integers(0, lit.shape[1], 3000)
        lit[ys, xs] = 255                     # 5% of the frame as hot pixels
        lit[100 - 30:100 + 30, 150 - 30:150 + 30] = np.asarray(Image.open(p))[100-30:100+30, 150-30:150+30]
        _write_png(p, lit)
        ch = lf.compute_features(tmp)["channels"]["1"]
        assert abs(ch["spot_x"] - 150) < 3 and abs(ch["spot_y"] - 100) < 3, ch
        assert ch["core"] > 50


def test_pre_single_session_scan_is_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        _make_scan(tmp)
        os.unlink(os.path.join(tmp, "exposure.json"))
        assert lf.compute_features(tmp) is None


def test_flood_background_reported_only_for_flood_channels():
    with tempfile.TemporaryDirectory() as tmp:
        _make_scan(tmp, flood_level=40.0)
        chs = lf.compute_features(tmp)["channels"]
        assert chs["4"]["flood"] is True and abs(chs["4"]["flood_background"] - 40.0) < 2.0
        assert chs["1"]["flood_background"] is None
