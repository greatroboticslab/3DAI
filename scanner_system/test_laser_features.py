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


def test_gain_normalization_makes_scans_comparable():
    """The same surface captured at gain 8 and gain 4 must give the same core
    and add_* once frames are scaled to GAIN_REF; saturation stays raw."""
    import json

    def scan_at(tmp, gain):
        # a frame at gain g is the reference frame times g/GAIN_REF
        k = gain / lf.GAIN_REF
        h, w = 200, 300
        dark_ref = np.full((h, w, 3), 30, np.float32)
        add_ref = _gaussian_spot(h, w, 100, 150, 8.0, 100.0).astype(np.float32)[..., None]
        _write_png(os.path.join(tmp, "dark.png"), np.clip(dark_ref * k, 0, 255).astype(np.uint8))
        _write_png(os.path.join(tmp, "las1.png"),
                   np.clip((dark_ref + add_ref) * k, 0, 255).astype(np.uint8))
        state = {"exposure_100ns": lf.EXPOSURE_REF_100NS, "gain": gain}
        with open(os.path.join(tmp, "exposure.json"), "w") as fh:
            json.dump({"dark": state, "ch1": state}, fh)
        return lf.compute_features(tmp, channels=(1,))["channels"]["1"]

    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        lo, hi = scan_at(a, 4.0), scan_at(b, 2.0)     # gain 2 = frames half as bright
        assert abs(lo["core"] - hi["core"]) < 3.0
        assert abs(lo["add_g"] - hi["add_g"]) < 3.0
        assert lo["gain_factor"] == 1.0 and hi["gain_factor"] == 2.0
        assert hi["saturated_core"] == 0.0

    # a scan whose exposure.json has no gain (older marker) is left unscaled
    with tempfile.TemporaryDirectory() as tmp:
        _make_scan(tmp)
        f = lf.compute_features(tmp)
        assert f["normalization"]["applied"] is False
        assert f["channels"]["1"]["gain_factor"] == 1.0


def test_side_camera_frames_get_their_own_feature_block():
    """cam_dark/cam_las<N> (manual-exposure side camera) are measured like the
    Kinect frames but kept in cam_channels, un-normalized, and flattened to
    cam_* columns."""
    with tempfile.TemporaryDirectory() as tmp:
        _make_scan(tmp, sigma=8.0, amp=120.0)
        h, w = 200, 300
        cam_dark = np.full((h, w, 3), 40, np.uint8)
        _write_png(os.path.join(tmp, "cam_dark.png"), cam_dark)
        add = _gaussian_spot(h, w, 90, 140, 12.0, 90.0).astype(np.float32)
        _write_png(os.path.join(tmp, "cam_las1.png"),
                   np.clip(cam_dark.astype(np.float32) + add[..., None], 0, 255).astype(np.uint8))
        f = lf.compute_features(tmp)
        assert set(f["cam_channels"]) == {"1"}
        cam = f["cam_channels"]["1"]
        assert cam["halo_r50"] > f["channels"]["1"]["halo_r50"]      # wider synthetic spot
        assert cam["saturated_core"] == 0.0
        rows = lf.feature_rows({"_id": "s", "label": "x"}, {"_id": "sc", "laser_features": f},
                               {1: 635, 2: 635, 3: 940, 4: 530})
        by_ch = {r["channel"]: r for r in rows}
        assert by_ch[1]["cam_halo_r50"] == cam["halo_r50"] and by_ch[1]["cam_core"] == cam["core"]
        assert by_ch[2]["cam_core"] is None                           # no cam frame for CH2


def test_small_exposure_step_between_frames_is_not_a_flood():
    """A 3% exposure difference between the dark and lit frames plus a sensor
    black-level offset must not become a whole-frame offset after scaling
    (the lights-on bench test at 11 ms / gain 1 produced exactly that)."""
    import json
    with tempfile.TemporaryDirectory() as tmp:
        h, w = 200, 300
        offset = 30.0                                   # black level, both frames
        scene = np.full((h, w, 3), 100.0, np.float32)   # room-lit table
        spot = _gaussian_spot(h, w, 100, 150, 8.0, 120.0).astype(np.float32)[..., None]
        _write_png(os.path.join(tmp, "dark.png"), np.clip(scene + offset, 0, 255).astype(np.uint8))
        # lit frame: 3% shorter exposure -> scene 3% dimmer, plus the spot
        _write_png(os.path.join(tmp, "las1.png"),
                   np.clip(scene * 0.97 + offset + spot, 0, 255).astype(np.uint8))
        with open(os.path.join(tmp, "exposure.json"), "w") as fh:
            json.dump({"dark": {"exposure_100ns": 116000, "gain": 1.0},
                       "ch1": {"exposure_100ns": 113000, "gain": 1.0}}, fh)
        f = lf.compute_features(tmp, channels=(1,))["channels"]["1"]
        assert f["flood"] is False and f["lit_fraction"] < 0.05   # the spot itself, not a flood
        assert f["halo_r50"] is not None and 6 <= f["halo_r50"] <= 12
        assert f["gain_factor"] > 20                    # large multiplier, still sane


def test_tail_features_ignore_the_clipped_core():
    """Two spots with the same clipped core but different scatter widths must
    differ in the ring features; the clipped disc itself is reported."""
    def scan(tmp, sigma):
        h, w = 260, 340
        dark = np.full((h, w, 3), 40, np.uint8)
        _write_png(os.path.join(tmp, "dark.png"), dark)
        with open(os.path.join(tmp, "exposure.json"), "w") as fh:
            fh.write("{}")
        add = _gaussian_spot(h, w, 130, 170, sigma, 2000.0)     # way over 255: clipped core
        lit = np.clip(dark.astype(np.float32) + add[..., None], 0, 255).astype(np.uint8)
        _write_png(os.path.join(tmp, "las1.png"), lit)
        return lf.compute_features(tmp, channels=(1,))["channels"]["1"]

    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        narrow, wide = scan(a, 6.0), scan(b, 14.0)
        assert narrow["saturated_core"] == 1.0 and wide["saturated_core"] == 1.0
        assert narrow["clip_radius"] > 0 and wide["clip_radius"] > narrow["clip_radius"]
        assert wide["annulus_mean"] > narrow["annulus_mean"]          # more light in the ring
        assert wide["tail_efold_px"] is not None and narrow["tail_efold_px"] is not None
        assert wide["tail_efold_px"] > narrow["tail_efold_px"]        # slower fall-off
        assert narrow["bg_noise"] < 1.0                                # clean synthetic frame
        for k in ("annulus_r", "annulus_g", "annulus_b", "annulus_speckle"):
            assert wide[k] is not None
