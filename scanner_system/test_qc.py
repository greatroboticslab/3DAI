"""Capture-time QC verdicts, from synthetic scan packages (no hardware)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scanner_system import qc


def _pkg(cores=(40, 40, 5, 40), gains=(4.6, 4.6, 4.6, 4.6, 4.6), dark_ms=66.5,
         wavelengths=None, results=None, object_px=25000, sat=(0, 0, 0, 0),
         valid=(True, True, True, True), fringe=True):
    wavelengths = wavelengths or {"1": 635, "2": 635, "3": 940, "4": 530}
    channels = {}
    for i, ch in enumerate("1234"):
        channels[ch] = {"core": cores[i], "saturated_core": sat[i], "valid": valid[i]}
        if not valid[i]:
            channels[ch]["invalid_reason"] = "loose wire"
    exposure = {"dark": {"exposure_100ns": int(dark_ms * 10000), "gain": gains[0]}}
    for i, ch in enumerate("1234"):
        exposure[f"ch{ch}"] = {"exposure_100ns": 664719, "gain": gains[i + 1]}
    pkg = {
        "_id": "scan", "laser_features": {"channels": channels, "exposure": exposure},
        "capture_config": {"wavelengths_nm": wavelengths},
        "results": results or {"kinect": {"status": "ok"}, "projector": {"status": "ok"},
                               "laser": {"status": "ok"}},
    }
    if fringe:
        pkg["fringe_features"] = {"fringe_object_px": object_px}
    return pkg


def test_good_scan_passes():
    v = qc.check(_pkg(), [1, 2, 3, 4])
    assert v["ok"] and not v["problems"] and not v["redo"]


def test_dead_visible_laser_flagged():
    v = qc.check(_pkg(cores=(40, 40, 5, 7)), [1, 2, 3, 4])   # CH4 core 7 = dead
    assert v["redo"]
    assert any("CH4" in p and "fired" in p for p in v["problems"])
    # a low NIR (CH3) core is NOT flagged: the IR floods that sensor
    assert not any("CH3" in p for p in v["problems"])


def test_nir_low_core_not_flagged():
    v = qc.check(_pkg(cores=(40, 40, 1, 40)), [1, 2, 3, 4])
    assert v["ok"]


def test_explicit_invalid_channel_flagged():
    v = qc.check(_pkg(valid=(True, True, True, False)), [1, 2, 3, 4])
    assert v["redo"] and any("CH4" in p and "invalid" in p for p in v["problems"])


def test_exposure_drift_flagged():
    v = qc.check(_pkg(gains=(6.1, 4.4, 4.4, 4.4, 4.4)), [1, 2, 3, 4])   # spread 1.39
    assert v["redo"] and any("exposure moved" in p for p in v["problems"])


def test_lights_on_flagged():
    v = qc.check(_pkg(dark_ms=33.0), [1, 2, 3, 4])
    assert v["redo"] and any("room lights" in p for p in v["problems"])


def test_instrument_failure_flagged():
    v = qc.check(_pkg(results={"kinect": {"status": "ok"},
                               "projector": {"status": "failed", "detail": "fringe timed out"}}),
                 [1, 2, 3, 4])
    assert v["redo"] and any("projector failed" in p for p in v["problems"])
    # a derived-step failure must NOT trigger a redo (it is recomputable)
    v2 = qc.check(_pkg(results={"kinect": {"status": "ok"}, "projector": {"status": "ok"},
                                "reconstruction": {"status": "failed"}}), [1, 2, 3, 4])
    assert not any("reconstruction" in p for p in v2["problems"])


def test_warnings_do_not_force_redo():
    v = qc.check(_pkg(sat=(1.0, 1.0, 0, 1.0), object_px=100, fringe=True), [1, 2, 3, 4])
    assert v["ok"] and not v["redo"]
    assert any("saturated" in w for w in v["warnings"])
    assert any("little object" in w for w in v["warnings"])


def test_missing_laser_frame_flagged():
    pkg = _pkg()
    del pkg["laser_features"]["channels"]["2"]
    v = qc.check(pkg, [1, 2, 3, 4])
    assert v["redo"] and any("CH2 produced no laser frame" in p for p in v["problems"])


def test_only_requested_channels_checked():
    # CH4 dead but not requested -> no problem
    v = qc.check(_pkg(cores=(40, 40, 5, 7)), [1, 2, 3])
    assert v["ok"]
