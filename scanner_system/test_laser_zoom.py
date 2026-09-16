"""The readable laser view renders from synthetic frames and degrades cleanly."""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scanner_system import laser_features as lf
from scanner_system import laser_zoom
from scanner_system.test_laser_features import _make_scan


def test_zoom_renders_all_channels_and_marks_invalid():
    with tempfile.TemporaryDirectory() as tmp:
        _make_scan(tmp, sigma=6.0, amp=140.0)
        feats = lf.compute_features(tmp)
        feats["channels"]["4"]["valid"] = False
        out = os.path.join(tmp, "out", "laser_zoom.jpg")
        assert laser_zoom.make_laser_zoom(tmp, feats, out) == out
        from PIL import Image
        with Image.open(out) as im:
            tile = 2 * laser_zoom.WINDOW * laser_zoom.SCALE
            assert im.size[0] == 3 * tile
            assert im.size[1] == 4 * (tile + 22)          # four channel rows
            a = np.asarray(im.convert("L"))
        assert a.mean() > 10                                 # not a black image

        # a channel with no spot centre is skipped, not fatal
        del feats["channels"]["2"]["spot_x"]
        assert laser_zoom.make_laser_zoom(tmp, feats, out) == out
        with Image.open(out) as im:
            assert im.size[1] == 3 * (tile + 22)

    with tempfile.TemporaryDirectory() as empty:
        assert laser_zoom.make_laser_zoom(empty, {"channels": {}}, os.path.join(empty, "x.jpg")) is None


def test_zoom_adds_side_camera_rows_when_present():
    with tempfile.TemporaryDirectory() as tmp:
        _make_scan(tmp, sigma=6.0, amp=140.0)
        h, w = 200, 300
        import numpy as np
        from PIL import Image
        cam_dark = np.full((h, w, 3), 40, np.uint8)
        Image.fromarray(cam_dark).save(os.path.join(tmp, "cam_dark.png"))
        from scanner_system.test_laser_features import _gaussian_spot
        add = _gaussian_spot(h, w, 100, 150, 10.0, 90.0).astype(np.float32)
        Image.fromarray(np.clip(cam_dark + add[..., None], 0, 255).astype(np.uint8)).save(
            os.path.join(tmp, "cam_las1.png"))
        feats = lf.compute_features(tmp)
        out = os.path.join(tmp, "z.jpg")
        assert laser_zoom.make_laser_zoom(tmp, feats, out) == out
        with Image.open(out) as im:
            tile = 2 * laser_zoom.WINDOW * laser_zoom.SCALE
            assert im.size[1] == 5 * (tile + 22)      # 4 Kinect rows + 1 side-camera row
