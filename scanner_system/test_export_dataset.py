"""Folder export for the GitHub dataset: filtering, previews, metadata."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scanner_system import export_dataset, scanner_db
from scanner_system.test_scanner_db import FakeDB


def _png(path, mode="RGB", size=(64, 48)):
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if mode == "I;16":
        import numpy as np
        a = (np.arange(size[0] * size[1], dtype=np.uint16).reshape(size[1], size[0]) * 300)
        Image.fromarray(a, mode="I;16").save(path)
    else:
        Image.new(mode, size, (200, 30, 30)).save(path)


def _complete_scan(db, sid, root, roles):
    scan_id = scanner_db.start_scan(sid, db=db)
    for inst in ("laser", "kinect", "projector"):
        scanner_db.record_instrument(scan_id, inst, "ok", db=db)
    scanner_db.finish_scan(scan_id, db=db)
    for role, mode in roles:
        rel = f"scans/{scan_id}/{role}.png"
        _png(os.path.join(root, rel), mode)
        scanner_db.register_artifact(scan_id, sid, "laser", role, rel,
                                     media_type="image/png", db=db)
    return scan_id


def test_export_folder_filters_and_previews():
    db = FakeDB()
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "storage")
        wood = scanner_db.create_sample("oak block 01", material_class="wood",
                                        context={"surface": "matte"}, db=db)
        demo = scanner_db.create_sample("DEMO thing", material_class="demo", db=db)
        nolabel = scanner_db.create_sample("mystery", db=db)
        good = _complete_scan(db, wood, root,
                              [("laser_ch1_png", "RGB"), ("laser_ch1_ir_png", "I;16")])
        _complete_scan(db, demo, root, [("laser_ch1_png", "RGB")])
        _complete_scan(db, nolabel, root, [("laser_ch1_png", "RGB")])
        # an unfinished scan of the good sample must not leak into the export
        bad = scanner_db.start_scan(wood, db=db)
        scanner_db.record_instrument(bad, "kinect", "failed", detail="x", db=db)
        _png(os.path.join(root, f"scans/{bad}/laser_ch1_png.png"))
        scanner_db.register_artifact(bad, wood, "laser", "laser_ch1_png",
                                     f"scans/{bad}/laser_ch1_png.png", db=db)
        scanner_db.set_laser_features(good, {"1": {"core": 1.0}}, db=db)

        out = os.path.join(tmp, "dataset")
        s = export_dataset.export_folder(out, db=db, storage_root=root)

        assert s["objects"] == 1 and s["scans"] == 1 and s["images"] == 2
        assert s["by_class"] == {"wood": 1}
        scan_dir = os.path.join(out, "images", "wood", f"oak-block-01__angle1of1__{good[:8]}")
        assert sorted(os.listdir(scan_dir)) == ["laser_ch1.jpg", "laser_ch1_ir.jpg"]
        assert not os.path.isdir(os.path.join(out, "images", "demo"))
        assert not os.path.isdir(os.path.join(out, "images", "unlabeled"))
        for name in ("metadata.xlsx", "laser_features.csv", "README.md"):
            assert os.path.isfile(os.path.join(out, name)), name
        readme = open(os.path.join(out, "README.md"), encoding="utf-8").read()
        assert "Objects: **1**" in readme

        from PIL import Image
        with Image.open(os.path.join(scan_dir, "laser_ch1_ir.jpg")) as ir:
            assert ir.mode == "L" and max(ir.size) <= export_dataset.PREVIEW_MAX_PX

        # re-export drops stale files instead of accumulating them
        stale = os.path.join(out, "images", "wood", "old", "x.jpg")
        os.makedirs(os.path.dirname(stale)); open(stale, "w").close()
        export_dataset.export_folder(out, db=db, storage_root=root)
        assert not os.path.exists(stale)
