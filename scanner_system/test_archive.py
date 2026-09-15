"""Archiving non-dataset samples out of the live collections, reversibly."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scanner_system import archive, scanner_db
from scanner_system.test_scanner_db import FakeDB


def _sample_with_scan(db, label, cls):
    sid = scanner_db.create_sample(label, material_class=cls, db=db)
    scan = scanner_db.start_scan(sid, db=db)
    scanner_db.register_artifact(scan, sid, "kinect", "color_png", f"{scan}/c.png", db=db)
    return sid


def test_archive_before_and_restore():
    db = FakeDB()
    demo = _sample_with_scan(db, "DEMO", "demo")
    real = _sample_with_scan(db, "cardboard piece 01", "cardboard")
    later = _sample_with_scan(db, "white plastic 01", "plastic")
    # created_at ordering: make the demo strictly older than the real one
    db["samples"].update_one({"_id": demo}, {"$set": {"created_at": "2026-07-31T00:00:00"}})
    db["samples"].update_one({"_id": real}, {"$set": {"created_at": "2026-09-05T15:55:30"}})
    db["samples"].update_one({"_id": later}, {"$set": {"created_at": "2026-09-15T16:00:00"}})

    ids = archive.samples_before(real[:8], db=db)
    assert ids == [demo]
    moved = archive.archive_samples(ids, "old", db=db)
    assert moved == {"samples": 1, "scans": 1, "artifacts": 1}

    live = {s["_id"] for s in scanner_db.list_samples(db=db)}
    assert live == {real, later}
    assert db["scans"].find({"sample_id": demo}) == []
    arch = archive.list_archived(db=db)
    assert [a["_id"] for a in arch] == [demo] and arch[0]["archive_reason"] == "old"

    assert archive.restore_samples([demo], db=db) == {"samples": 1, "scans": 1, "artifacts": 1}
    assert scanner_db.get_sample(demo, db=db)["label"] == "DEMO"
    assert "archived_at" not in scanner_db.get_sample(demo, db=db)
    assert len(db["scans"].find({"sample_id": demo})) == 1
    assert archive.list_archived(db=db) == []
