"""Unit tests for the scanner data layer + schema.

Runs without pymongo and without a live Mongo, by injecting a fake db. Covers
the schema builders, capture-mode validation, per-instrument status rollup, and
the query/grouping helpers the GUI depends on.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scanner_system import schema, scanner_db


# ── Fake Mongo ──────────────────────────────────────────────────────────────

def _dotted_get(doc, key):
    """Resolve a possibly-dotted key like 'material.class' against a doc,
    mirroring how real Mongo matches nested fields in queries."""
    cur = doc
    for part in key.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


class FakeCollection:
    def __init__(self):
        self._docs = []

    def insert_one(self, doc):
        self._docs.append(dict(doc))

    def _match(self, d, query):
        return all(_dotted_get(d, k) == v for k, v in query.items())

    def find_one(self, query):
        for d in self._docs:
            if self._match(d, query):
                return dict(d)
        return None

    def find(self, query=None):
        query = query or {}
        return [dict(d) for d in self._docs if self._match(d, query)]

    def update_one(self, query, update):
        for d in self._docs:
            if all(d.get(k) == v for k, v in query.items()):
                for key, val in update.get("$set", {}).items():
                    # support dotted paths like "results.kinect"
                    parts = key.split(".")
                    tgt = d
                    for p in parts[:-1]:
                        tgt = tgt.setdefault(p, {})
                    tgt[parts[-1]] = val
                return
        # upsert not needed for these tests

    def create_index(self, keys):
        pass


class FakeDB:
    def __init__(self):
        self._colls = {}

    def __getitem__(self, name):
        return self._colls.setdefault(name, FakeCollection())


def expect(cond, msg):
    if not cond:
        raise AssertionError(msg)


# ── Schema builder tests ────────────────────────────────────────────────────

def test_build_sample_material_label():
    s = schema.build_sample("plank #3", material_class="wood",
                            material_subclass="oak", context={"moisture": "31"})
    expect(s["label"] == "plank #3", "label kept")
    expect(s["material"]["class"] == "wood", "material class first-class")
    expect(s["material"]["subclass"] == "oak", "material subclass first-class")
    expect(s["context"]["moisture"] == "31", "context kept separate from material")
    expect(s["_id"] and s["created_at"] and s["updated_at"], "id + timestamps set")


def test_laser_state_wavelength_and_ir():
    # visible red
    red = schema.build_laser_state(1, wavelength_nm=650)
    expect(red["wavelength_nm"] == 650 and red["ir"] is False, "650nm not IR")
    # near-IR
    ir = schema.build_laser_state(3, wavelength_nm=850)
    expect(ir["ir"] is True, "850nm is IR")
    expect(schema.is_ir(940) and not schema.is_ir(520), "is_ir threshold")


def test_build_scan_rejects_bad_mode():
    try:
        schema.build_scan("sid", mode="nonsense")
        expect(False, "should reject unknown mode")
    except ValueError:
        pass


def test_build_artifact_rejects_bad_modality():
    try:
        schema.build_artifact("scan", "sid", "not_a_modality", "role", "p.png")
        expect(False, "should reject unknown modality")
    except ValueError:
        pass


def test_laser_state_only_on_request():
    a = schema.build_artifact("scan", "sid", "kinect", "color_png", "k.png")
    expect("laser_state" not in a, "kinect artifact has no laser_state")
    b = schema.build_artifact("scan", "sid", "laser", "lit_png", "l.png",
                              laser_state={"on": [3], "ir": True})
    expect(b["laser_state"]["on"] == [3], "laser_state carried on laser artifact")


def test_resolve_scan_status():
    ok = {"kinect": {"status": "ok"}, "projector": {"status": "ok"}}
    expect(schema.resolve_scan_status(ok) == "complete", "all ok -> complete")
    mix = {"kinect": {"status": "ok"}, "laser": {"status": "failed"}}
    expect(schema.resolve_scan_status(mix) == "partial", "mixed -> partial")
    bad = {"kinect": {"status": "failed"}}
    expect(schema.resolve_scan_status(bad) == "failed", "all failed -> failed")
    skipped = {"laser": {"status": "skipped"}}
    expect(schema.resolve_scan_status(skipped) == "running", "only skipped -> running")
    expect(schema.resolve_scan_status({}) == "running", "empty -> running")


# ── Data layer tests (fake db) ──────────────────────────────────────────────

def test_end_to_end_scan_flow():
    db = FakeDB()
    sid = scanner_db.create_sample("plank", material_class="wood",
                                   material_subclass="oak", db=db)
    expect(scanner_db.get_sample(sid, db=db)["label"] == "plank", "sample stored")
    expect(scanner_db.get_sample(sid, db=db)["material"]["class"] == "wood",
           "material label stored")

    scan_id = scanner_db.start_scan(sid, mode="full", operator="adain", db=db)
    scan = scanner_db.get_scan(scan_id, db=db)
    expect(scan["status"] == "running", "scan starts running")
    expect(scan["mode"] == "full", "mode stored")

    # kinect ok, projector ok, laser fails (blocked by hardware)
    scanner_db.record_instrument(scan_id, "kinect", "ok", db=db)
    scanner_db.record_instrument(scan_id, "projector", "ok", db=db)
    scanner_db.record_instrument(scan_id, "laser", "failed",
                                 detail="laser blocked kinect view", db=db)
    scan = scanner_db.get_scan(scan_id, db=db)
    expect(scan["status"] == "partial", "one failed -> partial")
    expect(scan["results"]["laser"]["detail"] == "laser blocked kinect view",
           "failure detail surfaced")

    # register artifacts
    scanner_db.register_artifact(scan_id, sid, "kinect", "color_png",
                                 "scans/x/kinect/c.png", media_type="image/png", db=db)
    scanner_db.register_artifact(scan_id, sid, "laser", "lit_png",
                                 "scans/x/laser/l3.png", media_type="image/png",
                                 laser_state=schema.build_laser_state(3, wavelength_nm=850),
                                 db=db)

    pkg = scanner_db.scan_package(scan_id, db=db)
    expect(len(pkg["artifacts"]["kinect"]) == 1, "kinect artifact grouped")
    expect(len(pkg["artifacts"]["laser"]) == 1, "laser artifact grouped")
    expect(pkg["artifacts"]["laser"][0]["laser_state"]["ir"] is True, "laser IR flag in package")
    expect(pkg["artifacts"]["laser"][0]["laser_state"]["wavelength_nm"] == 850,
           "laser wavelength in package")

    # sample-level query via denormalized sample_id (no join)
    all_for_sample = scanner_db.artifacts_for_sample(sid, db=db)
    expect(len(all_for_sample) == 2, "both artifacts found by sample_id")
    laser_only = scanner_db.artifacts_for_sample(sid, modality="laser", db=db)
    expect(len(laser_only) == 1, "modality filter works on sample query")

    final = scanner_db.finish_scan(scan_id, db=db)
    expect(final == "partial", "finish keeps derived status")
    expect(scanner_db.get_scan(scan_id, db=db)["completed_at"] is not None,
           "completed_at set")


def test_set_material_and_export_dataset():
    db = FakeDB()
    # two wood samples, one metal; each with one laser feature artifact
    for label, cls, sub, wl in [
        ("oak plank", "wood", "oak", 850),
        ("pine board", "wood", "pine", 650),
        ("alu sheet", None, None, 940),   # unlabeled at capture time
    ]:
        sid = scanner_db.create_sample(label, material_class=cls, material_subclass=sub, db=db)
        scan = scanner_db.start_scan(sid, db=db)
        scanner_db.register_artifact(scan, sid, "laser", "lit_png",
                                     f"scans/{sid}/laser/l.png",
                                     laser_state=schema.build_laser_state(1, wavelength_nm=wl), db=db)
        if cls is None:
            # label it later (as would happen in the GUI)
            scanner_db.set_material(sid, "metal", "aluminum", db=db)

    # export just wood
    wood = scanner_db.export_dataset(material_class="wood", db=db)
    expect(len(wood) == 2, "two wood feature rows")
    expect(all(r["material_class"] == "wood" for r in wood), "all wood")
    expect(any(r["wavelength_nm"] == 850 for r in wood), "wavelength carried into export")

    # full export includes the later-labeled metal
    allrows = scanner_db.export_dataset(db=db)
    expect(any(r["material_subclass"] == "aluminum" for r in allrows),
           "set_material relabeling reflected in export")


def test_scans_for_sample_newest_first():
    db = FakeDB()
    sid = scanner_db.create_sample("s", db=db)
    a = scanner_db.start_scan(sid, db=db)
    # force distinct started_at ordering
    db["scans"].update_one({"_id": a}, {"$set": {"started_at": "2026-01-01T00:00:00Z"}})
    b = scanner_db.start_scan(sid, db=db)
    db["scans"].update_one({"_id": b}, {"$set": {"started_at": "2026-02-01T00:00:00Z"}})
    scans = scanner_db.scans_for_sample(sid, db=db)
    expect(scans[0]["_id"] == b, "newest scan first")


def main():
    test_build_sample_material_label()
    test_laser_state_wavelength_and_ir()
    test_build_scan_rejects_bad_mode()
    test_build_artifact_rejects_bad_modality()
    test_laser_state_only_on_request()
    test_resolve_scan_status()
    test_end_to_end_scan_flow()
    test_set_material_and_export_dataset()
    test_scans_for_sample_newest_first()
    print("OK: scanner_db unit tests")


if __name__ == "__main__":
    main()
