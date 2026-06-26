"""Unit tests for fourdai_client.

Runs without pymongo and without a live Mongo. The Mongo-touching path is
exercised by injecting a fake db object, and the graceful-degrade paths are
checked directly. These cover both branches required by docs/4dai_integration.md:
4DAI configured and resolvable, and 4DAI not configured / not available.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fourdai_client


class FakeCollection:
    def __init__(self, docs):
        self._docs = docs

    def find_one(self, query):
        for doc in self._docs:
            if all(doc.get(k) == v for k, v in query.items()):
                return doc
        return None

    def find(self, query):
        return [
            doc for doc in self._docs
            if all(doc.get(k) == v for k, v in query.items())
        ]


class FakeDB:
    def __init__(self, collections):
        self._collections = collections

    def __getitem__(self, name):
        return self._collections[name]


def expect(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_shape_record_strips_id_and_copies_images():
    doc = {
        "_id": "abc",
        "soil_type": "loam",
        "soil_moisture": "7",
        "date": "2026-06-26",
        "notes": "n",
    }
    shaped = fourdai_client.shape_record("soil", doc, ["i1", "i2"])
    expect(shaped["kind"] == "soil", "kind preserved")
    expect("_id" not in shaped["record"], "_id stripped from record")
    expect(shaped["record"]["soil_moisture"] == "7", "moisture field carried")
    expect(shaped["image_ids"] == ["i1", "i2"], "image ids carried")

    # shape_record must not alias the caller's list.
    src = ["i1"]
    out = fourdai_client.shape_record("soil", doc, src)
    src.append("mutated")
    expect(out["image_ids"] == ["i1"], "image_ids list is copied, not aliased")


def test_fetch_sample_finds_vegetable():
    db = FakeDB({
        "vegetable": FakeCollection([
            {"_id": "veg-1", "vegetable_name": "basil", "vegetable_health": "Healthy",
             "date": "2026-06-26", "notes": ""},
        ]),
        "soil": FakeCollection([]),
        "images": FakeCollection([
            {"_id": "img-1", "sample_id": "veg-1", "image_path": "x"},
            {"_id": "img-2", "sample_id": "veg-1", "image_path": "y"},
            {"_id": "img-3", "sample_id": "other", "image_path": "z"},
        ]),
    })
    rec = fourdai_client.fetch_sample("veg-1", db=db)
    expect(rec is not None, "vegetable found")
    expect(rec["kind"] == "vegetable", "kind is vegetable")
    expect(rec["record"]["vegetable_name"] == "basil", "veg name")
    expect(rec["image_ids"] == ["img-1", "img-2"], "only this sample's images")


def test_fetch_sample_falls_back_to_soil():
    db = FakeDB({
        "vegetable": FakeCollection([]),
        "soil": FakeCollection([
            {"_id": "soil-1", "soil_type": "clay", "soil_moisture": "3",
             "date": "2026-06-26", "notes": ""},
        ]),
        "images": FakeCollection([]),
    })
    rec = fourdai_client.fetch_sample("soil-1", db=db)
    expect(rec is not None and rec["kind"] == "soil", "soil found via fallback")
    expect(rec["image_ids"] == [], "no images is fine")


def test_fetch_sample_missing_returns_none():
    db = FakeDB({
        "vegetable": FakeCollection([]),
        "soil": FakeCollection([]),
        "images": FakeCollection([]),
    })
    expect(fourdai_client.fetch_sample("nope", db=db) is None, "missing sample -> None")


def test_fetch_sample_swallows_db_errors():
    class BoomDB:
        def __getitem__(self, name):
            raise RuntimeError("mongo is down")

    expect(fourdai_client.fetch_sample("x", db=BoomDB()) is None,
           "db errors degrade to None")


def test_not_configured_paths():
    saved = os.environ.pop("FOURDAI_MONGO_URL", None)
    try:
        fourdai_client.reset_client_cache()
        expect(fourdai_client.is_configured() is False, "not configured without env")
        # With no db injected and nothing configured, fetch returns None.
        expect(fourdai_client.fetch_sample("anything") is None,
               "unconfigured fetch -> None")
    finally:
        if saved is not None:
            os.environ["FOURDAI_MONGO_URL"] = saved
        fourdai_client.reset_client_cache()


def test_db_name_default_and_override():
    saved = os.environ.pop("FOURDAI_DB_NAME", None)
    try:
        expect(fourdai_client.get_db_name() == "Collections", "default db name")
        os.environ["FOURDAI_DB_NAME"] = "Other"
        expect(fourdai_client.get_db_name() == "Other", "override db name")
    finally:
        os.environ.pop("FOURDAI_DB_NAME", None)
        if saved is not None:
            os.environ["FOURDAI_DB_NAME"] = saved


def main():
    test_shape_record_strips_id_and_copies_images()
    test_fetch_sample_finds_vegetable()
    test_fetch_sample_falls_back_to_soil()
    test_fetch_sample_missing_returns_none()
    test_fetch_sample_swallows_db_errors()
    test_not_configured_paths()
    test_db_name_default_and_override()
    print("OK: fourdai_client unit tests")


if __name__ == "__main__":
    main()
