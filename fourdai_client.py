"""Optional, lazy read-only client for the 4DAI MongoDB.

4DAI (greatroboticslab/4DAI, Torres) stores plant/material context, 2D images,
and manual moisture in MongoDB. This module lets the 3DAI API enrich its 3D
scan data with the matching 4DAI record for a shared ``sample_id`` (which is the
4DAI record ``_id``).

Design constraints (see docs/4dai_integration.md):

- 3DAI must start and serve all of its own endpoints even if ``pymongo`` is not
  installed or Mongo is unreachable. Nothing here is imported at module load of
  ``main.py`` in a way that can fail; the Mongo client is created lazily and all
  errors degrade to "not configured / not available" rather than raising.
- Read-only. No writes, no hardware, no camera/Kinect imports.

4DAI schema (Server/db.py, Server/main.py in greatroboticslab/4DAI):
    db "Collections"
      vegetable: _id, vegetable_name, vegetable_health, date, notes
      soil:      _id, soil_type, soil_moisture, date, notes
      images:    _id, sample_id, image_path
"""

import os
from typing import Any, Optional


DEFAULT_DB_NAME = "Collections"

# Module-level cache so we build at most one MongoClient per process.
_client = None
_client_built = False


def is_configured() -> bool:
    """True if a 4DAI Mongo URL has been configured via the environment."""
    return bool(os.getenv("FOURDAI_MONGO_URL", "").strip())


def get_db_name() -> str:
    return os.getenv("FOURDAI_DB_NAME", "").strip() or DEFAULT_DB_NAME


def _get_client():
    """Build (once) and return a MongoClient, or None if unavailable.

    Returns None if 4DAI is not configured, if pymongo is not installed, or if
    the client cannot be constructed. Never raises.
    """
    global _client, _client_built
    if _client_built:
        return _client

    _client_built = True
    url = os.getenv("FOURDAI_MONGO_URL", "").strip()
    if not url:
        _client = None
        return _client

    try:
        from pymongo import MongoClient
    except ImportError:
        _client = None
        return _client

    try:
        # serverSelectionTimeoutMS keeps a down Mongo from hanging the request.
        timeout_ms = int(os.getenv("FOURDAI_MONGO_TIMEOUT_MS", "1500"))
        _client = MongoClient(url, serverSelectionTimeoutMS=timeout_ms)
    except Exception:
        _client = None
    return _client


def reset_client_cache() -> None:
    """Drop the cached client. Intended for tests that change the environment."""
    global _client, _client_built
    _client = None
    _client_built = False


def shape_record(kind: str, doc: dict[str, Any], image_ids: list[str]) -> dict[str, Any]:
    """Shape a raw 4DAI Mongo document into the API record form.

    Pure function: no I/O, so it is unit-testable without pymongo or a live
    Mongo. ``kind`` is "vegetable" or "soil"; ``doc`` is the raw Mongo doc;
    ``image_ids`` are the 4DAI image ``_id`` values for the sample.
    """
    fields = {k: v for k, v in doc.items() if k != "_id"}
    return {
        "kind": kind,
        "record": fields,
        "image_ids": list(image_ids),
    }


def fetch_sample(sample_id: str, db=None) -> Optional[dict[str, Any]]:
    """Return the 4DAI record for ``sample_id`` (its Mongo ``_id``), or None.

    Looks in the ``vegetable`` collection first, then ``soil``. Resolves the
    sample's image ids from the ``images`` collection. ``db`` may be injected
    for testing; otherwise the configured Mongo database is used.

    Never raises on connection or driver problems; returns None instead so the
    caller can degrade gracefully.
    """
    if db is None:
        client = _get_client()
        if client is None:
            return None
        try:
            db = client[get_db_name()]
        except Exception:
            return None

    try:
        for kind, coll_name in (("vegetable", "vegetable"), ("soil", "soil")):
            doc = db[coll_name].find_one({"_id": sample_id})
            if doc is not None:
                image_ids = [
                    img["_id"]
                    for img in db["images"].find({"sample_id": sample_id})
                ]
                return shape_record(kind, doc, image_ids)
    except Exception:
        return None

    return None
