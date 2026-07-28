"""Data layer for the scanner system's local MongoDB.

One authority for reading/writing the ``samples`` / ``scans`` / ``artifacts``
collections. Everything (capture scripts, the Streamlit GUI) goes through here so
the schema stays consistent.

Design:
- Lazy, import-safe: importing this module opens no connection and needs no
  pymongo. The client is built on first use.
- A ``db`` may be injected (any object supporting ``db[name]`` returning a
  collection with find/find_one/insert_one/update_one). This lets tests run
  against a fake Mongo with no server.

Environment:
    SCANNER_MONGO_URL      default "mongodb://127.0.0.1:27017"
    SCANNER_DB_NAME        default "scanner"
    SCANNER_MONGO_TIMEOUT_MS  default "2000"
"""

from __future__ import annotations

import os
from typing import Any, Optional

from . import schema

DEFAULT_URL = "mongodb://127.0.0.1:27017"
DEFAULT_DB_NAME = "scanner"

_client = None
_client_built = False


# ── Connection (lazy) ───────────────────────────────────────────────────────

def get_url() -> str:
    return os.getenv("SCANNER_MONGO_URL", "").strip() or DEFAULT_URL


def get_db_name() -> str:
    return os.getenv("SCANNER_DB_NAME", "").strip() or DEFAULT_DB_NAME


def _get_client():
    """Build (once) and return a MongoClient, or None if unavailable.

    Never raises: if pymongo is missing or the client can't be constructed,
    returns None so callers can degrade rather than crash.
    """
    global _client, _client_built
    if _client_built:
        return _client
    _client_built = True
    try:
        from pymongo import MongoClient
    except ImportError:
        _client = None
        return _client
    try:
        timeout_ms = int(os.getenv("SCANNER_MONGO_TIMEOUT_MS", "2000"))
        _client = MongoClient(get_url(), serverSelectionTimeoutMS=timeout_ms)
    except Exception:
        _client = None
    return _client


def reset_client_cache() -> None:
    """Drop the cached client (for tests that change the environment)."""
    global _client, _client_built
    _client = None
    _client_built = False


def get_db(db=None):
    """Return the scanner database handle, or the injected ``db``.

    Raises RuntimeError if Mongo is unavailable and no db was injected, so
    callers get a clear failure instead of a silent None deref.
    """
    if db is not None:
        return db
    client = _get_client()
    if client is None:
        raise RuntimeError(
            "scanner MongoDB is not available (pymongo missing or server "
            f"unreachable at {get_url()})"
        )
    return client[get_db_name()]


def ensure_indexes(db=None) -> None:
    """Create the schema's indexes if missing. Safe to call repeatedly."""
    d = get_db(db)
    for coll_name, keys in schema.INDEXES.items():
        for field, direction in keys:
            try:
                d[coll_name].create_index([(field, direction)])
            except Exception:
                # A fake/injected db may not support create_index; ignore.
                pass


# ── Samples ─────────────────────────────────────────────────────────────────

def create_sample(label, material_class=None, material_subclass=None,
                  context=None, sample_id=None, db=None) -> str:
    """Create a sample document with its (hierarchical) material label; returns
    its sample_id."""
    d = get_db(db)
    doc = schema.build_sample(
        label, material_class=material_class, material_subclass=material_subclass,
        context=context, sample_id=sample_id,
    )
    d["samples"].insert_one(doc)
    return doc["_id"]


def set_material(sample_id, material_class, material_subclass=None, db=None) -> None:
    """(Re)label a sample's material. Labeling is often done in the GUI after
    capture, so this is a first-class operation."""
    get_db(db)["samples"].update_one(
        {"_id": sample_id},
        {"$set": {
            "material.class": material_class,
            "material.subclass": material_subclass,
            "updated_at": schema._now(),
        }},
    )


def get_sample(sample_id: str, db=None) -> Optional[dict[str, Any]]:
    return get_db(db)["samples"].find_one({"_id": sample_id})


def list_samples(db=None, limit: int = 200) -> list[dict[str, Any]]:
    """Newest samples first (by created_at)."""
    cur = get_db(db)["samples"].find({})
    docs = list(cur)
    docs.sort(key=lambda s: s.get("created_at", ""), reverse=True)
    return docs[:limit]


# ── Scans ───────────────────────────────────────────────────────────────────

def start_scan(sample_id, mode="full", operator=None, notes="", db=None) -> str:
    """Start a scan for a sample in the given capture mode; returns scan_id."""
    d = get_db(db)
    doc = schema.build_scan(sample_id, mode=mode, operator=operator, notes=notes)
    d["scans"].insert_one(doc)
    return doc["_id"]


def record_instrument(scan_id, instrument, status, detail="", extra=None, db=None) -> None:
    """Record one instrument's outcome on a scan and refresh overall status.

    ``instrument`` is a modality-ish label (kinect/projector/laser). ``status``
    is one of INSTRUMENT_STATUSES. The scan's overall status is recomputed from
    all recorded results so it stays honest (complete/partial/failed).
    """
    d = get_db(db)
    entry = schema.instrument_result(status, detail=detail, extra=extra)
    d["scans"].update_one({"_id": scan_id}, {"$set": {f"results.{instrument}": entry}})
    scan = d["scans"].find_one({"_id": scan_id})
    if scan is not None:
        overall = schema.resolve_scan_status(scan.get("results", {}))
        d["scans"].update_one({"_id": scan_id}, {"$set": {"status": overall}})


def finish_scan(scan_id, db=None) -> Optional[str]:
    """Mark a scan finished: set completed_at and the final derived status.

    Returns the final status.
    """
    d = get_db(db)
    scan = d["scans"].find_one({"_id": scan_id})
    if scan is None:
        return None
    overall = schema.resolve_scan_status(scan.get("results", {}))
    d["scans"].update_one(
        {"_id": scan_id},
        {"$set": {"status": overall, "completed_at": schema._now()}},
    )
    return overall


def get_scan(scan_id: str, db=None) -> Optional[dict[str, Any]]:
    return get_db(db)["scans"].find_one({"_id": scan_id})


def scans_for_sample(sample_id: str, db=None) -> list[dict[str, Any]]:
    """All scans for a sample, newest first."""
    docs = list(get_db(db)["scans"].find({"sample_id": sample_id}))
    docs.sort(key=lambda s: s.get("started_at", ""), reverse=True)
    return docs


# ── Artifacts ───────────────────────────────────────────────────────────────

def register_artifact(
    scan_id, sample_id, modality, role, file_path,
    media_type="application/octet-stream", size_bytes=None,
    laser_state=None, array_metadata=None, metadata=None, db=None,
) -> str:
    """Register one captured file's metadata; returns artifact_id.

    The file itself is expected to already exist on disk under STORAGE_ROOT;
    this stores only metadata + path (binaries never go in Mongo).
    """
    d = get_db(db)
    doc = schema.build_artifact(
        scan_id, sample_id, modality, role, file_path,
        media_type=media_type, size_bytes=size_bytes,
        laser_state=laser_state, array_metadata=array_metadata, metadata=metadata,
    )
    d["artifacts"].insert_one(doc)
    return doc["_id"]


def artifacts_for_scan(scan_id, modality=None, db=None) -> list[dict[str, Any]]:
    q: dict[str, Any] = {"scan_id": scan_id}
    if modality is not None:
        q["modality"] = modality
    return list(get_db(db)["artifacts"].find(q))


def artifacts_for_sample(sample_id, modality=None, db=None) -> list[dict[str, Any]]:
    """All artifacts for a sample (optionally one modality) via the denormalized
    sample_id index -- no join through scans needed."""
    q: dict[str, Any] = {"sample_id": sample_id}
    if modality is not None:
        q["modality"] = modality
    return list(get_db(db)["artifacts"].find(q))


def samples_by_material(material_class=None, material_subclass=None, db=None) -> list[dict[str, Any]]:
    """List samples filtered by material class/subclass (for dataset browsing)."""
    q: dict[str, Any] = {}
    if material_class is not None:
        q["material.class"] = material_class
    if material_subclass is not None:
        q["material.subclass"] = material_subclass
    return list(get_db(db)["samples"].find(q))


def export_dataset(material_class=None, modality=None, db=None) -> list[dict[str, Any]]:
    """Flatten labeled samples + their feature artifacts for ML training.

    Returns one row per artifact: the material label (the target) joined with the
    artifact's modality, wavelength (for laser features), and file_path (the
    feature file on disk). Optionally filter by material class and/or modality.
    This is the query a training pipeline runs to build a labeled dataset.
    """
    d = get_db(db)
    rows: list[dict[str, Any]] = []
    for sample in samples_by_material(material_class=material_class, db=d):
        mat = sample.get("material", {})
        for art in artifacts_for_sample(sample["_id"], modality=modality, db=d):
            rows.append({
                "sample_id": sample["_id"],
                "material_class": mat.get("class"),
                "material_subclass": mat.get("subclass"),
                "modality": art.get("modality"),
                "role": art.get("role"),
                "wavelength_nm": (art.get("laser_state") or {}).get("wavelength_nm"),
                "file_path": art.get("file_path"),
            })
    return rows


def scan_package(scan_id: str, db=None) -> Optional[dict[str, Any]]:
    """Assemble a full scan view: the scan doc plus its artifacts grouped by
    modality. This is the shape the GUI renders."""
    d = get_db(db)
    scan = d["scans"].find_one({"_id": scan_id})
    if scan is None:
        return None
    grouped: dict[str, list] = {m: [] for m in schema.MODALITIES}
    for art in d["artifacts"].find({"scan_id": scan_id}):
        grouped.setdefault(art.get("modality", "?"), []).append(art)
    scan = dict(scan)
    scan["artifacts"] = grouped
    return scan
