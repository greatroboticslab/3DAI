"""Schema constants and document builders for the scanner data system.

Pure Python: no Mongo, no hardware, no I/O. These builders produce the exact
documents stored in the ``samples`` / ``scans`` / ``artifacts`` collections, so
the schema has one authority and is unit-testable without a database.

Model (see design doc):
    sample          one physical object            -> collection "samples"
      └── scan      one capture session            -> collection "scans"
            └── artifact  one captured file, tagged by modality -> "artifacts"

Binaries live on disk under STORAGE_ROOT; documents hold metadata + file_path.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional


# ── Vocabularies ────────────────────────────────────────────────────────────

# Capture modalities. Each artifact belongs to exactly one. Lasers, Kinect, and
# the projector/fringe source are distinct instruments and stay in separate lanes.
MODALITIES = ("projector", "kinect", "laser", "fusion", "calibration")

# Capture modes requested for a scan. "full" is the default (the real idea:
# lasers + kinect + projector together). The others are graceful fallbacks for
# when instruments interfere (e.g. lasers blocking the Kinect) or for debugging.
CAPTURE_MODES = (
    "full",               # lasers + kinect + projector
    "kinect_projector",   # 3D without lasers
    "laser_only",         # lasers + a camera capture of the lit sample
    "kinect_only",
    "projector_only",
)

# Per-instrument outcome recorded on each scan, so the GUI can show exactly what
# succeeded vs. what was requested rather than silently producing partial data.
INSTRUMENT_STATUSES = ("ok", "skipped", "failed")

SCAN_STATUSES = ("running", "complete", "partial", "failed")

# The four lasers, by channel number, matching the ESP32 relay channels
# (CH1-4 -> GPIO18/19/25/26). See esp32-relay-provisioning.
LASER_CHANNELS = (1, 2, 3, 4)

# Each laser is a DIFFERENT wavelength/color. Per Dr. Zhang: "we used different
# colors of lasers to see different aspects of the same object ... for different
# materials-related modulated information." So the system is effectively
# multispectral: the wavelength IS a feature dimension for material recognition.
#
# NOMINAL wavelengths, set 2026-09-05 by bench inspection, NOT measured. The
# modules carry no labels. CH1/CH2 look like standard red laser pointers
# (red pointers are essentially always 650 nm) and their spots read
# red-dominant on the Kinect. CH4 is a bright green pointer (green pointers
# are essentially always 532 nm; camera reads G+190 under it). CH3 emits a
# faint visible smudge of unknown color and stays None until identified.
# Treat these as class labels ("red", "green"), not spectroscopy. Override
# with SCANNER_LASER_WAVELENGTHS if the modules are ever measured
# (CSV of "ch:nm", e.g. "1:650,2:650,3:780,4:532").
LASER_WAVELENGTHS_NM = {
    1: 650,    # nominal: red pointer, unlabeled module
    2: 650,    # nominal: red pointer, unlabeled module
    3: None,   # visible but faint ("smudge"); color/wavelength unidentified
    4: 532,    # nominal: green pointer, unlabeled module
}


def _apply_wavelength_env() -> None:
    """Apply SCANNER_LASER_WAVELENGTHS ("ch:nm,ch:nm") over the nominals.

    Promised by the comment above since this module was written, implemented
    2026-09-05. Malformed entries are ignored so a typo'd env var degrades to
    the nominal table instead of killing every import of the schema.
    """
    raw = os.getenv("SCANNER_LASER_WAVELENGTHS", "").strip()
    if not raw:
        return
    for part in raw.split(","):
        try:
            ch_txt, nm_txt = part.split(":", 1)
            ch = int(ch_txt)
            if ch in LASER_WAVELENGTHS_NM:
                LASER_WAVELENGTHS_NM[ch] = int(nm_txt)
        except ValueError:
            continue


_apply_wavelength_env()


def is_ir(wavelength_nm) -> bool:
    """True if a wavelength is (near-)infrared, i.e. beyond visible red."""
    return wavelength_nm is not None and wavelength_nm >= 750


def _now() -> str:
    """UTC ISO-8601 timestamp string (stable, sortable, timezone-explicit)."""
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    """A fresh UUID string, used for every _id in this system."""
    return str(uuid.uuid4())


# ── Document builders ───────────────────────────────────────────────────────

def build_sample(
    label: str,
    material_class: Optional[str] = None,
    material_subclass: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    sample_id: Optional[str] = None,
) -> dict[str, Any]:
    """Build a ``samples`` document for one physical object.

    The system's purpose is **material recognition**, so the material label is
    first-class and hierarchical: ``material_class`` (e.g. "wood", "metal") plus
    ``material_subclass`` (e.g. "oak", "aluminum"). This is the ML target; keep
    it out of freeform ``context`` so it stays cleanly queryable and groupable
    for dataset export/training. One sample = one material.

    ``sample_id`` may be supplied to share a key with another system (e.g.
    Torres's 4DAI ``Collections``); otherwise a fresh UUID is generated.
    """
    now = _now()
    return {
        "_id": sample_id or new_id(),
        "label": label,
        "material": {
            "class": material_class,
            "subclass": material_subclass,
        },
        "context": dict(context or {}),
        "created_at": now,
        "updated_at": now,
    }


def build_scan(
    sample_id: str,
    mode: str = "full",
    operator: Optional[str] = None,
    notes: str = "",
    angle: Optional[dict[str, int]] = None,
    known_height_mm: Optional[float] = None,
    scan_id: Optional[str] = None,
) -> dict[str, Any]:
    """Build a ``scans`` document for one capture session of a sample.

    ``mode`` must be one of CAPTURE_MODES. The per-instrument ``results`` start
    empty and are filled in as capture proceeds (via ``instrument_result``).

    ``angle`` records which viewing pose this scan is, for multi-angle
    collection ("shining lasers on it from different angles" per Dr. Zhang):
    ``{"index": k, "count": n}`` meaning pose k of n for this sample. None
    means single-pose. Stored structured rather than in ``notes`` so dataset
    export can carry it as a queryable field per training row.

    ``known_height_mm`` makes the scan a calibration-anchor candidate: the
    object's caliper-measured height as placed for THIS scan. Dual-purpose
    collection: flat matte objects contribute to the height-calibration fit
    while being collected. Per-scan, not per-sample, because a rotated object
    has a different height; the manifest applies it to pose 1 only.
    """
    if mode not in CAPTURE_MODES:
        raise ValueError(f"unknown capture mode {mode!r}; use one of {CAPTURE_MODES}")
    if angle is not None:
        idx, cnt = int(angle.get("index", 0)), int(angle.get("count", 0))
        if not (1 <= idx <= cnt):
            raise ValueError(f"angle index/count out of order: {angle!r}")
        angle = {"index": idx, "count": cnt}
    if known_height_mm is not None:
        known_height_mm = float(known_height_mm)
        if not (0.2 <= known_height_mm <= 200.0):
            raise ValueError(
                f"known_height_mm {known_height_mm} outside the plausible "
                "0.2-200 mm anchor range (caliper typo?)")
    now = _now()
    return {
        "_id": scan_id or new_id(),
        "sample_id": sample_id,
        "mode": mode,
        "status": "running",
        "operator": operator,
        "notes": notes,
        "angle": angle,
        "known_height_mm": known_height_mm,
        # requested vs. actual, per instrument, so the GUI can show honest state.
        "results": {},          # e.g. {"kinect": {"status":"ok"}, "laser": {...}}
        "started_at": now,
        "completed_at": None,
    }


def instrument_result(
    status: str,
    detail: str = "",
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build a per-instrument result entry for a scan's ``results`` map.

    ``status`` is one of INSTRUMENT_STATUSES. ``detail`` carries a human-readable
    reason (especially for "failed"/"skipped") that flows through to the GUI.
    """
    if status not in INSTRUMENT_STATUSES:
        raise ValueError(f"unknown instrument status {status!r}; use {INSTRUMENT_STATUSES}")
    entry = {"status": status, "detail": detail}
    if extra:
        entry.update(extra)
    return entry


def resolve_scan_status(results: dict[str, Any]) -> str:
    """Derive an overall scan status from its per-instrument results.

    - no results yet                       -> "running"
    - every attempted instrument ok        -> "complete"
    - all attempted instruments failed     -> "failed"
    - mix of ok and failed                 -> "partial"
    (skipped instruments don't count for/against; they were intentional.)
    """
    attempted = [r.get("status") for r in results.values()
                 if r.get("status") in ("ok", "failed")]
    if not attempted:
        return "running"
    if all(s == "ok" for s in attempted):
        return "complete"
    if all(s == "failed" for s in attempted):
        return "failed"
    return "partial"


def build_artifact(
    scan_id: str,
    sample_id: str,
    modality: str,
    role: str,
    file_path: str,
    media_type: str = "application/octet-stream",
    size_bytes: Optional[int] = None,
    laser_state: Optional[dict[str, Any]] = None,
    array_metadata: Optional[dict[str, Any]] = None,
    metadata: Optional[dict[str, Any]] = None,
    artifact_id: Optional[str] = None,
) -> dict[str, Any]:
    """Build an ``artifacts`` document for one captured file.

    ``modality`` must be one of MODALITIES. ``sample_id`` is denormalized here
    (also on the parent scan) so "all laser images for this sample" is a single
    indexed query with no join. ``laser_state`` (e.g. {"on":[3],"ir":true}) is
    only meaningful on ``laser`` artifacts.
    """
    if modality not in MODALITIES:
        raise ValueError(f"unknown modality {modality!r}; use one of {MODALITIES}")
    doc = {
        "_id": artifact_id or new_id(),
        "scan_id": scan_id,
        "sample_id": sample_id,
        "modality": modality,
        "role": role,
        "file_path": file_path,
        "media_type": media_type,
        "size_bytes": size_bytes,
        "metadata": dict(metadata or {}),
        "created_at": _now(),
    }
    if laser_state is not None:
        doc["laser_state"] = laser_state
    if array_metadata is not None:
        doc["array_metadata"] = array_metadata
    return doc


def build_laser_state(
    channel: int,
    wavelength_nm: Optional[int] = None,
) -> dict[str, Any]:
    """Build the ``laser_state`` for a laser artifact.

    Records WHICH laser fired and, crucially for material recognition, its
    WAVELENGTH -- the spectral feature dimension. If ``wavelength_nm`` is not
    given, falls back to the configured LASER_WAVELENGTHS_NM for the channel.
    """
    if wavelength_nm is None:
        wavelength_nm = LASER_WAVELENGTHS_NM.get(channel)
    return {
        "channel": channel,
        "wavelength_nm": wavelength_nm,
        "ir": is_ir(wavelength_nm),
    }


# Indexes to create on first use. Chosen for the common GUI/query paths:
# "scans for a sample", "artifacts for a scan", "artifacts for a sample",
# "artifacts of a modality". Keeps research-scale (100s-1000s samples, many
# more artifacts) queries as index hits.
INDEXES = {
    "samples": [
        ("material.class", 1),      # group/filter by material for dataset export
        ("material.subclass", 1),
    ],
    "scans": [
        ("sample_id", 1),
        ("started_at", -1),
    ],
    "artifacts": [
        ("scan_id", 1),
        ("sample_id", 1),
        ("modality", 1),
    ],
}
