"""Move samples that are not dataset material out of the live collections.

The July demos, API smoke tests and pipeline verification samples share the
database with the real collection, so they showed up in the live view and
had to be filtered out of every export. Archiving moves a sample, its
scans and its artifact records into ``samples_archive`` / ``scans_archive``
/ ``artifacts_archive`` with an ``archived_at`` stamp and reason. Nothing is
deleted and no file on disk is touched, so it is reversible with
``--restore``.

    python -m scanner_system.archive --before 2725693b        # older than that sample
    python -m scanner_system.archive --ids 9dbcddc7 21828196   # specific samples
    python -m scanner_system.archive --list
    python -m scanner_system.archive --restore 9dbcddc7
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from . import scanner_db, schema

COLLECTIONS = ("samples", "scans", "artifacts")


def _archive_name(coll: str) -> str:
    return f"{coll}_archive"


def archive_samples(sample_ids: list[str], reason: str, db=None) -> dict[str, int]:
    d = scanner_db.get_db(db)
    stamp = schema._now()
    moved = {c: 0 for c in COLLECTIONS}
    for sid in sample_ids:
        sample = d["samples"].find_one({"_id": sid})
        if sample is None:
            continue
        docs = {"samples": [sample],
                "scans": list(d["scans"].find({"sample_id": sid})),
                "artifacts": list(d["artifacts"].find({"sample_id": sid}))}
        for coll, items in docs.items():
            for doc in items:
                doc = dict(doc, archived_at=stamp, archive_reason=reason)
                d[_archive_name(coll)].insert_one(doc)
                d[coll].delete_one({"_id": doc["_id"]})
                moved[coll] += 1
    return moved


def restore_samples(sample_ids: list[str], db=None) -> dict[str, int]:
    d = scanner_db.get_db(db)
    moved = {c: 0 for c in COLLECTIONS}
    for sid in sample_ids:
        for coll in COLLECTIONS:
            q = {"_id": sid} if coll == "samples" else {"sample_id": sid}
            for doc in list(d[_archive_name(coll)].find(q)):
                doc = {k: v for k, v in doc.items() if k not in ("archived_at", "archive_reason")}
                d[coll].insert_one(doc)
                d[_archive_name(coll)].delete_one({"_id": doc["_id"]})
                moved[coll] += 1
    return moved


def samples_before(cutoff: str, db=None) -> list[str]:
    """Ids of samples created before ``cutoff``: a sample id (or its short
    prefix), whose creation time is the boundary, or an ISO timestamp."""
    d = scanner_db.get_db(db)
    try:
        ref = scanner_db.resolve_sample_id(cutoff, db=d)
        boundary = d["samples"].find_one({"_id": ref})["created_at"]
    except KeyError:
        boundary = cutoff
    return [s["_id"] for s in scanner_db.list_samples(db=d)
            if s.get("created_at", "") < boundary]


def list_archived(db=None) -> list[dict[str, Any]]:
    d = scanner_db.get_db(db)
    docs = list(d[_archive_name("samples")].find({}))
    docs.sort(key=lambda s: s.get("created_at", ""))
    return docs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", help="archive samples created before this sample id or ISO time")
    ap.add_argument("--ids", nargs="*", default=[], help="sample ids (short ok) to archive")
    ap.add_argument("--reason", default="not dataset material")
    ap.add_argument("--restore", nargs="*", help="sample ids to move back into the live collections")
    ap.add_argument("--list", action="store_true", help="show archived samples")
    args = ap.parse_args(argv)

    if args.list:
        for s in list_archived():
            print(f"  {s['created_at'][:16]} {s['_id'][:8]} {s.get('label')!r} "
                  f"({(s.get('material') or {}).get('class')}) archived {s['archived_at'][:16]}: {s['archive_reason']}")
        return 0
    if args.restore is not None:
        full = []
        for x in args.restore:
            hits = [s["_id"] for s in list_archived() if s["_id"].startswith(x)]
            if len(hits) != 1:
                print(f"ERR {x!r} matches {len(hits)} archived samples")
                return 2
            full.append(hits[0])
        print("restored:", restore_samples(full))
        return 0

    ids: list[str] = []
    if args.before:
        ids += samples_before(args.before)
    for x in args.ids:
        try:
            ids.append(scanner_db.resolve_sample_id(x))
        except KeyError as exc:
            print(f"ERR {exc}")
            return 2
    ids = list(dict.fromkeys(ids))
    if not ids:
        print("nothing to archive")
        return 0
    print("archived:", archive_samples(ids, args.reason))
    return 0


if __name__ == "__main__":
    sys.exit(main())
