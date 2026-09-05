"""Fit the height calibration from collection anchors.

Dual-purpose collection: any flat, matte object whose caliper height was
entered in the manifest's ``known_height_mm`` column carries a calibration
anchor inside its ordinary scan. This tool sweeps those scans, measures each
anchor's contrast-gated phase shift from the stored fringe stack, applies the
quality rules learned during the June/July calibration work, and fits the
constrained quadratic

    height_mm = a*dphi^2 + b*dphi          (c = 0: zero phase = zero height,
                                            a hard physical constraint)

Because every scan stores its raw fringe stack, a promoted curve can be used
to RE-derive the heights of every object collected so far: calibrate late,
apply retroactively.

Quality rules encoded from the June data:
- contrast gate: only ``reliable`` pixels count (same gate as reconstruction).
- enough object pixels (a washed-out top gave 483 px vs 7692 healthy).
- flat top: the object-region phase spread, converted to rough mm, must be
  small. Blue-taped plate 0.29 mm = excellent; bent cardboard 2.3 mm and a
  glossy rounded case 3.1 mm were the canonical bad anchors.

This tool NEVER touches the active calibration file. It prints a report and,
with --write-draft, writes a dated draft + JSON residual report next to the
active file. Promotion stays a deliberate human step.

USAGE
    python -m scanner_system.calibrate_from_anchors            # report only
    python -m scanner_system.calibrate_from_anchors --write-draft
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from typing import Any, Optional

import numpy as np

from . import scanner_db
from .export_bundle import STORAGE_ROOT

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REFERENCE = os.path.join(REPO_ROOT, "data", "scan_test", "calib_new",
                                 "ref20_20260905")

# Quality gates (June-derived; see module docstring).
MIN_OBJECT_PX = 3000
MAX_TOP_SPREAD_MM = 1.0
# Rough dphi->mm factor for the spread gate only (active global curve slope);
# the FIT itself never uses this number.
APPROX_MM_PER_RAD = 24.6


def measure_anchor(fringe_dir: str, reference_dir: str) -> dict[str, Any]:
    """Contrast-gated object dphi median + quality stats for one anchor scan."""
    from fpp_tools.temporal_unwrap import load_multifrequency_npz, temporal_delta_phase

    freqs, obj = load_multifrequency_npz(fringe_dir)
    ref_freqs, ref = load_multifrequency_npz(reference_dir)
    if not np.allclose(freqs, ref_freqs):
        return {"ok": False, "why": f"freq mismatch {freqs} vs {ref_freqs}"}

    result = temporal_delta_phase(ref, obj, freqs, method="wrapped")
    d = result.delta
    rel = result.reliable

    # Background = reliable pixels with near-zero shift; object = reliable
    # pixels clearly shifted. Raised objects shift NEGATIVE on this rig.
    bg = rel & (np.abs(d) < 0.08)
    if bg.sum() < 1000:
        return {"ok": False, "why": "no usable background region"}
    noise = float(np.std(d[bg]))
    obj_mask = rel & (d < -max(0.1, 5.0 * noise))
    obj_px = int(obj_mask.sum())
    if obj_px < MIN_OBJECT_PX:
        return {"ok": False, "why": f"only {obj_px} object px "
                                    f"(min {MIN_OBJECT_PX}; washed out or too small)"}

    dphi = float(np.median(d[obj_mask]))
    spread_mm = float(np.std(d[obj_mask])) * APPROX_MM_PER_RAD
    out = {"ok": True, "dphi": dphi, "object_px": obj_px,
           "top_spread_mm": round(spread_mm, 2),
           "background_noise_rad": round(noise, 4)}
    if spread_mm > MAX_TOP_SPREAD_MM:
        out["ok"] = False
        out["why"] = (f"top spread {spread_mm:.2f} mm > {MAX_TOP_SPREAD_MM} "
                      "(not flat/matte enough for an anchor)")
    return out


def fit_constrained(anchors: list[dict[str, Any]]) -> np.ndarray:
    """Least-squares [a, b] for height = a*dphi^2 + b*dphi (through origin)."""
    x = np.array([a["dphi"] for a in anchors], dtype=float)
    y = np.array([a["known_height_mm"] for a in anchors], dtype=float)
    if len(x) == 1:
        return np.array([0.0, y[0] / x[0]])
    A = np.stack([x * x, x], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def collect_anchor_scans(db=None) -> list[dict[str, Any]]:
    """All scans carrying a known_height_mm plus their fringe artifact path."""
    d = scanner_db.get_db(db)
    out = []
    for scan in d["scans"].find({"known_height_mm": {"$ne": None}}):
        arts = scanner_db.artifacts_for_scan(scan["_id"], db=d)
        npz = [a for a in arts if a.get("role") == "fringe_stack_npz"]
        sample = scanner_db.get_sample(scan["sample_id"], db=d) or {}
        out.append({
            "scan_id": scan["_id"],
            "label": sample.get("label", "?"),
            "known_height_mm": float(scan["known_height_mm"]),
            "fringe_dir": (os.path.dirname(os.path.join(STORAGE_ROOT, npz[0]["file_path"]))
                           if npz else None),
        })
    return out


def run(reference_dir: str = DEFAULT_REFERENCE, min_anchors: int = 3,
        write_draft: bool = False, db=None) -> dict[str, Any]:
    candidates = collect_anchor_scans(db=db)
    print(f"{len(candidates)} anchor-candidate scan(s) in the database")

    good, rejected = [], []
    for c in candidates:
        if not c["fringe_dir"] or not os.path.isfile(os.path.join(c["fringe_dir"], "scan.npz")):
            rejected.append({**c, "why": "no fringe stack stored (laser_only scan?)"})
            continue
        m = measure_anchor(c["fringe_dir"], reference_dir)
        if m.pop("ok"):
            good.append({**c, **m})
        else:
            rejected.append({**c, "why": m["why"]})

    for r in rejected:
        print(f"  REJECTED {r['label'][:30]:30} {r['known_height_mm']:6.2f}mm  {r['why']}")
    if len(good) < min_anchors:
        print(f"\nonly {len(good)} usable anchor(s); need {min_anchors}. "
              "Keep collecting flat matte objects with caliper heights.")
        return {"usable": len(good), "fitted": False}

    coef = fit_constrained(good)
    print(f"\nfit: height_mm = {coef[0]:+.6f}*dphi^2 {coef[1]:+.6f}*dphi   (c=0)")
    print(f"{'label':30} {'known':>7} {'dphi':>9} {'fit':>7} {'resid':>7} "
          f"{'px':>7} {'spread':>7}")
    resids = []
    for a in good:
        fit_mm = float(np.polyval([coef[0], coef[1], 0.0], a["dphi"]))
        resid = fit_mm - a["known_height_mm"]
        resids.append(resid)
        print(f"{a['label'][:30]:30} {a['known_height_mm']:7.2f} {a['dphi']:9.4f} "
              f"{fit_mm:7.2f} {resid:+7.2f} {a['object_px']:7d} "
              f"{a['top_spread_mm']:7.2f}")
    rmse = float(np.sqrt(np.mean(np.square(resids))))
    print(f"\nRMSE {rmse:.3f} mm over {len(good)} anchors "
          f"(max |resid| {max(abs(r) for r in resids):.3f} mm)")

    summary = {"usable": len(good), "fitted": True, "a": float(coef[0]),
               "b": float(coef[1]), "c": 0.0, "rmse_mm": round(rmse, 4)}

    if write_draft:
        tag = date.today().strftime("%Y%m%d")
        calib_dir = os.path.join(REPO_ROOT, "data", "scan_test", "calib_new")
        draft = os.path.join(calib_dir, f"calibration_anchors_draft_{tag}.txt")
        with open(draft, "w") as f:
            f.write(f"# height_mm = a*dphi^2 + b*dphi + c  "
                    f"(anchors draft {tag}, {len(good)} anchors, "
                    f"RMSE {rmse:.3f}mm; reference {os.path.basename(reference_dir)})\n")
            f.write(f"{coef[0]:.10f}\n{coef[1]:.10f}\n0.0000000000\n")
        report = draft.replace(".txt", "_report.json")
        with open(report, "w") as f:
            json.dump({"summary": summary, "anchors": good,
                       "rejected": [{k: v for k, v in r.items() if k != "fringe_dir"}
                                    for r in rejected],
                       "reference": reference_dir}, f, indent=2, default=str)
        print(f"\ndraft written -> {draft}")
        print("the ACTIVE calibration is untouched. To promote, set "
              "SCANNER_HEIGHT_CALIB to the draft (or update the capture.py "
              "default) after reviewing residuals, then re-derive stored scans.")
        summary["draft"] = draft
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", default=DEFAULT_REFERENCE)
    ap.add_argument("--min-anchors", type=int, default=3)
    ap.add_argument("--write-draft", action="store_true")
    args = ap.parse_args(argv)
    run(reference_dir=args.reference, min_anchors=args.min_anchors,
        write_draft=args.write_draft)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
