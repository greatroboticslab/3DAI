"""Live cross-talk demo: 4DAI <-> scanner, over HTTP, joined on sample_id.

Acts as the 4DAI side. It creates a sample identity (exactly as 4DAI does when a
user submits), asks the scanner to run a multimodal capture for that id, then
reads the result back -- demonstrating that the two systems interoperate on a
shared ``sample_id``.

Usage:
    python -m scanner_system.crosstalk_demo [mode]
      mode: full (default, ~1 min, all 3 modalities) | laser_only (~15s, fast)

Env: SCANNER_API_URL (default http://127.0.0.1:8600)
"""

import os
import sys
import uuid

import requests

API = os.getenv("SCANNER_API_URL", "http://127.0.0.1:8600").rstrip("/")


def line(msg=""):
    print(msg, flush=True)


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

    line("=" * 68)
    line("  4DAI  <-->  SCANNER   cross-talk demo (shared sample_id, over HTTP)")
    line("=" * 68)

    # 1) 4DAI creates the sample identity (its POST /collection/submission does
    #    exactly this: mints a UUID sample_id and owns it).
    sample_id = str(uuid.uuid4())
    line(f"\n[4DAI]  A user submitted a sample. 4DAI minted sample_id:")
    line(f"        {sample_id}")

    # 2) 4DAI asks the scanner to capture that sample (over HTTP).
    line(f"\n[4DAI -> scanner]  POST {API}/capture")
    line(f"        (asking the scanner to scan this sample, mode={mode})")
    line(f"        ...scanning across projector / lasers / kinect...")
    r = requests.post(f"{API}/capture", json={
        "sample_id": sample_id,
        "label": "cross-talk demo sample",
        "material_class": "demo",
        "material_subclass": "crosstalk",
        "mode": mode,
        "laser_channels": [1, 2, 3],
    }, timeout=300)
    r.raise_for_status()
    cap = r.json()
    line(f"\n[scanner]  capture complete: status={cap['status']}")
    line(f"        per-instrument: {cap['results']}")
    line(f"        artifacts: {cap['artifacts']}")

    # 3) 4DAI reads the multimodal result back, joined on the SAME sample_id.
    line(f"\n[4DAI -> scanner]  GET {API}/samples/{sample_id}")
    r = requests.get(f"{API}/samples/{sample_id}", timeout=30)
    r.raise_for_status()
    rec = r.json()
    line(f"\n[scanner -> 4DAI]  returned the 4D record for that sample_id:")
    line(f"        label   : {rec['label']}")
    line(f"        material: {rec['material']}")
    for sc in rec["scans"]:
        line(f"        scan {sc['scan_id'][:8]}  status={sc['status']}  mode={sc['mode']}")
        for modality, arts in sc["artifacts"].items():
            for a in arts:
                wl = (a.get("laser_state") or {}).get("wavelength_nm")
                tag = f" {wl}nm" if wl else ""
                line(f"           - {modality:9s} {a['role']}{tag}   {API}{a['download_url']}")

    line("\n" + "=" * 68)
    line("  Cross-talk works: 4DAI created the id, the scanner captured under it,")
    line("  and 4DAI read the multimodal result back -- all on one sample_id.")
    line("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
