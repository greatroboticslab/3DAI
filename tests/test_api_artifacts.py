import base64
import os
import sys
from pathlib import Path

import numpy as np
import requests


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
API = os.getenv("API_URL", "http://localhost:8000")
API_TOKEN_HEADER = os.getenv("API_TOKEN_HEADER", "X-API-Token").strip() or "X-API-Token"
ARTIFACT_DIR = Path("data/ci_artifacts")
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def auth_headers():
    token = (os.getenv("SCANNER_API_TOKEN") or os.getenv("API_TOKEN") or "").strip()
    if not token or token.lower() == "string":
        return {}
    return {API_TOKEN_HEADER: token}


def require_ok(response):
    try:
        response.raise_for_status()
    except requests.HTTPError:
        raise AssertionError(f"{response.request.method} {response.url} failed: {response.text}")
    return response.json() if response.content else {}


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = ARTIFACT_DIR / "preview.png"
    npz_path = ARTIFACT_DIR / "raw.npz"
    png_path.write_bytes(PNG_1X1)
    np.savez_compressed(
        npz_path,
        color=np.zeros((2, 3, 4), dtype=np.uint8),
        depth=np.zeros((2, 3), dtype=np.uint16),
    )

    sample_id = "ci-sample-artifacts"
    created = require_ok(requests.post(
        f"{API}/sessions",
        json={
            "total_steps": 0,
            "sample_id": sample_id,
            "metadata": {"operator": "ci"},
        },
        headers=auth_headers(),
    ))
    session_id = created["session_id"]
    assert created["sample_id"] == sample_id

    registered = require_ok(requests.post(
        f"{API}/sessions/{session_id}/artifacts",
        json={
            "artifacts": [
                {
                    "category": "kinect",
                    "role": "color_png",
                    "file_path": "ci_artifacts/preview.png",
                    "metadata": {"note": "api smoke"},
                },
                {
                    "category": "kinect",
                    "role": "raw_color_depth_npz",
                    "file_path": "ci_artifacts/raw.npz",
                },
            ]
        },
        headers=auth_headers(),
    ))
    assert len(registered["registered"]) == 2

    scan = require_ok(requests.get(f"{API}/sessions/{session_id}/scan-3d", headers=auth_headers()))
    assert scan["session_id"] == session_id
    assert scan["sample_id"] == sample_id
    assert len(scan["artifacts"]["kinect"]) == 2

    raw = next(a for a in scan["artifacts"]["kinect"] if a["role"] == "raw_color_depth_npz")
    arrays = raw["array_metadata"]["arrays"]
    assert arrays["color"]["shape"] == [2, 3, 4]
    assert arrays["depth"]["dtype"] == "<u2"

    by_sample = require_ok(requests.get(f"{API}/samples/{sample_id}/scan-3d", headers=auth_headers()))
    assert by_sample["sample_id"] == sample_id
    assert any(session["session_id"] == session_id for session in by_sample["sessions"])

    png = next(a for a in scan["artifacts"]["kinect"] if a["role"] == "color_png")
    downloaded = requests.get(f"{API}{png['download_url']}", headers=auth_headers())
    downloaded.raise_for_status()
    assert downloaded.headers["content-type"].startswith("image/png")
    assert downloaded.content == PNG_1X1

    placeholder = requests.post(
        f"{API}/sessions",
        json={"total_steps": 0, "sample_id": "string", "metadata": {}},
        headers=auth_headers(),
    )
    assert placeholder.status_code == 400

    too_many_steps = requests.post(
        f"{API}/sessions",
        json={"total_steps": 1001, "sample_id": "ci-too-many-steps", "metadata": {}},
        headers=auth_headers(),
    )
    assert too_many_steps.status_code == 400

    invalid_session_id = requests.get(f"{API}/sessions/not-a-uuid", headers=auth_headers())
    assert invalid_session_id.status_code == 422

    print("OK: artifact API smoke test")


if __name__ == "__main__":
    main()
