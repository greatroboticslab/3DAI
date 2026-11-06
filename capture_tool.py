"""
capture_tool.py - capture worker

Drives a full session against the API. By default it writes fake placeholder
images and needs no hardware. With --real it captures real frames from the
Kinect V2 (color + depth) instead; that mode must run under the patched
C:\\KinectEnv interpreter with the sensor connected and powered.

Usage:
    python capture_tool.py                  # 10 steps, fake images (any machine)
    python capture_tool.py --steps 4        # custom step count
    C:\\KinectEnv\\Scripts\\python.exe capture_tool.py --real --steps 4   # real Kinect
"""

import argparse
import base64
import os
import sys
import time

import requests

# The console output below uses Unicode glyphs (→, ✓). On Windows the default
# console encoding is cp1252, which cannot encode them and raises
# UnicodeEncodeError mid-session. Force UTF-8 so the worker runs anywhere.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

API = os.getenv("API_URL", "http://localhost:8000")
IMAGE_ROOT = os.getenv("IMAGE_ROOT", "./data/images")
API_TIMEOUT = float(os.getenv("API_TIMEOUT_S", "15"))
API_TOKEN_HEADER = os.getenv("API_TOKEN_HEADER", "X-API-Token").strip() or "X-API-Token"

FAKE_JPEG_BYTES = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////"
    "2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/"
    "8QAFQABAQAAAAAAAAAAAAAAAAAAAAX/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAH/xAAUEAEAAAAA"
    "AAAAAAAAAAAAAAAA/9oACAEBAAEFAqf/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAEDAQE/Aaf/xAAUEQEAAAAAAA"
    "AAAAAAAAAAAAAA/9oACAECAQE/Aaf/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAY/Aqf/xAAUEAEAAAAAAAAA"
    "AAAAAAAAAAAA/9oACAEBAAE/IV//2gAMAwEAAgADAAAAEP/EABQRAQAAAAAAAAAAAAAAAAAAABD/2gAIAQMBAT8QH//"
    "EABQRAQAAAAAAAAAAAAAAAAAAABD/2gAIAQIBAT8QH//EABQQAQAAAAAAAAAAAAAAAAAAABD/2gAIAQEAAT8QH//Z"
)

# ─────────────────────────── Capture backends ───────────────────────────
#
# Two backends share one signature: capture(session_id, step_index) -> path.
# fake_capture needs no hardware and runs anywhere. kinect_capture talks to the
# Kinect and only loads pykinect2/cv2/numpy when actually used, so fake mode
# still works on machines without those (e.g. the API .venv). Hardware is
# initialized lazily and once (see _get_kinect), never at import time.


def fake_capture(session_id: str, step_index: int) -> str:
    """Write a placeholder image for the step and return its absolute path."""
    time.sleep(0.3)  # simulate shutter / processing time

    out_dir = os.path.join(IMAGE_ROOT, session_id)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.abspath(os.path.join(out_dir, f"img_{step_index:03d}.jpg"))

    with open(path, "wb") as f:
        f.write(FAKE_JPEG_BYTES)

    return path


_kinect = None  # opened once on first real capture, reused across steps


def auth_headers() -> dict[str, str]:
    token = (os.getenv("SCANNER_API_TOKEN") or os.getenv("API_TOKEN") or "").strip()
    if not token or token.lower() == "string":
        return {}
    return {API_TOKEN_HEADER: token}


def _get_kinect():
    """Open the Kinect color+depth runtime once and reuse it."""
    global _kinect
    if _kinect is None:
        # Imported here, not at module scope, so fake mode never needs pykinect2.
        from pykinect2 import PyKinectV2, PyKinectRuntime
        print("  initializing Kinect (color + depth)...")
        _kinect = PyKinectRuntime.PyKinectRuntime(
            PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth
        )
        time.sleep(2.0)  # let the sensor start delivering frames
    return _kinect


def _close_kinect() -> None:
    global _kinect
    if _kinect is not None:
        _kinect.close()
        _kinect = None


def kinect_capture(session_id: str, step_index: int, frame_timeout: float = 15.0) -> str:
    """
    Capture a real Kinect color frame (and depth) for the given step.

    Saves the color frame as a PNG (the path recorded by the API) and the raw
    color + depth arrays alongside as an .npz for later 3D reconstruction.
    Returns the absolute path of the color PNG.
    """
    import numpy as np
    import cv2

    kinect = _get_kinect()
    time.sleep(0.2)  # small settle (placeholder for projector settle time later)

    color = None
    depth = None
    deadline = time.time() + frame_timeout
    while time.time() < deadline and (color is None or depth is None):
        if color is None and kinect.has_new_color_frame():
            cf = kinect.get_last_color_frame()  # flat uint8 BGRA
            h, w = kinect.color_frame_desc.Height, kinect.color_frame_desc.Width
            color = cf.reshape((h, w, 4))
        if depth is None and kinect.has_new_depth_frame():
            df = kinect.get_last_depth_frame()  # flat uint16 (mm)
            h, w = kinect.depth_frame_desc.Height, kinect.depth_frame_desc.Width
            depth = df.reshape((h, w))
        time.sleep(0.03)

    if color is None:
        raise RuntimeError(
            f"Kinect delivered no color frame within {frame_timeout}s "
            "(check the power brick and USB3 connection)."
        )

    out_dir = os.path.join(IMAGE_ROOT, session_id)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.abspath(os.path.join(out_dir, f"img_{step_index:03d}"))

    png_path = base + ".png"
    cv2.imwrite(png_path, color[:, :, :3])  # BGRA -> BGR for saving

    # Keep raw color + depth for downstream 3D processing.
    np.savez_compressed(
        base + ".npz",
        color=color,
        depth=depth if depth is not None else np.zeros(0),
    )

    return png_path


# ─────────────────────────── API helpers ───────────────────────────────

def create_session(total_steps: int) -> str:
    """POST /sessions → returns session_id."""
    res = requests.post(
        f"{API}/sessions",
        json={"total_steps": total_steps},
        headers=auth_headers(),
        timeout=API_TIMEOUT,
    )
    res.raise_for_status()
    return res.json()["session_id"]


def fetch_steps(session_id: str) -> list:
    """GET /sessions/{id}/steps → returns list of step dicts."""
    res = requests.get(
        f"{API}/sessions/{session_id}/steps",
        headers=auth_headers(),
        timeout=API_TIMEOUT,
    )
    res.raise_for_status()
    return res.json()


def report_image(session_id: str, step_id: str, file_path: str) -> None:
    """POST /report-image — mark step done and record the image path."""
    res = requests.post(f"{API}/report-image", json={
        "session_id": session_id,
        "step_id": step_id,
        "file_path": file_path,
    }, headers=auth_headers(), timeout=API_TIMEOUT)
    res.raise_for_status()


def get_session_status(session_id: str) -> dict:
    """GET /sessions/{id} → returns status dict."""
    res = requests.get(
        f"{API}/sessions/{session_id}",
        headers=auth_headers(),
        timeout=API_TIMEOUT,
    )
    res.raise_for_status()
    return res.json()


# ─────────────────────────── Session runner ────────────────────────────

def run_session(total_steps: int = 10, real: bool = False) -> None:
    """
    Run a complete capture session end-to-end.

    1. Create a session  (API generates steps + UUIDs)
    2. Fetch those steps (so we use the real step IDs)
    3. For each step: capture (fake or real Kinect) → report to API
    4. Print a final summary

    With real=True the Kinect is opened once and closed at the end.
    """
    capture_fn = kinect_capture if real else fake_capture
    mode = "REAL Kinect" if real else "fake"

    print(f"Creating session ({total_steps} steps, {mode} capture)...")
    session_id = create_session(total_steps)
    print(f"  Session ID: {session_id}\n")

    steps = fetch_steps(session_id)
    print(f"Fetched {len(steps)} steps from API\n")

    try:
        for step in steps:
            idx = step["step_index"]
            step_id = step["step_id"]

            print(f"Step {idx + 1}/{len(steps)}")
            path = capture_fn(session_id, idx)
            print(f"  captured  → {path}")

            report_image(session_id, step_id, path)
            print(f"  reported  ✓")
    finally:
        if real:
            _close_kinect()

    status = get_session_status(session_id)
    print(f"\n{'=' * 40}")
    print(f"Session complete")
    print(f"  ID:        {session_id}")
    print(f"  Status:    {status['status']}")
    print(f"  Progress:  {status['completed_steps']}/{status['total_steps']}")
    print(f"{'=' * 40}")


# ─────────────────────────── Entry point ───────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a capture session")
    parser.add_argument("--steps", type=int, default=10,
                        help="Number of capture steps (default: 10)")
    parser.add_argument("--real", action="store_true",
                        help="Capture from the real Kinect (requires C:\\KinectEnv "
                             "and a connected sensor). Without it, fake images.")
    args = parser.parse_args()

    run_session(total_steps=args.steps, real=args.real)
