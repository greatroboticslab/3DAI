"""
capture_tool.py — Fake capture worker

Drives a full session against the API using placeholder image files.
Once the USB webcam path is ready, replace fake_capture() with the
real acquisition function and nothing else needs to change.

Usage:
    python capture_tool.py                  # 10 steps (default)
    python capture_tool.py --steps 4        # custom step count
"""

import argparse
import os
import time

import requests

from lib_3dai.system import SystemManager
from lib_3dai.camera import Camera

API = "http://localhost:8000"
IMAGE_ROOT = "./data/images"

camera = Camera(SystemManager())  # Initialize the camera instance

# ─────────────────────────── Capture ────────────────────────────

def capture(session_id:str, step_index: int) -> str:
    """
    Simulate acquiring an image for the given step.

    Writes a placeholder file and returns its absolute path.
    Replace this function body with real webcam capture when ready —
    the signature and return type stay the same.

    Args:
        step_index: Zero-based index of the current step.

    Returns:
        Absolute path to the captured (or placeholder) image file.
    """
    time.sleep(0.3)  # simulate shutter / processing time

    path = f"{IMAGE_ROOT}/{session_id}"
    os.makedirs(path, exist_ok=True)
    path = os.path.abspath(f"{path}/img_{step_index:03d}.jpg")

    with open(path, "wb") as f:
        f.write(b"FAKE_IMAGE_PLACEHOLDER")

    return path


# ─────────────────────────── API helpers ───────────────────────────────

def create_session(total_steps: int) -> str:
    """POST /sessions → returns session_id."""
    res = requests.post(f"{API}/sessions", json={"total_steps": total_steps})
    res.raise_for_status()
    return res.json()["session_id"]


def fetch_steps(session_id: str) -> list:
    """GET /sessions/{id}/steps → returns list of step dicts."""
    res = requests.get(f"{API}/sessions/{session_id}/steps")
    res.raise_for_status()
    return res.json()


def report_image(session_id: str, step_id: str, file_path: str) -> None:
    """POST /report-image — mark step done and record the image path."""
    res = requests.post(f"{API}/report-image", json={
        "session_id": session_id,
        "step_id": step_id,
        "file_path": file_path,
    })
    res.raise_for_status()


def get_session_status(session_id: str) -> dict:
    """GET /sessions/{id} → returns status dict."""
    res = requests.get(f"{API}/sessions/{session_id}")
    res.raise_for_status()
    return res.json()


# ─────────────────────────── Session runner ────────────────────────────

def run_session(total_steps: int = 10) -> None:
    """
    Run a complete fake capture session end-to-end.

    1. Create a session  (API generates steps + UUIDs)
    2. Fetch those steps (so we use the real step IDs)
    3. For each step: fake capture → report to API
    4. Print a final summary
    """
    print(f"Creating session ({total_steps} steps)...")
    session_id = create_session(total_steps)
    print(f"  Session ID: {session_id}\n")

    steps = fetch_steps(session_id)
    print(f"Fetched {len(steps)} steps from API\n")

    for step in steps:
        idx = step["step_index"]
        step_id = step["step_id"]

        print(f"Step {idx + 1}/{len(steps)}")
        path = capture(session_id, idx)
        print(f"  captured  → {path}")

        report_image(session_id, step_id, path)
        print(f"  reported  ✓")

    status = get_session_status(session_id)
    print(f"\n{'=' * 40}")
    print(f"Session complete")
    print(f"  ID:        {session_id}")
    print(f"  Status:    {status['status']}")
    print(f"  Progress:  {status['completed_steps']}/{status['total_steps']}")
    print(f"{'=' * 40}")


# ─────────────────────────── Entry point ───────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a fake capture session")
    parser.add_argument("--steps", type=int, default=10,
                        help="Number of capture steps (default: 10)")
    args = parser.parse_args()

    run_session(total_steps=args.steps)