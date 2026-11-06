"""
capture_tool.py - capture worker

Drives a local session and syncs data to Project A across machines via Cloudflare. 
By default it writes fake placeholder images and needs no hardware. With --real 
it captures real frames from the Kinect V2 (color + depth) instead.

Usage:
    python capture_tool.py                  # 10 steps, fake images (any machine)
    python capture_tool.py --steps 4        # custom step count
    C:\KinectEnv\Scripts\python.exe capture_tool.py --real --steps 4   # real Kinect
"""

import argparse
import base64
from datetime import datetime
import os
import sys
import time
import uuid

import requests

# Force UTF-8 for Windows console encoding compatibility
for _stream in (sys.stdout, sys.stderr):
  if hasattr(_stream, "reconfigure"):
    _stream.reconfigure(encoding="utf-8")

# Point directly to your Cloudflare tunnel hub on Project A
API = "https://collect-educated-combine-figured.trycloudflare.com"
IMAGE_ROOT = os.getenv("IMAGE_ROOT", "./data/images")
API_TIMEOUT = float(os.getenv("API_TIMEOUT_S", "15"))
API_TOKEN_HEADER = os.getenv("API_TOKEN_HEADER", "X-API-Token").strip() or "X-API-Token"

FAKE_JPEG_BYTES = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////"
    "2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/"
    "8QAFQABAQAAAAAAAAAAAAAAAAAAAAX/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAEFAqf/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAEDAQE/Aaf/xAAUEQEAAAAAAA"
    "AAAAAAAAAAAAAA/9oACAECAQE/Aaf/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAY/Aqf/xAAUEAEAAAAAAAAA"
    "AAAAAAAAAAAA/9oACAEBAAE/IV//2gAMAwEAAgADAAAAEP/EABQRAQAAAAAAAAAAAAAAAAAAABD/2gAIAQMBAT8QH//"
    "EABQRAQAAAAAAAAAAAAAAAAAAABD/2gAIAQIBAT8QH//EABQQAQAAAAAAAAAAAAAAAAAAABD/2gAIAQEAAT8QH//Z"
)

# ─────────────────────────── Capture backends ───────────────────────────


def fake_capture(session_id: str, step_index: int) -> list[str]:
  """Write a placeholder image for the step and return its file path in a list."""
  time.sleep(0.3)

  out_dir = os.path.join(IMAGE_ROOT, session_id)
  os.makedirs(out_dir, exist_ok=True)
  path = os.path.abspath(os.path.join(out_dir, f"img_{step_index:03d}.jpg"))

  with open(path, "wb") as f:
    f.write(FAKE_JPEG_BYTES)

  return [path]


_kinect = None


def auth_headers() -> dict[str, str]:
  token = (os.getenv("SCANNER_API_TOKEN") or os.getenv("API_TOKEN") or "").strip()
  if not token or token.lower() == "string":
    return {}
  return {API_TOKEN_HEADER: token}


def _get_kinect():
  """Open the Kinect color+depth runtime once and reuse it."""
  global _kinect
  if _kinect is None:
    from pykinect2 import PyKinectRuntime, PyKinectV2

    print("  initializing Kinect (color + depth)...")
    _kinect = PyKinectRuntime.PyKinectRuntime(
        PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth
    )
    time.sleep(2.0)
  return _kinect


def _close_kinect() -> None:
  global _kinect
  if _kinect is not None:
    _kinect.close()
    _kinect = None


def kinect_capture(session_id: str, step_index: int, frame_timeout: float = 15.0) -> list[str]:
  """Capture a real Kinect color frame and depth matrix, returning paths for both files."""
  import cv2
  import numpy as np

  kinect = _get_kinect()
  time.sleep(0.2)

  color = None
  depth = None
  deadline = time.time() + frame_timeout
  while time.time() < deadline and (color is None or depth is None):
    if color is None and kinect.has_new_color_frame():
      cf = kinect.get_last_color_frame()
      h, w = kinect.color_frame_desc.Height, kinect.color_frame_desc.Width
      color = cf.reshape((h, w, 4))
    if depth is None and kinect.has_new_depth_frame():
      df = kinect.get_last_depth_frame()
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
  cv2.imwrite(png_path, color[:, :, :3])

  npz_path = base + ".npz"
  np.savez_compressed(
      npz_path,
      color=color,
      depth=depth if depth is not None else np.zeros(0),
  )

  # Return BOTH the PNG image path and the NPZ depth archive path
  return [png_path, npz_path]


# ─────────────────────────── Project A Sync Helper ───────────────────────────


def sync_to_project_a(
    material: str, object_name: str, surface: str, height_mm, image_paths: list
) -> None:
  """Pushes completed local scan metadata and all captured files directly to Project A's backend."""
  category = material.lower() if material else "kinect_scans"

  metadata_payload = {
      "category": category,
      "date": datetime.today().strftime("%Y-%m-%d"),
      "data": {
          "Object Name": object_name,
          "Material": material,
          "Surface": surface,
          "Height (mm)": height_mm if height_mm else "N/A",
      },
  }

  try:
    res = requests.post(
        f"{API}/collection/submission",
        json=metadata_payload,
        headers=auth_headers(),
        timeout=API_TIMEOUT,
    )
    res.raise_for_status()
    sample_id = res.json().get("sample_id")
    print(f"  Successfully created record | Sample ID: {sample_id}")

    for file_path in image_paths:
      if os.path.exists(file_path):
        with open(file_path, "rb") as f:
          files = {"file": f}
          data = {"sample_id": sample_id, "category": category}
          img_res = requests.post(
              f"{API}/collection/images/upload",
              data=data,
              files=files,
              headers=auth_headers(),
              timeout=API_TIMEOUT,
          )
          if img_res.status_code == 200:
            print(f"  Uploaded artifact: {os.path.basename(file_path)}")

    print("  Sync complete!")

  except Exception as e:
    print(f"  Warning: Failed to sync with Project A backend: {e}")


# ─────────────────────────── Session Runner ────────────────────────────


def run_session(total_steps: int = 10, real: bool = False) -> None:
  """Run local capture loop, collect operator terminal input, and sync all data at end."""
  print("\n" + "=" * 40)
  print(" ENTER SCAN METADATA ")
  print("=" * 40)
  object_name = input("Enter Object Name (e.g., 'Wooden Block 01'): ").strip() or "Unknown Object"
  material = input("Enter Material/Category (e.g., 'Wood', 'Plastic'): ").strip() or "kinect_scans"
  surface = input("Enter Surface Type (e.g., 'Matte', 'Glossy'): ").strip() or "Matte"
  height_input = input("Enter Height in mm (optional): ").strip()
  height_mm = float(height_input) if height_input.replace(".", "", 1).isdigit() else None

  # 1. Check existing categories via /home endpoint
  try:
    res = requests.get(f"{API}/home")
    existing_categories = res.json() if res.status_code == 200 else []
  except Exception:
    existing_categories = []

  matched = False
  for existing in existing_categories:
    if existing.lower() == material.lower():
      material = existing  # Silently reuse existing canonical name
      matched = True
      break

  # 2. If it does not exist, create the block JSON file on the server via /settings
  if not matched:
    try:
      block_payload = {"category": material, "block": True}
      requests.post(
          f"{API}/settings",
          json=block_payload,
          headers=auth_headers(),
          timeout=API_TIMEOUT,
      )
      print(f"\n[Notice] Category '{material}' is new. Created block configuration file on server.")
    except Exception as e:
      print(f"\n[Warning] Could not create block configuration: {e}")

  capture_fn = kinect_capture if real else fake_capture
  mode = "REAL Kinect" if real else "fake"
  session_id = str(uuid.uuid4())

  print(f"\nStarting local capture session ({total_steps} steps, {mode} capture)...")
  captured_paths = []

  try:
    for idx in range(total_steps):
      print(f"Step {idx + 1}/{total_steps}")
      step_file_paths = capture_fn(session_id, idx)
      for path in step_file_paths:
        print(f"  captured  → {path}")
        captured_paths.append(path)
  finally:
    if real:
      _close_kinect()

  print(f"\n{'=' * 40}")
  print("Local capture finished successfully.")
  sync_to_project_a(material, object_name, surface, height_mm, captured_paths)
  print(f"{'=' * 40}")


# ─────────────────────────── Entry Point ───────────────────────────────

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run a capture session")
  parser.add_argument(
      "--steps", type=int, default=10, help="Number of capture steps (default: 10)"
  )
  parser.add_argument(
      "--real",
      action="store_true",
      help=(
          "Capture from the real Kinect (requires C:\\KinectEnv and a connected"
          " sensor). Without it, fake images."
      ),
  )
  args = parser.parse_args()

  run_session(total_steps=args.steps, real=args.real)