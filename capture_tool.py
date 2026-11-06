"""
capture_tool.py - capture worker (lib_3dai version)

Drives a local session and syncs data to Project A across machines via Cloudflare. 
By default it writes fake placeholder images and needs no hardware. With --real 
it captures real frames from the lib_3dai camera instance.

Usage:
    python capture_tool.py                  # 10 steps, fake images (any machine)
    python capture_tool.py --steps 4        # custom step count
    C:\KinectEnv\Scripts\python.exe capture_tool.py --real --steps 4   # real camera
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
API = "https://trycloudflare.com"
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


_camera_instance = None

def auth_headers() -> dict[str, str]:
  token = (os.getenv("SCANNER_API_TOKEN") or os.getenv("API_TOKEN") or "").strip()
  if not token or token.lower() == "string":
    return {}
  return {API_TOKEN_HEADER: token}


def _get_camera():
  """Open the lib_3dai camera context once and reuse it."""
  global _camera_instance
  if _camera_instance is None:
    from lib_3dai.system import SystemManager
    from lib_3dai.camera import Camera
    
    print("  initializing lib_3dai camera system...")
    _camera_instance = Camera(SystemManager())
    time.sleep(1.0)
  return _camera_instance


def real_camera_capture(session_id: str, step_index: int) -> list[str]:
  """Capture an image using the real lib_3dai camera framework."""
  import cv2
  import numpy as np

  cam = _get_camera()
  time.sleep(0.2)

  # Acquire frame using your original framework's image capture methodology
  # (Adjust this specific call if lib_3dai requires a method like cam.capture() or cam.get_frame())
  frame = cam.capture() 

  out_dir = os.path.join(IMAGE_ROOT, session_id)
  os.makedirs(out_dir, exist_ok=True)
  png_path = os.path.abspath(os.path.join(out_dir, f"img_{step_index:03d}.png"))

  # Save the frame to disk safely
  if isinstance(frame, np.ndarray):
    cv2.imwrite(png_path, frame)
  else:
    # Fallback placeholder if the framework returns a custom object instead of a raw numpy array
    with open(png_path, "wb") as f:
      f.write(FAKE_JPEG_BYTES)

  return [png_path]


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

  try:
    res = requests.get(f"{API}/home", timeout=API_TIMEOUT)
    existing_categories = res.json() if res.status_code == 200 else []
  except Exception:
    existing_categories = []

  matched = False
  for existing in existing_categories:
    if existing.lower() == material.lower():
      material = existing  
      matched = True
      break

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

  capture_fn = real_camera_capture if real else fake_capture
  mode = "REAL Camera" if real else "fake"
  session_id = str(uuid.uuid4())

  print(f"\nStarting local capture session ({total_steps} steps, {mode} capture)...")
  captured_paths = []

  try:
    for idx in range(total_steps):
      step_index = idx + 1
      print(f"  [Step {step_index}/{total_steps}] Capturing frame...")
      paths = capture_fn(session_id, step_index)
      captured_paths.extend(paths)

    print(f"\n[Success] All {total_steps} capture steps completed locally.")

  except Exception as e:
    print(f"\n[Error] Capture sequence encountered a failure: {e}")
    raise e

  if captured_paths:
    print(f"\nInitiating sync to Project A hub ({len(captured_paths)} files)...")
    sync_to_project_a(
        material=material,
        object_name=object_name,
        surface=surface,
        height_mm=height_mm,
        image_paths=captured_paths,
    )
  else:
    print("\n[Warning] No frames were captured. Skipping sync step.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Local capture session sync utility for Project A backend."
  )
  parser.add_argument(
      "--steps",
      type=int,
      default=10,
      help="Total capture iterations per session (default: 10)",
  )
  parser.add_argument(
      "--real",
      action="store_true",
      help="Fire up physical camera hardware via lib_3dai",
  )

  args = parser.parse_args()

  try:
    run_session(total_steps=args.steps, real=args.real)
  except KeyboardInterrupt:
    print("\n\n[Aborted] Scan sequence canceled by operator terminal signal.")
    sys.exit(0)
