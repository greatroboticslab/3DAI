"""
Register an existing scan output folder with the 3DAI API.

This is scanner-side bookkeeping only. It never starts capture, projector,
Kinect, relay, serial, ESP32, or laser code.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from artifact_store import ArtifactError, get_artifact_root, inspect_artifact, resolve_artifact_path


API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
API_TIMEOUT = float(os.getenv("API_TIMEOUT_S", "15"))
API_TOKEN_HEADER = os.getenv("API_TOKEN_HEADER", "X-API-Token").strip() or "X-API-Token"


def auth_headers() -> dict[str, str]:
    token = (os.getenv("SCANNER_API_TOKEN") or os.getenv("API_TOKEN") or "").strip()
    if not token or token.lower() == "string":
        return {}
    return {API_TOKEN_HEADER: token} if token else {}


def role_for_file(path: Path, info: dict[str, Any]) -> tuple[str, str] | None:
    name = path.name.lower()
    suffix = path.suffix.lower()

    if suffix == ".npz":
        arrays = set(info.get("array_metadata", {}).get("arrays", {}).keys())
        if {"color", "depth"}.issubset(arrays):
            return "kinect", "raw_color_depth_npz"
        if "gray_stack" in arrays or any(key.startswith("gray_") for key in arrays):
            return "fringe", "fringe_stack_npz"
        return None

    if suffix in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
        if name.startswith("color_"):
            return "kinect", "color_image"
        if name.startswith("depth_") or "depth" in name:
            return "kinect", "depth_preview"
        if name == "white.png":
            return "fringe", "white_frame"
        if name.startswith("fringe_"):
            return "fringe", "fringe_frame"
        if name == "height_mm.png":
            return "fusion", "height_map_png"
        if name in {"shape.png", "object_shape.png"}:
            return "fusion", "shape_png"
        if name == "contrast.png":
            return "fusion", "contrast_png"
        if name == "amplitude_map.png":
            return "fusion", "amplitude_map_png"
        if name in {"phase_wrapped.png", "phase_unwrapped.png"}:
            return "fusion", name.removesuffix(".png")
        if name in {"roi_overlay.png", "flattened_residual.png"}:
            return "calibration", name.removesuffix(".png")
        return None

    if suffix == ".txt" and name.startswith("calibration"):
        return "calibration", "calibration_coefficients_txt"

    if suffix == ".json":
        return "calibration", "metadata_json"

    return None


def iter_candidate_files(folder: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(path for path in folder.glob(pattern) if path.is_file())


def resolve_folder_argument(folder: Path, artifact_root: Path) -> Path:
    if folder.is_absolute():
        resolved, _ = resolve_artifact_path(str(folder), root=artifact_root)
        return resolved

    cwd_candidate = folder.resolve()
    if cwd_candidate.exists():
        resolved, _ = resolve_artifact_path(str(cwd_candidate), root=artifact_root)
        return resolved

    resolved, _ = resolve_artifact_path(str(folder), root=artifact_root)
    if resolved.exists():
        return resolved

    parts = folder.parts
    if parts and parts[0].lower() == artifact_root.name.lower():
        stripped = Path(*parts[1:]) if len(parts) > 1 else Path(".")
        resolved, _ = resolve_artifact_path(str(stripped), root=artifact_root)

    return resolved


def collect_artifacts(folder: Path, recursive: bool = False) -> tuple[list[dict[str, Any]], list[str]]:
    artifact_root = get_artifact_root()
    resolved_folder = resolve_folder_argument(folder, artifact_root)
    if not resolved_folder.is_dir():
        raise ArtifactError(f"Scan folder does not exist under ARTIFACT_ROOT: {folder}")

    artifacts: list[dict[str, Any]] = []
    skipped: list[str] = []

    for path in iter_candidate_files(resolved_folder, recursive):
        try:
            _, relative_path = resolve_artifact_path(str(path), root=artifact_root)
            info = inspect_artifact(relative_path, root=artifact_root)
        except ArtifactError as exc:
            skipped.append(f"{path.name}: {exc}")
            continue

        role = role_for_file(path, info)
        if role is None:
            skipped.append(f"{relative_path}: unrecognized scan artifact")
            continue

        category, role_name = role
        artifacts.append({
            "category": category,
            "role": role_name,
            "file_path": relative_path,
            "metadata": {"registered_by": "scripts/register_scan_folder.py"},
        })

    return artifacts, skipped


def post_json(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.request(
        method,
        url,
        json=payload,
        headers=auth_headers(),
        timeout=API_TIMEOUT,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise SystemExit(f"{method} {url} failed: {response.status_code} {response.text}") from exc
    return response.json()


def register_scan_folder(args: argparse.Namespace) -> dict[str, Any]:
    artifacts, skipped = collect_artifacts(Path(args.folder), recursive=args.recursive)
    if not artifacts:
        raise SystemExit("No recognized, registerable artifacts found.")

    payload = {
        "total_steps": 0,
        "sample_id": args.sample_id,
        "metadata": {
            "source": "scripts/register_scan_folder.py",
            "scan_folder": str(args.folder).replace("\\", "/"),
            "recursive": args.recursive,
        },
    }

    if args.dry_run:
        return {"session": payload, "artifacts": artifacts, "skipped": skipped}

    session = post_json("POST", f"{args.api_url}/sessions", payload)
    post_json("POST", f"{args.api_url}/sessions/{session['session_id']}/artifacts", {"artifacts": artifacts})
    scan = post_json("GET", f"{args.api_url}/sessions/{session['session_id']}/scan-3d")
    return {"session": session, "scan_3d": scan, "skipped": skipped}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Register an existing scanner output folder.")
    parser.add_argument("--sample-id", required=True, help="Torres recording_name / sample_id join key.")
    parser.add_argument("--folder", required=True, help="Scan folder under ARTIFACT_ROOT, for example data/scan_test/calib/breadboard.")
    parser.add_argument("--api-url", default=API_URL, help="Scanner API base URL.")
    parser.add_argument("--recursive", action="store_true", help="Scan nested folders too.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be sent without writing to the API.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.api_url = args.api_url.rstrip("/")
    result = register_scan_folder(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
