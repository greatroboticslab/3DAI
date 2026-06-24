"""
Fetch scanner-side 3D artifacts for one Torres recording/sample.

This script is intentionally plain HTTP + JSON so it can be copied into the
Torres GUI project without requiring MongoDB or scanner hardware libraries.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import requests


API_URL = (
    os.getenv("SCANNER_API_URL")
    or os.getenv("API_URL")
    or "http://localhost:8000"
).rstrip("/")
API_TIMEOUT = float(os.getenv("API_TIMEOUT_S", "15"))
API_TOKEN_HEADER = os.getenv("API_TOKEN_HEADER", "X-API-Token").strip() or "X-API-Token"


def auth_headers() -> dict[str, str]:
    token = (os.getenv("SCANNER_API_TOKEN") or os.getenv("API_TOKEN") or "").strip()
    if not token or token.lower() == "string":
        return {}
    return {API_TOKEN_HEADER: token}


def fetch_scan_3d(api_url: str, sample_id: str) -> dict[str, Any]:
    response = requests.get(
        f"{api_url.rstrip('/')}/samples/{sample_id}/scan-3d",
        headers=auth_headers(),
        timeout=API_TIMEOUT,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise SystemExit(
            f"GET {response.url} failed: {response.status_code} {response.text}"
        ) from exc
    return response.json()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch 3DAI scan_3d JSON for one sample_id.")
    parser.add_argument("sample_id", help="Torres recording_name / scanner sample_id.")
    parser.add_argument("--api-url", default=API_URL, help="Scanner API base URL.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to stdout.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = fetch_scan_3d(args.api_url, args.sample_id)
    text = json.dumps(payload, indent=2)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
