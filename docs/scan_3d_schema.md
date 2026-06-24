# scan_3d API Contract

This document describes the read-only 3D scan package exposed by the 3DAI API
for Torres's GUI and later MongoDB integration. The API returns metadata as JSON
and exposes large binary artifacts through download URLs. It does not embed raw
image stacks, depth arrays, or point-cloud data inside JSON responses.

## Join Key

Torres's GUI should create a `sample_id` and pass it to the scanner API when it
creates a session:

```json
{
  "total_steps": 0,
  "sample_id": "plant-2026-06-23-001",
  "metadata": {
    "operator": "torres",
    "notes": "matte sample on reference panel"
  }
}
```

`sample_id` is opaque to this service. The scanner API stores and returns it
exactly after trimming leading and trailing whitespace.

## Endpoints

### Health Check

`GET /health`

Returns API liveness and whether token auth is required:

```json
{
  "status": "ok",
  "auth_required": true
}
```

### Auth

For localhost-only development, `API_TOKEN` may be empty. Before exposing the
API to another machine, set a real `API_TOKEN` and send it on each request with
the `X-API-Token` header unless `API_TOKEN_HEADER` has been changed.

Docker defaults to binding the API on `127.0.0.1`. If `API_BIND_ADDR` is changed
to a non-local address without `API_TOKEN`, the API fails fast at startup.

### Create a Session

`POST /sessions`

Optional fields:

- `total_steps`: number of capture steps to create. Use `0` for metadata-only or
  already-captured artifact registration.
- `sample_id`: Torres's sample join key.
- `metadata`: JSON object for session context.

### Register Artifacts

`POST /sessions/{session_id}/artifacts`

Registers files that already exist under `ARTIFACT_ROOT`.

```json
{
  "artifacts": [
    {
      "category": "kinect",
      "role": "color_png",
      "file_path": "images/session-id/img_000.png",
      "metadata": {
        "source": "capture_tool.py --real"
      }
    },
    {
      "category": "fringe",
      "role": "fringe_stack_npz",
      "file_path": "scan_test/calib/breadboard/scan.npz"
    },
    {
      "category": "fusion",
      "role": "height_map_png",
      "file_path": "scan_test/calib/breadboard/height_mm.png"
    }
  ]
}
```

The API rejects missing files, zero-byte files, unsupported extensions, and
paths that escape `ARTIFACT_ROOT`.

### Fetch One Scan Package

`GET /sessions/{session_id}/scan-3d`

### Fetch All Scans for a Sample

`GET /samples/{sample_id}/scan-3d`

### Download an Artifact

`GET /artifacts/{artifact_id}`

Streams the registered file with its stored media type.

## Response Shape

```json
{
  "session_id": "590d8135-4a7e-48d0-ae35-14211189ed5f",
  "sample_id": "plant-2026-06-23-001",
  "status": "running",
  "total_steps": 0,
  "completed_steps": 0,
  "metadata": {
    "operator": "torres"
  },
  "created_at": "2026-06-23T18:30:00.000000",
  "artifacts": {
    "kinect": [
      {
        "artifact_id": "c99fd369-2e5d-4ff2-890e-e5826b8a0266",
        "category": "kinect",
        "role": "raw_color_depth_npz",
        "file_path": "images/session-id/img_000.npz",
        "media_type": "application/x-npz",
        "size_bytes": 2695092,
        "array_metadata": {
          "arrays": {
            "color": {
              "dtype": "|u1",
              "shape": [1080, 1920, 4],
              "fortran_order": false
            },
            "depth": {
              "dtype": "<u2",
              "shape": [424, 512],
              "fortran_order": false
            }
          }
        },
        "metadata": {},
        "download_url": "/artifacts/c99fd369-2e5d-4ff2-890e-e5826b8a0266"
      }
    ],
    "fringe": [],
    "fusion": [],
    "calibration": []
  }
}
```

## Artifact Categories

- `kinect`: Kinect color, depth, and color-depth raw files.
- `fringe`: structured-light fringe inputs such as `white.png`, individual
  fringe frames, and fringe stack `.npz` files.
- `fusion`: reconstructed products such as `shape.png`, `height_mm.png`,
  confidence maps, point clouds, or future fused outputs.
- `calibration`: calibration coefficients and calibration-related result files.

## Binary Policy

Keep large binaries outside JSON. Store them as files under `ARTIFACT_ROOT` for
this first pass, and later map the same artifact records to Mongo GridFS or path
references when the shared MongoDB design is finalized.
