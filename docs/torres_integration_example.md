# Torres Integration Example

This first pass keeps the scanner and Torres projects loosely coupled. Torres's
GUI can keep using its own recording folder flow, while the scanner API exposes
3D scan products as JSON plus artifact download URLs.

## Shared Join Key

Use Torres's recording name as the scanner `sample_id`.

Example:

```text
Torres recording_name: basil_run_014
Scanner sample_id:    basil_run_014
```

The scanner API treats this as an opaque string. It does not need MongoDB, a
Torres database row, or a shared filesystem path.

## Scanner-Side Registration

After scan files already exist under `ARTIFACT_ROOT`, register that folder:

```powershell
$env:API_URL = "http://localhost:8000"
$env:ARTIFACT_ROOT = "C:\Users\Robotics_Lab\3DAI\3DAI\data"
.\.venv\Scripts\python.exe scripts\register_scan_folder.py `
  --sample-id basil_run_014 `
  --folder data\scan_test\calib\breadboard
```

If the API is exposed to another machine, set a real token first:

```powershell
$env:API_BIND_ADDR = "0.0.0.0"
$env:API_TOKEN = "replace-with-a-long-random-token"
docker compose up -d --build api
```

Keep the database port bound to localhost. Torres only needs the API port.

## Torres-Side Fetch

From Torres's machine or project environment:

```powershell
$env:SCANNER_API_URL = "http://scanner-pc-ip:8000"
$env:SCANNER_API_TOKEN = "replace-with-the-same-token"
python examples\torres_fetch_scan_3d.py basil_run_014 --out scan_3d.json
```

The JSON response contains session metadata and grouped artifact records. Large
files stay as URLs such as `/artifacts/{artifact_id}`.

## Minimal GUI Hook

Torres's GUI only needs to save the scan package next to its existing recording
outputs:

```python
from examples.torres_fetch_scan_3d import fetch_scan_3d

scan_3d = fetch_scan_3d("http://scanner-pc-ip:8000", recording_name)
```

`recording_name` is the join key. The returned JSON can be stored as a sidecar
file now and mapped to MongoDB later if that project adds MongoDB.

## Safety Boundaries

These endpoints and scripts only expose existing files. They do not start
Kinect capture, projector output, relay control, ESP32 code, serial ports, or
laser-control paths.
