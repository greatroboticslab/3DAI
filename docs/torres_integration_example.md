# Torres Integration Example

This first pass keeps the scanner and 4DAI projects loosely coupled. 4DAI
creates the canonical `sample_id`, while the scanner API exposes 3D scan
products as JSON plus artifact download URLs.

## Shared Join Key

Use the `sample_id` returned by 4DAI FastAPI `POST /collection/submission` as
the scanner `sample_id`.

Example:

```text
4DAI sample_id:   0f54282b-785b-452f-b3b8-dc2d9905779b
Scanner sample_id: 0f54282b-785b-452f-b3b8-dc2d9905779b
```

The scanner API treats this as an opaque string. The 3D artifact registration
path does not require MongoDB; the optional `/samples/{sample_id}/4d` fusion
endpoint can read 4DAI MongoDB when configured.

## Scanner-Side Registration

After scan files already exist under `ARTIFACT_ROOT`, register that folder:

```powershell
$env:API_URL = "http://localhost:8000"
$env:ARTIFACT_ROOT = "C:\Users\Robotics_Lab\3DAI\3DAI\data"
.\.venv\Scripts\python.exe scripts\register_scan_folder.py `
  --sample-id 0f54282b-785b-452f-b3b8-dc2d9905779b `
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
python examples\torres_fetch_scan_3d.py 0f54282b-785b-452f-b3b8-dc2d9905779b --out scan_3d.json
```

The JSON response contains session metadata and grouped artifact records. Large
files stay as URLs such as `/artifacts/{artifact_id}`.

## Minimal GUI Hook

4DAI only needs to save the scan package next to its existing sample outputs:

```python
from examples.torres_fetch_scan_3d import fetch_scan_3d

scan_3d = fetch_scan_3d("http://scanner-pc-ip:8000", sample_id)
```

`sample_id` is the join key. 4DAI already stores image and context metadata in
MongoDB with file path references, so the returned JSON can be stored as a
sidecar or joined through `GET /samples/{sample_id}/4d`.

## Safety Boundaries

These endpoints and scripts only expose existing files. They do not start
Kinect capture, projector output, relay control, ESP32 code, serial ports, or
laser-control paths.
