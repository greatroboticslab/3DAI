# 4DAI <-> 3DAI Integration Design

Status: design. Last updated 2026-06-26.

This document describes how the 4DAI data-collection system
(`greatroboticslab/4DAI`, Yarely Torres) and this 3DAI scanner system are
joined into a single per-sample record: context + 2D imagery + moisture from
4DAI, plus 3D scan artifacts from 3DAI.

It supersedes the "later MongoDB integration" and "if that project adds
MongoDB" language in `scan_3d_schema.md` and `torres_integration_example.md`,
which were written before the 4DAI repo was reviewed. 4DAI already uses
MongoDB; the concrete binding is specified here.

## Systems at a glance

| | 4DAI (Torres) | 3DAI (this repo) |
|---|---|---|
| Role | context, 2D images, manual moisture | 3D scan sessions and artifacts |
| Stack | Streamlit UI + FastAPI + MongoDB | FastAPI + PostgreSQL |
| Database | Mongo db `Collections` | Postgres db `capture` (localhost:55432) |
| Collections / tables | `vegetable`, `soil`, `images` | `sessions`, `steps`, `images`, `artifacts` |
| Sample key | record `_id` (UUID string) | `sessions.sample_id` (TEXT, nullable, indexed) |
| Image bytes | on disk under `images/{vegetables,soils}/{sample_id}/`, indexed in Mongo | files under `ARTIFACT_ROOT`, indexed in `artifacts` |
| Network exposure | ngrok tunnel, no auth | `API_TOKEN` required before binding off-localhost |

4DAI source of truth for the above: `Server/main.py`, `Server/db.py` in
`greatroboticslab/4DAI`.

## The join key

Decision: the canonical `sample_id` is the **4DAI record `_id`**.

4DAI already mints a UUID per sample in `create_vegetable` and
`create_soil_sample` (`str(uuid.uuid4())` stored as `_id`). That exact string
is what 3DAI stores in `sessions.sample_id`. No new ID scheme is introduced and
no change to 4DAI's ID generation is required.

3DAI already treats `sample_id` as an opaque trimmed string
(`main.py:_clean_sample_id`) and indexes it
(`migrations/001_init.sql:idx_sessions_sample_id`), so no 3DAI schema change is
required to accept 4DAI ids.

Note on granularity: 4DAI mints a *separate* `_id` for each vegetable record
and each soil record. A "sample" in the field-science sense (one physical
specimen with both a vegetable image and a soil reading) is therefore not yet a
single id in 4DAI. For now, one 4DAI record `_id` == one `sample_id` == one set
of 3D scans. Unifying a vegetable record and a soil record under one specimen id
is a separate 4DAI-side change and is out of scope here.

## End-to-end flow

```
                        canonical sample_id = 4DAI record _id
                                      |
 [4DAI Streamlit UI] --POST /soil--> [4DAI FastAPI] --insert--> (Mongo: soil/vegetable/images)
        |                                   |
        |  returns sample_id               |
        v                                   |
 operator triggers a 3D scan on the scanner PC (gated, hardware approval)
        |
        v
 [3DAI] POST /sessions {sample_id, total_steps:0}  -> session_id
 scan runs on this PC, outputs land under ARTIFACT_ROOT
 [3DAI] POST /sessions/{session_id}/artifacts      -> indexes kinect/fringe/fusion/calibration files
        |
        v
 [Fusion read] GET /samples/{sample_id}/4d  (NEW, this repo)
   -> Mongo lookup (4DAI context/2D/moisture) + Postgres lookup (3DAI scans)
   -> one combined JSON record for the sample
```

3DAI never starts scanner hardware as a side effect of any of these endpoints.
Registration indexes files that already exist. Running an actual scan remains a
separate, explicitly approved manual step per `AGENTS.md` and
`.claude/PROJECT_MEMORY.md`.

## Integration steps

### Step 1 - Pass the 4DAI id into 3DAI

When a 3D scan is wanted for a 4DAI sample, create the 3DAI session with the
4DAI record `_id` as `sample_id`:

```jsonc
POST /sessions
{
  "total_steps": 0,            // 0 = metadata-only / register already-captured files
  "sample_id": "<4DAI record _id>",
  "metadata": { "mode": "soil", "operator": "torres", "origin": "4dai" }
}
```

Who calls this is a workflow choice; it does not change the contract:
- Manual: operator copies the `_id` from 4DAI and runs
  `scripts/register_scan_folder.py --sample-id <_id> --folder <scan folder>`,
  which already wraps session creation + artifact registration.
- Automated later: 4DAI's collection page POSTs to 3DAI right after it gets its
  `sample_id`. Needs the scanner API reachable from the 4DAI machine and a
  shared `API_TOKEN` (see Step 4).

### Step 2 - Register scan artifacts (existing)

Already implemented: `POST /sessions/{session_id}/artifacts` with categories
`kinect | fringe | fusion | calibration`. See `scan_3d_schema.md` for the
artifact body and response shape. No change needed.

### Step 3 - Add the 4D fusion read (NEW, this repo)

Add a single read endpoint to `main.py` that combines both databases for one
sample:

`GET /samples/{sample_id}/4d`

Behavior:
1. Resolve the 4DAI side from Mongo: look up the record by `_id` in `vegetable`
   then `soil`, and its images in `images` (by `sample_id`). Expose 4DAI image
   bytes as 4DAI download URLs, not embedded blobs.
2. Resolve the 3DAI side by reusing the existing
   `GET /samples/{sample_id}/scan-3d` logic (sessions + grouped artifacts).
3. Return one document, e.g.:

```jsonc
{
  "sample_id": "<4DAI record _id>",
  "soil_or_vegetable": {          // whichever 4DAI collection held the _id
    "kind": "soil",
    "record": { "soil_type": "...", "soil_moisture": "7", "date": "...", "notes": "..." },
    "image_ids": ["..."]          // resolvable via the 4DAI API
  },
  "scan_3d": { "sessions": [ /* same shape as /samples/{id}/scan-3d */ ] },
  "found": { "fourdai": true, "threedai": true }
}
```

Design constraints for this endpoint:
- Read-only. No hardware, no writes to either DB.
- Mongo access must be lazy and optional. 3DAI's API must still start and serve
  its own endpoints if `pymongo` is absent or Mongo/ngrok is down. Degrade to
  `"found": { "fourdai": false }` rather than failing the whole request.
- Keep it behind the existing `require_api_token` dependency.
- Do not import camera/Kinect modules; this is pure DB I/O (respect the
  import-time safety rules in `AGENTS.md`).

Configuration (new env vars, all optional):
- `FOURDAI_MONGO_URL` (e.g. `mongodb://<4dai-host>:27017`) - if unset, the 4D
  endpoint reports the 4DAI side as not configured and still returns the 3D
  side.
- `FOURDAI_DB_NAME` default `Collections`.

### Step 4 - Network and auth

- 3DAI: set a real `API_TOKEN` and keep the Postgres port bound to localhost
  before exposing the API. `main.py` fails fast if `API_BIND_ADDR` is non-local
  without a token.
- 4DAI: backend is currently exposed via ngrok with no auth. If 3DAI's fusion
  endpoint reads 4DAI's Mongo directly (preferred), it connects to Mongo over
  the network/VPN rather than through ngrok; lock Mongo down to the scanner
  host. If instead it calls 4DAI's HTTP API, account for the rotating ngrok URL
  (4DAI hardcodes it in `UI/key.py`) by making `FOURDAI_*` configurable.

## Implementation status

Step 3 is implemented in this repo:

- `fourdai_client.py`: lazy, optional Mongo client. No `pymongo` import at load
  time; `is_configured()`, `get_db_name()`, and `fetch_sample()` all degrade to
  None / False when 4DAI is unconfigured, `pymongo` is missing, or Mongo is
  down. `shape_record()` is a pure function for the document shaping.
- `GET /samples/{sample_id}/4d` in `main.py`: combines the Postgres scan data
  with the optional 4DAI record and reports `found.fourdai` / `found.threedai`.
- `tests/test_fourdai_client.py`: unit tests (no live Mongo, no `pymongo`
  needed) covering shaping, vegetable/soil lookup, missing sample, db-error
  degrade, and the unconfigured path. Wired into CI.
- `pymongo` listed as an optional (commented) requirement.

Verified locally: `main.py` imports and registers the `/samples/{id}/4d` route
with `pymongo` absent, and the fusion endpoint returns the 3D side with
`found.fourdai=false` when 4DAI is not configured.

To enable the 4DAI side at runtime:

```powershell
$env:FOURDAI_MONGO_URL = "mongodb://127.0.0.1:27017"   # 127.0.0.1, not localhost (see gotchas)
$env:FOURDAI_DB_NAME   = "Collections"   # optional, this is the default
.\.venv\Scripts\python.exe -m pip install "pymongo>=4.6,<5.0"
```

## End-to-end verification (2026-06-26)

The full four-layer chain was run on real data on the scanner PC:

1. Seeded a real 4DAI soil record in MongoDB (canonical `sample_id` =
   the 4DAI record `_id`).
2. Ran a live structured-light scan (projector fringes [1,6,24] x 8 phases +
   Kinect color; no laser). Reconstructed at ~89% reliable; metric height map
   read 5.64 mm vs a 6.10 mm expected target (-0.46 mm / -7.5%), in line with
   prior white-flat 6 mm scans.
3. Registered the scan folder under the same `sample_id` via
   `scripts/register_scan_folder.py` (2 fringe + 4 fusion artifacts).
4. `GET /samples/{sample_id}/4d` returned both halves joined:
   `found = {fourdai: true, threedai: true}`, with the soil record (moisture,
   type, notes) from Mongo and the scan artifacts from Postgres.

The scratch harness used (seed, runbook, final check) is in `scratch_e2e/`.

### Operational gotchas found during the run

- MongoDB 8.x does not run on this PC's Windows 10 build 19045: `mongod.exe`
  fails immediately with `STATUS_ENTRYPOINT_NOT_FOUND` (0xC0000139), even after
  a reboot, because the 8.x build links against Windows APIs this build does
  not export. Use MongoDB 7.x (native MSI) or run Mongo in Docker
  (`docker run -d -p 127.0.0.1:27017:27017 --name 4dai-mongo mongo:7`).
- Use `127.0.0.1`, never `localhost`, in `DATABASE_URL` and `FOURDAI_MONGO_URL`.
  Docker binds these ports IPv4-only; `localhost` resolving to IPv6 `::1` first
  adds a ~5-8 s connect stall per request on Windows before falling back. The
  code defaults now use `127.0.0.1`.
- For the fusion endpoint to reach a host Mongo, run the 3DAI API on the host
  (not in its container), since the API container cannot reach host
  `127.0.0.1`. A containerized API would need `host.docker.internal` or a shared
  Docker network with the Mongo container.

## What already exists vs what is new

Already implemented in this repo (no work needed):
- `sample_id` accepted, validated, indexed.
- `POST /sessions`, `POST /sessions/{id}/artifacts`.
- `GET /samples/{sample_id}/scan-3d`, `GET /sessions/{id}/scan-3d`,
  `GET /artifacts/{id}`.
- `scripts/register_scan_folder.py`, `examples/torres_fetch_scan_3d.py`.
- Path-traversal and content guards in `artifact_store.py`; token auth.

Implemented for this design (see Implementation status above):
1. `GET /samples/{sample_id}/4d` in `main.py` (Step 3).
2. `fourdai_client.py`, a lazy Mongo client reading `FOURDAI_MONGO_URL` /
   `FOURDAI_DB_NAME` with an optional `pymongo` import.
3. `pymongo` listed as an optional (commented) requirement.
4. `tests/test_fourdai_client.py`, wired into CI.

Still open (workflow, not in this repo):
- Whether 4DAI's collection flow auto-creates the 3DAI session (Step 1
  automation), versus the manual `register_scan_folder.py` path used today.
- An end-to-end fusion smoke test against a live Mongo, deferred until a 4DAI
  Mongo is reachable from CI or the scanner PC.

## Open questions for Torres

- Should a vegetable record and a soil record for the same physical specimen
  share one id in 4DAI? If yes, that is the cleaner long-term `sample_id` and
  changes Step 1's granularity.
- Mongo reachable directly from the scanner PC, or only via the 4DAI HTTP API?
  This decides whether Step 3 talks to Mongo or to 4DAI's REST endpoints.
- Should `soil_moisture` stay a manual 1-10 dropdown, or become a sensor reading
  (relevant to the "moisture" layer of the 4D goal)?

## Safety

Everything in this design is read-only indexing and querying. No endpoint here
starts Kinect capture, projector output, relay/ESP32 code, serial ports, or
laser control. Running an actual 3D scan to produce artifacts remains a
separate, explicitly approved step.
