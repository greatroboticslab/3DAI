# 4DAI <-> 3DAI Integration Design

Status: implemented bridge, updated for dynamic 4DAI categories. Last updated
2026-07-20.

This document describes how the 4DAI data-collection system
(`greatroboticslab/4DAI`, Yarely Torres) and this 3DAI scanner system are
joined into a single per-sample record: dynamic context + 2D imagery + optional
moisture from 4DAI, plus 3D scan artifacts from 3DAI.

It supersedes the "later MongoDB integration" and "if that project adds
MongoDB" language in `scan_3d_schema.md` and `torres_integration_example.md`,
which were written before the 4DAI repo was clarified. 4DAI already uses
MongoDB; the concrete binding is specified here.

2026-07-20 confirmation from Torres:
- 4DAI FastAPI creates the UUID `sample_id` at `POST /collection/submission`.
- Streamlit does not create an id; it receives `sample_id` from the server and
  passes it around for image and metadata work.
- MongoDB runs locally on the Remote Desktop named `lab` at
  `mongodb://localhost:27017`, database `Collections`.
- Each dynamic category has its own collection, with `sample_id` as the Mongo
  document `_id`.
- Category context fields come from `Server/settings/{category}.json` and must
  be treated as dynamic.
- Moisture can be a form field or per-image metadata depending on the category
  JSON.
- File paths are preferred over GridFS. Images are on server disk and metadata
  records carry path strings.
- The category JSON may expose a Kinect/camera toggle, but 4DAI does not read
  or enforce that setting yet.

## Systems at a glance

| | 4DAI (Torres) | 3DAI (this repo) |
|---|---|---|
| Role | context, 2D images, manual moisture | 3D scan sessions and artifacts |
| Stack | Streamlit UI + FastAPI + MongoDB | FastAPI + PostgreSQL |
| Database | Mongo db `Collections` | Postgres db `capture` (localhost:55432) |
| Collections / tables | dynamic category collections plus image metadata | `sessions`, `steps`, `images`, `artifacts` |
| Sample key | category record `_id` / returned `sample_id` (UUID string) | `sessions.sample_id` (TEXT, nullable, indexed) |
| Image bytes | server-disk paths tied to metadata records | files under `ARTIFACT_ROOT`, indexed in `artifacts` |
| Network exposure | ngrok tunnel, no auth | `API_TOKEN` required before binding off-localhost |

4DAI source of truth for the dynamic category model:
`Server/main.py`, `Server/db.py`, and `Server/settings/{category}.json` in
`greatroboticslab/4DAI`.

## The join key

Decision: the canonical `sample_id` is the **4DAI FastAPI-generated UUID**.

4DAI mints this UUID string when Streamlit submits to
`POST /collection/submission`. The returned `sample_id` is stored as the Mongo
document `_id` in the selected category collection. That exact string is what
3DAI stores in `sessions.sample_id`. No new ID scheme is introduced and no
Streamlit-side ID generation is required.

3DAI already treats `sample_id` as an opaque trimmed string
(`main.py:_clean_sample_id`) and indexes it
(`migrations/001_init.sql:idx_sessions_sample_id`), so no 3DAI schema change is
required to accept 4DAI ids.

3DAI treats `sample_id` as opaque. It does not infer category, moisture shape,
camera mode, or physical sample meaning from the UUID.

## End-to-end flow

```
                        canonical sample_id = 4DAI record _id
                                      |
 [4DAI Streamlit UI] --POST /collection/submission--> [4DAI FastAPI]
                                      |
                                      v
                         insert into Mongo category collection
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
4DAI response `sample_id` as `sample_id`:

```jsonc
POST /sessions
{
  "total_steps": 0,            // 0 = metadata-only / register already-captured files
  "sample_id": "<4DAI FastAPI sample_id>",
  "metadata": { "category": "<4DAI category>", "operator": "torres", "origin": "4dai" }
}
```

Who calls this is a workflow choice; it does not change the contract:
- Manual: operator copies the returned `sample_id` from 4DAI and runs
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
1. Resolve the 4DAI side from Mongo: look up the record by `_id` in dynamic
   category collections. If `FOURDAI_CATEGORY_COLLECTIONS` is configured, those
   collection names are searched. Otherwise, list Mongo collections and skip
   obvious non-sample collections such as `images` and `system.*`.
   Image metadata is read from `FOURDAI_IMAGE_COLLECTIONS`, defaulting to
   `images`, when that collection exists. Large image bytes stay on disk.
2. Resolve the 3DAI side by reusing the existing
   `GET /samples/{sample_id}/scan-3d` logic (sessions + grouped artifacts).
3. Return one document, e.g.:

```jsonc
{
  "sample_id": "<4DAI record _id>",
  "fourdai": {                    // whichever 4DAI category collection held the _id
    "kind": "<category>",
    "category": "<category>",
    "collection": "<category>",
    "record": { "...": "dynamic fields from Server/settings/<category>.json" },
    "image_ids": ["..."],
    "images": [{ "image_id": "...", "image_path": "..." }]
  },
  "soil_or_vegetable": { "...": "legacy alias for older clients" },
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
- `FOURDAI_CATEGORY_COLLECTIONS` optional comma-separated category collection
  allowlist. Use this if Mongo contains non-sample collections beyond `images`.
- `FOURDAI_IMAGE_COLLECTIONS` optional comma-separated image metadata
  collection list. Defaults to `images` for backward compatibility.

### Step 4 - Network and auth

- 3DAI: set a real `API_TOKEN` and keep the Postgres port bound to localhost
  before exposing the API. `main.py` fails fast if `API_BIND_ADDR` is non-local
  without a token.
- 4DAI: current confirmed host is the Remote Desktop named `lab`, with Mongo at
  `mongodb://localhost:27017` / db `Collections`. When running 3DAI on the same
  Windows host, prefer `mongodb://127.0.0.1:27017` if `localhost` introduces the
  known IPv6 stall. If Mongo is remote, lock it down to the scanner/API host.

## Implementation status

Step 3 is implemented in this repo:

- `fourdai_client.py`: lazy, optional Mongo client. No `pymongo` import at load
  time; `is_configured()`, `get_db_name()`, `sample_collection_names()`, and
  `fetch_sample()` all degrade to None / False when 4DAI is unconfigured,
  `pymongo` is missing, or Mongo is down. Dynamic category collections are
  supported, with legacy `vegetable` / `soil` fallback. `shape_record()` is a
  pure function for document shaping and preserves image path metadata.
- `GET /samples/{sample_id}/4d` in `main.py`: combines the Postgres scan data
  with the optional 4DAI record and reports `found.fourdai` / `found.threedai`.
  The canonical response key is `fourdai`; `soil_or_vegetable` remains as a
  backward-compatible alias for the first prototype.
- `tests/test_fourdai_client.py`: unit tests (no live Mongo, no `pymongo`
  needed) covering shaping, legacy vegetable/soil lookup, dynamic category
  lookup, configured category allowlists, missing sample, db-error degrade, and
  the unconfigured path. Wired into CI.
- `pymongo` listed as an optional (commented) requirement.

Verified locally: `main.py` imports and registers the `/samples/{id}/4d` route
with `pymongo` absent, and the fusion endpoint returns the 3D side with
`found.fourdai=false` when 4DAI is not configured.

To enable the 4DAI side at runtime:

```powershell
$env:FOURDAI_MONGO_URL = "mongodb://127.0.0.1:27017"   # 127.0.0.1, not localhost (see gotchas)
$env:FOURDAI_DB_NAME   = "Collections"   # optional, this is the default
$env:FOURDAI_CATEGORY_COLLECTIONS = "soil,vegetable,<other-category>"   # optional allowlist
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
- Whether 3DAI should also write a compact `scan_3d` summary back into the 4DAI
  category document after artifact registration, or keep Postgres as the scan
  artifact index and expose joins through `GET /samples/{sample_id}/4d`.
- An end-to-end fusion smoke test against the updated dynamic 4DAI repo once
  that repo is available locally or on this machine.

## Resolved 4DAI questions

- `sample_id` is generated by 4DAI FastAPI, not Streamlit.
- Mongo is already stood up locally on the `lab` Remote Desktop under database
  `Collections`.
- Context fields and moisture placement are dynamic per
  `Server/settings/{category}.json`.
- File paths are preferred over GridFS.
- The 4DAI Kinect/camera toggle is configuration only right now; scanner code
  should not assume it is enforced by 4DAI.

## Safety

Everything in this design is read-only indexing and querying. No endpoint here
starts Kinect capture, projector output, relay/ESP32 code, serial ports, or
laser control. Running an actual 3D scan to produce artifacts remains a
separate, explicitly approved step.
