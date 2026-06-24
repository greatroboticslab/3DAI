import json
import os
import hmac
import uuid
from typing import Any, Literal, Optional

from fastapi import Body, Depends, FastAPI, HTTPException, Security
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from psycopg.types.json import Json
from pydantic import BaseModel, ConfigDict, Field

from artifact_store import (
    ArtifactError,
    inspect_artifact,
    resolve_artifact_path,
    validate_category,
    validate_role,
)
from db import get_conn

app = FastAPI()

MAX_SESSION_STEPS = int(os.getenv("MAX_SESSION_STEPS", "1000"))
MAX_ARTIFACTS_PER_REQUEST = int(os.getenv("MAX_ARTIFACTS_PER_REQUEST", "100"))
MAX_METADATA_BYTES = int(os.getenv("MAX_METADATA_BYTES", "65536"))
API_TOKEN = os.getenv("API_TOKEN", "").strip()
API_TOKEN_HEADER = os.getenv("API_TOKEN_HEADER", "X-API-Token").strip() or "X-API-Token"
API_BIND_ADDR = os.getenv("API_BIND_ADDR", "127.0.0.1").strip() or "127.0.0.1"

if API_TOKEN.lower() == "string":
    raise RuntimeError("API_TOKEN looks like placeholder text; set a real token or leave it empty for localhost-only dev.")

if not API_TOKEN and API_BIND_ADDR.lower() not in {"127.0.0.1", "localhost", "::1", "[::1]"}:
    raise RuntimeError(
        "API_TOKEN is required before exposing the API outside localhost. "
        "Set API_TOKEN or keep API_BIND_ADDR=127.0.0.1."
    )

api_token_header = APIKeyHeader(name=API_TOKEN_HEADER, auto_error=False)


def require_api_token(token: Optional[str] = Security(api_token_header)) -> None:
    if not API_TOKEN:
        return
    if not token or not hmac.compare_digest(token, API_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid or missing API token")


class CreateSessionPayload(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_steps": 0,
                "sample_id": "plant-2026-06-23-001",
                "metadata": {
                    "operator": "torres",
                    "notes": "matte sample on reference panel",
                },
            }
        }
    )

    total_steps: Optional[int] = Field(
        default=10,
        description="Number of capture steps to create. Use 0 for metadata-only artifact registration.",
        examples=[0],
    )
    sample_id: Optional[str] = Field(
        default=None,
        description="Opaque sample join key created by Torres's GUI.",
        examples=["plant-2026-06-23-001"],
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form JSON scan/session context.",
        examples=[{"operator": "torres", "notes": "matte sample on reference panel"}],
    )


class ReportImagePayload(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "590d8135-4a7e-48d0-ae35-14211189ed5f",
                "step_id": "0f54282b-785b-452f-b3b8-dc2d9905779b",
                "file_path": "C:/Users/Robotics_Lab/3DAI/3DAI/data/images/session/img_000.png",
            }
        }
    )

    session_id: uuid.UUID = Field(examples=["590d8135-4a7e-48d0-ae35-14211189ed5f"])
    step_id: uuid.UUID = Field(examples=["0f54282b-785b-452f-b3b8-dc2d9905779b"])
    file_path: str = Field(examples=["C:/Users/Robotics_Lab/3DAI/3DAI/data/images/session/img_000.png"])


class ArtifactRegistration(BaseModel):
    category: Literal["kinect", "fringe", "fusion", "calibration"] = Field(
        description="Artifact group: kinect, fringe, fusion, or calibration.",
        examples=["fringe"],
    )
    role: str = Field(
        description="Artifact role within its category.",
        examples=["fringe_stack_npz"],
    )
    file_path: str = Field(
        description="Path under ARTIFACT_ROOT.",
        examples=["scan_test/calib/breadboard/scan.npz"],
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional artifact-specific JSON metadata.",
        examples=[{"source": "capture_multifreq.py", "maxval": 100}],
    )


class RegisterArtifactsPayload(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "artifacts": [
                    {
                        "category": "kinect",
                        "role": "color_png",
                        "file_path": "images/session-id/img_000.png",
                        "metadata": {"source": "capture_tool.py --real"},
                    },
                    {
                        "category": "fringe",
                        "role": "fringe_stack_npz",
                        "file_path": "scan_test/calib/breadboard/scan.npz",
                    },
                    {
                        "category": "fusion",
                        "role": "height_map_png",
                        "file_path": "scan_test/calib/breadboard/height_mm.png",
                    },
                ]
            }
        }
    )

    artifacts: list[ArtifactRegistration] = Field(
        description="Existing files under ARTIFACT_ROOT to index for this session."
    )


@app.get("/health")
def health():
    return {"status": "ok", "auth_required": bool(API_TOKEN)}


def _clean_sample_id(sample_id: Optional[str]) -> Optional[str]:
    if sample_id is None:
        return None
    cleaned = sample_id.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="sample_id must be nonempty when provided")
    if cleaned.lower() == "string":
        raise HTTPException(
            status_code=400,
            detail="sample_id looks like Swagger placeholder text; provide a real sample id",
        )
    return cleaned


def _clean_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    value = metadata or {}
    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="metadata must be JSON serializable") from exc

    if len(encoded) > MAX_METADATA_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"metadata exceeds {MAX_METADATA_BYTES} bytes",
        )

    return value


def _session_dict(row) -> dict[str, Any]:
    return {
        "session_id": str(row[0]),
        "sample_id": row[1],
        "status": row[2],
        "total_steps": row[3],
        "completed_steps": row[4],
        "metadata": row[5] or {},
        "created_at": row[6].isoformat() if row[6] else None,
    }


def _artifact_dict(row) -> dict[str, Any]:
    artifact_id = str(row[0])
    return {
        "artifact_id": artifact_id,
        "session_id": str(row[1]),
        "category": row[2],
        "role": row[3],
        "file_path": row[4],
        "media_type": row[5],
        "size_bytes": row[6],
        "array_metadata": row[7] or {},
        "metadata": row[8] or {},
        "created_at": row[9].isoformat() if row[9] else None,
        "download_url": f"/artifacts/{artifact_id}",
    }


def _group_artifacts(artifacts: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = {
        "kinect": [],
        "fringe": [],
        "fusion": [],
        "calibration": [],
    }
    for artifact in artifacts:
        grouped.setdefault(artifact["category"], []).append(artifact)
    return grouped


def _fetch_session(cur, session_id: uuid.UUID):
    cur.execute(
        """
        SELECT id, sample_id, status, total_steps, completed_steps, metadata, created_at
        FROM sessions
        WHERE id=%s
        """,
        (session_id,),
    )
    return cur.fetchone()


def _fetch_artifacts(cur, session_id: uuid.UUID) -> list[dict[str, Any]]:
    cur.execute(
        """
        SELECT
            id, session_id, category, role, file_path, media_type, size_bytes,
            array_metadata, metadata, created_at
        FROM artifacts
        WHERE session_id=%s
        ORDER BY category, role, created_at, id
        """,
        (session_id,),
    )
    return [_artifact_dict(row) for row in cur.fetchall()]


def _scan_package_from_row(cur, session_row) -> dict[str, Any]:
    session = _session_dict(session_row)
    artifacts = _fetch_artifacts(cur, session_row[0])
    return {
        **session,
        "artifacts": _group_artifacts(artifacts),
    }


@app.post("/sessions", dependencies=[Depends(require_api_token)])
def create_session(
    payload: Optional[CreateSessionPayload] = Body(
        default=None,
        openapi_examples={
            "metadata_only": {
                "summary": "Create a sample-linked scan session",
                "value": {
                    "total_steps": 0,
                    "sample_id": "plant-2026-06-23-001",
                    "metadata": {
                        "operator": "torres",
                        "notes": "matte sample on reference panel",
                    },
                },
            },
            "fake_capture": {
                "summary": "Create a normal fake/worker capture session",
                "value": {
                    "total_steps": 3,
                    "sample_id": "demo-sample-001",
                    "metadata": {"mode": "fake_capture_smoke"},
                },
            },
        },
    )
):
    """
    Create a new capture session and pre-populate its steps.

    Body (optional JSON):
        total_steps: int
        sample_id: optional opaque join key from Torres's GUI
        metadata: optional JSON object for scan/session context
    """
    if payload is None:
        payload = CreateSessionPayload()

    session_id = uuid.uuid4()
    total_steps = 10 if payload.total_steps is None else payload.total_steps
    if total_steps < 0:
        raise HTTPException(status_code=400, detail="total_steps must be >= 0")
    if total_steps > MAX_SESSION_STEPS:
        raise HTTPException(
            status_code=400,
            detail=f"total_steps must be <= {MAX_SESSION_STEPS}",
        )

    sample_id = _clean_sample_id(payload.sample_id)
    metadata = _clean_metadata(payload.metadata)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO sessions (
                    id, status, total_steps, completed_steps, sample_id, metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (session_id, "running", total_steps, 0, sample_id, Json(metadata)),
            )
            for i in range(total_steps):
                step_id = uuid.uuid4()
                cur.execute(
                    "INSERT INTO steps (id, session_id, step_index, status) "
                    "VALUES (%s, %s, %s, %s)",
                    (step_id, session_id, i, "pending"),
                )
        conn.commit()

    return {
        "session_id": str(session_id),
        "total_steps": total_steps,
        "sample_id": sample_id,
        "metadata": metadata,
    }


@app.get("/sessions/{session_id}", dependencies=[Depends(require_api_token)])
def get_session(session_id: uuid.UUID):
    """Return the current status, progress, sample id, and scan metadata."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            row = _fetch_session(cur, session_id)

    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return _session_dict(row)


@app.get("/sessions/{session_id}/steps", dependencies=[Depends(require_api_token)])
def get_steps(session_id: uuid.UUID):
    """
    Return all steps for a session in index order.

    The capture tool calls this to get the real step UUIDs that were
    created by POST /sessions, so it can report back with matching IDs.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM sessions WHERE id=%s", (session_id,))
            if cur.fetchone() is None:
                raise HTTPException(status_code=404, detail="Session not found")

            cur.execute(
                """
                SELECT id, step_index, status
                FROM steps
                WHERE session_id = %s
                ORDER BY step_index
                """,
                (session_id,),
            )
            rows = cur.fetchall()

    return [
        {"step_id": str(row[0]), "step_index": row[1], "status": row[2]}
        for row in rows
    ]


@app.post("/sessions/{session_id}/claim-step", dependencies=[Depends(require_api_token)])
def claim_step(session_id: uuid.UUID):
    """
    Atomically claim the next available step for a session.

    Returns the lowest-index step that is still pending and unclaimed, marking
    it claimed so no other worker can take it.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM sessions WHERE id=%s", (session_id,))
            if cur.fetchone() is None:
                raise HTTPException(status_code=404, detail="Session not found")

            cur.execute(
                """
                UPDATE steps
                SET claimed = TRUE
                WHERE id = (
                    SELECT id
                    FROM steps
                    WHERE session_id = %s AND status = 'pending' AND claimed = FALSE
                    ORDER BY step_index
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                RETURNING id, step_index, status
                """,
                (session_id,),
            )
            row = cur.fetchone()
        conn.commit()

    if row is None:
        return {"step_id": None}

    return {"step_id": str(row[0]), "step_index": row[1], "status": row[2]}


@app.post("/report-image", dependencies=[Depends(require_api_token)])
def report_image(
    payload: ReportImagePayload = Body(
        openapi_examples={
            "report_worker_image": {
                "summary": "Report one captured step image",
                "value": {
                    "session_id": "590d8135-4a7e-48d0-ae35-14211189ed5f",
                    "step_id": "0f54282b-785b-452f-b3b8-dc2d9905779b",
                    "file_path": "C:/Users/Robotics_Lab/3DAI/3DAI/data/images/session/img_000.png",
                },
            }
        }
    )
):
    """
    Worker calls this after capturing (real or fake) an image for a step.

    The step must belong to the given session. Marks the step done, records
    the image path, and recomputes session progress from the actual number
    of completed steps.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE steps SET status='done' "
                "WHERE id=%s AND session_id=%s AND status<>'done'",
                (payload.step_id, payload.session_id),
            )

            if cur.rowcount == 0:
                cur.execute(
                    "SELECT 1 FROM steps WHERE id=%s AND session_id=%s",
                    (payload.step_id, payload.session_id),
                )
                if cur.fetchone() is None:
                    raise HTTPException(
                        status_code=404,
                        detail="Step not found for this session",
                    )
                return {"ok": True, "duplicate": True}

            cur.execute(
                "INSERT INTO images (id, session_id, step_id, file_path) "
                "VALUES (%s, %s, %s, %s)",
                (uuid.uuid4(), payload.session_id, payload.step_id, payload.file_path),
            )
            cur.execute(
                """
                UPDATE sessions
                SET
                    completed_steps = sub.done,
                    status = CASE WHEN sub.done >= total_steps THEN 'complete' ELSE status END
                FROM (
                    SELECT count(*) AS done
                    FROM steps
                    WHERE session_id = %s AND status = 'done'
                ) AS sub
                WHERE id = %s
                """,
                (payload.session_id, payload.session_id),
            )
        conn.commit()

    return {"ok": True}


@app.post("/sessions/{session_id}/artifacts", dependencies=[Depends(require_api_token)])
def register_artifacts(
    session_id: uuid.UUID,
    payload: RegisterArtifactsPayload = Body(
        openapi_examples={
            "register_scan_outputs": {
                "summary": "Register Kinect, fringe, and fusion files",
                "value": {
                    "artifacts": [
                        {
                            "category": "kinect",
                            "role": "color_png",
                            "file_path": "images/session-id/img_000.png",
                            "metadata": {"source": "capture_tool.py --real"},
                        },
                        {
                            "category": "fringe",
                            "role": "fringe_stack_npz",
                            "file_path": "scan_test/calib/breadboard/scan.npz",
                        },
                        {
                            "category": "fusion",
                            "role": "height_map_png",
                            "file_path": "scan_test/calib/breadboard/height_mm.png",
                        },
                    ]
                },
            }
        }
    ),
):
    """
    Register one or more existing read-only scan artifacts under ARTIFACT_ROOT.
    This endpoint indexes files only; it never starts scanner hardware.
    """
    if not payload.artifacts:
        raise HTTPException(status_code=400, detail="At least one artifact is required")
    if len(payload.artifacts) > MAX_ARTIFACTS_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot register more than {MAX_ARTIFACTS_PER_REQUEST} artifacts per request",
        )

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM sessions WHERE id=%s", (session_id,))
            if cur.fetchone() is None:
                raise HTTPException(status_code=404, detail="Session not found")

    inspected: list[dict[str, Any]] = []
    for artifact in payload.artifacts:
        try:
            category = validate_category(artifact.category)
            role = validate_role(artifact.role)
            file_info = inspect_artifact(artifact.file_path)
        except ArtifactError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        inspected.append({
            "category": category,
            "role": role,
            "file_path": file_info["file_path"],
            "media_type": file_info["media_type"],
            "size_bytes": file_info["size_bytes"],
            "array_metadata": file_info["array_metadata"],
            "metadata": _clean_metadata(artifact.metadata),
        })

    registered = []
    with get_conn() as conn:
        with conn.cursor() as cur:
            for artifact in inspected:
                cur.execute(
                    """
                    INSERT INTO artifacts (
                        id, session_id, category, role, file_path, media_type,
                        size_bytes, array_metadata, metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (session_id, category, role, file_path)
                    DO UPDATE SET
                        media_type = EXCLUDED.media_type,
                        size_bytes = EXCLUDED.size_bytes,
                        array_metadata = EXCLUDED.array_metadata,
                        metadata = EXCLUDED.metadata
                    RETURNING
                        id, session_id, category, role, file_path, media_type,
                        size_bytes, array_metadata, metadata, created_at
                    """,
                    (
                        uuid.uuid4(),
                        session_id,
                        artifact["category"],
                        artifact["role"],
                        artifact["file_path"],
                        artifact["media_type"],
                        artifact["size_bytes"],
                        Json(artifact["array_metadata"]),
                        Json(artifact["metadata"]),
                    ),
                )
                registered.append(_artifact_dict(cur.fetchone()))
        conn.commit()

    return {"registered": registered}


@app.get("/sessions/{session_id}/scan-3d", dependencies=[Depends(require_api_token)])
def get_session_scan_3d(session_id: uuid.UUID):
    """Return the scan_3d package for one session, grouped by artifact category."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            row = _fetch_session(cur, session_id)
            if row is None:
                raise HTTPException(status_code=404, detail="Session not found")
            return _scan_package_from_row(cur, row)


@app.get("/samples/{sample_id}/scan-3d", dependencies=[Depends(require_api_token)])
def get_sample_scan_3d(sample_id: str):
    """Return all 3D scan sessions for a Torres sample id, newest first."""
    cleaned = _clean_sample_id(sample_id)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, sample_id, status, total_steps, completed_steps, metadata, created_at
                FROM sessions
                WHERE sample_id=%s
                ORDER BY created_at DESC, id DESC
                """,
                (cleaned,),
            )
            sessions = [_scan_package_from_row(cur, row) for row in cur.fetchall()]

    return {"sample_id": cleaned, "sessions": sessions}


@app.get("/artifacts/{artifact_id}", dependencies=[Depends(require_api_token)])
def download_artifact(artifact_id: uuid.UUID):
    """Stream a registered artifact file with its stored media type."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT file_path, media_type FROM artifacts WHERE id=%s",
                (artifact_id,),
            )
            row = cur.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Artifact not found")

    try:
        path, _ = resolve_artifact_path(row[0])
    except ArtifactError as exc:
        raise HTTPException(status_code=404, detail="Artifact file is not available") from exc

    if not path.is_file() or path.stat().st_size <= 0:
        raise HTTPException(status_code=404, detail="Artifact file is not available")

    return FileResponse(str(path), media_type=row[1], filename=path.name)
