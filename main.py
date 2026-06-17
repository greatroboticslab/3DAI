import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from db import get_conn

app = FastAPI()


# ─────────────────────────── Request models ────────────────────────────

class CreateSessionPayload(BaseModel):
    total_steps: Optional[int] = 10


class ReportImagePayload(BaseModel):
    session_id: str
    step_id: str
    file_path: str


# ─────────────────────────── Endpoints ─────────────────────────────────

@app.post("/sessions")
def create_session(payload: CreateSessionPayload = None):
    """
    Create a new capture session and pre-populate its steps.

    Body (optional JSON):
        total_steps: int  – number of capture steps (default 10)
    """
    if payload is None:
        payload = CreateSessionPayload()

    session_id = str(uuid.uuid4())
    total_steps = payload.total_steps

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sessions VALUES (%s, %s, %s, %s)",
                (session_id, "running", total_steps, 0)
            )
            for i in range(total_steps):
                step_id = str(uuid.uuid4())
                cur.execute(
                    "INSERT INTO steps VALUES (%s, %s, %s, %s)",
                    (step_id, session_id, i, "pending")
                )
        conn.commit()

    return {"session_id": session_id, "total_steps": total_steps}


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    """Return the current status and progress of a session."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT status, total_steps, completed_steps FROM sessions WHERE id=%s",
                (session_id,)
            )
            row = cur.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "status": row[0],
        "total_steps": row[1],
        "completed_steps": row[2],
    }


@app.get("/sessions/{session_id}/steps")
def get_steps(session_id: str):
    """
    Return all steps for a session in index order.

    The capture tool calls this to get the real step UUIDs that were
    created by POST /sessions, so it can report back with matching IDs.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Verify session exists first
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
                (session_id,)
            )
            rows = cur.fetchall()

    return [
        {"step_id": row[0], "step_index": row[1], "status": row[2]}
        for row in rows
    ]


@app.post("/sessions/{session_id}/claim-step")
def claim_step(session_id: str):
    """
    Atomically claim the next available step for a session.

    Returns the lowest-index step that is still pending and unclaimed, marking
    it claimed so no other worker can take it. This lets multiple workers pull
    from the same session concurrently without grabbing the same step.

    Response:
        {"step_id": ..., "step_index": ..., "status": ...}  when a step is claimed
        {"step_id": None}                                   when none remain

    The SELECT ... FOR UPDATE SKIP LOCKED makes the claim safe under concurrency:
    each concurrent caller locks and takes a different row instead of blocking or
    double-claiming.
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
                (session_id,)
            )
            row = cur.fetchone()
        conn.commit()

    if row is None:
        # Session exists but has no pending, unclaimed steps left to hand out.
        return {"step_id": None}

    return {"step_id": row[0], "step_index": row[1], "status": row[2]}


@app.post("/report-image")
def report_image(payload: ReportImagePayload):
    """
    Worker calls this after capturing (real or fake) an image for a step.

    Marks the step as done, records the image path, increments the session
    progress counter, and flips the session to 'complete' once all steps
    are finished.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO images VALUES (%s, %s, %s, %s)",
                (str(uuid.uuid4()), payload.session_id, payload.step_id, payload.file_path)
            )
            cur.execute(
                "UPDATE steps SET status='done' WHERE id=%s",
                (payload.step_id,)
            )
            # Increment counter and auto-complete the session when all steps are done
            cur.execute(
                """
                UPDATE sessions
                SET
                    completed_steps = completed_steps + 1,
                    status = CASE
                        WHEN completed_steps + 1 >= total_steps THEN 'complete'
                        ELSE status
                    END
                WHERE id = %s
                """,
                (payload.session_id,)
            )
        conn.commit()

    return {"ok": True}