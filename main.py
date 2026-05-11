import uuid
from fastapi import FastAPI
from db import get_conn

app = FastAPI()

@app.post("/sessions")
def create_session():
    session_id = str(uuid.uuid4())
    total_steps = 10

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

    return {"session_id": session_id}


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT status, total_steps, completed_steps FROM sessions WHERE id=%s",
                (session_id,)
            )
            row = cur.fetchone()

    return {
        "status": row[0],
        "total_steps": row[1],
        "completed_steps": row[2]
    }


@app.post("/report-image")
def report_image(payload: dict):
    """
    Worker calls this after capturing an image
    """
    session_id = payload["session_id"]
    step_id = payload["step_id"]
    file_path = payload["file_path"]

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO images VALUES (%s, %s, %s, %s)",
                (str(uuid.uuid4()), session_id, step_id, file_path)
            )

            cur.execute(
                "UPDATE steps SET status='done' WHERE id=%s",
                (step_id,)
            )

            cur.execute("""
                UPDATE sessions
                SET completed_steps = completed_steps + 1
                WHERE id=%s
            """, (session_id,))

        conn.commit()

    return {"ok": True}