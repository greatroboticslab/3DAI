import time
import uuid
import requests
import os

API = "http://localhost:8000"
IMAGE_ROOT = "./data/images"


def fake_capture(step_index):
    time.sleep(1)

    os.makedirs(IMAGE_ROOT, exist_ok=True)

    path = f"{IMAGE_ROOT}/img_{step_index}.jpg"

    with open(path, "w") as f:
        f.write("fake image")

    return path


def get_steps():
    # MVP: just simulate steps locally
    return [
        {"step_id": str(uuid.uuid4()), "session_id": CURRENT_SESSION, "index": i}
        for i in range(10)
    ]


def run_session(session_id):
    global CURRENT_SESSION
    CURRENT_SESSION = session_id

    steps = get_steps()

    for step in steps:
        path = fake_capture(step["index"])

        requests.post(f"{API}/report-image", json={
            "session_id": session_id,
            "step_id": step["step_id"],
            "file_path": path
        })


if __name__ == "__main__":
    res = requests.post(f"{API}/sessions")
    session_id = res.json()["session_id"]

    run_session(session_id)