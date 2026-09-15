"""Plain-English go/no-go check run before every scanning session.

Double-clicking scan.bat runs this first. It answers one question for the
person at the bench: "can I scan right now, and if not, what do I do?"
Every line is written for someone who has never opened this code.

It also does the one bit of setup that kept biting us: the database runs in
Docker, and Docker Desktop stops whenever the PC restarts. If the database is
unreachable this starts Docker Desktop and the ``scanner-mongo`` container
and waits for them, so nobody has to know what Docker is.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

DOCKER_DESKTOP = r"C:\Program Files\Docker\Docker\Docker Desktop.exe"
MONGO_CONTAINER = "scanner-mongo"


def _docker(*args, timeout=20):
    try:
        return subprocess.run(["docker", *args], capture_output=True, text=True,
                              timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None


def ensure_mongo(wait_s: int = 150) -> dict:
    """Make the scanner database reachable, starting Docker if needed."""
    from . import hardware, scanner_db
    st = hardware.probe_mongo()
    if st["ok"]:
        return st

    print("  database is down, starting it (this can take a minute)...")
    daemon = _docker("ps")
    if daemon is None or daemon.returncode != 0:
        if os.path.isfile(DOCKER_DESKTOP):
            subprocess.Popen([DOCKER_DESKTOP], close_fds=True)
        t0 = time.time()
        while time.time() - t0 < wait_s:
            daemon = _docker("ps")
            if daemon is not None and daemon.returncode == 0:
                break
            time.sleep(2)
        else:
            st["message"] = ("Docker Desktop did not start. Open 'Docker Desktop' "
                             "from the Start menu, wait for it to say 'running', "
                             "then try again.")
            return st

    started = _docker("start", MONGO_CONTAINER)
    if started is None or started.returncode != 0:
        # First time on a fresh Docker install: create the container.
        _docker("run", "-d", "--name", MONGO_CONTAINER, "--restart", "unless-stopped",
                "-p", "127.0.0.1:27017:27017", "-v", "scanner-mongo-data:/data/db",
                "mongo:7", timeout=180)

    t0 = time.time()
    while time.time() - t0 < 60:
        scanner_db.reset_client_cache()
        st = hardware.probe_mongo()
        if st["ok"]:
            return st
        time.sleep(2)
    return st


def main(argv=None) -> int:
    from . import hardware

    print("Checking the scanner...")
    problems = []

    db = ensure_mongo()
    print(f"  [{'OK' if db['ok'] else 'XX'}] Database:    {db['message']}")
    if not db["ok"]:
        problems.append("database")

    k = hardware.probe_kinect()
    print(f"  [{'OK' if k['ok'] else 'XX'}] Kinect:      {k['message']}")
    if not k["ok"]:
        problems.append("kinect")

    e = hardware.probe_esp32()
    msg = e.message or ("Connected." if e.connected else "Not found.")
    print(f"  [{'OK' if e.connected else 'XX'}] Laser board: {msg}")
    if not e.connected:
        problems.append("laser board")

    p = hardware.probe_projector()
    print(f"  [{'OK' if p['ok'] else '??'}] Projector:   {p['message']}")
    if not p["ok"]:
        print("       (if the projector is on and showing an image, ignore this line)")

    if problems:
        print()
        print("STOP: fix the " + " and ".join(problems) + " above, then run scan.bat again.")
        print("  Kinect: plug in its power brick and the USB3 cable, wait 10 s.")
        print("  Laser board: the small ESP32 board on COM3; unplug/replug its USB cable.")
        return 1

    print()
    print("All good. Turn the room lights OFF and the projector ON.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
