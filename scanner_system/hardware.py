"""Hardware probing for the scanner GUI's status/self-test panel.

Goal: make the rig legible to someone who cannot read the wiring. Every function
here reports, in plain terms, what the software can and cannot see -- and never
pretends. There is no simulation/fake mode: if a device is not connected, that
is exactly what it reports.

Safety (see AGENTS.md):
- Importing this module touches no hardware and opens no port.
- Probing the ESP32 opens its serial port, which RESETS the board; on reset the
  firmware drives all relays to their safe (OFF) state before it reports READY,
  so a probe is safe. Probes are read-only (PING / STATUS) unless you explicitly
  call a set/blink function.
- Firing a channel (blink_channel) energizes a relay/laser. It is a deliberate,
  clearly-labeled action, never part of a status probe. It always turns the
  channel back OFF afterward.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional


# ── Serial port / ESP32 presence (no connection, pure enumeration) ──────────

CP210X_HINTS = ("cp210", "silicon labs", "usb-serial", "ch340", "uart")

# Longest a deliberate fire (blink test) may energize a channel. Bounded so a
# stuck slider/typo can't leave a laser on indefinitely; raise deliberately.
MAX_FIRE_SECONDS = 15.0

# Interpreter that can actually talk to the Kinect (has pykinect2). Real grabs
# run here, not in the GUI's venv. Keep in sync with capture.KINECT_PYTHON.
KINECT_PYTHON = os.getenv("SCANNER_KINECT_PYTHON", "").strip() or r"C:\KinectEnv\Scripts\python.exe"

_relay_controller_cls = None


def _connect_with_retry(rc, attempts: int = 3) -> bool:
    """Try to connect to the ESP32 a few times. The port can be briefly held by
    another action (a capture just finished) or the board mid-reset; a couple of
    retries turns a transient miss into a success instead of 'no response'."""
    import time as _t
    for i in range(attempts):
        try:
            if rc.connect():
                return True
        except Exception:
            pass
        _t.sleep(0.8)
    return False


def _load_relay_controller():
    """Load RelayController from lib_3dai/relay_controller.py by FILE PATH.

    Deliberately does NOT do ``from lib_3dai.relay_controller import ...``: that
    triggers lib_3dai/__init__.py, which imports the Kinect/projector modules and
    can initialize hardware just from importing (see AGENTS.md). relay_controller
    only needs pyserial, so we import the single file in isolation. Returns the
    class, or None if pyserial is missing.
    """
    global _relay_controller_cls
    if _relay_controller_cls is not None:
        return _relay_controller_cls
    try:
        import importlib.util
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(here, "lib_3dai", "relay_controller.py")
        spec = importlib.util.spec_from_file_location("_relay_controller_isolated", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _relay_controller_cls = mod.RelayController
    except Exception:
        _relay_controller_cls = None
    return _relay_controller_cls


def list_serial_ports() -> list[dict[str, str]]:
    """List serial ports with descriptions. Does NOT open them."""
    try:
        import serial.tools.list_ports as lp
    except Exception:
        return []
    out = []
    for p in lp.comports():
        out.append({
            "device": p.device,
            "description": p.description or "",
            "hwid": p.hwid or "",
        })
    return out


def likely_esp32_port() -> Optional[str]:
    """Best guess at the ESP32's port from the USB descriptor, without opening it.

    Returns a device name (e.g. "COM3") or None. Purely from enumeration -- a
    match here means "a board that looks like the ESP32 is plugged in", not that
    it is responding.
    """
    for p in list_serial_ports():
        blob = f"{p['description']} {p['hwid']}".lower()
        if any(h in blob for h in CP210X_HINTS):
            return p["device"]
    return None


# ── ESP32 status (opens the port -> resets the board -> read-only PING/STATUS)

@dataclass
class Esp32Status:
    connected: bool = False           # port opened and PONG received
    port: Optional[str] = None
    message: str = ""
    channels: list[dict[str, Any]] = field(default_factory=list)  # relay channels
    laser: Optional[dict[str, Any]] = None                        # laser PWM state


def probe_esp32(port: Optional[str] = None) -> Esp32Status:
    """Open the ESP32, PING it, and read its relay + laser status. Read-only.

    ``port`` may be given; otherwise we auto-detect. Never raises: any failure
    (no board, wrong port, pyserial missing) comes back as ``connected=False``
    with a human message explaining what to check.
    """
    st = Esp32Status()

    RelayController = _load_relay_controller()
    if RelayController is None:
        st.message = "relay controller unavailable (pyserial missing?)"
        return st

    if port is None:
        port = likely_esp32_port()
    if port is None:
        st.message = ("No ESP32-like serial port found. Is the board plugged in "
                      "with a DATA usb cable (not charge-only)?")
        return st
    st.port = port

    rc = RelayController(port)
    try:
        if not _connect_with_retry(rc):
            st.message = (f"Found {port} but no response to PING after retries. The "
                          "port may be busy (a capture running?), or the board needs "
                          "a replug.")
            return st
        st.connected = True
        # relay channels
        for c in rc.status():
            st.channels.append({
                "ch": c.ch,
                "pin": c.pin,
                "polarity": "active-HIGH" if c.active_high else "active-LOW",
                "safe": "ON" if c.safe_on else "OFF",
                "state": "ON" if c.state else "OFF",
            })
        # laser subsystem (LASER STATUS); tolerate older firmware without it
        try:
            laser_line = "\n".join(rc._send("LASER STATUS"))
            if "UNCONFIGURED" in laser_line.upper():
                st.laser = {"configured": False, "raw": laser_line}
            elif laser_line.upper().startswith("LASER"):
                st.laser = {"configured": True, "raw": laser_line}
        except Exception:
            st.laser = None
        st.message = (f"Connected on {port}. {len(st.channels)} relay channel(s) "
                      "configured. All shown states are what the firmware reports.")
        return st
    finally:
        rc.disconnect()


# ── Blink test (DELIBERATE: energizes one channel briefly, then OFF) ─────────

def blink_channel(ch: int, port: Optional[str] = None, seconds: float = 1.0) -> dict[str, Any]:
    """Turn one relay channel ON for ``seconds`` then OFF. Deliberate action.

    This is how you find out whether *anything* is physically wired to a channel:
    fire it and watch/listen. It always returns the channel to OFF. Returns a
    dict with ok/message. Never leaves a channel energized on error.
    """
    result = {"ok": False, "message": ""}
    RelayController = _load_relay_controller()
    if RelayController is None:
        result["message"] = "relay controller unavailable (pyserial missing?)"
        return result

    if port is None:
        port = likely_esp32_port()
    if port is None:
        result["message"] = "No ESP32 port found."
        return result

    rc = RelayController(port)
    try:
        if not _connect_with_retry(rc):
            result["message"] = f"No response on {port} after retries (port busy or board needs replug)."
            return result
        # make sure it's a configured channel before firing
        status = rc.get_channel(ch)
        if status is None or not status.configured:
            result["message"] = f"CH{ch} is not configured; nothing to fire."
            return result
        # Defensive: explicitly drive every OTHER configured channel OFF so only
        # this one is energized. The board already reset to all-OFF on connect;
        # this makes the single-channel guarantee explicit. It CANNOT cure
        # electrical cross-talk -- if another relay still fires while its GPIO is
        # held LOW, that is a wiring/power fault (shared/underpowered rail,
        # coil back-EMF), not software.
        for other in rc.status():
            if other.ch != ch:
                rc.set_channel(other.ch, False)
        rc.set_channel(ch, True)
        import time
        fired = max(0.1, min(seconds, MAX_FIRE_SECONDS))   # clamp 0.1-15s
        time.sleep(fired)
        rc.set_channel(ch, False)
        result["ok"] = True
        result["message"] = (f"Fired CH{ch} for {fired:.1f}s then turned it OFF. "
                             "Did a laser/relay light or click? If not, nothing is "
                             "wired to that channel (or its power is off).")
        return result
    finally:
        # belt and suspenders: force the channel off no matter what
        try:
            rc.set_channel(ch, False)
        except Exception:
            pass
        rc.disconnect()


def blink_channels(chs, port: Optional[str] = None, seconds: float = 1.0) -> dict[str, Any]:
    """Fire several relay channels together for ``seconds``, then turn them all
    OFF. Same safety contract as blink_channel: unconfigured channels are
    skipped, and every requested channel is forced OFF no matter what. Used to
    fire all lasers at once.
    """
    result = {"ok": False, "message": ""}
    RelayController = _load_relay_controller()
    if RelayController is None:
        result["message"] = "relay controller unavailable (pyserial missing?)"
        return result
    if port is None:
        port = likely_esp32_port()
    if port is None:
        result["message"] = "No ESP32 port found."
        return result

    fired = max(0.1, min(seconds, MAX_FIRE_SECONDS))
    rc = RelayController(port)
    try:
        if not _connect_with_retry(rc):
            result["message"] = f"No response on {port} after retries (port busy or board needs replug)."
            return result
        usable = [ch for ch in chs
                  if (s := rc.get_channel(ch)) is not None and s.configured]
        if not usable:
            result["message"] = "None of the requested channels are configured; nothing to fire."
            return result

        # Prefer the atomic SETM command so all channels switch together. Old
        # firmware without SETM replies ERR (set_channels -> False); fall back to
        # per-channel SETs, which stagger over serial round-trips.
        def _drive(state: bool) -> str:
            if rc.set_channels(usable, state):
                return "together"
            for ch in usable:
                rc.set_channel(ch, state)
            return "sequentially (board firmware lacks SETM; reflash for simultaneous)"

        how = _drive(True)
        import time
        time.sleep(fired)
        _drive(False)
        result["ok"] = True
        skipped = [c for c in chs if c not in usable]
        note = f" (skipped unconfigured {skipped})" if skipped else ""
        result["message"] = f"Fired channels {usable} {how} for {fired:.1f}s then turned them OFF{note}."
        return result
    finally:
        # force every requested channel off no matter what
        for ch in chs:
            try:
                rc.set_channel(ch, False)
            except Exception:
                pass
        rc.disconnect()


# ── Projector aiming target ──────────────────────────────────────────────────

# GUI-capable OpenCV lives in the KinectEnv interpreter (the project venv ships
# headless OpenCV, which can't open a window), so the target is projected by
# shelling out to it -- same pattern as the capture pipeline's projector stage.
_KINECT_PYTHON = os.getenv("SCANNER_KINECT_PYTHON", "").strip() or r"C:\KinectEnv\Scripts\python.exe"
_TARGET_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "project_target.py")


def start_target_projection(seconds: float):
    """Project the laser-aiming target (bullseye grid) on the projector for
    ``seconds`` in a background process. Returns the Popen handle (or None if it
    couldn't start). Runs concurrently with firing so the laser dot can be
    walked onto the center while it's lit. Always pair with stop_projection().
    A margin is added so the target outlives the fire's connect overhead; the
    caller terminates it when the fire finishes.
    """
    import subprocess
    try:
        return subprocess.Popen([_KINECT_PYTHON, _TARGET_SCRIPT, str(max(1.0, seconds) + 5.0)])
    except Exception:
        return None


def stop_projection(proc) -> None:
    """Terminate a start_target_projection() process. Safe on None."""
    if proc is not None:
        try:
            proc.terminate()
        except Exception:
            pass


# ── MongoDB status ──────────────────────────────────────────────────────────

def probe_mongo() -> dict[str, Any]:
    """Check the scanner MongoDB. Never raises."""
    from . import scanner_db
    try:
        db = scanner_db.get_db()
        db["samples"].find_one({})
        n = len(scanner_db.list_samples(db=db, limit=1_000_000))
        return {"ok": True, "url": scanner_db.get_url(),
                "db": scanner_db.get_db_name(), "samples": n,
                "message": f"Connected ({n} samples)."}
    except Exception as exc:
        return {"ok": False, "url": scanner_db.get_url(),
                "db": scanner_db.get_db_name(),
                "message": f"Not reachable: {exc}. Start it with "
                           "`docker run -d -p 127.0.0.1:27017:27017 mongo:7`."}


# ── Windows device presence (asks the OS what's actually plugged in) ─────────

def _pnp_present(pattern: str) -> list[dict[str, str]]:
    """Return present PnP devices whose name/id matches ``pattern`` (regex).

    Asks Windows directly, so it reflects what's ACTUALLY connected right now --
    not what libraries happen to be installed in this interpreter. Returns a list
    of {status, name}; empty if none present or the query fails.
    """
    import subprocess
    ps = (
        "Get-PnpDevice -PresentOnly -ErrorAction SilentlyContinue | "
        f"Where-Object {{ $_.FriendlyName -match '{pattern}' -or $_.InstanceId -match '{pattern}' }} | "
        "ForEach-Object { \"$($_.Status)|$($_.FriendlyName)\" }"
    )
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=12)
    except Exception:
        return []
    rows = []
    for line in (out.stdout or "").splitlines():
        line = line.strip()
        if "|" in line:
            status, name = line.split("|", 1)
            rows.append({"status": status.strip(), "name": name.strip()})
    return rows


def _kinect_interpreter_ok() -> tuple[bool, str]:
    """Is C:\\KinectEnv present AND carrying the pykinect2 patches capture needs?

    A file-exists check passed an unpatched interpreter, which then failed
    inside kinect_laser_sequence.py after the dark frame was written, leaving
    a scan with a dark reference and zero laser frames. The laser path needs
    the 2026-09-05 infrared patch (PyKinectRuntime.get_last_infrared_frame);
    probe for it without touching the sensor.
    """
    if not os.path.isfile(KINECT_PYTHON):
        return False, f"capture interpreter missing: {KINECT_PYTHON}"
    import subprocess
    try:
        r = subprocess.run(
            [KINECT_PYTHON, "-c",
             "from pykinect2.PyKinectRuntime import PyKinectRuntime as R; "
             "assert hasattr(R, 'get_last_infrared_frame')"],
            capture_output=True, text=True, timeout=20, check=False)
    except Exception as exc:
        return False, f"capture interpreter did not start: {exc}"
    if r.returncode != 0:
        return False, ("KinectEnv pykinect2 lacks the infrared patch needed for laser "
                       "capture (see PROJECT_MEMORY.md, KinectEnv patches)")
    return True, ""


def probe_kinect() -> dict[str, Any]:
    """Report whether the Kinect V2 is actually connected to this PC.

    Asks Windows whether the Kinect sensor device is present (not whether some
    library is installed). Also confirms the capture interpreter exists, since
    real grabs run under C:\\KinectEnv. Does NOT initialize the sensor.
    """
    devices = _pnp_present(r"Xbox NUI Sensor|Kinect|VID_045E")
    present = [d for d in devices if d["status"].upper() == "OK"]
    interp_ok, interp_why = _kinect_interpreter_ok()
    if present:
        names = ", ".join(sorted({d["name"] for d in present}))
        msg = f"Connected: {names}."
        msg += "" if interp_ok else f"  ({interp_why})"
        return {"ok": bool(interp_ok), "present": True, "interpreter_ok": interp_ok,
                "message": msg}
    if devices:
        return {"ok": False, "present": True, "interpreter_ok": interp_ok,
                "message": f"Kinect detected but not ready (status "
                           f"{devices[0]['status']}). Check power brick + USB3."}
    return {"ok": False, "present": False, "interpreter_ok": interp_ok,
            "message": "Kinect not detected on USB. Check it's powered (needs its "
                       "power brick) and on a USB3 port."}


def probe_projector() -> dict[str, Any]:
    """Report whether the DLP projector is connected (as a second display)."""
    devices = _pnp_present(r"DLP4500|DLP|LightCrafter")
    present = [d for d in devices if d["status"].upper() == "OK"]
    if present:
        names = ", ".join(sorted({d["name"] for d in present}))
        return {"ok": True, "present": True,
                "message": f"Connected as a display: {names}."}
    return {"ok": False, "present": False,
            "message": "Projector (DLP) not detected as a display. Check the HDMI "
                       "cable and that it's powered on."}
