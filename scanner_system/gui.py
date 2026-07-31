"""Streamlit GUI for the scanner material-recognition system.

Browse samples, label their material, view per-modality scan artifacts (with
laser wavelengths), see honest per-instrument capture status, and export a
labeled dataset for training.

Run locally:
    streamlit run scanner_system/gui.py

Reach it from other machines on the lab LAN:
    streamlit run scanner_system/gui.py --server.address 0.0.0.0
    # then open http://<scanner-pc-ip>:8501

From anywhere (off-network): put a tunnel (ngrok / Cloudflare Tunnel) in front
of port 8501. Add auth before exposing it publicly.

The GUI degrades gracefully: if MongoDB is unreachable it shows a clear message
instead of crashing.
"""

from __future__ import annotations

import os
import sys

import streamlit as st

# Allow "streamlit run scanner_system/gui.py" from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scanner_system import scanner_db, schema, hardware

# Root under which artifact file_paths are stored, so the GUI can load images.
# Same default as the capture orchestrator, so images written by a capture are
# found here.
from scanner_system.capture import STORAGE_ROOT  # noqa: E402


st.set_page_config(page_title="Scanner · Material Recognition", page_icon="🔬", layout="wide")


def _db_or_message():
    """Return a live db handle, or None after showing a friendly message."""
    try:
        db = scanner_db.get_db()
        # touch the server so an unreachable Mongo fails here, not mid-page
        db["samples"].find_one({})
        return db
    except Exception as exc:
        st.error(
            "Cannot reach the scanner MongoDB.\n\n"
            f"URL: `{scanner_db.get_url()}`  ·  DB: `{scanner_db.get_db_name()}`\n\n"
            f"Details: {exc}\n\n"
            "Start MongoDB (e.g. `docker run -d -p 127.0.0.1:27017:27017 mongo:7`) "
            "and reload."
        )
        return None


def _artifact_abs_path(file_path: str) -> str:
    return os.path.join(STORAGE_ROOT, file_path)


# ── Sidebar: navigation ─────────────────────────────────────────────────────

st.sidebar.title("🔬 Scanner")
st.sidebar.caption("Material recognition dataset")
page = st.sidebar.radio("View", ["Samples", "New sample", "Dataset export", "Hardware"])


# ── Page: Samples ───────────────────────────────────────────────────────────

def page_samples():
    st.header("Samples")
    samples = scanner_db.list_samples(db=db)
    if not samples:
        st.info("No samples yet. Create one from **New sample**.")
        return

    # material filter
    classes = sorted({(s.get("material") or {}).get("class") for s in samples
                      if (s.get("material") or {}).get("class")})
    pick = st.selectbox("Filter by material class", ["(all)"] + classes)
    if pick != "(all)":
        samples = [s for s in samples if (s.get("material") or {}).get("class") == pick]

    for s in samples:
        mat = s.get("material") or {}
        label = mat.get("class") or "unlabeled"
        sub = mat.get("subclass")
        title = f"{s.get('label','(no label)')}  —  {label}" + (f" / {sub}" if sub else "")
        with st.expander(title):
            _render_sample(s)


def _render_sample(s):
    sid = s["_id"]
    st.caption(f"sample_id: `{sid}`")

    # material labeling (can happen after capture)
    mat = s.get("material") or {}
    c1, c2, c3 = st.columns([2, 2, 1])
    cls = c1.text_input("Material class", value=mat.get("class") or "", key=f"cls_{sid}")
    sub = c2.text_input("Subclass", value=mat.get("subclass") or "", key=f"sub_{sid}")
    if c3.button("Save label", key=f"save_{sid}"):
        scanner_db.set_material(sid, cls or None, sub or None, db=db)
        st.success("Label saved.")
        st.rerun()

    # capture a new scan
    _render_capture(sid)

    # scans for this sample
    scans = scanner_db.scans_for_sample(sid, db=db)
    if not scans:
        st.info("No scans recorded for this sample yet.")
        return

    for scan in scans:
        _render_scan(scan)


def _render_capture(sid):
    from scanner_system import capture

    with st.expander("📷 Run a capture", expanded=False):
        mode = st.selectbox(
            "Mode", list(schema.CAPTURE_MODES), key=f"mode_{sid}",
            help="full = lasers + kinect (+projector). Fallbacks let you run "
                 "instruments individually if they interfere.",
        )
        chans = st.multiselect(
            "Laser channels to fire", list(schema.LASER_CHANNELS),
            default=[1, 2, 3], key=f"chans_{sid}",  # CH4 diode lead broken; solder fix pending
            help="Which of the 4 lasers to capture the sample under. "
                 "Each is a different wavelength.",
        )
        st.caption(
            "This drives real hardware: it fires the selected lasers and captures "
            "a Kinect frame under each. Lasers are turned OFF between shots and at "
            "the end. If the ESP32/Kinect isn't connected, the scan still records "
            "what worked and why the rest didn't."
        )
        if st.button("▶ Run capture", key=f"cap_{sid}", type="primary"):
            with st.spinner("Capturing… (firing lasers + grabbing frames)"):
                pkg = capture.run_capture(sid, mode=mode, laser_channels=chans, db=db)
            status = pkg.get("status", "?")
            if status == "complete":
                st.success(f"Capture complete ({status}).")
            elif status == "partial":
                st.warning(f"Capture partial — some instruments failed (see below).")
            else:
                st.error(f"Capture {status} — check the per-instrument detail below.")
            st.rerun()


def _render_scan(scan):
    status = scan.get("status", "?")
    badge = {"complete": "🟢", "partial": "🟡", "failed": "🔴",
             "running": "⚪"}.get(status, "•")
    st.markdown(f"**Scan** `{scan['_id'][:8]}` · mode `{scan.get('mode','?')}` · {badge} {status}")

    # per-instrument honest status
    results = scan.get("results") or {}
    if results:
        cols = st.columns(len(results))
        for col, (inst, r) in zip(cols, results.items()):
            mark = {"ok": "🟢", "failed": "🔴", "skipped": "⚪"}.get(r.get("status"), "•")
            col.markdown(f"{mark} **{inst}**")
            if r.get("detail"):
                col.caption(r["detail"])

    # artifacts grouped by modality
    pkg = scanner_db.scan_package(scan["_id"], db=db)
    for modality in schema.MODALITIES:
        arts = pkg["artifacts"].get(modality, [])
        if not arts:
            continue
        st.markdown(f"*{modality}* ({len(arts)})")
        thumbs = st.columns(4)
        for i, art in enumerate(arts):
            with thumbs[i % 4]:
                _render_artifact(art)


def _render_artifact(art):
    cap = art.get("role", "")
    ls = art.get("laser_state")
    if ls:
        wl = ls.get("wavelength_nm")
        cap += f" · {wl}nm" if wl else ""
        cap += " (IR)" if ls.get("ir") else ""
    path = _artifact_abs_path(art.get("file_path", ""))
    media = art.get("media_type", "")
    if media.startswith("image/") and os.path.isfile(path):
        st.image(path, caption=cap, use_container_width=True)
    else:
        st.caption(f"{cap}\n\n`{art.get('file_path','')}`")


# ── Page: New sample ────────────────────────────────────────────────────────

def page_new_sample():
    st.header("New sample")
    with st.form("new_sample"):
        label = st.text_input("Label", placeholder="e.g. oak plank #3")
        c1, c2 = st.columns(2)
        cls = c1.text_input("Material class", placeholder="e.g. wood")
        sub = c2.text_input("Subclass", placeholder="e.g. oak")
        notes = st.text_area("Notes / context", placeholder="anything worth recording")
        submitted = st.form_submit_button("Create sample")
    if submitted:
        if not label.strip():
            st.error("Label is required.")
            return
        sid = scanner_db.create_sample(
            label.strip(), material_class=cls or None, material_subclass=sub or None,
            context={"notes": notes} if notes else None, db=db,
        )
        st.success(f"Created sample `{sid}`.")
        st.caption("Capture scans against this sample_id from the capture script.")


# ── Page: Dataset export ────────────────────────────────────────────────────

def page_export():
    st.header("Dataset export")
    st.caption("Flatten labeled samples + their feature artifacts into training rows.")
    samples = scanner_db.list_samples(db=db)
    classes = sorted({(s.get("material") or {}).get("class") for s in samples
                      if (s.get("material") or {}).get("class")})
    c1, c2 = st.columns(2)
    cls = c1.selectbox("Material class", ["(all)"] + classes)
    modality = c2.selectbox("Modality", ["(all)"] + list(schema.MODALITIES))
    rows = scanner_db.export_dataset(
        material_class=None if cls == "(all)" else cls,
        modality=None if modality == "(all)" else modality,
        db=db,
    )
    st.write(f"**{len(rows)}** feature rows")
    if rows:
        st.dataframe(rows, use_container_width=True)
        import csv, io
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
        st.download_button("Download CSV", buf.getvalue(), "dataset.csv", "text/csv")


# ── Page: Hardware (status + self-test) ─────────────────────────────────────

def page_hardware():
    st.header("Hardware")
    st.caption(
        "What the software can actually see. If you don't know what's wired, this "
        "tells you what's connected and responding -- in plain terms."
    )

    # --- MongoDB ---
    st.subheader("MongoDB (data store)")
    m = hardware.probe_mongo()
    (st.success if m["ok"] else st.error)(m["message"])
    st.caption(f"`{m['url']}` · db `{m['db']}`")

    # --- Kinect ---
    st.subheader("Kinect (depth + color camera)")
    k = hardware.probe_kinect()
    (st.success if k.get("present") else st.error)(k["message"])
    if k.get("present") and st.button("Test Kinect capture (grab a live frame)"):
        import tempfile
        out = os.path.join(tempfile.gettempdir(), "kinect_hwtest.png")
        with st.spinner("Grabbing a frame from the Kinect…"):
            from scanner_system import capture
            r = capture._kinect_grab(out)
        if r["ok"] and os.path.isfile(out):
            st.success("Live frame captured — the Kinect is delivering.")
            st.image(out, caption="Live Kinect color frame", use_container_width=True)
        else:
            st.error(f"Grab failed: {r['detail']}")

    # --- Projector ---
    st.subheader("Projector (DLP structured-light)")
    pj = hardware.probe_projector()
    (st.success if pj["ok"] else st.error)(pj["message"])

    # --- ESP32 / lasers ---
    st.subheader("ESP32 (laser + relay controller)")
    ports = hardware.list_serial_ports()
    with st.expander(f"Serial ports seen: {len(ports)}"):
        for p in ports:
            st.write(f"- `{p['device']}` — {p['description']}")
    guess = hardware.likely_esp32_port()
    if guess is None:
        st.error(
            "No ESP32-like board detected on any serial port.\n\n"
            "Check: is the ESP32 plugged into USB? Is it a **data** cable (not a "
            "charge-only one)? A charge-only cable powers the board (LED on) but "
            "shows no port."
        )
        return
    st.info(f"A board that looks like the ESP32 is on `{guess}`.")

    st.warning(
        "Reading its status opens the serial port, which **resets the board**. "
        "On reset the firmware drives all relays to their safe (OFF) state first, "
        "so this is safe."
    )
    if st.button("Probe ESP32 (read-only: PING + status)"):
        with st.spinner(f"Talking to {guess}…"):
            es = hardware.probe_esp32(guess)
        st.session_state["esp32"] = es

    es = st.session_state.get("esp32")
    if es is not None:
        (st.success if es.connected else st.error)(es.message)
        if es.connected:
            if es.channels:
                st.write("**Relay channels** (each may drive a laser/device):")
                st.table(es.channels)
            else:
                st.info("No relay channels configured on the board.")
            if es.laser:
                st.write("**Laser PWM:**", "configured" if es.laser.get("configured") else "not configured")
                st.caption(f"`{es.laser.get('raw','')}`")

            st.divider()
            st.subheader("Blink test — find out what's wired")
            st.caption(
                "Fire one channel briefly to see if anything physically responds. "
                "It turns the channel back OFF automatically. **Only do this if it's "
                "safe for that channel to activate** (e.g. a low-power laser pointed "
                "somewhere safe, eye protection on if it's a laser)."
            )
            chans = [c["ch"] for c in es.channels]
            if chans:
                c1, c2, c3 = st.columns([1, 1, 2])
                ch = c1.selectbox("Channel", chans)
                secs = c2.slider("Seconds", 0.2, 3.0, 1.0, 0.1)
                if c3.button(f"⚡ Fire CH{ch} briefly", type="primary"):
                    with st.spinner(f"Firing CH{ch}…"):
                        r = hardware.blink_channel(ch, port=guess, seconds=secs)
                    (st.success if r["ok"] else st.error)(r["message"])


# ── Dispatch ────────────────────────────────────────────────────────────────

if page == "Hardware":
    # The hardware page must work even when Mongo is down (diagnosing that is
    # part of its job), so it runs before the Mongo gate.
    page_hardware()
else:
    db = _db_or_message()
    if db is None:
        st.stop()
    try:
        scanner_db.ensure_indexes(db)
    except Exception:
        pass
    {
        "Samples": page_samples,
        "New sample": page_new_sample,
        "Dataset export": page_export,
    }[page]()
