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
# SCANNER_GUI_READONLY=1 is how the GUI is shared over Tailscale: the Capture
# and Hardware pages fire lasers and must never be reachable from outside.
READONLY = os.getenv("SCANNER_GUI_READONLY", "").strip() in ("1", "true", "yes")
if READONLY:
    st.sidebar.caption("read-only view")
    page = st.sidebar.radio("View", ["Samples", "Dataset export"])
else:
    page = st.sidebar.radio("View", ["Capture", "Samples", "Dataset export", "Hardware"])


# ── Page: Samples ───────────────────────────────────────────────────────────

def page_samples():
    st.header("Samples")
    samples = scanner_db.list_samples(db=db)
    if not samples:
        st.info("No samples yet. Run one from the **Capture** page.")
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
            default=list(schema.LASER_CHANNELS), key=f"chans_{sid}",
            help="Which of the 4 lasers to capture the sample under. "
                 "CH1/CH2 red, CH3 near-infrared, CH4 green.",
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
    has_images = False
    for modality in schema.MODALITIES:
        arts = pkg["artifacts"].get(modality, [])
        if not arts:
            continue
        has_images = True
        st.markdown(f"*{modality}* ({len(arts)})")
        thumbs = st.columns(4)
        for i, art in enumerate(arts):
            with thumbs[i % 4]:
                _render_artifact(art)

    # interactive 3D surface from the structured-light height map
    _render_height_3d_for_scan(pkg, scan["_id"])

    # publish this scan into 4DAI (via 4DAI's own public API; 4DAI unchanged)
    if has_images and status in ("complete", "partial"):
        _render_publish(scan)


def _render_publish(scan):
    import os
    fourdai_url = os.getenv("FOURDAI_URL", "http://127.0.0.1:8000")
    scan_id = scan["_id"]
    c1, c2 = st.columns([1, 2])
    if c1.button("⬆ Publish to 4DAI", key=f"pub_{scan_id}"):
        from scanner_system import push_to_4dai
        with st.spinner("Pushing scan into 4DAI…"):
            try:
                res = push_to_4dai.publish_scan(
                    scan_id, category=os.getenv("FOURDAI_CATEGORY", "materials"),
                    fourdai_url=fourdai_url, db=db)
                st.success(
                    f"Published to 4DAI: sample `{res['fourdai_sample_id'][:8]}`, "
                    f"{len(res['uploaded_images'])} images. Viewable in 4DAI's UI.")
            except Exception as exc:
                st.error(f"Publish failed: {exc}  (is 4DAI up at {fourdai_url}?)")
    c2.caption(f"Pushes this scan's images into 4DAI at `{fourdai_url}` "
               "using 4DAI's public API — 4DAI code is not modified.")


def _render_artifact(art):
    cap = art.get("role", "")
    ls = art.get("laser_state")
    role = str(art.get("role", ""))
    if ls:
        wl = ls.get("wavelength_nm")
        cap += f" · {wl}nm" if wl else ""
        # laser_state.ir describes the LASER; the camera is told by the role.
        cap += " NIR laser" if ls.get("ir") else ""
    if role.endswith("_ir_png"):
        cap += " · IR camera 512x424"
    path = _artifact_abs_path(art.get("file_path", ""))
    media = art.get("media_type", "")
    if media.startswith("image/") and os.path.isfile(path):
        st.image(_displayable(path), caption=cap, use_container_width=True)
    else:
        st.caption(f"{cap}\n\n`{art.get('file_path','')}`")


def _displayable(path: str):
    """Return something st.image can render for any of our PNGs.

    The infrared frames are 16-bit grayscale (mode I;16). Streamlit hands
    non-RGB(A)/P images to Pillow for JPEG re-encoding, and Pillow refuses
    mode I;16, so st.image(path) raised OSError and took the whole Samples
    page down for every scan captured since IR frames were added. Raw Kinect
    IR values also sit near the bottom of the 0-65535 range, so a plain cast
    renders black: stretch between the 1st and 99.5th percentiles instead.
    """
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(path)
        if not img.mode.startswith("I"):
            return path
        a = np.asarray(img, dtype=np.float32)
        lo, hi = np.percentile(a, 1.0), np.percentile(a, 99.5)
        if hi <= lo:
            hi = lo + 1.0
        return np.clip((a - lo) / (hi - lo) * 255.0, 0, 255).astype("uint8")
    except Exception:
        return path


def _render_height_3d(npy_path):
    """Interactive 3D surface of a structured-light height map (mm)."""
    import numpy as np
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.info("3D view needs plotly:  `scanner_system/.venv/Scripts/pip install plotly`")
        return
    h = np.load(npy_path).astype(float)
    finite = np.isfinite(h)
    if int(finite.sum()) < 100:
        st.caption("Height map too sparse to render in 3D.")
        return
    zmin = float(np.nanpercentile(h, 1))
    zmax = float(np.nanpercentile(h, 99))
    step = max(1, max(h.shape) // 220)                 # keep the surface light + smooth
    hs = np.clip(h[::step, ::step], zmin, zmax)        # clip edge spikes for a clean surface
    rows, cols = hs.shape
    fig = go.Figure(go.Surface(
        z=hs, colorscale="Turbo", cmin=zmin, cmax=zmax,
        colorbar=dict(title="mm"), connectgaps=False))
    fig.update_layout(
        height=520, margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(xaxis_title="x (px)", yaxis_title="y (px)", zaxis_title="height (mm)",
                   aspectmode="manual",
                   aspectratio=dict(x=1.0, y=rows / max(cols, 1), z=0.35)))
    st.plotly_chart(fig, use_container_width=True)
    obj = h[finite & (h > 3.0)]
    if obj.size:
        st.caption(f"Object height: median **{np.median(obj):.1f} mm** · "
                   f"peak {zmax:.1f} mm · {int(finite.sum()):,} reliable points")


def _render_height_3d_for_scan(pkg, scan_id):
    """Show the 3D height surface for a scan, if it produced a height map."""
    fusion = pkg["artifacts"].get("fusion", [])
    npy = next((a for a in fusion if a.get("role") == "height_map_npy"), None)
    if not npy:
        return
    npy_path = _artifact_abs_path(npy.get("file_path", ""))
    if not os.path.isfile(npy_path):
        return
    with st.expander("🧊 3D reconstruction", expanded=False):
        # Lazy: only build the Plotly figure when asked (keeps multi-scan pages fast).
        if st.checkbox("Render interactive 3D surface", key=f"3d_{scan_id}"):
            _render_height_3d(npy_path)
        else:
            st.caption("Per-pixel structured-light height map — tick to load the interactive 3D view.")


# ── Page: Capture (the main workflow) ───────────────────────────────────────

_MODE_LABELS = {
    "full": "Full AIO — 3D (projector) + multispectral lasers + Kinect",
    "kinect_projector": "3D (projector) + Kinect, no lasers",
    "laser_only": "Multispectral lasers only",
    "kinect_only": "Kinect RGB-D only",
    "projector_only": "Structured-light 3D only",
}


def page_capture():
    st.header("📷 New Capture")
    st.caption("Scan a sample across the three modalities: **structured-light 3D** "
               "(projector), **multispectral lasers** (per-wavelength), and **Kinect** "
               "RGB-D. Everything lands in MongoDB, grouped by modality.")

    # 1. Sample: new or existing
    st.subheader("1 · Sample")
    existing = scanner_db.list_samples(db=db)
    which = st.radio("Sample", ["Create new", "Use existing"], horizontal=True,
                     label_visibility="collapsed")
    label = cls = sub = None
    sid = None
    if which == "Create new":
        c1, c2, c3 = st.columns(3)
        label = c1.text_input("Label", placeholder="e.g. oak plank #3")
        cls = c2.text_input("Material class", placeholder="wood")
        sub = c3.text_input("Subclass", placeholder="oak")
    else:
        if not existing:
            st.info("No samples yet — switch to **Create new**.")
            return
        opts = {f"{s.get('label','?')}  ·  {(s.get('material') or {}).get('class') or 'unlabeled'}"
                f"  ·  {s['_id'][:8]}": s["_id"] for s in existing}
        sid = opts[st.selectbox("Existing sample", list(opts))]

    # 2. What to run
    st.subheader("2 · What to capture")
    mode = st.selectbox("Mode", list(schema.CAPTURE_MODES),
                        format_func=lambda m: _MODE_LABELS.get(m, m))
    chans = st.multiselect(
        "Laser wavelengths (channels)", list(schema.LASER_CHANNELS),
        default=list(schema.LASER_CHANNELS),
        help="CH1/CH2 red (~635 nm), CH3 near-infrared (~940 nm, captured on the "
             "Kinect IR sensor), CH4 green (~530 nm, floods the scene).")
    st.caption("⚠️ The laser stage fires real lasers (goggles on, safe beam). The "
               "projector goes black during it so the lasers are the only light.")

    # 3. Run
    st.subheader("3 · Run")
    if st.button("▶ Run capture", type="primary", use_container_width=True):
        if which == "Create new":
            if not (label or "").strip():
                st.error("Label is required to create a sample.")
                return
            sid = scanner_db.create_sample(
                label.strip(), material_class=cls or None, material_subclass=sub or None, db=db)
        from scanner_system import capture
        with st.spinner("Capturing… a full scan takes ~a minute (fringe + lasers + Kinect)."):
            pkg = capture.run_capture(sid, mode=mode, laser_channels=chans, db=db)
        status = pkg.get("status")
        emit = {"complete": st.success, "partial": st.warning}.get(status, st.error)
        emit(f"Capture {status} — scan `{pkg['_id'][:8]}` for sample `{sid[:8]}`.")
        for inst, r in (pkg.get("results") or {}).items():
            mark = {"ok": "🟢", "failed": "🔴", "skipped": "⚪"}.get(r.get("status"), "•")
            st.write(f"{mark} **{inst}** — {r.get('status')}"
                     + (f"  ·  {r.get('detail')}" if r.get("detail") else ""))
        st.info(f"Open **Samples → {sid[:8]}** to view the images and publish to 4DAI.")


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
                "Fire a channel to see if anything physically responds. Channels "
                "are turned back OFF automatically. **Only do this if it's safe for "
                "those channels to activate** (lasers pointed somewhere safe, eye "
                "protection on)."
            )
            chans = [c["ch"] for c in es.channels]
            if chans:
                c1, c2 = st.columns([1, 2])
                ch = c1.selectbox("Channel", chans)
                secs = c2.slider("Seconds", 0.2, 15.0, 1.0, 0.1)
                aim = st.checkbox(
                    "Project aiming target while firing",
                    help="Shows the bullseye/grid on the projector at the same time "
                         "so you can walk the laser dot onto the center while it's lit.",
                )

                def _fire(label, fn):
                    proc = hardware.start_target_projection(secs) if aim else None
                    try:
                        with st.spinner(f"{label} for {secs:.1f}s…"):
                            r = fn()
                    finally:
                        hardware.stop_projection(proc)
                    (st.success if r["ok"] else st.error)(r["message"])

                b1, b2 = st.columns(2)
                if b1.button(f"⚡ Fire CH{ch}", type="primary"):
                    _fire(f"Firing CH{ch}",
                          lambda: hardware.blink_channel(ch, port=guess, seconds=secs))
                if b2.button(f"⚡⚡ Fire ALL {len(chans)} channels"):
                    _fire(f"Firing all channels {chans}",
                          lambda: hardware.blink_channels(chans, port=guess, seconds=secs))


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
        "Capture": page_capture,
        "Samples": page_samples,
        "Dataset export": page_export,
    }[page]()
