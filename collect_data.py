"""
collect_data.py

Runs a full fringe projection capture session, saves the results in a format
optimised for AI training, and optionally generates an interactive HTML report.

Save format — NumPy .npz archive:
    colors   uint8   (1080, 1920, 3, N) — raw BGR color frames
    depths   uint16  (424,  512,    N)  — raw depth in mm, 0 = no data
    patterns object                     — array of pattern descriptor strings

Color and depth are at their native Kinect resolutions and are NOT pre-aligned.
Use the Kinect SDK coordinate mapper in post-processing to align them if needed.

Usage:
    python collect_data.py                  # capture and save
    python collect_data.py --visualize      # capture, save, open HTML report
    python collect_data.py --visualize --output-dir ./my_data
    python collect_data.py --report path/to/capture_20240315_143022.npz  # report from existing file
"""

import argparse
import base64
import io
import time
import webbrowser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — renders to PNG buffers
import matplotlib.pyplot as plt
import numpy as np

import config
from lib_3dai import capture_projections_single


# ---------------------------------------------------------------------------
# Image rendering helpers
# ---------------------------------------------------------------------------

def _array_to_b64_png(arr, cmap="gray", vmin=None, vmax=None) -> str:
    """
    Renders a 2-D or 3-D (H, W, 3) numpy array to a base64-encoded PNG string.
    For 3-channel arrays the cmap/vmin/vmax arguments are ignored.
    """
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    if arr.ndim == 3:
        ax.imshow(arr, aspect="auto")
    else:
        ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.axis("off")
    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _depth_to_b64_png(depth_2d: np.ndarray,
                       low_pct: float = 2.0,
                       high_pct: float = 98.0) -> str:
    """
    Renders a depth frame with viridis colormap.
    Pixels with value 0 (no data) are drawn grey.

    The colormap range is set by percentiles of the valid (non-zero) depth
    values rather than the absolute min/max. This focuses the colour scale
    on the object of interest and prevents a single distant background pixel
    from washing out all the near-object detail.

    Args:
        depth_2d:  2-D uint16 depth array in mm.
        low_pct:   Lower percentile for colormap minimum (default 2%).
        high_pct:  Upper percentile for colormap maximum (default 98%).
    """
    masked = np.where(depth_2d > 0, depth_2d.astype(np.float32), np.nan)

    valid = masked[~np.isnan(masked)]
    if valid.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.percentile(valid, low_pct))
        vmax = float(np.percentile(valid, high_pct))
        # Ensure vmin != vmax so the colorbar renders correctly
        if vmin == vmax:
            vmin = max(0.0, vmin - 1.0)
            vmax = vmax + 1.0

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#aaaaaa")  # grey for no-data pixels
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mm")
    cbar.ax.set_title(f"p{low_pct:.0f}–p{high_pct:.0f}", fontsize=7, pad=4)
    ax.axis("off")
    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _depth_stats(depth_2d: np.ndarray) -> dict:
    """Returns basic statistics for a depth frame."""
    valid = depth_2d[depth_2d > 0]
    if valid.size == 0:
        return {"min": "—", "max": "—", "mean": "—",
                "coverage": "0.0%", "valid_px": 0}
    return {
        "min":      f"{valid.min():.0f} mm",
        "max":      f"{valid.max():.0f} mm",
        "mean":     f"{valid.mean():.1f} mm",
        "coverage": f"{100.0 * valid.size / depth_2d.size:.1f}%",
        "valid_px": int(valid.size),
    }


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_capture(data: dict, output_dir: str = "data") -> Path:
    """
    Saves a capture session as a compressed NumPy .npz archive.

    Arrays in the archive:
        colors   uint8   (1080, 1920, 3, N) — raw BGR color frames
        depths   uint16  (424,  512,    N)  — raw depth in mm, 0 = no data
        patterns object                     — 1-D string array of pattern descriptors

    Color and depth are at their native Kinect resolutions and are NOT
    pre-aligned. Use the Kinect SDK coordinate mapper in post-processing
    to align them if needed.

    Reload with:
        d = np.load("capture_....npz", allow_pickle=True)
        colors, depths, patterns = d["colors"], d["depths"], d["patterns"]
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    filepath = out / f"capture_{timestamp}.npz"

    np.savez_compressed(
        filepath,
        colors=data["colors"],
        depths=data["depths"],
        patterns=np.array(data["patterns"], dtype=object),
    )

    print(f"\n  Saved → {filepath}")
    print(f"  colors  : {data['colors'].shape}  dtype={data['colors'].dtype}")
    print(f"  depths  : {data['depths'].shape}  dtype={data['depths'].dtype}")
    print(f"  patterns: {data['patterns']}")

    return filepath


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def generate_report(data: dict, save_path: Path) -> Path:
    """
    Writes a self-contained HTML file with a tab per captured pattern showing
    the raw color image and depth map side by side. All images are base64
    embedded so the file is fully portable.

    Args:
        data:      Return value of capture_projections_single (or reloaded npz).
        save_path: Path of the .npz file — report is saved alongside it.

    Returns:
        Path to the generated HTML report.
    """
    colors   = data["colors"]    # (1080, 1920, 3, N)
    depths   = data["depths"]    # (424,  512,    N)
    patterns = data["patterns"]
    n        = colors.shape[3]

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    tab_buttons = []
    tab_panels  = []

    for i, pat in enumerate(patterns):
        # Kinect returns BGR — convert to RGB for correct matplotlib rendering
        color_rgb = colors[:, :, :, i][:, :, ::-1]
        img_b64   = _array_to_b64_png(color_rgb)
        depth_b64 = _depth_to_b64_png(depths[:, :, i])
        stats     = _depth_stats(depths[:, :, i])
        active    = "active" if i == 0 else ""

        tab_buttons.append(f"""
            <button class="tab-btn {active}" onclick="showTab({i})">
                {pat}
            </button>""")

        tab_panels.append(f"""
            <div class="tab-panel {active}" id="panel-{i}">
                <p class="res-note">
                    Color: 1920&times;1080 px &nbsp;|&nbsp;
                    Depth: 512&times;424 px &nbsp;|&nbsp;
                    <em>Frames are at native resolution — not pre-aligned.</em>
                </p>
                <div class="panel-grid">
                    <div class="panel-cell">
                        <h3>Color Image</h3>
                        <img src="data:image/png;base64,{img_b64}" alt="Color {pat}">
                    </div>
                    <div class="panel-cell">
                        <h3>Depth Map</h3>
                        <img src="data:image/png;base64,{depth_b64}" alt="Depth {pat}">
                    </div>
                </div>
                <table class="stats-table">
                    <tr>
                        <th>Min depth</th><th>Max depth</th><th>Mean depth</th>
                        <th>Valid pixel coverage</th><th>Valid pixels</th>
                    </tr>
                    <tr>
                        <td>{stats['min']}</td><td>{stats['max']}</td>
                        <td>{stats['mean']}</td><td>{stats['coverage']}</td>
                        <td>{stats['valid_px']:,}</td>
                    </tr>
                </table>
            </div>""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Fringe Capture Report — {timestamp}</title>
<style>
  body            {{ font-family: Arial, sans-serif; margin: 0; padding: 20px;
                     background: #f4f4f4; color: #222; }}
  h1              {{ margin-bottom: 4px; }}
  .meta           {{ color: #666; font-size: 0.9em; margin-bottom: 20px; }}
  .res-note       {{ color: #888; font-size: 0.85em; margin: 0 0 12px; }}
  .summary        {{ background: #fff; border-radius: 8px; padding: 16px 20px;
                     margin-bottom: 20px; box-shadow: 0 1px 4px #0002; }}
  .summary code   {{ background: #f4f4f4; padding: 4px 8px; border-radius: 4px;
                     display: inline-block; margin-top: 6px; line-height: 1.8; }}
  .tab-bar        {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 16px; }}
  .tab-btn        {{ padding: 8px 18px; border: none; border-radius: 6px;
                     background: #ddd; cursor: pointer; font-size: 0.95em;
                     transition: background 0.15s; }}
  .tab-btn:hover  {{ background: #bbb; }}
  .tab-btn.active {{ background: #3a7bd5; color: #fff; }}
  .tab-panel      {{ display: none; background: #fff; border-radius: 8px;
                     padding: 20px; box-shadow: 0 1px 4px #0002; }}
  .tab-panel.active {{ display: block; }}
  .panel-grid     {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px;
                     margin-bottom: 16px; }}
  .panel-cell h3  {{ margin: 0 0 8px; font-size: 1em; color: #444; }}
  .panel-cell img {{ width: 100%; border-radius: 4px; display: block; }}
  .stats-table    {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
  .stats-table th {{ background: #f0f4ff; padding: 8px 12px; text-align: left;
                     border-bottom: 2px solid #ccd; }}
  .stats-table td {{ padding: 8px 12px; border-bottom: 1px solid #eee; }}
  @media (max-width: 700px) {{ .panel-grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>

<h1>Fringe Projection Capture Report</h1>
<p class="meta">
    Generated: {timestamp} &nbsp;|&nbsp;
    File: <code>{save_path.name}</code> &nbsp;|&nbsp;
    Patterns: {n}
</p>

<div class="summary">
    <strong>Dataset summary</strong> — load this file in Python with:<br>
    <code>
        d = np.load("{save_path.name}", allow_pickle=True)<br>
        colors, depths, patterns = d["colors"], d["depths"], d["patterns"]<br>
        # colors  → uint8  {tuple(colors.shape)}&nbsp;&nbsp;(height, width, BGR, num_patterns)<br>
        # depths  → uint16 {tuple(depths.shape)}&nbsp;&nbsp;(height, width, num_patterns) depth mm, 0=no data<br>
        # patterns → {list(patterns)}
    </code>
</div>

<div class="tab-bar">{''.join(tab_buttons)}</div>
{''.join(tab_panels)}

<script>
  function showTab(idx) {{
    document.querySelectorAll('.tab-panel').forEach((p, i) => p.classList.toggle('active', i === idx));
    document.querySelectorAll('.tab-btn').forEach((b, i)   => b.classList.toggle('active', i === idx));
  }}
</script>

</body>
</html>"""

    report_path = save_path.with_suffix(".html")
    report_path.write_text(html, encoding="utf-8")
    print(f"  Report → {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fringe projection data collector")
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Generate and open an HTML report after capture",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="data",
        help="Directory to save files into (default: ./data)",
    )
    parser.add_argument(
        "--report", "-r",
        default=None,
        metavar="NPZ_FILE",
        help="Generate a report from an existing .npz file without capturing",
    )
    args = parser.parse_args()

    # --- Report-only mode (no capture needed) ---
    if args.report:
        npz_path = Path(args.report).resolve()
        d = np.load(npz_path, allow_pickle=True)
        data = {
            "colors":   d["colors"],
            "depths":   d["depths"],
            "patterns": list(d["patterns"]),
        }
        report_path = generate_report(data, npz_path)
        webbrowser.open(report_path.resolve().as_uri())
        return

    # --- Normal capture mode ---
    print("=== Starting capture session ===")
    data = capture_projections_single(config.PROJECTOR_RES, config.PHASES_DEG)

    if not data["patterns"]:
        print("No patterns captured successfully — nothing saved.")
        return

    npz_path = save_capture(data, output_dir=args.output_dir)

    if args.visualize:
        report_path = generate_report(data, npz_path)
        webbrowser.open(report_path.resolve().as_uri())


if __name__ == "__main__":
    main()