import base64
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.register_scan_folder import collect_artifacts


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "data"
        scan = root / "scan_test" / "calib" / "breadboard"
        scan.mkdir(parents=True)

        np.savez_compressed(
            scan / "scan.npz",
            freqs=np.array([1, 6, 24]),
            phases=np.array([0.0, 1.0]),
            gray_0=np.zeros((2, 3, 2), dtype=np.uint8),
        )
        (scan / "white.png").write_bytes(PNG_1X1)
        (scan / "height_mm.png").write_bytes(PNG_1X1)
        (scan / "contrast.png").write_bytes(PNG_1X1)
        (scan / "calibration_temporal.txt").write_text("1.0\n0.0\n0.0\n", encoding="utf-8")
        (scan / "notes.md").write_text("ignored", encoding="utf-8")

        old_root = os.environ.get("ARTIFACT_ROOT")
        old_cwd = os.getcwd()
        os.environ["ARTIFACT_ROOT"] = str(root)
        try:
            artifacts, skipped = collect_artifacts(Path("scan_test/calib/breadboard"))
            os.chdir(root.parent)
            artifacts_from_data_path, _ = collect_artifacts(Path("data/scan_test/calib/breadboard"))
        finally:
            os.chdir(old_cwd)
            if old_root is None:
                os.environ.pop("ARTIFACT_ROOT", None)
            else:
                os.environ["ARTIFACT_ROOT"] = old_root

        roles = {(a["category"], a["role"], a["file_path"]) for a in artifacts}
        assert ("fringe", "fringe_stack_npz", "scan_test/calib/breadboard/scan.npz") in roles
        assert ("fringe", "white_frame", "scan_test/calib/breadboard/white.png") in roles
        assert ("fusion", "height_map_png", "scan_test/calib/breadboard/height_mm.png") in roles
        assert ("fusion", "contrast_png", "scan_test/calib/breadboard/contrast.png") in roles
        assert ("calibration", "calibration_coefficients_txt", "scan_test/calib/breadboard/calibration_temporal.txt") in roles
        assert any("notes.md" in item for item in skipped)
        assert artifacts_from_data_path == artifacts

    print("OK: register_scan_folder classification and path handling")


if __name__ == "__main__":
    main()
