import base64
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import artifact_store
from artifact_store import ArtifactError, inspect_artifact, media_type_for_path, validate_role


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def expect_artifact_error(fn):
    try:
        fn()
    except ArtifactError:
        return
    raise AssertionError("Expected ArtifactError")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        png_path = root / "scan" / "preview.png"
        png_path.parent.mkdir()
        png_path.write_bytes(PNG_1X1)

        info = inspect_artifact("scan/preview.png", root=root)
        assert info["file_path"] == "scan/preview.png"
        assert info["media_type"] == "image/png"
        assert info["size_bytes"] == len(PNG_1X1)
        assert info["array_metadata"] == {}

        npz_path = root / "scan" / "raw.npz"
        np.savez_compressed(
            npz_path,
            color=np.zeros((2, 3, 4), dtype=np.uint8),
            depth=np.zeros((2, 3), dtype=np.uint16),
        )
        npz_info = inspect_artifact("scan/raw.npz", root=root)
        arrays = npz_info["array_metadata"]["arrays"]
        assert arrays["color"]["shape"] == [2, 3, 4]
        assert arrays["color"]["dtype"] == "|u1"
        assert arrays["depth"]["shape"] == [2, 3]
        assert arrays["depth"]["dtype"] == "<u2"

        zero_path = root / "scan" / "empty.png"
        zero_path.write_bytes(b"")

        outside = root.parent / "outside.png"
        outside.write_bytes(PNG_1X1)

        expect_artifact_error(lambda: inspect_artifact("../outside.png", root=root))
        expect_artifact_error(lambda: inspect_artifact("scan/missing.png", root=root))
        expect_artifact_error(lambda: inspect_artifact("scan/empty.png", root=root))
        expect_artifact_error(lambda: media_type_for_path(root / "bad.exe"))
        expect_artifact_error(lambda: validate_role("string"))

        many_arrays = root / "scan" / "too_many_arrays.npz"
        np.savez_compressed(
            many_arrays,
            first=np.zeros((1,), dtype=np.uint8),
            second=np.zeros((1,), dtype=np.uint8),
        )
        old_limit = artifact_store.MAX_NPZ_ARRAYS
        artifact_store.MAX_NPZ_ARRAYS = 1
        try:
            expect_artifact_error(lambda: inspect_artifact("scan/too_many_arrays.npz", root=root))
        finally:
            artifact_store.MAX_NPZ_ARRAYS = old_limit

    print("OK: artifact path validation, allowlist, zero-byte rejection, and npz headers")


if __name__ == "__main__":
    main()
