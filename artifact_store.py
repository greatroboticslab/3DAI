import ast
import mimetypes
import os
import struct
import zipfile
from pathlib import Path
from typing import Any


ALLOWED_CATEGORIES = {"kinect", "fringe", "fusion", "calibration"}

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".npz",
    ".json",
    ".txt",
}

MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".npz": "application/x-npz",
    ".json": "application/json",
    ".txt": "text/plain",
}

MAX_NPZ_ARRAYS = int(os.getenv("MAX_NPZ_ARRAYS", "256"))
MAX_NPY_HEADER_BYTES = int(os.getenv("MAX_NPY_HEADER_BYTES", "65536"))


class ArtifactError(ValueError):
    """Raised when an artifact path or file is not safe to expose."""


def get_artifact_root() -> Path:
    return Path(os.getenv("ARTIFACT_ROOT", "./data")).resolve()


def validate_category(category: str) -> str:
    value = category.strip().lower()
    if value not in ALLOWED_CATEGORIES:
        allowed = ", ".join(sorted(ALLOWED_CATEGORIES))
        raise ArtifactError(f"Unsupported artifact category '{category}'. Allowed: {allowed}.")
    return value


def validate_role(role: str) -> str:
    value = role.strip()
    if not value:
        raise ArtifactError("Artifact role must be a nonempty string.")
    if value.lower() == "string":
        raise ArtifactError("Artifact role looks like Swagger placeholder text.")
    return value


def resolve_artifact_path(file_path: str, root: Path | None = None) -> tuple[Path, str]:
    if not file_path or not file_path.strip():
        raise ArtifactError("Artifact file_path must be a nonempty string.")

    artifact_root = (root or get_artifact_root()).resolve()
    normalized = file_path.strip().replace("\\", "/")
    candidate = Path(normalized)
    resolved = candidate.resolve() if candidate.is_absolute() else (artifact_root / candidate).resolve()

    try:
        relative = resolved.relative_to(artifact_root)
    except ValueError as exc:
        raise ArtifactError("Artifact path must stay inside ARTIFACT_ROOT.") from exc

    return resolved, relative.as_posix()


def media_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ArtifactError(f"Unsupported artifact extension '{suffix}'. Allowed: {allowed}.")
    return MEDIA_TYPES.get(suffix) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def inspect_artifact(file_path: str, root: Path | None = None) -> dict[str, Any]:
    resolved, relative_path = resolve_artifact_path(file_path, root=root)
    media_type = media_type_for_path(resolved)

    if not resolved.exists():
        raise ArtifactError(f"Artifact file does not exist: {relative_path}")
    if not resolved.is_file():
        raise ArtifactError(f"Artifact path is not a file: {relative_path}")

    size_bytes = resolved.stat().st_size
    if size_bytes <= 0:
        raise ArtifactError(f"Artifact file is empty: {relative_path}")

    array_metadata: dict[str, Any] = {}
    if resolved.suffix.lower() == ".npz":
        array_metadata = read_npz_header_metadata(resolved)

    return {
        "file_path": relative_path,
        "media_type": media_type,
        "size_bytes": size_bytes,
        "array_metadata": array_metadata,
    }


def read_npz_header_metadata(path: Path) -> dict[str, Any]:
    arrays: dict[str, Any] = {}

    try:
        with zipfile.ZipFile(path) as archive:
            array_count = 0
            for member_name in archive.namelist():
                if not member_name.endswith(".npy"):
                    continue
                array_count += 1
                if array_count > MAX_NPZ_ARRAYS:
                    raise ArtifactError(f".npz artifact has more than {MAX_NPZ_ARRAYS} arrays.")
                with archive.open(member_name) as member:
                    arrays[member_name[:-4]] = _read_npy_header(member)
    except zipfile.BadZipFile as exc:
        raise ArtifactError(f"Invalid .npz artifact: {path.name}") from exc

    return {"arrays": arrays}


def _read_npy_header(member) -> dict[str, Any]:
    magic = member.read(6)
    if magic != b"\x93NUMPY":
        raise ArtifactError("Invalid .npy member inside .npz artifact.")

    version = member.read(2)
    if len(version) != 2:
        raise ArtifactError("Truncated .npy header inside .npz artifact.")

    major = version[0]
    if major == 1:
        header_len_raw = member.read(2)
        if len(header_len_raw) != 2:
            raise ArtifactError("Truncated .npy header length inside .npz artifact.")
        header_len = struct.unpack("<H", header_len_raw)[0]
        encoding = "latin1"
    elif major in (2, 3):
        header_len_raw = member.read(4)
        if len(header_len_raw) != 4:
            raise ArtifactError("Truncated .npy header length inside .npz artifact.")
        header_len = struct.unpack("<I", header_len_raw)[0]
        encoding = "utf-8" if major == 3 else "latin1"
    else:
        raise ArtifactError(f"Unsupported .npy format version {major}.")

    if header_len > MAX_NPY_HEADER_BYTES:
        raise ArtifactError(f".npy header exceeds {MAX_NPY_HEADER_BYTES} bytes.")

    header_raw = member.read(header_len)
    if len(header_raw) != header_len:
        raise ArtifactError("Truncated .npy header inside .npz artifact.")

    try:
        header = header_raw.decode(encoding)
        parsed = ast.literal_eval(header)
    except (SyntaxError, ValueError, UnicodeDecodeError) as exc:
        raise ArtifactError("Invalid .npy header inside .npz artifact.") from exc

    try:
        return {
            "dtype": parsed["descr"],
            "shape": list(parsed["shape"]),
            "fortran_order": bool(parsed["fortran_order"]),
        }
    except (KeyError, TypeError) as exc:
        raise ArtifactError("Invalid .npy metadata inside .npz artifact.") from exc
