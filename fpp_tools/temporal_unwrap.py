from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .fpp_tools import estimate_phi_N_uniform_frames


@dataclass(frozen=True)
class Footprint:
    row0: int
    row1: int
    col0: int
    col1: int
    mask: np.ndarray


@dataclass(frozen=True)
class DeltaPhaseResult:
    delta: np.ndarray
    reliable: np.ndarray
    contrast: np.ndarray
    footprint: Footprint


def load_multifrequency_npz(path: str | Path) -> tuple[np.ndarray, list[np.ndarray]]:
    """Load a capture_multifreq.py scan.npz file or its parent directory."""
    scan_path = Path(path)
    if scan_path.is_dir():
        scan_path = scan_path / "scan.npz"

    with np.load(scan_path) as scan:
        freqs = np.asarray(scan["freqs"], dtype=float)
        stacks = [scan[f"gray_{i}"].astype(np.float32) for i in range(len(freqs))]

    return freqs, stacks


def denoise_wrapped_phase(phi: np.ndarray, kernel_size: int = 1) -> np.ndarray:
    """Smooth a wrapped phase image in the complex domain."""
    if kernel_size <= 1:
        return np.asarray(phi)
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    z = np.exp(1j * phi)
    try:
        import cv2

        real = cv2.GaussianBlur(z.real.astype(np.float32), (kernel_size, kernel_size), 0)
        imag = cv2.GaussianBlur(z.imag.astype(np.float32), (kernel_size, kernel_size), 0)
        return np.arctan2(imag, real)
    except ImportError:
        pass

    real = _box_mean(z.real.astype(np.float32), kernel_size)
    imag = _box_mean(z.imag.astype(np.float32), kernel_size)
    return np.arctan2(imag, real)


def estimate_temporal_absolute_phase(
    stacks: Iterable[np.ndarray],
    freqs: Iterable[float],
    *,
    denoise_kernel: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate absolute phase from low-to-high frequency phase-shift stacks.

    The lowest frequency is spatially unwrapped first. Each higher frequency is
    then snapped to the previous absolute phase per pixel, preserving the
    high-frequency precision while avoiding whole-fringe order jumps.
    """
    # estimate_phi_N_uniform_frames mutates its input accumulator, so copy here.
    stack_list = [np.array(stack, dtype=np.float32, copy=True) for stack in stacks]
    freq_arr = np.asarray(list(freqs), dtype=float)
    if len(stack_list) != len(freq_arr):
        raise ValueError("stacks and freqs must have the same length")
    if not stack_list:
        raise ValueError("at least one phase stack is required")
    if np.any(freq_arr <= 0):
        raise ValueError("all frequencies must be positive")

    wrapped_phases = []
    contrast = None
    for index, stack in enumerate(stack_list):
        if stack.ndim != 3:
            raise ValueError("each phase stack must have shape (height, width, phases)")
        phi, stack_contrast, _ = estimate_phi_N_uniform_frames(stack)
        wrapped_phases.append(denoise_wrapped_phase(phi, denoise_kernel))
        if index == len(stack_list) - 1:
            contrast = stack_contrast

    absolute = np.unwrap(np.unwrap(wrapped_phases[0], axis=1), axis=0)
    for index in range(1, len(freq_arr)):
        ratio = freq_arr[index] / freq_arr[index - 1]
        wrapped = wrapped_phases[index]
        absolute = wrapped + 2.0 * np.pi * np.round(
            (ratio * absolute - wrapped) / (2.0 * np.pi)
        )

    return absolute, np.asarray(contrast)


def footprint_from_mask(mask: np.ndarray) -> Footprint:
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.ndim != 2:
        raise ValueError("footprint mask must be 2D")
    rows = np.where(mask_bool.any(axis=1))[0]
    cols = np.where(mask_bool.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        raise ValueError("footprint mask is empty")
    return Footprint(
        row0=int(rows.min()),
        row1=int(rows.max()) + 1,
        col0=int(cols.min()),
        col1=int(cols.max()) + 1,
        mask=mask_bool,
    )


def footprint_from_amplitude(
    stack: np.ndarray,
    *,
    threshold_fraction: float = 0.08,
) -> Footprint:
    """Build a projector footprint from phase-stack modulation amplitude."""
    if not 0.0 < threshold_fraction < 1.0:
        raise ValueError("threshold_fraction must be between 0 and 1")
    amplitude = np.asarray(stack, dtype=np.float32).std(axis=2)
    try:
        return _opencv_footprint_from_amplitude(amplitude)
    except ImportError:
        pass

    threshold = float(amplitude.max()) * threshold_fraction
    return footprint_from_mask(amplitude > threshold)


def temporal_delta_phase(
    ref_stacks: Iterable[np.ndarray],
    obj_stacks: Iterable[np.ndarray],
    freqs: Iterable[float],
    *,
    footprint: Footprint | np.ndarray | None = None,
    contrast_floor_fraction: float = 0.08,
    denoise_kernel: int = 1,
    median_kernel: int = 5,
    level: bool = True,
) -> DeltaPhaseResult:
    """Reference-subtracted temporal-unwrapped phase inside the projector footprint."""
    ref_list = [np.asarray(stack, dtype=np.float32) for stack in ref_stacks]
    obj_list = [np.asarray(stack, dtype=np.float32) for stack in obj_stacks]
    if len(ref_list) != len(obj_list):
        raise ValueError("ref_stacks and obj_stacks must have the same length")
    if not ref_list:
        raise ValueError("at least one phase stack is required")
    if any(ref.shape != obj.shape for ref, obj in zip(ref_list, obj_list)):
        raise ValueError("reference and object stacks must have matching shapes")

    if footprint is None:
        fp = footprint_from_amplitude(ref_list[-1])
    elif isinstance(footprint, Footprint):
        fp = footprint
    else:
        fp = footprint_from_mask(footprint)

    row_slice = slice(fp.row0, fp.row1)
    col_slice = slice(fp.col0, fp.col1)
    ref_crop = [stack[row_slice, col_slice] for stack in ref_list]
    obj_crop = [stack[row_slice, col_slice] for stack in obj_list]
    mask_crop = fp.mask[row_slice, col_slice]

    phi_ref, _ = estimate_temporal_absolute_phase(
        ref_crop,
        freqs,
        denoise_kernel=denoise_kernel,
    )
    phi_obj, contrast = estimate_temporal_absolute_phase(
        obj_crop,
        freqs,
        denoise_kernel=denoise_kernel,
    )

    delta = median_filter2d((phi_obj - phi_ref).astype(np.float32), median_kernel)
    contrast_floor = float(contrast.max()) * contrast_floor_fraction
    reliable = mask_crop & (contrast > contrast_floor)
    if not reliable.any():
        raise ValueError("no reliable pixels passed the contrast threshold")
    if level:
        delta = delta - float(np.median(delta[reliable]))

    return DeltaPhaseResult(delta=delta, reliable=reliable, contrast=contrast, footprint=fp)


def fit_height_curve(dphi: Iterable[float], height_mm: Iterable[float]) -> np.ndarray:
    """
    Fit height_mm = a*dphi^2 + b*dphi + c.

    With two or more nonzero heights, this returns a quadratic through the
    supplied points. With one nonzero height, it falls back to a line through
    the origin, matching the prototype calibration script's rough mode.
    """
    dphi_arr = np.asarray(list(dphi), dtype=float)
    height_arr = np.asarray(list(height_mm), dtype=float)
    if dphi_arr.shape != height_arr.shape:
        raise ValueError("dphi and height_mm must have the same shape")
    if dphi_arr.ndim != 1:
        raise ValueError("dphi and height_mm must be 1D")

    nonzero = np.abs(height_arr) > 0.1
    count = int(nonzero.sum())
    if count >= 2:
        return np.polyfit(dphi_arr, height_arr, deg=2)
    if count == 1:
        index = int(np.where(nonzero)[0][0])
        if abs(dphi_arr[index]) < 1e-12:
            raise ValueError("cannot fit a line from zero dphi")
        return np.array([0.0, height_arr[index] / dphi_arr[index], 0.0])
    raise ValueError("at least one nonzero height is required")


def median_filter2d(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    if kernel_size <= 1:
        return np.asarray(image)
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    try:
        from scipy.signal import medfilt2d

        return medfilt2d(image, kernel_size).astype(np.float32)
    except ImportError:
        pass

    pad = kernel_size // 2
    padded = np.pad(image, pad_width=pad, mode="edge")
    windows = []
    for row in range(kernel_size):
        for col in range(kernel_size):
            windows.append(padded[row:row + image.shape[0], col:col + image.shape[1]])
    return np.median(np.stack(windows, axis=0), axis=0).astype(np.float32)


def _box_mean(image: np.ndarray, kernel_size: int) -> np.ndarray:
    pad = kernel_size // 2
    padded = np.pad(image, pad_width=pad, mode="reflect")
    acc = np.zeros_like(image, dtype=np.float32)
    for row in range(kernel_size):
        for col in range(kernel_size):
            acc += padded[row:row + image.shape[0], col:col + image.shape[1]]
    return acc / float(kernel_size * kernel_size)


def _opencv_footprint_from_amplitude(amplitude: np.ndarray) -> Footprint:
    import cv2

    amp8 = np.clip(
        amplitude / max(float(amplitude.max()), 1.0) * 255.0,
        0,
        255,
    ).astype(np.uint8)
    _, lit = cv2.threshold(amp8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lit = cv2.morphologyEx(lit, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(lit)
    if count > 1:
        component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = labels == component
    else:
        mask = lit > 0
    return footprint_from_mask(mask)
