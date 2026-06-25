import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpp_tools.temporal_unwrap import (
    fit_height_curve,
    footprint_from_mask,
    estimate_temporal_absolute_phase,
    temporal_delta_phase,
)


def synthesize_stacks(freqs, shift_px, height=40, width=96, phases=8):
    _, x = np.indices((height, width), dtype=np.float32)
    shift = np.asarray(shift_px, dtype=np.float32)
    stacks = []
    for freq in freqs:
        phi = 2.0 * np.pi * float(freq) * (x + shift) / float(width)
        stack = np.stack(
            [
                0.5 + 0.4 * np.cos(phi - 2.0 * np.pi * n / phases)
                for n in range(phases)
            ],
            axis=2,
        ).astype(np.float32)
        stacks.append(stack)
    return stacks


def main():
    freqs = np.array([1, 6, 24], dtype=float)
    height, width = 40, 96
    shift_px = 0.35
    expected_high_phase_shift = 2.0 * np.pi * freqs[-1] * shift_px / width

    ref = synthesize_stacks(freqs, np.zeros((height, width), dtype=np.float32))
    shifted = synthesize_stacks(freqs, np.full((height, width), shift_px, dtype=np.float32))
    phi_ref, _ = estimate_temporal_absolute_phase(ref, freqs)
    phi_shifted, _ = estimate_temporal_absolute_phase(shifted, freqs)
    assert np.allclose(
        np.median(phi_shifted - phi_ref),
        expected_high_phase_shift,
        atol=1e-4,
    )

    local_shift = np.zeros((height, width), dtype=np.float32)
    local_shift[12:28, 30:66] = shift_px
    obj = synthesize_stacks(freqs, local_shift)
    footprint = footprint_from_mask(np.ones((height, width), dtype=bool))
    result = temporal_delta_phase(ref, obj, freqs, footprint=footprint)

    object_delta = np.median(result.delta[12:28, 30:66])
    panel = result.delta.copy()
    panel[12:28, 30:66] = np.nan
    panel_delta = np.nanmedian(panel)

    assert result.reliable.mean() > 0.99
    assert abs(panel_delta) < 1e-4
    assert np.isclose(object_delta, expected_high_phase_shift, atol=1e-4)

    ref_before = [stack.copy() for stack in ref]
    first, _ = estimate_temporal_absolute_phase(ref, freqs)
    second, _ = estimate_temporal_absolute_phase(ref, freqs)
    for before, after in zip(ref_before, ref):
        assert np.array_equal(before, after), "input stacks must not be mutated"
    assert np.allclose(first, second)

    coeffs = fit_height_curve([0.0, expected_high_phase_shift], [0.0, 9.0])
    assert np.allclose(coeffs, [0.0, 9.0 / expected_high_phase_shift, 0.0])

    print("OK: temporal unwrap, reference subtraction, and height fit")


if __name__ == "__main__":
    main()
