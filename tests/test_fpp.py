"""
Unit test for the fringe-projection phase estimator. Pure math, no hardware -
synthesizes a known phase ramp, builds the N phase-shifted frames, and checks
the estimator recovers the phase (and contrast). Run: python tests/test_fpp.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpp_tools.fpp_tools import estimate_phi_N_uniform_frames


def main():
    H, W, N = 64, 96, 8
    bias, amp = 0.5, 0.4
    _, x = np.indices((H, W))
    true_phi = 2.0 * np.pi * 3.0 * x / W          # 3 fringes across the width

    # I_n = bias + amp*cos(phi - 2*pi*n/N): this convention makes the estimator
    # return phi directly (see derivation in the FPP docs).
    stack = np.stack(
        [bias + amp * np.cos(true_phi - 2.0 * np.pi * n / N) for n in range(N)],
        axis=-1,
    ).astype(np.float32)

    phi, contrast, est_bias = estimate_phi_N_uniform_frames(stack)

    # Compare wrapped phases via the complex domain (handles the +/-pi seam).
    wrapped_true = np.angle(np.exp(1j * true_phi))
    phase_err = np.abs(np.angle(np.exp(1j * (phi - wrapped_true))))

    assert phase_err.max() < 1e-3, f"phase error too high: {phase_err.max()}"
    assert abs(float(contrast.mean()) - amp) < 1e-3, f"contrast off: {contrast.mean()} vs {amp}"
    assert abs(float(est_bias.mean()) - bias) < 1e-3, f"bias off: {est_bias.mean()} vs {bias}"

    print(f"OK: max phase error {phase_err.max():.2e}, "
          f"contrast {contrast.mean():.3f} (expected {amp}), "
          f"bias {est_bias.mean():.3f} (expected {bias})")


if __name__ == "__main__":
    main()
