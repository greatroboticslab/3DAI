"""fpp_tools must run under C:\KinectEnv (Python 3.9), which executes the live
height reconstruction. A 3.10+ construct (zip(strict=...)) silently killed
every height map on 2026-09-05 while the 3.14 test venv stayed green. This
runs a tiny synthetic delta-phase under that interpreter. Skips if absent."""

import os
import subprocess
import sys

import pytest

KINECT_PY = os.getenv("SCANNER_KINECT_PYTHON", r"C:\KinectEnv\Scripts\python.exe")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SNIPPET = r"""
import sys, numpy as np
sys.path.insert(0, REPO)
from fpp_tools.temporal_unwrap import temporal_delta_phase, reconstruct_height_map
h, w = 60, 100
_, x = np.indices((h, w))
lit = np.zeros((h, w)); lit[5:55, 10:90] = 1
def stack(shift=0.0):
    out = []
    for nf in (1, 6, 24):
        k = 2*np.pi*nf/w
        out.append(np.stack([np.clip(20 + 60*lit + 40*lit*np.cos(k*x + p + shift), 0, 255)
                             for p in [2*np.pi*i/8 for i in range(8)]], -1).astype(np.uint8))
    return out
ref, obj = stack(0.0), stack(0.3)
height, res = reconstruct_height_map(ref, obj, np.array([1., 6., 24.]), [0.0, -8.0, 0.0], method="wrapped")
assert height.shape == res.delta.shape
print("OK", sys.version.split()[0])
""".replace("REPO", repr(REPO))


@pytest.mark.skipif(not os.path.isfile(KINECT_PY), reason="KinectEnv interpreter not present")
def test_fpp_tools_runs_under_kinectenv():
    r = subprocess.run([KINECT_PY, "-c", SNIPPET], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-800:]
    assert r.stdout.startswith("OK 3.9"), r.stdout
