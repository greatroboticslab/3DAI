import os
import config
import fpp_tools as fpp
import numpy as np
from glob import glob

# ------------------- Remove existing patterns -------------------
for old_file in glob(f"{config.PATTERN_DIR}/*.png"):
        try:
            os.remove(old_file)
        except:
            pass
print(f"   Cleared old images in {config.PATTERN_DIR}")

# ------------------- Generate Patterns -------------------
print("Generating patterns using fpp_tools.generate_fringe_patterns() ...")
#sinusoidal fringes
phases_rad = np.deg2rad(config.PHASES_DEG)
fpp.generate_fringe_patterns(
    Nx=config.PROJECTOR_RES[1],
    Ny=config.PROJECTOR_RES[0],
    phases=phases_rad,
    num_fringes=config.NUM_FRINGES,
    # vertical fringes
    gamma=1.0,
    filebase=f"{config.PATTERN_DIR}/fringe_"
)