import os
import config
import fpp_tools as fpp
import numpy as np
from glob import glob
from PIL import Image

# ------------------- Ask user about BMP generation -------------------
gen_bmp = input("Generate BMP versions for LightCrafter? [y/N]: ").strip().lower() == "y"

bmp_dir = f"{config.PATTERN_DIR}_bmp"

# ------------------- Remove existing patterns -------------------
for old_file in glob(f"{config.PATTERN_DIR}/*.png"):
    try:
        os.remove(old_file)
    except:
        pass

if gen_bmp:
    os.makedirs(bmp_dir, exist_ok=True)
    for old_file in glob(f"{bmp_dir}/*.bmp"):
        try:
            os.remove(old_file)
        except:
            pass

print(f"   Cleared old PNG images in {config.PATTERN_DIR}")
if gen_bmp:
    print(f"   Cleared old BMP images in {bmp_dir}")

# ------------------- Generate Patterns -------------------
print("Generating patterns using fpp_tools.generate_fringe_patterns() ...")

phases_rad = np.deg2rad(config.PHASES_DEG)

imagestack = fpp.generate_fringe_patterns(
    Nx=config.PROJECTOR_RES[1],
    Ny=config.PROJECTOR_RES[0],
    phases=phases_rad,
    num_fringes=config.NUM_FRINGES,
    gamma=1.0,
    filebase=f"{config.PATTERN_DIR}/fringe_"
)

# ------------------- Save BMP versions (optional) -------------------
if gen_bmp:
    print("Saving BMP versions ...")

    for i, phase in enumerate(phases_rad):
        img = imagestack[:, :, i]

        # Match fpp_tools quantization exactly
        img_8bit = np.uint8(np.rint(img))

        bmp = Image.fromarray(img_8bit, mode="L")
        filename = f"{bmp_dir}/fringe_{int(np.rad2deg(phase)):03}.bmp"
        bmp.save(filename)

    print(f"   BMP images saved to {bmp_dir}")