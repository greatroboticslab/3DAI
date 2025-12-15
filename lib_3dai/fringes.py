import numpy as np
import fpp_tools as fpp
from glob import glob
import os

# ------------------- Generate All Patterns -------------------
def generate_patterns(resolution,dir,phases_deg,num_fringes):

    print("Clearing old patterns")

    for old_file in glob(f"{dir}/*.png"):
        try:
            os.remove(old_file)
        except:
            pass
    print(f"   Cleared old images in {dir}")

    print("Generating patterns using fpp_tools.generate_fringe_patterns() ...")


    #sinusoidal fringes
    phases_rad = np.deg2rad(phases_deg)
    fpp.generate_fringe_patterns(
        Nx=resolution[1],     
        Ny=resolution[0],      
        phases=phases_rad,
        num_fringes=num_fringes,
        gamma=1.0,
        filebase=f"{dir}/fringe_"   # ← this creates fringe_000.png, fringe_090.png, etc.
    )


def get_patterns(dir):
    all_files = os.listdir(dir)

    # Add only the actual fringe files that exist
    patterns = sorted([f for f in all_files if f.endswith(".png")])
    
    for p in patterns:
        print("   ", p)
    print()

    return patterns