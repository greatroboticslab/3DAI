import os

# ========================== CONFIG ==========================
PROJECTOR_RES = (1140, 912)           # Your projector resolution
SECOND_SCREEN_X = 1920                # X offset of your second monitor

# === CALIBRATION PLATES (in mm) ===
KNOWN_THICKNESSES_MM = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]   # ← change to your actual gauges

# Kinect color ROI — adjust once by eye (where the projector pattern fully appears)
ROI_Y, ROI_X = slice(200, 725), slice(250, 1050)   

# Fringe parameters (tune once for your setup)
NUM_FRINGES = 24          # How many vertical fringes across projector width → ~50–60 px per fringe is ideal
N_PHASES = 9             # Number of evenly spaced fringe projections
PHASES_DEG = [i * 360 / N_PHASES for i in range(N_PHASES)] 

CHANNEL_OF_INTEREST = 0b111  # White

# SETTLE_TIME: Time in seconds to wait after projecting before capturing
#   Allows projector to stabilize and camera to adjust (if using auto-exposure)
#   Increase if you see motion blur or inconsistent captures
SETTLE_TIME = 0.2  # seconds

# Number of Kinect color frames to capture and average together
# Higher values = less noise / better quality but longer capture time
# Typical good values: 3–15
NUMBER_OF_CAPTURES = 11 # not used for data collection