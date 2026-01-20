import os

# ========================== CONFIG ==========================
PROJECTOR_RES = (1140, 912)           # Your projector resolution
SECOND_SCREEN_X = 1920                # X offset of your second monitor
CAPTURE_DIR = "captures_kinect"
PATTERN_DIR = "patterns_auto"         # Will be created automatically
os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(PATTERN_DIR, exist_ok=True)

# === CALIBRATION PLATES (in mm) ===
KNOWN_THICKNESSES_MM = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]   # ← change to your actual gauges

# Kinect color ROI — adjust once by eye (where the projector pattern fully appears)
ROI_Y, ROI_X = slice(200, 725), slice(250, 1050)   

# Fringe parameters (tune once for your setup)
NUM_FRINGES = 24          # How many vertical fringes across projector width → ~50–60 px per fringe is ideal
N_PHASES = 120             # Number of evenly spaced fringe projections
PHASES_DEG = [i * 360 / N_PHASES for i in range(N_PHASES)] 

# =========================================================