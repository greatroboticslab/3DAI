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
N_PHASES = 16             # Number of evenly spaced fringe projections
PHASES_DEG = [i * 360 / N_PHASES for i in range(N_PHASES)] 

# =========================================================
# ─── Color Channel Selection for Fringe Projection ───────────────────────────
# CHANNEL_OF_INTEREST: Binary mask specifying which color channels to use
#   Use binary notation (0bXXX) where each bit represents a channel:
#     0b001 = Blue channel only
#     0b010 = Green channel only (recommended - best SNR for most projectors)
#     0b100 = Red channel only
#     0b011 = Blue + Green (cyan)
#     0b101 = Blue + Red (magenta)
#     0b110 = Green + Red (yellow)
#     0b111 = All channels (white light - use for maximum brightness)
#   
#   For multi-channel selections, the captured intensities are averaged.
#   Single green channel (0b010) is recommended for most structured light systems
#   due to better signal-to-noise ratio and lower chromatic aberration.
CHANNEL_OF_INTEREST = 0b111  # White

# SETTLE_TIME: Time in seconds to wait after projecting before capturing
#   Allows projector to stabilize and camera to adjust (if using auto-exposure)
#   Increase if you see motion blur or inconsistent captures
SETTLE_TIME = 1.2  # seconds

# Number of Kinect color frames to capture and average together
# Higher values = less noise / better quality but longer capture time
# Typical good values: 3–15
NUMBER_OF_CAPTURES = 15