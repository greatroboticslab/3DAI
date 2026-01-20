import os
from glob import glob
from .fringes import get_patterns
from .projector import project_image
from .kinect import capture_kinect
import cv2
import time
import config
import numpy as np


# Assuming these exist in your config or as globals/constants
CHANNEL_OF_INTEREST = 1          # 0 = Blue, 1 = Green, 2 = Red
SETTLE_TIME = 0.4
# You can also make it a parameter or read from config

def generate_and_estimate_gamma(
        capture_dir="gamma_calibration",
        num_gray_levels=9,           # 3, 5, 7, 9, 17… — odd number is nice
        channel_idx=1,               # 0=B, 1=G, 2=R — usually green for fringe systems
        min_gray=0.02,               # avoid pure 0 (camera black offset)
        max_gray=0.98,               # avoid pure 255 (projector saturation
        roi_y=config.ROI_Y,          # reuse your existing ROI if defined
        roi_x=config.ROI_X,
        black_filename="black.png",
        white_filename="white.png"
    ):
        """
        Projects black, white + multiple gray levels → captures responses
        → estimates gamma (simple power-law fit per pixel or globally)
        
        Returns:
            global_gamma : float
            gamma_map    : np.ndarray (h,w) or None if fit failed
        """

        # Get projector resolution from config (assuming you have these)
        try:
            proj_w = config.PROJECTOR_RES[0]
            proj_h = config.PROJECTOR_RES[1]
        except AttributeError:
            # fallback — capture one image and use its size (not perfect but works)
            print("Projector resolution not in config → using captured size")
            proj_w, proj_h = None, None
        
        # ── Generate images ─────────────────────────────────────────────
        images = {}
        
        # Black
        black = np.zeros((proj_h, proj_w, 3), dtype=np.uint8) if proj_h else None
        images[black_filename] = black
        
        # White
        white = np.full((proj_h, proj_w, 3), 255, dtype=np.uint8) if proj_h else None
        images[white_filename] = white
        
        # Gray levels (linear in display value)
        gray_values = np.linspace(min_gray, max_gray, num_gray_levels)
        gray_int = np.round(gray_values * 255).astype(np.uint8)
        
        for i, val in enumerate(gray_int):
            gray_img = np.full((proj_h, proj_w, 3), val, dtype=np.uint8) if proj_h else None
            name = f"gray_{val:03d}.png"
            images[name] = gray_img
        
        captured_means = {}     # display_value → mean intensity (channel)
        
        print(f"Starting gamma calibration – {len(images)} patterns")
        
        for name, img_proj in images.items():
            if img_proj is None:
                # If we don't know resolution yet, project a dummy and get size from capture
                if name == black_filename:
                    dummy = np.zeros((480, 640, 3), np.uint8)  # arbitrary small
                    project_image(dummy, config.SECOND_SCREEN_X)
                    time.sleep(SETTLE_TIME)
                    _, sample = capture_kinect("temp", capture_dir, roi_y, roi_x)
                    proj_h, proj_w = sample.shape[:2]
                    # Now recreate all images with correct size
                    print(f"Detected projector area size: {proj_w}×{proj_h}")
                    # Re-create all images with correct size
                    images[black_filename] = np.zeros((proj_h, proj_w, 3), np.uint8)
                    images[white_filename] = np.full((proj_h, proj_w, 3), 255, np.uint8)
                    for i, val in enumerate(gray_int):
                        name_g = f"gray_{val:03d}.png"
                        images[name_g] = np.full((proj_h, proj_w, 3), val, np.uint8)
                    img_proj = images[name]  # now correct size
            
            print(f"  Projecting {name} ...")
            project_image(img_proj, config.SECOND_SCREEN_X)
            time.sleep(SETTLE_TIME)
            
            # Capture
            _, live = capture_kinect(f"calib_{name}", capture_dir, roi_y, roi_x)
            if live is None or live.size == 0:
                print(f"  Capture failed for {name}")
                continue
            
            # Extract channel of interest
            ch = live[:, :, channel_idx].astype(np.float32)
            
            # Save mean for quick global estimate
            mean_val = np.mean(ch)
            captured_means[name] = mean_val
            
            # Keep map for per-pixel gamma (memory heavy — optional)
            # captured_maps[name] = ch.copy()
            
            print(f"    → mean({channel_idx}) = {mean_val:.1f}")
        
        # ── Simple global gamma estimation ──────────────────────────────
        # We use gray levels only (black/white used as anchors)
        display_vals = []
        captured_vals = []
        
        for name in images:
            if "gray_" in name:
                try:
                    val = int(name.split('_')[1].split('.')[0])
                    display_vals.append(val / 255.0)
                    captured_vals.append(captured_means.get(name, np.nan))
                except:
                    pass
        
        display_vals = np.array(display_vals)
        captured_vals = np.array(captured_vals)
        
        # Normalize captured to [0,1] roughly using black/white
        black_mean = captured_means.get(black_filename, 0)
        white_mean = captured_means.get(white_filename, 255)
        if white_mean <= black_mean + 10:
            print("Warning: very low dynamic range detected!")
            return 2.2, None
        
        norm_captured = (captured_vals - black_mean) / (white_mean - black_mean + 1e-6)
        norm_captured = np.clip(norm_captured, 0, 1)
        
        # Fit gamma:  captured = display ^ gamma
        # → log(captured) = gamma * log(display) + 0
        valid = (display_vals > 0.01) & (norm_captured > 0.01)
        if np.sum(valid) < 3:
            print("Too few valid points for gamma fit → returning default 2.2")
            return 2.2, None
        
        log_disp = np.log(display_vals[valid])
        log_cap  = np.log(norm_captured[valid])
        
        gamma, _ = np.polyfit(log_disp, log_cap, 1)   # slope = gamma
        
        print(f"\nEstimated global gamma = {gamma:.3f}")
        print(f"(based on {np.sum(valid)} gray levels)")
        
        return float(gamma)

def apply_single_channel_and_gamma_correction(img_bgr, channel_idx=1, gamma=2.2):
    """
    Extract one channel → apply gamma correction → return float32 image
    """
    # Extract desired channel (BGR order)
    single_channel = img_bgr[:, :, channel_idx].astype(np.float32)
    
    # Normalize to [0,1]
    single_channel /= 255.0
    
    # Apply gamma correction:  out = in^(1/gamma)
    # (most projectors have gamma > 1 → images look darker → we brighten them)
    corrected = np.power(single_channel, 1.0 / gamma)
    
    # Back to 0–255 range (optional — depends on your later processing)
    corrected = (corrected * 255.0).clip(0, 255).astype(np.uint8)
    
    return corrected


def capture_projections(dir):
    os.makedirs(dir, exist_ok=True)
    
    # SAFETY: Remove old captures
    for old_file in glob(f"{dir}/*.png"):
        try:
            os.remove(old_file)
        except:
            pass
    print(f"   Cleared old images in {dir}")

    gamma = generate_and_estimate_gamma()

    patterns = get_patterns(config.PATTERN_DIR)   # assuming this returns filenames 

    for pat in patterns:
        img_path = f"{config.PATTERN_DIR}/{pat}"
        img_proj = cv2.imread(img_path)
        if img_proj is None:
            raise FileNotFoundError(f"Pattern not found: {img_path}")
        
        # Project the original RGB image
        project_image(img_proj, config.SECOND_SCREEN_X)
        time.sleep(SETTLE_TIME)  # let projection stabilize
        
        # Capture from Kinect
        _, live_img = capture_kinect("temp", dir, config.ROI_Y, config.ROI_X)
        
        # live_img is most likely BGR (OpenCV default)
        if live_img is None or live_img.size == 0:
            print(f"Capture failed for {pat}")
            continue
        
        # ── Preprocess: single channel + gamma correction ──
        processed = apply_single_channel_and_gamma_correction(
            live_img,
            channel_idx=CHANNEL_OF_INTEREST,
            gamma=gamma
        )
        
        # Simple clean name
        clean_name = f"{pat}" 
        final_path = os.path.join(dir, clean_name)
        
        # Save the processed image (as PNG for lossless quality)
        cv2.imwrite(final_path, processed)
        print(f"   Saved processed: {final_path}")
        
        # Optional: remove the original temp file if capture_kinect still created it
        temp_files = glob(f"{dir}/cap_*.jpg")
        for tf in temp_files:
            try:
                os.remove(tf)
            except:
                pass