import os
from glob import glob
from .fringes import get_patterns
from .projector import project_image
from .kinect import capture_kinect
import cv2
import time
import config
import numpy as np


def parse_channel_mask(channel_mask):
    """
    Parse binary channel mask to determine which channels to use.
    
    Args:
        channel_mask: Binary mask (e.g., 0b001=Blue, 0b010=Green, 0b100=Red, 0b111=White/All)
    
    Returns:
        list of channel indices to use [0=Blue, 1=Green, 2=Red]
    """
    channels = []
    if channel_mask & 0b001:  # Blue
        channels.append(0)
    if channel_mask & 0b010:  # Green
        channels.append(1)
    if channel_mask & 0b100:  # Red
        channels.append(2)
    
    if not channels:
        raise ValueError("Invalid channel mask - at least one channel must be selected")
    
    return channels


def generate_and_estimate_gamma(
        capture_dir="gamma_calibration",
        num_gray_levels=9,
        channel_mask=0b010,  # Default to green only
        min_gray=0.02,
        max_gray=0.98,
        roi_y=config.ROI_Y,
        roi_x=config.ROI_X,
        black_filename="black.png",
        white_filename="white.png",
        smooth_kernel_size=5
    ):
    """
    Projects black, white + multiple gray levels → captures responses
    → estimates per-pixel gamma map for specified color channels
    
    Args:
        channel_mask: Binary mask for channels (0b001=B, 0b010=G, 0b100=R, 0b111=All/White)
                     Examples: 0b010 = Green only
                              0b111 = All channels (white light)
                              0b101 = Blue + Red (magenta)
    
    Returns:
        global_gamma : float (for fallback/reporting)
        gamma_map    : np.ndarray (h,w) - per-pixel gamma values
        black_offset : np.ndarray (h,w) - per-pixel black level
        white_level  : np.ndarray (h,w) - per-pixel white level
    """
    
    # Parse which channels to use
    active_channels = parse_channel_mask(channel_mask)
    print(f"Using channels: {['Blue', 'Green', 'Red'][i] for i in active_channels}")
    
    # Get projector resolution from config
    try:
        proj_w = config.PROJECTOR_RES[0]
        proj_h = config.PROJECTOR_RES[1]
    except AttributeError:
        print("Projector resolution not in config → will detect from capture")
        proj_w, proj_h = None, None
    
    # ── Generate calibration images ─────────────────────────────────────
    images = {}
    
    # Black
    black = np.zeros((proj_h, proj_w, 3), dtype=np.uint8) if proj_h else None
    images[black_filename] = black
    
    # White (based on channel mask)
    if proj_h:
        white = np.zeros((proj_h, proj_w, 3), dtype=np.uint8)
        for ch_idx in active_channels:
            white[:, :, ch_idx] = 255
    else:
        white = None
    images[white_filename] = white
    
    # Gray levels (linear in display value)
    gray_values = np.linspace(min_gray, max_gray, num_gray_levels)
    gray_int = np.round(gray_values * 255).astype(np.uint8)
    
    for i, val in enumerate(gray_int):
        if proj_h:
            gray_img = np.zeros((proj_h, proj_w, 3), dtype=np.uint8)
            for ch_idx in active_channels:
                gray_img[:, :, ch_idx] = val
        else:
            gray_img = None
        name = f"gray_{val:03d}.png"
        images[name] = gray_img
    
    # Store captured images for per-pixel fitting
    captured_maps = {}
    captured_means = {}
    
    print(f"Starting gamma calibration – {len(images)} patterns")
    
    for name, img_proj in images.items():
        if img_proj is None:
            # Auto-detect projector size from first capture
            if name == black_filename:
                dummy = np.zeros((480, 640, 3), np.uint8)
                project_image(dummy, config.SECOND_SCREEN_X)
                time.sleep(config.SETTLE_TIME)
                _, sample = capture_kinect("temp", capture_dir, roi_y, roi_x)
                proj_h, proj_w = sample.shape[:2]
                print(f"Detected projector area size: {proj_w}×{proj_h}")
                
                # Re-create all images with correct size
                images[black_filename] = np.zeros((proj_h, proj_w, 3), np.uint8)
                
                white_new = np.zeros((proj_h, proj_w, 3), dtype=np.uint8)
                for ch_idx in active_channels:
                    white_new[:, :, ch_idx] = 255
                images[white_filename] = white_new
                
                for i, val in enumerate(gray_int):
                    name_g = f"gray_{val:03d}.png"
                    gray_new = np.zeros((proj_h, proj_w, 3), dtype=np.uint8)
                    for ch_idx in active_channels:
                        gray_new[:, :, ch_idx] = val
                    images[name_g] = gray_new
                img_proj = images[name]
        
        print(f"  Projecting {name} ...")
        project_image(img_proj, config.SECOND_SCREEN_X)
        time.sleep(config.SETTLE_TIME)
        
        # Capture
        _, live = capture_kinect(f"calib_{name}", capture_dir, roi_y, roi_x)
        if live is None or live.size == 0:
            print(f"  Capture failed for {name}")
            continue
        
        # Combine channels based on mask
        # If single channel, use it directly
        # If multiple channels, average them
        if len(active_channels) == 1:
            ch = live[:, :, active_channels[0]].astype(np.float32)
        else:
            # Average the active channels
            ch = np.mean([live[:, :, idx].astype(np.float32) for idx in active_channels], axis=0)
        
        # Store the full captured map
        captured_maps[name] = ch.copy()
        
        # Also store mean for global estimation
        mean_val = np.mean(ch)
        captured_means[name] = mean_val
        
        channel_names = ', '.join([['Blue', 'Green', 'Red'][i] for i in active_channels])
        print(f"    → mean({channel_names}) = {mean_val:.1f}")
    
    # ── Per-pixel gamma estimation ──────────────────────────────────────
    
    # Get black and white reference images
    black_map = captured_maps.get(black_filename)
    white_map = captured_maps.get(white_filename)
    
    if black_map is None or white_map is None:
        print("ERROR: Black or white capture missing!")
        return 2.2, None, None, None
    
    h, w = black_map.shape
    
    # Check dynamic range and warn if poor
    mean_black = np.mean(black_map)
    mean_white = np.mean(white_map)
    dynamic_range_global = mean_white - mean_black
    
    print(f"\nDynamic range check:")
    print(f"  Black level: {mean_black:.1f}")
    print(f"  White level: {mean_white:.1f}")
    print(f"  Range: {dynamic_range_global:.1f} / 255")
    
    if dynamic_range_global < 50:
        print("  ⚠ WARNING: Very low dynamic range detected!")
        print("  Suggestions:")
        print("    - Reduce camera exposure/gain")
        print("    - Turn off room lights")
        print("    - Try different color channels")
        print("    - Check if camera sensor is saturated")
    
    # Initialize gamma map with default value
    gamma_map = np.full((h, w), 2.2, dtype=np.float32)
    
    # Collect display values and corresponding captured intensities
    display_vals = []
    for name in images:
        if "gray_" in name:
            try:
                val = int(name.split('_')[1].split('.')[0])
                display_vals.append(val / 255.0)
            except:
                pass
    
    display_vals = np.array(display_vals)
    
    # Build a 3D array: (height, width, num_gray_levels)
    captured_stack = np.zeros((h, w, len(display_vals)), dtype=np.float32)
    
    gray_idx = 0
    for name in sorted(captured_maps.keys()):
        if "gray_" in name:
            captured_stack[:, :, gray_idx] = captured_maps[name]
            gray_idx += 1
    
    # Normalize each pixel's response using its black and white levels
    dynamic_range = white_map - black_map
    
    # Avoid division by zero - mark invalid pixels
    valid_mask = dynamic_range > 10  # Threshold for minimum dynamic range
    
    # Normalize captured values to [0, 1] per pixel
    normalized_stack = np.zeros_like(captured_stack)
    for i in range(captured_stack.shape[2]):
        normalized_stack[:, :, i] = np.where(
            valid_mask,
            (captured_stack[:, :, i] - black_map) / (dynamic_range + 1e-6),
            0
        )
    
    # Clip to valid range
    normalized_stack = np.clip(normalized_stack, 0.001, 0.999)
    
    print("\nFitting per-pixel gamma (this may take a moment)...")
    
    # Fit gamma per pixel using log-log linear regression
    # captured = display^gamma  =>  log(captured) = gamma * log(display)
    
    log_display = np.log(display_vals + 1e-6)
    
    for y in range(h):
        if y % 50 == 0:  # Progress indicator
            print(f"  Row {y}/{h}")
        
        for x in range(w):
            if not valid_mask[y, x]:
                continue
            
            # Get normalized captured values for this pixel
            log_captured = np.log(normalized_stack[y, x, :] + 1e-6)
            
            # Filter out invalid points (too close to 0 or 1)
            valid_points = (display_vals > 0.02) & (normalized_stack[y, x, :] > 0.02)
            
            if np.sum(valid_points) < 3:
                continue
            
            # Linear fit: log(captured) = gamma * log(display)
            try:
                # Using polyfit for robust fitting
                coeffs = np.polyfit(
                    log_display[valid_points],
                    log_captured[valid_points],
                    1
                )
                gamma_pixel = coeffs[0]
                
                # Sanity check: gamma should be in reasonable range
                if 0.5 < gamma_pixel < 5.0:
                    gamma_map[y, x] = gamma_pixel
            except:
                # If fit fails, keep default value
                pass
    
    # Smooth gamma map to reduce noise while preserving edges
    if smooth_kernel_size > 0:
        gamma_map = cv2.bilateralFilter(
            gamma_map, 
            smooth_kernel_size, 
            sigmaColor=0.5, 
            sigmaSpace=smooth_kernel_size
        )
    
    # Calculate global gamma for reporting
    global_gamma = np.median(gamma_map[valid_mask])
    
    print(f"\nPer-pixel gamma calibration complete!")
    print(f"Global median gamma: {global_gamma:.3f}")
    print(f"Gamma range: [{np.min(gamma_map[valid_mask]):.3f}, {np.max(gamma_map[valid_mask]):.3f}]")
    
    # Ensure capture directory exists before saving
    os.makedirs(capture_dir, exist_ok=True)
    
    # Save calibration data for visualization/debugging
    calib_save_path = os.path.join(capture_dir, "gamma_calibration.npz")
    np.savez(
        calib_save_path,
        gamma_map=gamma_map,
        black_offset=black_map,
        white_level=white_map,
        valid_mask=valid_mask,
        global_gamma=global_gamma,
        channel_mask=channel_mask
    )
    print(f"Saved calibration data to {calib_save_path}")
    
    # Save visualization of gamma map
    gamma_vis = ((gamma_map - 0.5) / 4.5 * 255).clip(0, 255).astype(np.uint8)
    gamma_vis = cv2.applyColorMap(gamma_vis, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(capture_dir, "gamma_map_visualization.png"), gamma_vis)
    
    return float(global_gamma), gamma_map, black_map, white_map


def apply_perpixel_gamma_correction(
    img_bgr, 
    channel_mask=0b010,
    gamma_map=None,
    black_offset=None,
    white_level=None,
    global_gamma=2.2
):
    """
    Apply per-pixel gamma correction using calibrated gamma map.
    Falls back to global gamma if map is not available.
    
    Args:
        img_bgr: Input BGR image
        channel_mask: Binary mask for channels (0b001=B, 0b010=G, 0b100=R, 0b111=All)
        gamma_map: Per-pixel gamma values (h, w)
        black_offset: Per-pixel black level (h, w)
        white_level: Per-pixel white level (h, w)
        global_gamma: Fallback gamma if per-pixel data unavailable
    
    Returns:
        Corrected single-channel or multi-channel averaged image (uint8)
    """
    # Parse which channels to use
    active_channels = parse_channel_mask(channel_mask)
    
    # Extract and combine channels
    if len(active_channels) == 1:
        single_channel = img_bgr[:, :, active_channels[0]].astype(np.float32)
    else:
        # Average the active channels
        single_channel = np.mean([img_bgr[:, :, idx].astype(np.float32) 
                                  for idx in active_channels], axis=0)
    
    if gamma_map is None or black_offset is None or white_level is None:
        # Fallback to simple global correction
        print("Using global gamma correction (per-pixel data not available)")
        normalized = single_channel / 255.0
        corrected = np.power(normalized, 1.0 / global_gamma)
        corrected = (corrected * 255.0).clip(0, 255).astype(np.uint8)
        return corrected
    
    # Per-pixel correction with black/white normalization
    # First, subtract black offset
    corrected = single_channel - black_offset
    
    # Normalize by dynamic range
    dynamic_range = white_level - black_offset
    normalized = np.where(
        dynamic_range > 10,
        corrected / (dynamic_range + 1e-6),
        0
    )
    normalized = np.clip(normalized, 0, 1)
    
    # Apply per-pixel gamma correction
    corrected = np.power(normalized + 1e-6, 1.0 / gamma_map)
    
    # Scale back to 0-255
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
    
    # Perform gamma calibration
    print("\n=== Starting Gamma Calibration ===")
    global_gamma, gamma_map, black_offset, white_level = generate_and_estimate_gamma(
        channel_mask=config.CHANNEL_OF_INTEREST
    )
    print("=== Gamma Calibration Complete ===\n")
    
    # Get fringe patterns
    patterns = get_patterns(config.PATTERN_DIR)
    
    for pat in patterns:
        img_path = f"{config.PATTERN_DIR}/{pat}"
        img_proj = cv2.imread(img_path)
        if img_proj is None:
            raise FileNotFoundError(f"Pattern not found: {img_path}")
        
        # Project the original RGB image
        project_image(img_proj, config.SECOND_SCREEN_X)
        time.sleep(config.SETTLE_TIME)
        
        # Capture from Kinect
        _, live_img = capture_kinect("temp", dir, config.ROI_Y, config.ROI_X)
        
        if live_img is None or live_img.size == 0:
            print(f"Capture failed for {pat}")
            continue
        
        # Apply per-pixel gamma correction
        processed = apply_perpixel_gamma_correction(
            live_img,
            channel_mask=config.CHANNEL_OF_INTEREST,
            gamma_map=gamma_map,
            black_offset=black_offset,
            white_level=white_level,
            global_gamma=global_gamma
        )
        
        # Save processed image
        clean_name = f"{pat}"
        final_path = os.path.join(dir, clean_name)
        cv2.imwrite(final_path, processed)
        print(f"   Saved processed: {final_path}")
        
        # Clean up temp files
        temp_files = glob(f"{dir}/cap_*.jpg")
        for tf in temp_files:
            try:
                os.remove(tf)
            except:
                pass
    
    print("\n=== Capture and Processing Complete ===")