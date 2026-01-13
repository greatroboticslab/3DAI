import time
import pycrafter4500  # Import the library

# Your camera capture function (assuming you have this)
def capture_image():
    # Your code to capture an image here, e.g., using OpenCV or similar
    print("Capturing image...")
    time.sleep(1)  # Placeholder for capture time
    print("Image captured.")

# List of flash pattern indices to display sequentially (match your firmware/GUI)
pattern_indices = [0, 1, 2, 3]  # Example: patterns at flash indices 0, 1, 2

with pycrafter4500.connect_usb() as dev:
    dmd = pycrafter4500.dlpc350(dev)
    
    # Power up and set to pattern mode
    dmd.set_power_mode(do_standby=False)
    dmd.set_dmd_park(park=False)
    dmd.set_display_mode('pattern')
    dmd.set_pattern_input_source('flash')  # Or 3 if string not accepted
    
    # Set trigger mode (assume 1 for command/internal; test 0 or 'vsync' if issues)
    dmd.set_pattern_trigger_mode(1)  # Or try 0/'vsync' based on your needs
    
    # Long exposure/frame period in microseconds (e.g., 10 seconds for static display during capture)
    long_exposure = 10000000  # 10 seconds; make longer if needed (max ~4e9 us)
    dmd.set_exposure_frame_period(long_exposure, long_exposure)
    
    for pat_index in pattern_indices:
        # Stop any running sequence
        dmd.pattern_display('stop')  # Or action=0
        
        # Configure sequence: single pattern, no repeat (displays once for the long exposure)
        dmd.set_pattern_config(num_lut_entries=1, do_repeat=False, num_pats_for_trig_out2=1, num_images=1)
        
        # Open pattern mailbox and set address
        dmd.open_mailbox(2)
        dmd.mailbox_set_address(0)
        
        # Send LUT entry for this pattern
        trig_type = 0  # No trigger/auto for static
        bit_depth = 8  # Match your patterns (1-8)
        led_select = 7  # White (all LEDs); adjust as needed
        do_invert_pat = False
        do_insert_black = False  # Don't clear after exposure to keep pattern loaded
        do_buf_swap = False
        do_trig_out_prev = False
        dmd.send_pattern_lut(trig_type, pat_index, bit_depth, led_select, do_invert_pat, do_insert_black, do_buf_swap, do_trig_out_prev)
        
        # Validate the LUT
        dmd.start_pattern_lut_validate()
        
        # Check status (optional but recommended; print or handle errors)
        status = dmd.get_main_status(pretty_print=True)
        print(status)  # Inspect for validation errors
        
        # Start displaying the pattern
        dmd.pattern_display('start')  # Or action=2
        
        # Wait briefly for DMD to load and stabilize
        time.sleep(0.1)
        
        # Now the pattern is displayed; capture your image
        capture_image()
        
        # Optional: Pause or wait if needed before next pattern
        # time.sleep(1)
    
    # Clean up: stop sequence, park DMD, power down
    dmd.pattern_display('stop')
    dmd.set_dmd_park(park=True)
    dmd.set_power_mode(do_standby=True)