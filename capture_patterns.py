
import config
import lib_3dai


print("=== START CAPTURE ===\n")
print("Place reference plane (0 mm) and press ENTER")
input()

for idx, h_mm in enumerate(config.KNOWN_THICKNESSES_MM):
    print(f"\n=== STEP {idx+1}/{len(config.KNOWN_THICKNESSES_MM)} : {h_mm:.1f} mm ===")
    if h_mm > 0:
        input(f"   → Place {h_mm:.1f} mm plate → press ENTER when ready...")
    
    height_dir = f"{config.CAPTURE_DIR}/h{h_mm:05.1f}mm"
    lib_3dai.capture_projections(height_dir)
        
    print(f"Finished {h_mm:.1f} mm")

print("\nCapture phase complete!")