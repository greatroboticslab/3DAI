import lib_3dai
import config
import numpy as np
import matplotlib.pyplot as plt
import cv2
from glob import glob
import fpp_tools as fpp

# Load your calibration
coeffs = np.loadtxt("calibration_kinect.txt")
print(f"Loaded calibration: a={coeffs[0]:.8f}, b={coeffs[1]:.7f}, c={coeffs[2]:.7f}")

DIRECTORIES = ('obj','ref')
PHI = []

for dir in DIRECTORIES:
    files = sorted(glob(f'{config.CAPTURE_DIR}/{dir}/*.png'))

    ## Make a list of the images to use, and then convert the list into a Numpy array 3D image stack.
    imagestack = []
    stack = [cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32) for f in files]
    imagestack = np.dstack(stack)

    (Nx,Ny,num_images) = imagestack.shape
    print('imagestack shape =', imagestack.shape)

    deltas = np.array([0.0, np.pi/2.0, np.pi, 3.0*np.pi/2.0])

    (phi_image) = fpp.estimate_phi_N_nonuniform_frames(imagestack, deltas)


    deltas -= deltas[0]

    print(f"est. deltas [deg] = {np.array2string(deltas*180.0/np.pi, formatter={'float': lambda x:f'{x:.2f}'}, separator=', ')}")

    plt.figure('img0')
    plt.imshow(imagestack[:,:,0])

    plt.figure('phi')
    plt.imshow(phi_image)

    ## In the unwrapped phase image, make sure that the smallest phase is zero. If the phase is increasing, then we subtract the
    ## smallest value. If decreasing, then we add the smallest value.
    unwrapped_phi = np.unwrap(phi_image)
    avg_gradient = np.mean(np.gradient(unwrapped_phi[:,Ny//2]))

    if (avg_gradient > 0.0):
        unwrapped_phi += np.amax(unwrapped_phi)
    else:
        unwrapped_phi -= np.amax(unwrapped_phi)

    plt.figure('phi_unwrapped')
    plt.imshow(unwrapped_phi)

    plt.show()

    ## ====================================================================================
    ## ====================================================================================

    order = 1       ## do the fitting to what order multiple of the main modulation frequency (use 1, 2, 3, or 4)
    noise = 0.04
    #nphases = 6
    #phases = arange(nphases) * 2.0 * pi / nphases
    phases = [0, 90, 180, 270]


    ## Normalize the intensities so that the maximum is 1.
    imagestack /= np.amax(imagestack)
    (Nx,Ny,num_images) = imagestack.shape
    print('imagestack shape =', imagestack.shape)

    if (noise > 0.0):
        imagestack += np.random.normal(0.0, noise, imagestack.shape)
        imagestack -= np.amin(imagestack)
        imagestack /= np.amax(imagestack)

    plt.figure('raw img0')
    plt.imshow(imagestack[:,:,0])
    plt.colorbar()

    if (order == 1):
        (phi_img, amplitude_img, bias_img, deltas) = fpp.estimate_deltas_and_phi_lsq(imagestack, order=order)
        gamma_img = None
    elif (order > 1):
        (phi_img, amplitude_img, bias_img, gamma_img, deltas) = fpp.estimate_deltas_and_phi_lsq(imagestack, order=2)

    deltas -= np.amin(deltas)
    deltas[deltas > 360.0] = deltas[deltas > 360.0] - 360.0
    deltas = np.sort(deltas)
    print(f"est. deltas [deg] = {np.array2string(deltas*180.0/np.pi, formatter={'float': lambda x:f'{x:.2f}'}, separator=', ')}")

    contrast_img = amplitude_img / bias_img

    plt.figure('contrast')
    plt.imshow(contrast_img)
    plt.title(f'mean(contrast)= {np.mean(contrast_img):.2f}')
    plt.colorbar()

    plt.figure('bias')
    plt.imshow(bias_img)
    plt.title(f'mean(bias)= {np.mean(bias_img):.2f}')
    plt.colorbar()

    if gamma_img is not None:
        plt.figure('gamma')
        plt.imshow(gamma_img)
        plt.title(f'mean(gamma)= {np.mean(gamma_img):.2f}')
        plt.colorbar()
        print(f'Average gamma value: {np.mean(gamma_img):.2f}')

    plt.figure('phi')
    plt.imshow(phi_img)

    ## In the unwrapped phase image, make sure that the smallest phase is zero. If the phase is increasing, then we subtract the
    ## smallest value. If decreasing, then we add the smallest value.
    unwrapped_phi = np.unwrap(phi_img)
    avg_gradient = np.mean(np.gradient(unwrapped_phi[:,Ny//2]))
    if (avg_gradient > 0.0):
        unwrapped_phi += np.amax(unwrapped_phi)
    else:
        unwrapped_phi -= np.amax(unwrapped_phi)

    unwrapped_phi -= unwrapped_phi[0,0]

    plt.figure('phi_unwrapped')
    plt.imshow(unwrapped_phi)
    plt.colorbar()

    phi_curve = unwrapped_phi[100,:]

    plt.figure('phi_unwrapped[100,:]')
    plt.plot(unwrapped_phi[100,:])
    plt.plot(unwrapped_phi[101,:])

    ## Subtract a linear fit to the phi curve.
    x = np.arange(len(unwrapped_phi[100,:]))
    phi_curve = np.zeros_like(unwrapped_phi[100,:])
    for i in range(100,200):
        fit_x = np.polyfit(x, unwrapped_phi[i,:], 1)
        linear_baseline = np.poly1d(fit_x) # create the linear baseline function
        phi_curve += unwrapped_phi[i,:] - linear_baseline(x) # subtract the baseline from μ_Y

    phi_curve_avg = phi_curve / 100.0

    plt.figure('nonlinear error: phi_curve - baseline')
    plt.plot(phi_curve_avg)

    plt.show()

    PHI.append(unwrapped_phi)


# Height map
delta_phi = PHI[1] - PHI[0]
height_mm = np.polyval(coeffs, delta_phi)



print(f"Height range: {np.nanmin(height_mm):.1f} mm to {np.nanmax(height_mm):.1f} mm")

# Display
plt.figure(figsize=(10, 6))
im = plt.imshow(height_mm, cmap='jet', vmin=0, vmax=30)  # adjust vmax to your expected range
plt.title("Height Map (mm)")
plt.colorbar(im, label="Height (mm)")
plt.axis('off')
plt.show()