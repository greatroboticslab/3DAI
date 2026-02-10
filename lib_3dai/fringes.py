import numpy as np


def generate_pattern(resolution, phase, num_fringes=10, gamma=1.0):
    '''
    Create a single image containing a sinusoidal fringe pattern.

    Parameters
    ----------
    resolution : tuple
        (width, height) - The image dimensions in pixels.
    phase : float
        The phase (in radians) of the pattern to generate.
    num_fringes : float
        How many fringes should appear inside each projection frame.
    gamma : float
        The gamma value to use for nonlinear conversion of the sinusoid profile.

    Returns
    -------
    fringe_pattern : ndarray
        A 2D array of shape (height, width) containing the fringe pattern with 
        uint8 values in the range [0, 255].

    Example
    -------
    pattern = generate_pattern((640, 480), pi/2, num_fringes=12)
    '''
    Ny = resolution[0]  # width
    Nx = resolution[1]  # height

    (proj_xcoord, proj_ycoord) = np.indices((Nx, Ny))
    k = 2.0 * np.pi * num_fringes / Ny
    fringe_pattern = pow(0.5 + 0.5 * np.cos(k * proj_ycoord + phase), gamma)
    fringe_pattern = 255.0 * fringe_pattern / np.amax(fringe_pattern)
    
    # Convert to uint8 for proper display
    fringe_pattern = np.uint8(np.rint(fringe_pattern))
    
    return fringe_pattern
    

    

