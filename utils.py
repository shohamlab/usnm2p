import numpy as np

def get_stack_baseline(stack: np.array, n: int, noisefactor: float=1.) -> np.array:
    '''
    Construct baseline stack from image stack. 
    
    :param stack: 3D image stack
    :param n: number of frames to put in the baseline
    :param noisefactor: uniform multiplying factor applied to the noise matrix to construct the baseline
    :return: generated baseline stack
    '''
    # Extract frame dimensions from stack
    _, *framedims = stack.shape
    # Estimate median and standard deviation of each pixel across the images of the stack
    pixel_med, pixel_std = np.median(stack, axis=0), np.std(stack, axis=0)
    # Tile both matrices along the required number of baseline frames
    pixel_med, pixel_std = [np.tile(x, (n, 1, 1)) for x in [pixel_med, pixel_std]]
    # Construct noise matrix spanning [-noisefactor, noisefactor] and matching the required baseline stack dimensions
    noise = noisefactor * (2 * np.random.rand(n, *framedims) - 1)
    # Construct baseline stack by summing up median and std-scale noise matrices
    baseline = pixel_med + noise * pixel_std
    # Bound to [0, MAX] interval and return
    return np.clip(baseline, 0, np.amax(stack))
