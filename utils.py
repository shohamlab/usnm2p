import os
import numpy as np

from logger import logger
from tifffile import imread, imsave


def loadtif(fpath):
    ''' Load stack/image from .tif file '''
    stack = imread(fpath)
    if stack.ndim > 2:
        logger.info(f'loaded {stack.shape} {stack.dtype} stack from "{fpath}"')
    return stack


def savetif(fpath, stack):
    ''' Save stack/image as .tif file '''
    if stack.ndim > 2:
        logger.info(f'saving {stack.shape} {stack.dtype} stack as "{fpath}"...')
    imsave(fpath, stack)


def get_stack_baseline(stack: np.array, n: int, noisefactor: float=1.) -> np.array:
    '''
    Construct baseline stack from image stack. 
    
    :param stack: 3D image stack
    :param n: number of frames to put in the baseline
    :param noisefactor: uniform multiplying factor applied to the noise matrix to construct the baseline
    :return: generated baseline stack
    '''
    # Extract frame dimensions from stack
    nframes, *framedims = stack.shape
    logger.info(f'constructing {n}-frames baseline from {nframes}-frames stack...')
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


def get_output_equivalent(inpath, basein, baseout):
    '''
    Get the "output equivalent" of a given file or directory, i.e. its corresponding path in
    an identified output branch of the file tree structure, while creating the intermediate
    output subdirectories if needed.

    :param inpath: absolute path to the input file or directory
    :param basein: name of the base folder containing the input data (must contain inpath)
    :param baseout: name of the base folder containing the output data (must not necessarily exist)
    :return: absolute path to the equivalent output file or directory
    '''
    if not os.path.exists(inpath):
        raise ValueError(f'"{inpath}" does not exist')
    pardir, dirname = os.path.split(inpath)
    logger.debug(f'input path: "{inpath}"')
    subdirs = []
    if os.path.isdir(inpath):
        subdirs.append(dirname)
        fname = None
    else:
        fname = dirname
    logger.debug(f'moving up the filetree to find "{basein}"')
    while dirname != basein:
        if len(pardir) < 2:
            raise ValueError(f'"{basein}"" is not a parent of "{inpath}"')
        pardir, dirname = os.path.split(pardir)
        subdirs.append(dirname)
    logger.debug(f'found "{basein}" in "{pardir}"')
    subdirs = subdirs[::-1]
    subdirs[0] = baseout
    logger.debug(f'moving down the file tree in "{baseout}"')
    outpath = pardir
    for subdir in subdirs:
        outpath = os.path.join(outpath, subdir)
        if not os.path.isdir(outpath):
            logger.info(f'creating "{outpath}" directory')
            os.mkdir(outpath)
    if fname is not None:
        outpath = os.path.join(outpath, fname)
    return outpath
