from multiprocessing import Value
import re
import os
import warnings
import logging
import numpy as np

from constants import *
from logger import logger
from utils import loadtif, savetif, get_output_equivalent, check_for_existence


def get_sorted_tif_list(dir):
    ''' Get the list of TIF files contained in a directory (in alphabetical order). '''
    if not os.path.isdir(dir):
        raise ValueError('"{dir}" is not a directory')
    # List directory files in alphabetical order
    fnames = sorted(os.listdir(dir))
    if len(fnames) == 0:
        raise ValueError(f'"{dir}" folder is empty')
    # Clean list of anything which is not .tif
    fnames = list(filter(TIF_PATTERN.match, fnames)) 
    if len(fnames) == 0:
        raise ValueError(f'"{dir}" folder does not contain any tifs')
    return fnames


def is_tif_dir(dir):
    ''' Assess whether or not a directory contains any TIF files. '''
    try:
        get_sorted_tif_list(dir)
        return True
    except ValueError:
        return False


def mergetifs(tifdir, overwrite='?'):
    '''
    Merge individual tif files into a tif stack.

    :param tifdir: absolute path to directory containing the tif images
    :param overwrite: one of (True, False, '?') defining what to do if output stack file already exists 
    :return: filepath to the created tif stack
    '''
    # Cast tifdir to absolute path
    tifdir = os.path.abspath(tifdir)
    # Get output file name
    pardir, dirname = os.path.split(tifdir)
    outdir = get_output_equivalent(pardir, 'raw', 'stacked')
    stack_fpath = os.path.join(outdir, f'{dirname}.tif')
    # Check for file existence and decide whether to move fowrard or not
    move_forward = check_for_existence(stack_fpath, overwrite)
    if not move_forward:
        return stack_fpath

    # Get tif files list
    try:
        fnames = get_sorted_tif_list(tifdir)
    except ValueError:
        return None
    # Initialize stack array
    stack = []
    refshape = None
    # For eaach filename
    for i, fname in enumerate(fnames):
        # Load corresponding image while tunring off warnings
        with warnings.catch_warnings(record=True):
            image = loadtif(os.path.join(tifdir, fname))
        # Implement fix for first file that contains 10 frames (a mystery) -> we just take the last one.
        if image.ndim > 2:
            nframes = image.shape[0]
            logger.warning(f'image {i} ("{fname}") is corrupted (shape = {image.shape}) -> ommitting first {nframes - 1} frames')
            image = image[-1]
        # Assign reference image shape or ensure match of current image with reference
        if refshape is None:
            refshape = image.shape
        else:
            assert image.shape == refshape, 'image {i} shape {image.shape} does not match reference {refshape}'
        # Append image to stack
        stack.append(image)
    # Check that final stack size is correct
    nframes = len(stack)
    if nframes != REF_NFRAMES:
        logger.warning(f'final stack size = {nframes} frames, seems suspicious...')
    logger.info(f'generated {nframes}-frames image stack')
    # Convert stack to numpy array
    stack = np.stack(stack)
    # Save stack as single file
    savetif(stack_fpath, stack)
    return stack_fpath


def get_data_folders(basedir, recursive=True, exclude_patterns=[], include_patterns=[]):
    '''
    Get data folders inside a root directory by searching (recursively or not) throughout
    a tree-like folder architecture.

    :param basedir: base directory from which the search is initiated.
    :param recursive: whether to search recursively or not.
    :param exclude_patterns: list of exclusion patterns (any folder paths containing any of these patterns are excluded)
    :param include_patterns: list of inclusion patterns (only folder paths containing all of these patterns are included) 
    :return: list of data folders
    '''
    logger.debug(f'Searching through {basedir}')
    # Populate folder list
    datafolders = []
    for item in os.listdir(basedir):
        absitem = os.path.join(basedir, item)
        if is_tif_dir(absitem):
            datafolders.append(absitem)
        if recursive and os.path.isdir(absitem):
            datafolders += get_data_folders(absitem)
    # Filter out excluded folders
    for k in exclude_patterns:
        datafolders = list(filter(lambda x: k not in x, datafolders))
    # Restrict to included patterns
    for k in include_patterns:
        datafolders = list(filter(lambda x: k in x, datafolders))
    return datafolders
