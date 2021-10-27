# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-14 18:28:46
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-10-27 15:45:39

''' Collection of utilities for operations on files and directories. '''

import os
import glob
import pprint
import datetime
from tifffile import imread, imsave

from parsers import P_TIFFILE
from logger import logger
from utils import is_iterable
from filters import StackFilter, NoFilter
from viewers import get_stack_viewer
from constants import *


def get_data_root():
    ''' Get the root directory for the raw data to analyze '''
    try:
        from config import dataroot
    except ModuleNotFoundError:
        raise ValueError(f'user-specific "config.py" file is missing')
    if not os.path.isdir(dataroot):
        raise ValueError(f'data root directory "{dataroot}" does not exist')
    return dataroot


def check_for_existence(fpath, overwrite):
    '''
    Check file for existence
    
    :param fpath: full path to candidate file
    :param overwrite: one of (True, False, '?') defining what to do if file already exists.
    :return: boolean stating whether to move forward or not.
    '''
    if os.path.exists(fpath):
        logger.warning(f'"{fpath}" already exists')
        return parse_overwrite(overwrite)
    else:
        return True


def get_sorted_filelist(dir, pattern=None):
    '''
    Get the list of files contained in a directory (in alphabetical order).

    :param pattern (optional): naming pattern files must match with.
    :return: list of filenames
    '''
    # Throw error if directory does not exist
    if not os.path.isdir(dir):
        raise ValueError('"{dir}" is not a directory')
    # List directory files in alphabetical order
    fnames = sorted(os.listdir(dir))
    if len(fnames) == 0:
        raise ValueError(f'"{dir}" folder is empty')
    # If pattern provided, clean list of anything that does not match
    if pattern is not None:
        fnames = list(filter(pattern.match, fnames)) 
    # Throw error if no file was selected
    if len(fnames) == 0:
        raise ValueError(f'"{dir}" folder does not contain any tifs')
    return fnames


def is_tif_dir(dir):
    ''' Assess whether or not a directory contains any TIF files. '''
    try:
        get_sorted_filelist(dir, P_TIFFILE)
        return True
    except ValueError:
        return False


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
    logger.debug(f'moving down the file tree in "{baseout}"')
    outpath = os.path.join(pardir, baseout, *subdirs[::-1][1:])
    if not os.path.isdir(outpath):
        os.makedirs(outpath)
    if fname is not None:
        outpath = os.path.join(outpath, fname)
    return outpath

    
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


def sort_folders_by_runID(datafolders, pdicts):
    '''
    Sort folders by runID
    
    :param datafolders: list of full paths to data folders
    :param pdicts: list of parsed parameters dictionaries for each datafolder
    '''
    # Check that all data folders are from the same parent
    pardirs = [os.path.split(x)[0] for x in datafolders]
    assert pardirs.count(pardirs[0]) == len(pardirs), 'Data folders have different parrents'
    # Sort folders once asserted that they belong to the same parent
    datafolders, pdicts = zip(*sorted(zip(datafolders, pdicts), key=lambda x: x[1]['runID']))
    logger.info(f'Sorted data folders:\n{pprint.pformat(datafolders)}')
    return datafolders, pdicts


def loadtif(fpath):
    ''' Load stack/image from .tif file '''
    stack = imread(fpath)
    if stack.ndim > 2:
        logger.info(f'loaded {stack.shape} {stack.dtype} stack from "{fpath}"')
    return stack


def savetif(fpath, stack, overwrite=True):
    ''' Save stack/image as .tif file '''
    move_forward = check_for_existence(fpath, overwrite)
    if not move_forward:
        return
    if stack.ndim > 2:
        logger.info(f'saving {stack.shape} {stack.dtype} stack as "{fpath}"...')
    imsave(fpath, stack)


def filter_and_save(filter, input_fpath, overwrite=False):
    '''
    Load input stack file, apply filter and save output stack in specific directory.
    
    :param filter: filter object
    :param input_fpath: absolute filepath to the input TIF stack (or list of TIF stacks).
    :return: absolute filepath to the output (filtered) TIF stack (or list of TIF stacks).
    :param overwrite: one of (True, False, '?') defining what to do if file already exists.
    '''
    # If list of filepaths provided as input -> apply function to all of them
    if is_iterable(input_fpath):
        return [filter_and_save(filter, x) for x in input_fpath]
    # Check that filter instance is a known class type
    if not isinstance(filter, StackFilter):
        raise ValueError(f'unknown filter type: {filter}')
    # If NoFilter object provided -> do nothing and return input file path
    if isinstance(filter, NoFilter):
        return input_fpath
    # Get output filepath
    output_fpath = get_output_equivalent(input_fpath, 'stacked', f'filtered/{filter.code}')
    # If already existing, act according to overwrite parameter
    if os.path.isfile(output_fpath):
        logger.warning(f'"{output_fpath}" already exists')
        overwrite = parse_overwrite(overwrite)
        if not overwrite:
            return output_fpath
    # Load input, filter stack, and save output
    stack = loadtif(input_fpath)
    filtered_stack = filter.filter(stack)
    savetif(output_fpath, filtered_stack, overwrite=overwrite)
    # Return output filepath
    return output_fpath


def parse_overwrite(overwrite):
    '''
    Parse a user input overwrite parameter.
    
    :param overwrite: one of (True, False, '?') defining what to do in case of overwrite dilemma.
    :return: parsed overwrite decision (True/False)
    '''
    # Check that input is valid
    valids = (True, False, '?')
    if overwrite not in valids:
        raise ValueError(f'"overwrite must be one of {valids}')
    # If not a question, must be True or False -> return as is
    if overwrite != '?':
        return overwrite
    # Otherwise ask user to input (y/n) response
    overwrite = input('overwrite (y/n)?:')
    if overwrite not in ['y', 'n']:
        raise ValueError('"overwrite" argument must be one of ("y", "n")')
        # Parse response into True/False and return boolean
    return {'y': True, 'n': False}[overwrite]


def get_figdir(figsroot):
    figsroot = os.path.abspath(figsroot)
    today = datetime.date.today().strftime('%Y-%m-%d')
    figsdir = os.path.join(figsroot, today)
    if not os.path.isdir(figsdir):
        os.makedirs(figsdir)
    return figsdir


def save_figs(figsroot, figs, ext='png'):
    ''' Save figures dictionary in specific directory. '''
    figsdir = get_figdir(figsroot)
    for k, v in figs.items():
        fname = f'{k}.{ext}'
        logger.info(f'saving "{fname}"')
        v.savefig(os.path.join(figsdir, fname), transparent=True, bbox_inches='tight')


def save_stack_to_gif(figsroot, *args, **kwargs):
    ''' High level function to save stacks to gifs. '''
    figsdir = get_figdir(figsroot)
    fps = kwargs.pop('fps', FPS)
    norm = kwargs.pop('norm', True)
    cmap = kwargs.pop('cmap', 'viridis')
    bounds = kwargs.pop('bounds', None)
    ilabels = kwargs.pop('ilabels', None)
    viewer = get_stack_viewer(*args, **kwargs)
    viewer.init_render(norm=norm, cmap=cmap, bounds=bounds, ilabels=ilabels)
    viewer.save_as_gif(figsdir, fps)


def locate_datafiles(line, layer, filter_key=None):
    ''' Construct a list of suite2p data files to be used as input for an analysis. '''
    
    # Determine data directory and potential exclusion patterns based on input parameters
    exclude = []
    if line == 'sarah_line3':
        base_dir = '/gpfs/scratch/asuacd01/sarah_usnm/line3/20210804/mouse28/region1'
        file_dir = f'{base_dir}/suite2p'
        # base_dir = '/gpfs/scratch/asuacd01/sarah_usnm/line3'
        # assert os.path.isdir(base_dir), f'"{base_dir}" directory does not exist'
        # for x in [date, mouse, region]:
        #     base_dir = f'{base_dir}/{x}'
        #     assert os.path.isdir(base_dir), f'"{base_dir}" directory does not exist'
        # file_dir = f'{base_dir}/suite2p'
    if line == 'yi_line3':        
        base_dir = '/gpfs/scratch/asuacd01/yi_usnm/line3'
        file_dir = f'{base_dir}/suite2p_results'
        exclude = ['mouse9', 'mouse10', 'mouse1_region2'] # list of excluded subjects, empty list if all included, for yi_line3
    elif line == 'sst':
        base_dir = '/gpfs/data/shohamlab/shared_data/yi_recordings/yi_new_holder_results/sst'
        file_dir = f'{base_dir}/suite2p_results_frame_norm'
        exclude = ['mouse6'] # list of excluded subjects, empty list if all included, for sst
    elif line == 'celia_line3':
        base_dir = '/gpfs/data/shohamlab/shared_data/celia/line3'
        file_dir = f'{base_dir}/suite2p_results' # new data
    elif line == 'pv':
        base_dir = '/gpfs/data/shohamlab/shared_data/yi_recordings/yi_new_holder_results/PV/'
        file_dir = f'{base_dir}/suite2p_results_framenorm'        
        exclude = ['mouse6'] # list of excluded subjects, empty list if all included, for PV

    # Locate and sort all files in the file directory
    group_files = sorted(glob.glob(os.path.join(file_dir,'*')))

    # Remove unwanted layers from the analysis
    if layer == 'layer5':
        group_files = [x for x in group_files if 'layer5' in x] 
    elif layer == 'layer2_3':
        group_files = [x for x in group_files if 'layer5' not in x]
    else:
        logger.warning('Performing analysis disregarding layer parameter')

    # Exclude relevant subjects from the analysis
    for subject in exclude:
        group_files = [x for x in group_files if subject not in x]
    
    # Restrict analysis to files containing the filter key, if any
    if filter_key is not None:
        group_files = [x for x in group_files if filter_key in x]

    # Get file base names
    file_basenames = [os.path.basename(x) for x in group_files]

    # # Add potential suffixes to line label
    # if layer:
    #     line = line + '_' + layer
    # if exclude:
    #     line = line + '_exclude'+ '-'.join(exclude)
    # if filter_key:
    #     line = line + '_individual'+ '-'.join(filter_key)

    return file_dir, file_basenames

