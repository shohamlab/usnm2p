# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-14 18:28:46
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-06-07 11:26:46

''' Collection of utilities for operations on files and directories. '''

import os
import glob
import pprint
import datetime
import pandas as pd
from tifffile import imread, imsave
import matplotlib.backends.backend_pdf
from tqdm import tqdm
from natsort import natsorted

from parsers import P_TIFFILE, parse_date_mouse_region
from logger import logger
from utils import *
from viewers import get_stack_viewer
from constants import *
from postpro import *

def get_data_root():
    ''' Get the root directory for the raw data to analyze '''
    try:
        from config import dataroot
    except ModuleNotFoundError:
        raise ValueError(f'user-specific "config.py" file is missing')
    except ImportError:
        raise ValueError(f'"dataroot" variable is missing from user-specific "config.py" file')
    if not os.path.isdir(dataroot):
        raise ValueError(f'data root directory "{dataroot}" does not exist')
    return dataroot


def get_subfolder_names(dirpath):
    ''' Return a list of sub-folders for a given directory '''
    return [f.name for f in os.scandir(dirpath) if f.is_dir()]


def get_dataset_params(root='.', excludes=None, includes=['region']):
    '''
    Construct a list of (line, date, mouse, region) combinations that contain
    experiment datasets inside a given root directory.
    
    :param root: root directory (typically a mouse line) containing the dataset folders
    :param excludes: list of exlusion patterns
    :param includes: list of inclusion patterns
    :return: list of dictionaries representing (line, date, mouse, region) combinations found
        inside the root folder.
    '''
    logger.info(f'Searching for data folders in {root} ...')
    datasets = []
    # Loop through lines, dates, mice, and regions, and add data root folders to list  
    for line in get_subfolder_names(root):
        linedir = os.path.join(root, line)
        for folder in get_subfolder_names(linedir):
            try:
                date, mouse, region, layer = parse_date_mouse_region(folder)
                dataset_dirpath = os.path.join(linedir, folder)
                datasets.append((line, date, mouse, region, layer, dataset_dirpath))
            except ValueError as err:
                logger.warning(err)

    # Remove unwanted patterns from list
    if excludes is not None:
        excludes = as_iterable(excludes)
        for k in excludes:
            datasets = list(filter(lambda x: k not in os.path.basename(x[-1]), datasets))
    for k in includes:
        datasets = list(filter(lambda x: k in os.path.basename(x[-1]), datasets))

    # Return line, date, mouse, region combinations
    return [
        {'mouseline': x[0], 'expdate': x[1], 'mouseid': x[2], 'region': x[3], 'layer': x[4]}
        for x in datasets]


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


def split_path_at(in_path, split_dir):
    '''
    Split an absolute path at a specific parent directory located somewhere upstream in the filesystem.
    
    :param in_path: absolute path to the input file or directory
    :param split_dir: name of the directory at which to split the path (must contain in_path)
    :return: 3-tuple with the upstream path, the dplit directory, and the downstream path
    '''
    if not os.path.exists(in_path):
        raise ValueError(f'"{in_path}" does not exist')
    logger.debug(f'input path: "{in_path}"')
    # Create empty downstream path list
    downstream_path = []
    # Split input path
    pardir, dirname = os.path.split(in_path)
    # Add current basename to downstream path list if not empty
    if dirname:
        downstream_path.append(dirname)
    # Move up file-tree until split directory is found
    logger.debug(f'moving up the file-tree to find "{split_dir}"')
    while dirname != split_dir:
        if len(pardir) < 2:  # if gone all the way up to root level without match -> raise error
            raise ValueError(f'"{split_dir}"" is not a parent of "{split_dir}"')
        pardir, dirname = os.path.split(pardir)
        # Add current basename to downstream path list
        downstream_path.append(dirname)
    logger.debug(f'found "{split_dir}" in "{pardir}"')
    # Reverse downstream path list (to match descending order) and isolate split directory
    found_split_dir, *downstream_path = downstream_path[::-1]
    # Make sure found split directory matches input
    assert found_split_dir == split_dir, 'mismatch in path parsing'
    # Return (upstream path, dplit dir, downstream path) tuple
    if len(downstream_path) > 0:
        downstream_path = os.path.join(*downstream_path)
    else:
        downstream_path = ''
    return pardir, split_dir, downstream_path


def get_output_equivalent(in_path, basein, baseout, mkdirs=True):
    '''
    Get the "output equivalent" of a given file or directory, i.e. its corresponding path in
    an identified output branch of the file tree structure, while creating the intermediate
    output subdirectories if needed.

    :param inpath: absolute path to the input file or directory
    :param basein: name of the base folder containing the input data (must contain inpath)
    :param baseout: name of the base folder containing the output data (must not necessarily exist)
    :return: absolute path to the equivalent output file or directory
    '''
    # Get the upstream and downstream paths according to basein split 
    upstream_path, _, downstream_path = split_path_at(in_path, basein)
    logger.debug(f'moving down the file tree in "{baseout}"')
    # Construct output path
    out_path = os.path.join(upstream_path, baseout, downstream_path)
    logger.debug(f'output path: "{out_path}"')
    # Create required subdirectories if output path does not exist
    if not os.path.exists(out_path) and mkdirs:
        if os.path.isdir(in_path):  # if input path was a directory -> include all elements
            os.makedirs(out_path)
        else:  # if input path was a file -> fetch parent directory
            pardir = os.path.split(out_path)[0]
            if not os.path.isdir(pardir):
                os.makedirs(pardir)
    # Return output path
    return out_path

    
def get_data_folders(basedir, recursive=True, exclude_patterns=[], include_patterns=[], rec_call=False):
    '''
    Get data folders inside a root directory by searching (recursively or not) throughout
    a tree-like folder architecture.

    :param basedir: base directory from which the search is initiated.
    :param recursive: whether to search recursively or not.
    :param exclude_patterns: list of exclusion patterns (any folder paths containing any of these patterns are excluded)
    :param include_patterns: list of inclusion patterns (only folder paths containing all of these patterns are included) 
    :param rec_call (default: False): whether or not this is a recursive function call
    :return: list of data folders
    '''
    logger.debug(f'Searching through {basedir}')
    if not rec_call:
        logger.info(basedir)
    # Populate folder list
    datafolders = []
    # Loop through content of base directory 
    for item in os.listdir(basedir):
        absitem = os.path.join(basedir, item)
        # If content item is a directory containing TIF files, add to list
        if is_tif_dir(absitem):
            datafolders.append(absitem)
        # If content item is a directory and recursive call enabled, call function
        # recursively on child folder and add output to list
        if recursive and os.path.isdir(absitem):
            datafolders += get_data_folders(
                absitem,
                exclude_patterns=exclude_patterns, include_patterns=include_patterns,
                rec_call=True)
    logger.debug(f'raw list: {datafolders}')
    # Filter out excluded folders
    for k in exclude_patterns:
        datafolders = list(filter(lambda x: k not in os.path.basename(x), datafolders))
    # Restrict to included patterns
    for k in include_patterns:
        datafolders = list(filter(lambda x: k in os.path.basename(x), datafolders))
    logger.debug(f'filtered list: {datafolders}')
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


def loadtif(fpath, verbose=True):
    ''' Load stack/image from .tif file '''
    stack = imread(fpath)
    if stack.ndim > 2:
        func = logger.info if verbose else logger.debug
        func(f'loaded {stack.shape} {stack.dtype} stack from "{fpath}"')
    return stack


def savetif(fpath, stack, overwrite=True):
    ''' Save stack/image as .tif file '''
    move_forward = check_for_existence(fpath, overwrite)
    if not move_forward:
        return
    if stack.ndim > 2:
        logger.info(f'saving {stack.shape} {stack.dtype} stack as "{fpath}"...')
    imsave(fpath, stack)


def process_and_save(processor, input_fpath, input_root, *args, overwrite=False, **kwargs):
    '''
    Load input stack file, apply processing function and save output stack in specific directory.
    
    :param processor: processor object
    :param input_fpath: absolute filepath to the input TIF stack (or list of TIF stacks).
    :param input_root: name of the directory that constitutes the root level of the input filepath
    :param overwrite: one of (True, False, '?') defining what to do if file already exists.
    :return: absolute filepath to the output TIF stack (or list of TIF stacks).
    '''
    # If list of filepaths provided as input -> apply function to all of them
    if is_iterable(input_fpath):
        return [process_and_save(processor, x, input_root, *args, overwrite=overwrite, **kwargs) for x in input_fpath]
    # Check that input root is indeed found in input filepath
    if input_root not in input_fpath:
        raise ValueError(f'input root "{input_root}" not found in input file path "{input_fpath}"')
    # Check that processor instance is a known class type
    if not isinstance(processor, StackProcessor):
        raise ValueError(f'unknown processor type: {processor}')
    # If NoProcessor object provided -> do nothing and return input file path
    if isinstance(processor, NoProcessor):
        return input_fpath
    # Get output filepath
    output_fpath = get_output_equivalent(
        input_fpath, input_root, f'{processor.rootcode}/{processor.code}')
    # If already existing, act according to overwrite parameter
    if os.path.isfile(output_fpath):
        logger.warning(f'"{output_fpath}" already exists')
        overwrite = parse_overwrite(overwrite)
        if not overwrite:
            return output_fpath
    # Load input, process stack, and save output
    input_stack = loadtif(input_fpath)
    output_stack = processor.run(input_stack, *args, **kwargs)
    savetif(output_fpath, output_stack, overwrite=overwrite)
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


def save_figs_book(figsroot, figs, suffix=None):
    ''' Save figures dictionary as consecutive pages in single PDF document. '''
    now = datetime.datetime.now()
    today = now.strftime('%Y.%m.%d')
    figsdir = os.path.join(figsroot, today)
    if not os.path.isdir(figsdir):
        os.mkdir(figsdir)
    fcode = 'figs'
    if suffix is not None:
        fcode = f'{fcode}_{suffix}'
    fname = f'{fcode}.pdf'
    fpath = os.path.join(figsdir, fname)
    file = matplotlib.backends.backend_pdf.PdfPages(fpath)
    logger.info(f'saving figures in {fpath}:')
    for v in tqdm(figs.values()):
        file.savefig(v, transparent=True, bbox_inches='tight')
    file.close()


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
    fps = kwargs.pop('fps', 10)
    norm = kwargs.pop('norm', True)
    cmap = kwargs.pop('cmap', 'viridis')
    bounds = kwargs.pop('bounds', None)
    ilabels = kwargs.pop('ilabels', None)
    viewer = get_stack_viewer(*args, **kwargs)
    viewer.init_render(norm=norm, cmap=cmap, bounds=bounds, ilabels=ilabels)
    viewer.save_as_gif(figsdir, fps)


def save_postpro_dataset(fpath, timeseries, info_table, ROI_masks):
    '''
    Save processed dataset into HDF5 file. 
    
    :param fpath: absolute path to data file
    :param timeseries: multi-indexed (ROI, run, trial, frame) timeseries dataframe
    :param info_table: dataframe containing the information about experimental parameters for each run
    :param ROI_masks: ROI-indexed dataframe of (x, y) coordinates and weights
    '''
    # Remove output file if it exists
    if os.path.isfile(fpath):
        os.remove(fpath)
    # Create HDF store object
    with pd.HDFStore(fpath) as store:
        # Save experiment info table
        logger.info('saving experiment info table...')
        store['info_table'] = info_table 
        # Save timeseries table
        logger.info('saving processed timeseries data...')
        store['timeseries'] = timeseries
        # Save ROI masks
        logger.info('saving ROI masks...')
        store['ROI_masks'] = ROI_masks
    logger.info('data successfully saved')


def load_postpro_dataset(fpath):
    '''
    Load processed dataset from HDF5 file
    
    :param fpath: absolute path to data file
    :return: 3-tuple containing multi-indexed (ROI, run, trial, frame) timeseries dataframe,
        experiment info dataframe, and ROI-indexed dataframe of (x, y) coordinates and weights.
    '''
    # Check that output file is present in directory
    if not os.path.isfile(fpath):
        raise FileNotFoundError('post-processed data file not found in directory')
    # Create HDF store object
    with pd.HDFStore(fpath) as store:
        # Load experiment info table
        logger.info('loading experiment info table...')
        info_table = store['info_table']
        # Load timeseries table
        logger.info('loading processed timeseries data...')
        timeseries = store['timeseries']
        # Load ROI masks
        logger.info('loading ROI masks...')
        ROI_masks = store['ROI_masks']
    logger.info('data successfully loaded')
    return timeseries, info_table, ROI_masks


def save_trialavg_dataset(fpath, timeseries, stats):
    '''
    Save trial-averaged dataset to a HDF5 file
    
    :param fpath: absolute path to data file
    :param timeseries: multi-indexed (ROI, run, frame) trial-averaged timeseries dataframe
    :param stats: multi-indexed (ROI, run) trial-averaged stats dataframe
    '''
    # Remove output file if it exists
    if os.path.isfile(fpath):
        os.remove(fpath)
    # Create HDF store object
    with pd.HDFStore(fpath) as store:
        # Save timeseries table
        logger.info('saving trial-averaged timeseries data...')
        store['timeseries'] = timeseries
        # Save stats table
        logger.info('saving trial-averaged stats data...')
        store['stats'] = stats
    logger.info('data successfully saved')


def load_trialavg_dataset(fpath):
    '''
    Load dataset of a particular date-mouse-region from a HDF5 file
    
    :param fpath: absolute path to data file
    :return: multi-indexed timeseries and stats dataframes
    '''
    # Check that output file is present in directory
    if not os.path.isfile(fpath):
        raise FileNotFoundError('trial-averaged data file not found in directory')
    # Create HDF store object
    with pd.HDFStore(fpath) as store:
        # Load data
        logger.info(f'loading trial-averaged data from {os.path.basename(fpath)}')
        timeseries = store['timeseries']
        stats = store['stats']
    return timeseries, stats


def load_trialavg_datasets(dirpath, include_patterns=None, exclude_patterns=None, **kwargs):
    '''
    Load multiple mouse-region datasets
    
    :param include_patterns (optional): inclusion pattern(s)
    :param exclude_patterns (optional): exclusion pattern(s)
    '''
    # List data filepaths
    fpaths = natsorted(glob.glob(os.path.join(dirpath, f'*.h5')))

    # Filter according to inclusion & exclusion patterns, if any
    if include_patterns is not None:
        include_patterns = as_iterable(include_patterns)
        logger.warning(f'excluding datasets not having the following patterns: {include_patterns}')
        fpaths = list(filter(lambda x: all(e in x for e in include_patterns), fpaths))
    if exclude_patterns is not None:
        exclude_patterns = as_iterable(exclude_patterns)
        logger.warning(f'excluding datasets with the following patterns: {exclude_patterns}')
        fpaths = list(filter(lambda x: not any(e in x for e in exclude_patterns), fpaths))
    
    # Load timeseries and stats datasets
    datasets = [load_trialavg_dataset(fpath) for fpath in fpaths]
    timeseries, stats = list(zip(*datasets))

    # Get dataset IDs
    logger.info('gathering dataset IDs...')
    dataset_ids = []
    for fpath in fpaths:
        fname = os.path.basename(fpath)
        dataset_id = os.path.splitext(fname)[0]
        while dataset_id.startswith('_'):
            dataset_id = dataset_id[1:]
        dataset_ids.append(dataset_id)
    
    # Concatenate datasets while adding their respective IDs
    timeseries = pd.concat(timeseries, keys=dataset_ids, names=[Label.DATASET])
    stats = pd.concat(stats, keys=dataset_ids, names=[Label.DATASET])

    # Sort index for each dataset
    logger.info('sorting dataset indexes...')
    timeseries.sort_index(
        level=[Label.DATASET, Label.ROI, Label.RUN, Label.FRAME], inplace=True) 
    stats.sort_index(
        level=[Label.DATASET, Label.ROI, Label.RUN], inplace=True)

    try:
        # Check run order consistency across datasets
        check_run_order(stats)
    except ValueError as err:
        # If needed, harmonize run indexes in stats & timeseries
        logger.warning(err)
        timeseries, stats = harmonize_run_index(timeseries, stats) 

    # Add missing change metrics, if any
    for ykey in [Label.ZSCORE, Label.DFF]:
        ykey_avg = f'avg {ykey}' 
        ykey_prestim_avg, ykey_poststim_avg = f'pre-stim {ykey_avg}', f'post-stim {ykey_avg}'
        ykey_diff = f'{ykey_poststim_avg} - {ykey_prestim_avg}'
        if ykey_diff not in stats:
            stats = add_change_metrics(timeseries, stats, ykey)
    
    # Return stats and timeseries as a dictionary
    return {'timeseries': timeseries, 'stats': stats}
