# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-14 18:28:46
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-05-17 16:39:28

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

from parsers import P_TIFFILE
from logger import logger
from utils import is_iterable, StackProcessor, NoProcessor
from viewers import get_stack_viewer
from constants import *
from postpro import slide_along_trial, detect_across_trials, find_response_peak, apply_in_window


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


def get_dataset_params(root='.', excludes=['layer5'], includes=['region']):
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
        for date in get_subfolder_names(linedir):
            datedir = os.path.join(linedir, date)
            for mouse in get_subfolder_names(datedir):
                mousedir = os.path.join(datedir, mouse)
                for region in get_subfolder_names(mousedir):
                    regiondir = os.path.join(mousedir, region)
                    datasets.append((line, date, mouse, region, regiondir))

    # Remove unwanted patterns from list
    for k in excludes:
        datasets = list(filter(lambda x: k not in os.path.basename(x[-1]), datasets))
    for k in includes:
        datasets = list(filter(lambda x: k in os.path.basename(x[-1]), datasets))

    # Return line, date, mouse, region combinations
    return [
        {'mouseline': x[0], 'expdate': x[1], 'mouseid': x[2], 'region': x[3]}
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


def get_output_equivalent(in_path, basein, baseout):
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
    if not os.path.exists(out_path):
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


def save_figs_book(figsroot, figs, prefix=None):
    ''' Save figures dictionary as consecutive pages in single PDF document. '''
    now = datetime.datetime.now()
    s = now.strftime('%Y.%m.%d-%H.%M')
    if prefix is not None:
        s = f'{prefix}_{s}'
    fname = f'figs_{s}.pdf'
    fpath = os.path.join(figsroot, fname)
    file = matplotlib.backends.backend_pdf.PdfPages(fpath)
    for i, (k, v) in enumerate(figs.items()):
        logger.info(f'saving figure "{k}" on page {i}')
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


def save_timeseries_data(timeseries, info_table, ROI_masks, outdir):
    '''
    Save processed timeseries data
    
    :param timeseries: multi-indexed (ROI, run, trial, frame) dataframe containing the z-score timeseries
    :param info_table: dataframe containing the information about experimental parameters for each run
    :param outdir: output directory
    :param ROI_masks: ROI-indexed dataframe of (x, y) coordinates and weights
    '''
    # Create output directory if needed
    if not os.path.isdir(outdir):
        logger.info(f'creating "{outdir}" folder...')
        os.makedirs(outdir)
    # Save experiment info table
    logger.info('saving experiment info table...')
    info_table.to_csv(os.path.join(outdir, f'info_table.csv'))
    # Save timeseries split by run
    logger.info('saving processed timeseries data (split by run)...')
    for ir, df in tqdm(timeseries.groupby(Label.RUN)):
        fpath = os.path.join(outdir, f'timeseries_run{ir}.csv')
        df.to_csv(fpath)
    # Save ROI masks
    logger.info('saving ROI masks...')
    ROI_masks.to_csv(os.path.join(outdir, f'ROI_masks.csv'))
    logger.info('data successfully saved')


def load_timeseries_data(outdir, nruns, check=False):
    '''
    Load processed timeseries data
    
    :param outdir: output directory
    :param nruns: number of runs
    :param check (default: False): whether to sjut check for data availability or to actually load it
    :return: 3-tuple containing multi-indexed (ROI, run, trial, frame) timeseries per ROI,
        experiment info dataframe, and ROI-indexed dataframe of (x, y) coordinates and weights.
    '''
    # Check that output directory exists
    if not os.path.isdir(outdir):
        raise ValueError(f'"{outdir}" does not exist')
    # Check that info table file is present in directory
    info_table_fpath = os.path.join(outdir, f'info_table.csv')
    if not os.path.isfile(info_table_fpath):
        raise ValueError('info table file not found in directory')
    # Check that ROI masks file is present in directory
    ROI_masks_fpath = os.path.join(outdir, f'ROI_masks.csv')
    if not os.path.isfile(ROI_masks_fpath):
        raise ValueError('ROI masks file not found in directory')
    # Check that all timeseries files are present in directory
    timeseriesfiles = glob.glob(os.path.join(outdir, 'timeseries_run*.csv'))
    if len(timeseriesfiles) != nruns:
        raise ValueError(f'number of saved timeseries files ({len(timeseriesfiles)}) does not correspond to number of runs ({nruns})')
    if check:  # check mode -> return None
        return None
    # Load experiment info table
    logger.info('loading experiment info table...')
    info_table = pd.read_csv(info_table_fpath).set_index(Label.RUN, drop=True)
    # Load timeseries split by run
    timeseries = []
    logger.info(f'loading processed timeseries data for {nruns} runs...')
    for ir in tqdm(range(nruns)):
        fpath = os.path.join(outdir, f'timeseries_run{ir}.csv')
        timeseries.append(pd.read_csv(fpath))
    # Concatenate and re-order index
    timeseries = pd.concat(timeseries, axis=0)
    logger.info('re-organizing timeseries index...')
    timeseries.set_index([Label.ROI, Label.RUN, Label.TRIAL, Label.FRAME], inplace=True)
    timeseries.sort_index(inplace=True)
    # Load ROI masks
    logger.info('loading ROI masks...')
    ROI_masks = pd.read_csv(ROI_masks_fpath).set_index(Label.ROI, drop=True)
    logger.info('data successfully loaded')
    return timeseries, info_table, ROI_masks


def check_timeseries_data(outdir, nruns):
    '''
    Check for availability of processed output data
    
    :return: boolean stating whether the data is available
    '''
    try:
        load_timeseries_data(outdir, nruns, check=True)
        logger.info(f'processed timeseries data is available in "{outdir}" directory')
        return True
    except ValueError:
        logger.info(f'processed timeseries data not found in "{outdir}" directory')
        return False


def get_peaks_along_trial(fpath, data, wlen, nseeds):
    '''
    Compute (or load) the detected activity peaks by sliding a detection window
    along the trial interval for evey trial

    :param fpath: output filepath
    :param data: z-score timeseries data
    :param wlen: window length (number of samples)
    :param nseeds: number of starting indexes for the detection window along the interval
    :return: multi-indexed series of detected peaks for each window position
    '''
    if os.path.isfile(fpath):
        logger.info('loading detected peaks data...')
        return pd.read_csv(fpath).set_index(
            [Label.ROI, Label.RUN, Label.TRIAL, Label.ISTART])
    else:
        logger.info(f'detecting activity events in {nseeds} windows along the trial interval for each trial...')
        peaks = slide_along_trial(
            lambda *args, **kwargs: detect_across_trials(find_response_peak, *args, **kwargs),
            data, wlen, nseeds)
        peaks.rename(columns={Label.ZSCORE: Label.PEAK_ZSCORE}, inplace=True)
        logger.info('saving detected peaks data...')
        peaks.to_csv(fpath)
        return peaks


def load_mousereg_dataset(fpath, prefix=None):
    '''
    Load dataset of a particular mouse-region from a CSV file
    
    :param fpath: absolute path to the data file
    :return: multi-indexed dataframe with mouse-region as an extra index dimension
    '''
    fname = os.path.basename(fpath)
    # Load data
    s = 'data'
    if prefix is not None:
        s = f'{prefix} {s}'
    logger.info(f'loading {s} from {fname}')
    data = pd.read_csv(fpath)
    # Add dataset ID column
    dataset_id = os.path.splitext(fname)[0]
    if prefix is not None:
        dataset_id = dataset_id.replace(prefix, '')
    while dataset_id.startswith('_'):
        dataset_id = dataset_id[1:]
    data[Label.MOUSEREG] = dataset_id
    # Re-generate data index 
    data.set_index(Label.MOUSEREG, inplace=True)
    indexcols = [Label.ROI, Label.RUN]
    for k in [Label.TRIAL, Label.FRAME]:
        if k in data.columns:
            indexcols.append(Label.FRAME)
    for k in indexcols:
        if k not in data:
            raise ValueError(f'index field "{k}" not found in "{fname}" dataframe')
        data.set_index(k, append=True, inplace=True)    
    # Return data
    return data


def load_mousereg_datasets(dirpath, **kwargs):
    ''' Load multiple mouse-region datasets '''
    filetypes = ['timeseries', 'stats']
    # List filepaths of each category
    fpaths = {
        k: natsorted(glob.glob(os.path.join(dirpath, f'{k}_*.csv')))
        for k in filetypes
    }
    # Load and concatenate datasets for each category 
    data = {k: pd.concat([
        load_mousereg_dataset(fpath, prefix=k, **kwargs) for fpath in v], axis=0)
        for k, v in fpaths.items()
    }
    # Sort index for each dataset
    logger.info('sorting dataset indexes...')
    data['timeseries'].sort_index(
        level=[Label.MOUSEREG, Label.ROI, Label.RUN, Label.FRAME], inplace=True) 
    data['stats'].sort_index(
        level=[Label.MOUSEREG, Label.ROI, Label.RUN], inplace=True)

    # Add missing change metrics, if any
    for ykey in [Label.ZSCORE, Label.DFF]:
        ykey_resp = f'average {ykey}'
        if ykey_resp not in data['stats']:
            logger.info(f'adding {ykey_resp} metrics to stats dataset...')
            data['stats'][ykey_resp] = apply_in_window(
                lambda x: x.mean(), data['timeseries'], ykey, FrameIndex.RESPONSE)
    # Harmonize run index for for stats dataset
    # logger.info('harmonizing stats run indexes...')
    # data['stats'] = harmonize_run_index(data['stats'])
    return data
