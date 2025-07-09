# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-14 18:28:46
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-07-09 15:52:00

''' Collection of utilities for operations on files and directories. '''

import os
import abc
import glob
import pprint
import datetime
import json
import pandas as pd
import h5py
from multiprocessing import Pool
from tifffile import imsave, TiffFile
import matplotlib.backends.backend_pdf
from tqdm import tqdm
from natsort import natsorted

from .parsers import parse_experiment_parameters, P_TIFFILE, parse_date_mouse_region
from .logger import logger
from .utils import *
from .constants import *
from .postpro import *


def get_data_root(kind=None):
    ''' 
    Get the path to the data root directory (or sub-directory containing a certain type of data)
    
    :param kind (optional): DataRoot folder type
    :return: full path to the data root directory
    '''
    # Get user-specific data root directory from config file
    try:
        from .config import dataroot
    except ModuleNotFoundError:
        raise ValueError(f'user-specific "config.py" file is missing')
    except ImportError:
        raise ValueError(f'"dataroot" variable is missing from user-specific "config.py" file')

    # If data root directory does not exist, raise error
    if not os.path.isdir(dataroot):
        raise ValueError(f'data root directory "{dataroot}" does not exist')

    # If no sub-directory type specified, return root directory
    if kind is None:
        return dataroot
    
    # If sub-directory type specified, check its existence
    subroot = os.path.join(dataroot, kind)
    if not os.path.isdir(subroot):
        raise ValueError(f'data root sub-directory "{subroot}" does not exist')
    
    # Return sub-directory
    return subroot


def get_subfolder_names(dirpath):
    ''' Return a list of sub-folders for a given directory '''
    return [f.name for f in os.scandir(dirpath) if f.is_dir()]


def get_dataset_params(root='.', analysis=DEFAULT_ANALYSIS, excludes=None, includes=['region']):
    '''
    Construct a list of (analysis type, line, date, mouse, region) combinations that contain
    experiment datasets inside a given root directory.
    
    :param root: root directory (typically a mouse line) containing the dataset folders
    :param analysis: string representing the analysis type and determining root sub-directory 
    :param excludes: list of exlusion patterns
    :param includes: list of inclusion patterns
    :return: list of dictionaries representing (line, date, mouse, region) combinations found
        inside the root folder.
    '''
    logger.info(f'searching for data folders in {root} ...')
    datasets = []
    subroot = os.path.join(root, analysis)
    # Loop through lines, dates, mice, and regions, and add data folders to list  
    for line in get_subfolder_names(subroot):
        linedir = os.path.join(subroot, line)
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
        {'analysis': analysis, 'mouseline': x[0], 'expdate': x[1], 'mouseid': x[2], 'region': x[3], 'layer': x[4]}
        for x in datasets]


def restrict_datasets(datasets, **kwargs):
    '''
    Restrict dataset list to those matching specific parameters.

    :param datasets: list of dictionaries representing (analysis type, line, date, mouse, region, layer) combinations
    :param kwargs: dictionary of parameters to match
    '''
    # Restrict parameters to those that are not None
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    # For each specified parameter
    for k, v in kwargs.items():
        # For date, apply loose "startswith" matching (e.g. to enable year-month filtering)
        if k == 'expdate':
            filtfunc = lambda x: x[k].startswith(v)
        # For all other arguments, apply strict matching
        else:
            filtfunc = lambda x: x[k] == v
        logger.info(f'restricting datasets to {k} = {v}')
        datasets = list(filter(filtfunc, datasets))
    
    # Return filtered dataset list
    return datasets


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
        raise ValueError(f'"{dir}" is not a directory')
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
        get_sorted_filelist(dir, pattern=P_TIFFILE)
        return True
    except ValueError:
        return False


def split_path_at(in_path, split_dir):
    '''
    Split an absolute path at a specific parent directory located somewhere upstream in the filesystem.
    
    :param in_path: absolute path to the input file or directory
    :param split_dir: name of the directory at which to split the path (must contain in_path)
    :return: 3-tuple with the upstream path, the split directory, and the downstream path
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
                # Adding exception handling to make process MPI-proof
                try:
                    os.makedirs(pardir)
                except FileExistsError as err:
                    logger.warning(err)
    # Return output path
    return out_path

    
def get_data_folders(basedir, recursive=True, exclude_patterns=[], include_patterns=[], rec_call=False, sortby=None):
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
    # Log
    s = f'searching through {basedir}'
    if rec_call:
        logger.debug(s)
    else:
        logger.info(s)

    # Initialize empty folders list
    datafolders = []

    # For each sub-item in base directory 
    for item in os.listdir(basedir):
        # If sub-item is not an exclusion pattern
        if item not in exclude_patterns:
            # Reconstruct full path to sub-item
            absitem = os.path.join(basedir, item)

            # If sub-item is a directory containing TIF files, add to list
            if is_tif_dir(absitem):
                datafolders.append(absitem)

            # If content item is a directory and recursive call enabled, 
            # call function recursively on child folder and add output to list
            if recursive and os.path.isdir(absitem):
                datafolders += get_data_folders(
                    absitem,
                    exclude_patterns=exclude_patterns, include_patterns=include_patterns,
                    rec_call=True
                )
    
    # Log 
    nfolders = len(datafolders)
    if not rec_call:
        logger.info(f'found {len(datafolders)} folders containing TIF files')

    # Log raw list
    logger.debug(f'raw list: {datafolders}')

    # Filter out excluded folders
    for k in exclude_patterns:
        datafolders = list(filter(lambda x: k not in os.path.basename(x), datafolders))

    # Restrict to included patterns
    for k in include_patterns:
        datafolders = list(filter(lambda x: k in os.path.basename(x), datafolders))

    # Log filtered list
    logger.debug(f'filtered list: {datafolders}')
    if not rec_call and len(datafolders) != nfolders:
        logger.info(f'{len(datafolders)} folders remain after filtering')

    # If sorting key specified, sort output according to it
    if sortby is not None:
        params_by_folder = [parse_experiment_parameters(os.path.basename(f)) for f in datafolders]
        if any(sortby not in p for p in params_by_folder):
            raise ValueError(f'"{sortby}" is not a valid input sorting key')
        if not rec_call:
            logger.info(f'sorting folders by {sortby}')
        vals_by_folder = [p[sortby] for p in params_by_folder]
        _, datafolders = zip(*sorted(zip(vals_by_folder, datafolders)))
        logger.debug(f'sorted list: {datafolders}')

    # Return
    return datafolders


def get_input_files(inputdir, sortby=None):
    '''
    Get input TIF files inside a directory.

    :param inputdir: full path to the input directory.
    :param sortby: sorting key for the input files.
    :return: list of full paths to constituent TIF files.
    '''
    # Get list of TIF files
    tif_files = get_sorted_filelist(inputdir, pattern=P_TIFFILE)
    logger.info(f'found {len(tif_files)} TIF files in "{inputdir}"')

    # Check validity of each input file name by attempting to parse experiment parameters from it,
    # and keep only those that are valid
    valid_tif_files, params_by_file = [], []
    for f in tif_files:
        try:
            params = parse_experiment_parameters(os.path.basename(f))
            params_by_file.append(params)
            valid_tif_files.append(f)
        except ValueError as err:
            logger.warning(f'{err} -> excluding')
    
    # If sorting key specified, sort output according to it
    if sortby is not None:
        if any(sortby not in p for p in params_by_file):
            raise ValueError(f'"{sortby}" is not a valid input sorting key')
        logger.info(f'sorting files by {sortby}')
        vals_by_file = [p[sortby] for p in params_by_file]
        _, valid_tif_files = zip(*sorted(zip(vals_by_file, valid_tif_files)))

    # Construct full paths to TIF files 
    valid_tif_fpaths = [os.path.join(inputdir, f) for f in valid_tif_files]
    
    # Return
    return valid_tif_fpaths


def save_acquisition_settings(folder, daq_settings):
    '''
    Save acquisition settings to JSON file in specific folder.

    :param folder: full path to the folder where to save the acquisition settings
    :param daq_settings: acquisition settings pandas Series object
    :return: full path to the saved JSON file
    '''
    logger.info(f'saving acquisition settings to "{folder}"')
    daq_settings_fpath = os.path.join(folder, 'daq_settings.json')
    with open(daq_settings_fpath, 'w') as f:
        json.dump(daq_settings.to_dict(), f, indent=4)
    return daq_settings_fpath


def load_acquisition_settings(folder):
    '''
    Load acquisition settings from JSON file in specific folder.

    :param folder: full path to the folder where acquisition settings are saved
    :return: acquisition settings pandas Series object
    '''
    daq_settings_fpath = os.path.join(folder, 'daq_settings.json')
    logger.info(f'loading acquisition settings from "{daq_settings_fpath}"')
    with open(daq_settings_fpath, 'r') as f:
        daq_settings = pd.Series(json.load(f))
    return daq_settings


def loadtif(fpath, verbose=True, metadata=False, nchannels=1):
    ''' Load stack/image from .tif file '''
    logfunc = logger.info if verbose else logger.debug
    logfunc(f'loading data from {os.path.basename(fpath)}')
    with TiffFile(fpath) as tif:
        # Load tiff stack
        stack = tif.asarray()
        # If TIF is from scanimage
        if tif.is_scanimage:
            # Load metadata
            meta = tif.scanimage_metadata
            # Check number of channels and reshape TIF stack if necessary
            nchannels = len(meta['FrameData']['SI.hChannels.channelSave'])
        else:
            meta = None
        if stack.ndim < 4 and nchannels > 1:
            logfunc(f'splitting {nchannels} channels {stack.shape} shaped array')
            stack = np.reshape(
                stack, (stack.shape[0] // nchannels, nchannels, *stack.shape[1:]))
        # Reassign metadata to None if not requested for output 
        if not metadata:
            meta = None
    # If stack has more than 2 dimensions (i.e. contains multiple frames), log info
    if stack.ndim > 2:
        logfunc(f'loaded {stack.shape} {stack.dtype} stack')
    # Return
    if meta is None:
        return stack
    else:
        return stack, meta


def get_tif_dtype(fpath):
    ''' Get data type of a TIF file '''
    with TiffFile(fpath) as t:
        dtype = t.pages[0].dtype
    return dtype


def load_tif_metadata(fpath):
    ''' Load metadata from .tif file '''
    with TiffFile(fpath) as tif:
        return tif.scanimage_metadata
    

def load_tif_nframes(fpath, nchannels=1):
    ''' 
    Load number of frames from .tif file
    
    :param fpath: full path to input TIF file
    :param nchannels: number of channels in the TIF file
    :return: number of frames per channel in the TIF file
    '''
    with TiffFile(fpath) as tif:
        npages = len(tif.pages)
    return npages // nchannels


def savetif(fpath, stack, overwrite=True, metadata=None):
    ''' Save stack/image as .tif file '''
    move_forward = check_for_existence(fpath, overwrite)
    if not move_forward:
        return
    if stack.ndim > 2:
        logger.info(f'saving {stack.shape} {stack.dtype} stack as "{fpath}"...')
    imsave(fpath, stack, metadata=metadata)


def get_stack_frame_aggregate(fpath, aggfunc=None):
    '''
    Load TIF file and compute frame-aggregate metrics for each frame
    
    :param fpath: full path to input TIF file
    :param aggfunc: aggregation function (default = average)
    :return: frame-average array
    '''
    if aggfunc is None:
        aggfunc = np.mean
    # Load TIF stack
    stack = loadtif(fpath)
    # Return frame aggregate metrics
    return aggfunc(stack, axis=(-2, -1))


class StackProcessor(metaclass=abc.ABCMeta):
    ''' Generic intrface for processor objects '''

    def __init__(self, overwrite=False, warn_if_exists=True, nchannels=1):
        self.overwrite = overwrite
        self.warn_if_exists = warn_if_exists
        self.nchannels = nchannels

    @abc.abstractmethod
    def run(self, stack: np.array) -> np.ndarray:
        ''' Abstract run method. '''
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def code(self):
        ''' Abstract code attribute. '''
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def rootcode(self):
        ''' Abstract root code attribute '''
        raise NotImplementedError

    def get_target_fname(self, fname):
        ''' Default method for target file name, conserving input name '''
        return fname

    def get_target_fpath(self, fpath):
        ''' Get target (i.e. post-processing) file path for an input filepath '''
        # Get output filepath
        fpath = get_output_equivalent(
            fpath, self.input_root, f'{self.rootcode}/{self.code}')
        # Modify output filepath according to processor
        fdir, fname = os.path.split(fpath)
        return os.path.join(fdir, self.get_target_fname(fname))
    
    def load_stack(self, fpath):
        ''' Load stack from TIF file '''
        return loadtif(fpath, nchannels=self.nchannels)
   
    def save_stack(self, fpath, stack, **kwargs):
        ''' Save stack to TIF file. '''
        savetif(fpath, stack, **kwargs)

    def _run(self, stack):
        '''
        Wrapper around run method to handle multi-channel stacks

        :param stack: input stack array
        :return: output stack array (s)
        '''
        # If stack is 3D, simply process it 
        if stack.ndim == 3:
            return self.run(stack)
        
        # If stack is 4D (i.e., multi-channel), process each channel (2nd dimension) 
        # separately and recombine
        elif stack.ndim == 4:
            chax = 1  # channel axis
            outstack = []
            for i in range(stack.shape[chax]):
                logger.info(f'working on channel {i + 1}...')
                outstack.append(self.run(stack[:, i]))
            return np.stack(outstack, axis=chax)
        
        # Otherwise, raise error
        else:
            raise ValueError(f'input stack has unsupported shape: {stack.shape}')
    
    def load_run_save(self, input_fpath):
        '''
        Load input stack file, apply processing function and save output stack in specific directory.
        
        :param input_fpath: absolute filepath to the input TIF stack (or list of TIF stacks).
        :return: absolute filepath to the output TIF stack (or list of TIF stacks).
        '''
        # Check that input root is indeed found in input filepath
        if self.input_root not in input_fpath:
            raise ValueError(f'input root "{self.input_root}" not found in input file path "{input_fpath}"')
        
        # If NoProcessor object provided -> do nothing and return input file path
        if isinstance(self, NoProcessor):
            return input_fpath
        
        # Get post-processing output filepath
        output_fpath = self.get_target_fpath(input_fpath)
        
        # If already existing, act according to overwrite parameter
        if os.path.isfile(output_fpath):
            if self.warn_if_exists:
                logger.warning(f'"{output_fpath}" already exists')
            overwrite = parse_overwrite(self.overwrite)
            if not overwrite:
                return output_fpath
        else:
            overwrite = self.overwrite
        
        # Load input
        input_stack = self.load_stack(input_fpath)

        # Process stack
        output_stack = self._run(input_stack)

        # Save output
        self.save_stack(output_fpath, output_stack, overwrite=overwrite)
        
        # Return output filepath
        return output_fpath
    
    @staticmethod
    def get_dtype_bounds(dtype):
        '''
        Get numerical bounds of data type
        
        :param dtype: data type
        :return: minimum and maximum allowed numerical values for data type
        '''
        info_func = np.finfo if np.issubdtype(dtype, np.floating) else np.iinfo
        dinfo = info_func(dtype)
        return dinfo.min, dinfo.max
    
    @classmethod
    def adapt_stack_range(cls, stack, dtype):
        '''
        Adapt stack data range to fit within numerical bounds of reference data type.

        :param stack: image stack array
        :param dtype: reference data type
        :return: adapted stack
        '''
        # Get input data type bounds
        dmin, dmax = cls.get_dtype_bounds(dtype)

        # If values lower than lower bound are found, offset stack accordingly
        if stack.min() < dmin:
            logger.warning(f'values lower than {dmin} found in corrected stack -> offseting')
            stack = stack - stack.min() + dmin + 1

        # If values higher than input data type maximum are found, 
        # rescale stack within bounds
        if stack.max() > dmax:
            logger.warning(f'values higher than {dmax} found in corrected stack -> rescaling')
            ratio = 0.5 * dmax / stack.max()
            stack = stack * ratio

        # Return adapted stack        
        return stack
    
    @classmethod
    def check_stack_range(cls, stack, dtype):
        '''
        Check that stack data range fits within numerical bounds of reference data type.

        :param stack: image stack array
        :param dtype: reference data type
        '''
        # Get input data type numerical bounds
        dmin, dmax = cls.get_dtype_bounds(dtype)
        # Get stack value bounds
        vbounds = stack.min(), stack.max()
        # Check that value bounds fit within data type bounds, raise error otherwise
        for v in vbounds:
            if not dmin <= v <= dmax:
                raise ValueError(f'stack data range {vbounds} is outside of {dtype} data type bounds: {dmin, dmax}') 


class NoProcessor(StackProcessor):
    ''' Dummy class for no-processor objects '''

    def run(self, stack: np.array, *args):
        raise NotImplementedError
    
    def get_target_fpath(self, fpath):
        return fpath

    @property
    def ptype(self):
        return self.__class__.__name__[2:].lower()

    def __str__(self) -> str:
        return f'no {self.ptype}'

    @property
    def code(self):
        return f'no_{self.ptype}'
    
    @property
    def rootcode(self):
        raise NotImplementedError


def process_and_save(processor, input_fpath, input_root, overwrite=False, warn_if_exists=True, mpi=False):
    '''
    Wrapper around StackProcessor load_run_save method

    :param processor: processor object
    :param input_fpath: absolute filepath to the input TIF stack (or list of TIF stacks).
    :param input_root: name of the directory that constitutes the root level of the input filepath
    :param overwrite: one of (True, False, '?') defining what to do if file already exists.
    :param: whether to use multiprocessing or not
    :return: absolute filepath to the output TIF stack (or list of TIF stacks).
    '''
    # Pass on overwrite and input_root to processor
    processor.overwrite = overwrite
    processor.input_root = input_root
    processor.warn_if_exists = warn_if_exists
    # If list of filepaths provided as input
    if is_iterable(input_fpath):
        # If they all exist already and overwrite set to False -> return directly
        output_fpaths = list(map(processor.get_target_fpath, input_fpath))
        if overwrite == False and all(os.path.isfile(x) for x in output_fpaths):
            logger.info('all output files already exist -> skipping')
            return output_fpaths
        # Otherwise, apply function to all of them, with/without multiprocessing 
        else:
            if mpi:
                with Pool() as pool:
                    return pool.map(processor.load_run_save, input_fpath)
            else:
                return list(map(processor.load_run_save, input_fpath))
    # Call processor method
    return processor.load_run_save(input_fpath)


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
    today = datetime.date.today().strftime('%Y.%m.%d')
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


def save_rowavg_dataset(fpath, dFF, info_table):
    '''
    Save row-average dFF dataset into HDF5 file.

    :param fpath: absolute path to data file
    :param dFF: (run, trial, time) row-average dFF series
    :param info_table: dataframe containing information about each acquisition run
    :return: None
    '''
    # Remove output file if it exists
    if os.path.isfile(fpath):
        os.remove(fpath)
    # Create HDF store object
    with pd.HDFStore(fpath) as store:
        # Save row-average dFF data
        logger.info('saving row-average dFF data...')
        store['rowavg_dFF'] = dFF
        # Save experiment info table
        logger.info('saving experiment info table...')
        store['info_table'] = info_table
    # Log success
    logger.info('data successfully saved')


def load_rowavg_dataset(fpath):
    '''
    Load row-average dFF dataset from HDF5 file.

    :param fpath: absolute path to data file
    :return: (run, trial, time) row-average dFF series, and run-indexed info table
    '''
    # Check that output file is present in directory
    if not os.path.isfile(fpath):
        raise FileNotFoundError('row-average dFF data file not found in directory')
    # Create HDF store object
    with pd.HDFStore(fpath) as store:
        # Load row-average dFF data
        logger.info(f'loading row-average dFF data from {os.path.basename(fpath)}')
        dFF = store['rowavg_dFF']
        # Load experiment info table
        logger.info('loading experiment info table...')
        info_table = store['info_table']
    return dFF, info_table


def save_conditioned_dataset(fpath, timeseries, popagg_timeseries, info_table, ROI_masks, isch2ROI=None):
    '''
    Save conditioned dataset into HDF5 file. 
    
    :param fpath: absolute path to data file
    :param timeseries: multi-indexed (ROI, run, trial, frame) timeseries dataframe
    :param popagg_timeseries: multi-indexed (run, trial, frame) population-average timeseries dataframe
    :param info_table: dataframe containing the information about experimental parameters for each run
    :param ROI_masks: ROI-indexed dataframe of (x, y) coordinates and weights
    :param isch2ROI (optional): ROI-indexed series defining whether each ROI is also detected on channel 2 (for 2-channel data only)
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
        logger.info('saving conditioned timeseries data...')
        store['timeseries'] = timeseries
        # Save population-average timeseries table
        logger.info('saving conditioned population-average timeseries data...')
        store['popagg_timeseries'] = popagg_timeseries
        # Save ROI masks
        logger.info('saving ROI masks...')
        store['ROI_masks'] = ROI_masks
        # Save isch2ROI mapping if provided
        if isch2ROI is not None:
            logger.info('saving isch2ROI mapping...')
            store['isch2ROI'] = isch2ROI
    logger.info('data successfully saved')


def load_conditioned_dataset(fpath):
    '''
    Load conditioned dataset from HDF5 file
    
    :param fpath: absolute path to data file
    :return: 3-tuple containing multi-indexed (ROI, run, trial, frame) timeseries dataframe,
        experiment info dataframe, and ROI-indexed dataframe of (x, y) coordinates and weights.
    '''
    # Check that output file is present in directory
    if not os.path.isfile(fpath):
        raise FileNotFoundError('conditioned data file not found in directory')
    # Create HDF store object
    with pd.HDFStore(fpath) as store:
        # Load experiment info table
        logger.info('loading experiment info table...')
        info_table = store['info_table']
        # Load timeseries table
        logger.info('loading conditioned timeseries data...')
        timeseries = store['timeseries']
        # Load population-average timeseries table
        logger.info('loading conditioned population-average timeseries data...')
        popagg_timeseries = store['popagg_timeseries']
        # Load ROI masks
        logger.info('loading ROI masks...')
        ROI_masks = store['ROI_masks']
        # Load isch2ROI mapping if present
        isch2ROI = None
        if 'isch2ROI' in store:
            logger.info('loading isch2ROI mapping...')
            isch2ROI = store['isch2ROI']
    logger.info('data successfully loaded')
    return timeseries, popagg_timeseries, info_table, ROI_masks, isch2ROI


def save_processed_dataset(fpath, trialagg_timeseries, popagg_timeseries, stats, triagg_stats, ROI_masks, map_ops):
    '''
    Save processed dataset to a HDF5 file
    
    :param fpath: absolute path to data file
    :param trialagg_timeseries: multi-indexed (ROI, run, frame) trial-aggregated timeseries dataframe
    :param popagg_timeseries: multi-indexed (run, trial, frame) population average timeseries dataframe
    :param stats: multi-indexed (ROI, run, trial) stats dataframe
    :param triagg_stats: multi-indexed (ROI, run) trial-aggregated stats dataframe
    :param ROI_masks: ROI-indexed dataframe of (x, y) coordinates and weights
    :param map_ops: dictionary containing the necessary fields to plot cell maps
    '''
    # Remove output file if it exists
    if os.path.isfile(fpath):
        logger.info(f'deleting pre-existing data file...')
        os.remove(fpath)
    # Create HDF stream to store pandas objects
    with pd.HDFStore(fpath) as store:
        # Save trial aggregated timeseries table
        logger.info('saving trial-aggregated timeseries data...')
        store['trialagg_timeseries'] = trialagg_timeseries
        # Save population-average timeseries table
        logger.info('saving population-average timeseries data...')
        store['popagg_timeseries'] = popagg_timeseries
        # Save extended stats table
        logger.info('saving extended stats data...')
        store['stats'] = stats
        # Save trial-aggregated stats table
        logger.info('saving trial-aggregated stats data...')
        store['triagg_stats'] = triagg_stats
        # Save ROI masks
        logger.info('saving ROI masks...')
        store['ROI_masks'] = ROI_masks
        # Save map_ops in the same object
        logger.info('saving mapping options...')
        store['map_ops'] = pd.Series(map_ops)
    logger.info(f'all data fields saved to "{fpath}"')


def load_processed_dataset(fpath):
    '''
    Load processed data of a particular date-mouse-region dataset from a HDF5 file
    
    :param fpath: absolute path to data file
    :return: multi-indexed timeseries and stats dataframes
    '''
    # Check that output file is present in directory
    if not os.path.isfile(fpath):
        raise FileNotFoundError('processed data file not found in directory')
    # Create HDF store object
    with pd.HDFStore(fpath) as store:
        # Load data
        logger.info(f'loading mouse-region data from {os.path.basename(fpath)}')
        trialagg_timeseries = store['trialagg_timeseries']
        popagg_timeseries = store['popagg_timeseries']
        stats = store['stats']
        trialagg_stats = store['triagg_stats']
        ROI_masks = store['ROI_masks']
        map_ops = store['map_ops'].to_dict()
    # Return as dictionary
    return {
        'trialagg_timeseries': trialagg_timeseries,
        'popagg_timeseries': popagg_timeseries,
        'stats': stats,
        'trialagg_stats': trialagg_stats,
        'ROI_masks': ROI_masks,
        'map_ops': map_ops
    }


def load_processed_datasets(dirpath, layer=None, include_patterns=None, exclude_patterns=None,
                            on_duplicate_runs='raise', harmonize_runs=True, include_mode='all', 
                            **kwargs):
    '''
    Load multiple mouse-region datasets
    
    :param layer: cortical layer
    :param include_patterns (optional): inclusion pattern(s)
    :param exclude_patterns (optional): exclusion pattern(s)
    :param on_duplicate_runs (optional): what to do if duplicate runs are found
    :param harmonize_runs: whether run index should be harmonized across datasets according to some condition
    '''
    if not os.path.isdir(dirpath):
        raise ValueError(f'"{dirpath}" is not a directory')
    
    # List data filepaths
    fpaths = natsorted(glob.glob(os.path.join(dirpath, f'*.h5')))

    # If layer specified
    if layer is not None:
        # If default layer, remove any files with "layer" in filename 
        if layer == DEFAULT_LAYER:
            if exclude_patterns is not None:
                exclude_patterns.append('layer')
            else:
                exclude_patterns = ['layer']
        # Otherwise, add it to include patterns
        else:
            if include_patterns is not None:
                include_patterns.append(layer)
            else:
                include_patterns = [layer]

    # Filter according to inclusion & exclusion patterns, if any
    if include_patterns is not None:
        include_patterns = as_iterable(include_patterns)
        logger.warning(f'excluding datasets not having {include_mode} of the following patterns:\n{itemize(include_patterns)}')
        if include_mode == 'all':
            fpaths = list(filter(lambda x: all(e in x for e in include_patterns), fpaths))
        else:
            fpaths = list(filter(lambda x: any(e in x for e in include_patterns), fpaths))
    if exclude_patterns is not None:
        exclude_patterns = as_iterable(exclude_patterns)
        logger.warning(f'excluding datasets with the following patterns:\n{itemize(exclude_patterns)}')
        fpaths = list(filter(lambda x: not any(e in x for e in exclude_patterns), fpaths))
    
    # Get dataset IDs
    logger.info('gathering dataset IDs...')
    dataset_ids = []
    for fpath in fpaths:
        fname = os.path.basename(fpath)
        dataset_id = os.path.splitext(fname)[0]
        while dataset_id.startswith('_'):
            dataset_id = dataset_id[1:]
        dataset_ids.append(dataset_id)
    
    # Load timeseries and stats datasets
    datasets = [load_processed_dataset(fpath) for fpath in fpaths]
    if len(datasets) == 0:
        raise ValueError(f'no valid datasets found in "{dirpath}"')
        
    # For each dataset
    for i, dataset_id in enumerate(dataset_ids):
        # Check for potential run duplicates in trial aggregated stats
        dup_table = get_duplicated_runs(datasets[i], **kwargs)
        # If dupliactes are found
        if dup_table is not None:
            dupstr = f'duplicated runs in {dataset_id}:\n{dup_table}'
            # Raise error if specified
            if on_duplicate_runs == 'raise':
                raise ValueError(dupstr)
            # Otherwise, issue warning
            else:
                logger.warning(dupstr)
                # If specified, drop one first of these runs
                if on_duplicate_runs.startswith('drop'):
                    # Drop first or last run according to specified mode
                    idrop = dup_table.index[0] if on_duplicate_runs == 'drop_first' else dup_table.index[-1]
                    logger.warning(f'dropping run {idrop} from dataset...')
                    for k, v in datasets[i].items():
                        if isinstance(v, (pd.DataFrame, pd.Series)) and Label.RUN in v.index.names:
                            datasets[i][k] = v.drop(idrop, level=Label.RUN)
                    
    # Concatenate datasets while adding their respective IDs
    logger.info('assembling data structures...')
    keys = list(datasets[0].keys())
    data_structures = {}
    for k in keys:
        ds = [d[k] for d in datasets]
        if isinstance(ds[0], (pd.DataFrame, pd.Series)):
            data_structures[k] = pd.concat(ds, keys=dataset_ids, names=[Label.DATASET])
        else:
            data_structures[k] = dict(zip(dataset_ids, ds))

    # If specified, harmonize run index of all run-indexed pandas objects 
    # across datasets according to some condition
    if harmonize_runs:
        dfs = {k: v for k, v in data_structures.items() if 
               isinstance(v, (pd.DataFrame, pd.Series)) and Label.RUN in v.index.names} 
        dfs = harmonize_run_index(dfs, **kwargs)
        for k, v in dfs.items():
            data_structures[k] = v

    # Sort index for each data structure
    logger.info('sorting data structures index...')
    for k in data_structures.keys():
        if isinstance(data_structures[k], (pd.DataFrame, pd.Series)):
            data_structures[k].sort_index(
                level=list(data_structures[k].index.names), inplace=True)
    
    # Process run IDs to uncover run sequences
    logger.info('processing run IDs to uncover run sequences...')
    for k, v in data_structures.items():
        if isinstance(v, pd.DataFrame) and Label.RUNID in v.columns:
            data_structures[k][Label.RUNID] = process_runids(v[Label.RUNID])

    # Return data_structures
    return data_structures


def load_lineagg_data(dirpath, errprop='intra'):
    '''
    Load line-aggregated data for multiple mouse lines
    
    :param dirpath: path to input directory (or dictionary of paths to input directories)
    :param errprop: error propagation method (intra/inter)
    :return: 2-tuple with:
        - multi-indexed (line, responder type, run, trial) stats dataframe
        - multi-indexed (line, dataset) cell count series
    '''
    logger.info(f'loading line-average data (with {errprop}-propagated SE)')

    # Dictionary of files fuffixes (with appropriate propagation method)
    suffixes = {
        'stats': f'_{errprop}.csv',
        'counts': '_counts.csv'
    }

    # Initialize empty data dictionary
    data = {}

    # If input is a dictionary of paths
    if isinstance(dirpath, dict):
        # Fill in data dictionary with empty stats and counts dictionaries
        data['stats'] = {}
        data['counts'] = {}

        # Go through each line and associated directory
        for line, dirp in dirpath.items():
            logger.info(f'loading {line} data from {dirp} folder')
            # Initialize empty line data dictionary
            linedata = {}
            # Attempt to load line data
            try:
                # For each file type
                for ftype, suffix in suffixes.items():
                    # Construct file path
                    fpath = os.path.join(dirp, f'{line}{suffix}')
                    # If file does not exist, raise error
                    if not os.path.isfile(fpath):
                        raise FileNotFoundError(f'"{fpath}" not found')
                    # Load file content and add to data dictionary
                    linedata[ftype] = pd.read_csv(fpath)

                # Add line data fields to global data dictionary
                for k, v in linedata.items():
                    data[k][line] = v

            # If line data could not be loaded, issue warning and continue
            except FileNotFoundError as err:
                logger.warning(err)
                continue
        
        # Concatenate stats and counts as separates dataframes
        stats = pd.concat(data['stats'], names=[Label.LINE])
        counts = pd.concat(data['counts'], names=[Label.LINE])
    
    # Otherwise, if input is a single path
    else:                      
        # Define file name patterns for each file type
        fpatterns = {k: f'*{v}' for k, v in suffixes.items()}

        logger.info(f'input folder: {dirpath}')
        # Load and concatenate data for each file type
        for k, pattern in fpatterns.items():
            fpaths = natsorted(glob.glob(os.path.join(dirpath, pattern)))
            if len(fpaths) == 0:
                raise ValueError(f'no {k} data (pattern = {pattern}) found in "{dirpath}"')
            data[k] = pd.concat([pd.read_csv(fpath) for fpath in fpaths])

        # Unpack data
        stats, counts = data['stats'], data['counts']
    
    # Create stats multi-index
    muxcols = [Label.LINE, Label.ROI_RESP_TYPE, Label.RUN]
    stats.index = pd.MultiIndex.from_arrays([stats.pop(k) for k in muxcols])

    # Create counts multi-index
    muxcols = [Label.LINE, Label.DATASET]
    counts.index = pd.MultiIndex.from_arrays([counts.pop(k) for k in muxcols])
    counts = counts[Label.ROI_COUNT]

    # Check that all datasets have the same number of runs
    lines = stats.index.unique(level=Label.LINE)
    countlines = counts.index.unique(level=Label.LINE)
    if not lines.equals(countlines):
        raise ValueError(f'inconsistent mouse lines between stats ({lines}) and counts data ({countlines})')

    # Return stats and counts
    logger.info(f'line-aggregated data successfully loaded for lines {lines.values}')
    return stats, counts


def extract_reference_distribution(fpath, projfunc, verbose=True):
    '''
    Extract pixel intensity distribution from reference image of a TIF file.

    :param fpath: Path to tif file.
    :param projfunc: Projection function to get reference image from 3D stack.
    :param verbose: Whether to print verbose output.
    :return: Reference distribution as pandas Series.
    '''
    # Load tif stack
    stack = loadtif(fpath, verbose=verbose)
    # Compute reference image 
    refimg = projfunc(stack, axis=0)
    # Create series from serialized pixel distribution
    refdist = pd.Series(refimg.ravel(), name='intensity')
    refdist.index.name = 'pixel'
    # Return reference distribution
    return refdist


def extract_reference_distributions(folder, projfunc, *args, **kwargs):
    '''
    Extract reference distributions from all TIF files in a folder

    :param folder: Path to folder containing TIF files.
    :param projfunc: Projection function to get reference image from 3D stack.
    :return: Reference distributions as multi-indexed pandas Series.
    '''
    # Check if reference distributions have already been extracted
    output_fpath = os.path.join(folder, f'{projfunc.__name__}_refdists.csv') 

    # If so, load them
    if os.path.exists(output_fpath):
        logger.info(f'loading reference distributions from {os.path.basename(folder)} folder')
        refdists = pd.read_csv(output_fpath, index_col=['file', 'pixel'])['intensity']

    # Otherwise, extract and save them
    else:
        logger.info(f'extracting stack {projfunc.__name__} projection images from {os.path.basename(folder)} folder')
        fpaths = glob.glob(os.path.join(folder, '*.tif'))
        fnames = [os.path.basename(fpath) for fpath in fpaths]
        refdists = []
        for fpath in tqdm(fpaths):
            refdists.append(
                extract_reference_distribution(fpath, projfunc, *args, verbose=False, **kwargs))
        refdists = pd.concat(
            refdists, axis=0, keys=fnames, names=['file'])
        logger.info(f'saving reference distributions to {os.path.basename(folder)} folder')
        refdists.to_csv(output_fpath)
    
    # Return reference distributions
    return refdists


def save_post_window_size(dirpath, n):
    '''
    Save information about the size of a post-stimulus window in a file.

    :param dirpath: path to directory where to save the file.
    :param n: size of the post-stimulus window (in number of samples)
    '''
    # Check that directory exists
    if not os.path.isdir(dirpath):
        raise ValueError(f'"{dirpath}" is not a directory')
    
    # Assemble file path
    fpath = os.path.join(dirpath, 'post_window_size.txt')
    
    # Log saving/overwiting process
    if os.path.isfile(fpath):
        logger.warning(f'overwriting post window size in "{fpath}"')
    else:
        logger.info(f'saving post window size in "{fpath}"')
    
    # Save window size
    with open(fpath, 'w') as file:
        file.write(str(n))


def load_post_window_size(dirpath):
    '''
    Load information about the size of a post-stimulus window from a file.

    :param dirpath: path to directory where to load the file.
    :return: size of the post-stimulus window (in number of samples)
    '''
    # Check that directory exists
    if not os.path.isdir(dirpath):
        raise ValueError(f'"{dirpath}" is not a directory')
    
    # Assemble file path
    fpath = os.path.join(dirpath, 'post_window_size.txt')
    
    # Check that file exists
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f'"{fpath}" not found')
    
    # Load window size
    with open(fpath, 'r') as file:
        return int(file.read())

