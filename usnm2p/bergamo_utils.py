# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-08-15 16:34:13
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2024-08-09 14:34:44

''' Utilities for Bergamo data pre-processing '''

# External packages
import os
import numpy as np

# Internal modules
from .constants import *
from .utils import get_singleton
from .logger import logger
from .parsers import get_info_table, find_suffixes, group_by_run
from .correctors import correct_tifs
from .resamplers import resample_tifs
from .substitutors import StackSubstitutor
from .fileops import load_tif_metadata, process_and_save, get_input_files, get_subfolder_names, get_dataset_params, restrict_datasets, split_path_at, load_acquisition_settings, loadtif, savetif
from .stackers import TifStacker
from .fileops import get_output_equivalent, save_acquisition_settings


def parse_scanimage_meta(fpath):
    '''
    Parse ScanImage metadata from TIF file

    :param fpath: full path to TIF file
    :return: metadata dictionary
    '''
    # Load metadata from TIF file
    meta = load_tif_metadata(fpath)

    # Restrict to FrameData information
    meta = meta['FrameData']

    # Remove unnecessary keys
    delkeys = [
        'SI.LINE_FORMAT_VERSION',
        'SI.TIFF_FORMAT_VERSION',
        'SI.VERSION_COMMIT',
        'SI.VERSION_MAJOR',
        'SI.VERSION_MINOR',
        'SI.hScan2D.mask',
        'SI.hWaveformManager.optimizedScanners',
        'SI.hStackManager.stackStartPower',
        'SI.hStackManager.stackZEndPos',
        'SI.hUserFunctions.userFunctionsUsr',
        'SI.hUserFunctions.userFunctionsCfg',
        'SI.hMotionManager.motionMarkersXY',
        'SI.hBeams.extPowerScaleFnc',
        'SI.hBeams.pzCustom',
        'SI.hDisplay.selectedZs',
        'SI.hFastZ.zAlignment',
        'SI.hMotors.userDefinedPositions',
        'SI.hPhotostim.sequenceSelectedStimuli',
        'SI.hPhotostim.stimSelectionAssignment',
        'SI.hPhotostim.stimSelectionTerms',
        'SI.hPhotostim.stimSelectionTriggerTerm',
        'SI.hPhotostim.syncTriggerTerm',
        'SI.hScan2D.channels',
        'SI.hStackManager.actualStackZStepSize',
        'SI.hStackManager.stackEndPower',
        'SI.hStackManager.stackZStartPos',
        'SI.hBeams.powerBoxes.powers',
        'SI.hFastZ.positionAbsoluteRaw',
        'SI.hPmts.bandwidths',
        'SI.hPmts.offsets',
        'SI.hChannels.channelLUT', 
        'SI.hChannels.channelOffset',
        'SI.hMotors.axesPosition',
        'SI.hMotors.motorPosition',
        'SI.hMotors.samplePosition',
        'SI.hScan2D.channelOffsets',
        'SI.hStackManager.zPowerReference',
        'SI.hStackManager.zs',
    ]
    for k in delkeys:
        del meta[k]

    # Extract coordinates of the corners of the imaging field of view in microns
    xyfov_um = dict(zip(['ll', 'lr', 'tr', 'ul'], np.array(meta['SI.hRoiManager.imagingFovUm'])))

    # Compute size of field of view in microns, and add to metadata dictionary
    fovdims_um = xyfov_um['tr'] - xyfov_um['ll']
    assert fovdims_um[0] == fovdims_um[1]
    meta['imagingFovUm'] = fovdims_um[0]
    
    # Add micronsPerPixel key, calculated from objective resolution and scan zoom factor
    meta['micronsPerPixel'] = meta['SI.objectiveResolution'] / meta['SI.hRoiManager.scanZoomFactor'] 

    # Add explicit key for frame period
    meta['framePeriod'] = 1. / meta['SI.hRoiManager.scanFrameRate']  # in seconds

    # Return metadata dictionary
    return meta


def stack_trial_tifs(stacker, in_fpaths, out_fpath, **kwargs):
    '''
    Stack TIFs of consecutive trials together
    
    :param stacker: TifStacker object
    :param in_fpaths: absolute paths to input trial stacks
    :param out_fpath: absolute path to output stack
    :param save_meta (default=True): whether to save metadata to JSON file or not
    :return: metadata shared across all trials, as a pandas Series
    '''
    # Load metadata from all TIF files
    codes = find_suffixes(in_fpaths)
    meta_by_trial = {}
    for fpath, code in zip(in_fpaths, codes):
        meta_by_trial[code] = pd.Series(parse_scanimage_meta(fpath))
    meta_by_trial = pd.concat(meta_by_trial, axis=1, names='trial')

    # Check that metadata is consistent across all TIF files
    ref_meta = meta_by_trial.mode(axis=1).iloc[:, 0].rename('settings')

    # If some settings vary across runs, raise an error
    ismatch = meta_by_trial.eq(ref_meta, axis=0).all(axis=1)
    nonmatching_settings = ismatch[~ismatch].index.values
    if len(nonmatching_settings) > 0:
        raise ValueError(f'Inconsistent metadata across trials for the following settings: {nonmatching_settings}: \n{meta_by_trial.loc[nonmatching_settings, :]}')

    # Add number of trial files to reference metadata
    ref_meta['nTrials'] = len(in_fpaths)

    # Rename explicitly number of frames per trial for compatibility with analysis pipeline
    ref_meta['nFramesPerTrial'] = ref_meta['SI.hStackManager.framesPerSlice']

    # Stack trial TIFs together
    stacker.stack(in_fpaths, out_fpath, **kwargs)

    # Return reference metadata
    return ref_meta


def stack_trial_tifs_across_runs(input_fpaths, input_key, align=False, save_meta=True, **kwargs):
    '''
    Stack trial TIFs for each run identified in a file list.
    
    :param input_fpaths: absolute paths to input stacks
    :param input_key: input key for output path replacement
    :param align (default=False): whether to align stacks or not
    :param save_meta (default=True): whether to save metadata to JSON file or not
    :return: filepaths to the created tif stacks per run
    '''
    # Get TIF stacker object
    stacker = TifStacker(input_type='stack', align=align)

    # Get input and output directories
    input_dir = os.path.split(input_fpaths[0])[0]
    outdir = get_output_equivalent(input_dir, input_key, stacker.code)

    # Group filepaths by run
    fpaths_by_run = group_by_run(input_fpaths, on_mismatch='warn', key_type='fname')
    logger.info(f'stacking trial TIFs across {len(fpaths_by_run)} runs')

    # Initialize output filepaths list
    output_fpaths = []

    # Initialize dictionary of metadata by run
    meta_by_run = {}

    # For each run
    for run_fname, trial_fpaths in fpaths_by_run.items():
        # Get output filepath
        out_fpath = os.path.join(outdir, run_fname)

        # Stack trial TIFs together, and store shared trials metadata
        meta_by_run[run_fname] = stack_trial_tifs(stacker, trial_fpaths, out_fpath, **kwargs)

        # Append output filepath to list
        output_fpaths.append(out_fpath)
    
    # Concatenate metadata by run
    meta_by_run = pd.concat(meta_by_run, axis=1, names='run')

    # Compare metadata across runs
    ref_meta = meta_by_run.mode(axis=1).iloc[:, 0].rename('reference')
    ismatch = meta_by_run.eq(ref_meta, axis=0).all(axis=1)
    nonmatching_settings = ismatch[~ismatch].index.values
    if len(nonmatching_settings) > 0:
        raise ValueError(f'Inconsistent metadata across runs for the following settings: {nonmatching_settings}: \n{meta_by_run.loc[nonmatching_settings, :]}')

    # Save metadata to JSON file, if requested
    if save_meta:
        save_acquisition_settings(outdir, ref_meta)

    # Return list of output filepaths
    return output_fpaths


def split_multichannel_tifs(input_fpaths, input_key, **kwargs):
    '''
    Split channels for each stack file in a list
    
    :param input_fpaths: list of absolute paths to input stack files 
    :param input_key: input key for output path replacement
    :return: filepaths to the created tif stacks per run
    '''
    # Initialize list of output filepaths
    output_fpaths = []

    # Load daq settings from first input file
    meta = load_acquisition_settings(os.path.dirname(input_fpaths[0]))

    # Extract number of channels from DAQ metadata
    nchannels = len(meta['SI.hChannels.channelSave'])

    # For each multi-channel stack file
    for input_fpath in input_fpaths:
        
        # Initialize output file found flag
        output_found = False

        # Check iteratively for existing output files
        for ichannel in range(nchannels):
            # Derive output key for current channel
            output_key = f'{DataRoot.SPLIT}/channel{ichannel + 1}'
            # Assemble associated output directory
            channeldir = os.path.join(split_path_at(input_fpath, input_key)[0], output_key)
            # If directory exists
            if os.path.isdir(channeldir):
                # Check if output file foir that channel already exists
                output_fpath_check = get_output_equivalent(
                    input_fpath, input_key, output_key)
                
                # If output file exists, append to output list and move to next channel
                if os.path.isfile(output_fpath_check):
                    logger.warning(f'{output_fpath_check} already exists -> skipping')
                    output_found = True
                    output_fpaths.append(output_fpath_check)

        # If no output output files were found, perform the split
        if not output_found:
            # Load input stack
            stack = loadtif(input_fpath, nchannels=nchannels)
            # If stack has more than 3 dimensions (i.e. multi-channel)
            if stack.ndim > 3:
                # Extract number of channels
                nchannels = stack.shape[1]
                # Loop through stack channels
                for ichannel in range(nchannels):
                    # Derive channel output filepath
                    output_fpath = get_output_equivalent(
                        input_fpath, input_key, f'{DataRoot.SPLIT}/channel{ichannel + 1}')
                    # Save channel data to specific file
                    savetif(output_fpath, stack[:, ichannel], **kwargs)
                    # Append output filepath to list
                    output_fpaths.append(output_fpath)

    # If existing, save DAQ metadata to output directory of each channel
    pardir = os.path.dirname(input_fpaths[0])
    daq_file = os.path.join(pardir, 'daq_settings.json')
    if os.path.exists(daq_file):
        for output_fpath in output_fpaths[-nchannels:]:
            output_pardir = os.path.dirname(output_fpath)
            output_daq_file = os.path.join(output_pardir, 'daq_settings.json')
            if not os.path.exists(output_daq_file):
                logger.info(f'copying {daq_file} to {output_daq_file}')
                os.system(f'cp {daq_file} {output_daq_file}')

    # Return list of output filepaths
    return output_fpaths


def preprocess_bergamo_dataset(fpaths, fps=None, smooth=False, detrend=False, mpi=False, overwrite=False, **kwargs):
    ''' 
    Pre-process Bergamo dataset
    
    :param fpaths: filepaths to input (raw) stack files
    :param fps (optional): target sampling rate for the output array, if resampling is requested (Hz)
    :param smooth (default=False): whether to smooth stacks upon resampling or not
    :param detrend (default=False): boolean indicating whether to detrend dataset or not
    :param mpi: whether to use multiprocessing or not
    :param overwrite: whether to overwrite input files if they exist
    :return: 2-tuple with:
        - filepaths to pre-processed functional stacks (list)
        - number of vframes per trial (int)
    '''
    # Set input root code
    input_root = DataRoot.RAW_BERGAMO

    # # If dataset is corrupted, detrend TIF stacks using linear regression 
    # if detrend:
    #     fpaths = correct_tifs(
    #         fpaths, input_root=input_root, overwrite=overwrite, mpi=mpi, **kwargs)
    #     input_root = 'corrected'
    
    # Stack trial TIFs of every run in the stack list
    fpaths = stack_trial_tifs_across_runs(
        fpaths, input_root, align=detrend, overwrite=overwrite)
    input_root = DataRoot.STACKED

    # Resample TIF stacks
    if fps is not None:
        fpaths = resample_tifs(
            fpaths, input_root, fps, smooth=smooth, mpi=mpi, **kwargs)
        input_root = DataRoot.RESAMPLED
    
    # Split channels from run stacks
    fpaths = split_multichannel_tifs(
        fpaths, input_root, overwrite=overwrite)
    input_root = DataRoot.SPLIT

    # Keep only files from functional channel
    channel_key = f'channel{FUNC_CHANNEL}'
    fpaths = list(filter(lambda x: channel_key in x, fpaths))

    # Return list of pre-processed stack files 
    return fpaths


def preprocess_bergamo_datasets(dataroot, analysis=None, **kwargs):
    ''' Pre-process Bergamo datasets '''
    # If analysis type not specified, extract analysis subfolders from input data directory
    # and call function recursively for each analysis type
    if analysis is None:
        for analysis in get_subfolder_names(dataroot):
            logger.info(f'processing Bergamo inputs for "{analysis}" analysis')
            preprocess_bergamo_datasets(dataroot, analysis=analysis, **kwargs)
        return
    
    # List datasets in input data directory, filtered by analysis type
    datasets = get_dataset_params(root=dataroot, analysis=analysis)
    logger.info(f'found {len(datasets)} datasets for "{analysis}" analysis')

    # Filter datasets to match related input parameters
    input_keys = ('mouseline', 'expdate', 'mouseid', 'region', 'layer')
    found_keys = [k for k in kwargs.keys() if k in input_keys]
    input_kwargs = {k: kwargs.pop(k) for k in found_keys}
    datasets = restrict_datasets(datasets, **input_kwargs)
    logger.info(f'found {len(datasets)} dataset(s) matching input parameters')
    
    # Process each dataset
    for dataset in datasets:
        # Construct dataset ID
        dataset_id = f'{dataset["expdate"]}_{dataset["mouseid"]}_{dataset["region"]}'
        if dataset["layer"] != DEFAULT_LAYER:
            dataset_id = f'{dataset_id}_{dataset["layer"]}'

        # Extract input data directory
        datadir = os.path.join(dataroot, analysis, dataset["mouseline"], dataset_id)
        if not os.path.exists:
            raise ValueError(f'input data directory "{datadir}" does not exist')

        # Get list of raw TIF files from dataset
        raw_fpaths = get_input_files(datadir)

        # Pre-process files
        logger.info(f'pre-processing {len(raw_fpaths)} Bergamo input files from "{datadir}"')
        preprocess_bergamo_dataset(raw_fpaths, **kwargs)
