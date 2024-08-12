# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-11 11:59:10
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2024-08-09 10:07:13

''' Collection of image stacking utilities. '''

import abc
import os
import warnings
import numpy as np
import pandas as pd

from .constants import *
from .logger import logger
from .fileops import loadtif, savetif, get_sorted_filelist, get_output_equivalent, check_for_existence, split_path_at, load_acquisition_settings
from .parsers import P_TIFFILE, parse_experiment_parameters


class ImageStacker(metaclass=abc.ABCMeta):
    ''' Generic interface to an image stacker. '''

    def __init__(self, input_type='image', align=False, verbose=True):
        self.input_type = input_type
        self.align = align
        self.verbose = verbose
    
    @property
    def rootcode(self):
        return 'stacked'
    
    @property
    def code(self):
        s = self.rootcode
        if self.align:
            s = f'{self.rootcode}/aligned'
        return s

    @abc.abstractmethod
    def load_image(self, fpath):
        ''' Abstract image loading method. '''
        raise NotImplementedError
    
    @abc.abstractmethod
    def load_stack(self, fpath):
        ''' Abstract stack loading method. '''
        raise NotImplementedError
   
    @abc.abstractmethod
    def save_stack(self, fpath, stack):
        ''' Abstract stack saving method. '''
        raise NotImplementedError
    
    def align_baselines(self, stacks):
        ''' Offsets stacks to align their baseline fluorescence levels '''
        # Save reference data type
        ref_dtype = stacks[0].dtype
        # Cast stacks as float64
        stacks = [s.astype(np.float64) for s in stacks]
        # Compute baselines as grand mean of each stack
        baselines = np.array([np.mean(s) for s in stacks])
        # Compute grand baseline as mean of baselines 
        b0 = np.mean(baselines)
        # Compute deltas for each baseline
        deltas = baselines - b0
        # Offsets stacks to align baselines to grand baseline
        stacks = [s - delta for (s, delta) in zip(stacks, deltas)]
        baselines = np.array([np.mean(s) for s in stacks])
        # Round, cast as input integer type and return
        stacks = [np.round(s).astype(ref_dtype) for s in stacks]
        baselines = np.array([np.mean(s.astype(np.float64)) for s in stacks])
        return stacks
        
    def stack(self, input_fpaths, output_fpath, overwrite='?', full_output=False, **kwargs):
        '''
        Merge individual image files into an image stack.

        :param input_fpaths: list of absolute paths to the input image files list
        :param output_fpath: absolute path to output image stack file
        :param overwrite: one of (True, False, '?') defining what to do if output image stack file already exists 
        :return: filepath to the created tif stack
        '''
        # Check for output file existence and decide whether to move forward or not
        move_forward = check_for_existence(output_fpath, overwrite)
        if not move_forward:
            if full_output:
                with warnings.catch_warnings(record=True):
                    dims = loadtif(output_fpath, verbose=False).shape
                return output_fpath, dims
            else:
                return output_fpath
    
        # Initialize stack array
        stack = []
        refshape = None

        # Log message
        logger.info(f'stacking {len(input_fpaths)} {self.input_type}s from "{os.path.dirname(input_fpaths[0])}"...')

        # For each input filepath
        for i, fpath in enumerate(input_fpaths):
            # Load input
            if self.input_type == 'image':
                stack_input = self.load_image(fpath)
            else:
                stack_input = self.load_stack(fpath)
          
            # Assign reference shape or ensure match of current input with reference
            if refshape is None:
                refshape = stack_input.shape
            else:
                if stack_input.shape != refshape:
                    raise ValueError(
                        f'{self.input_type} {i} shape {stack_input.shape} does not match reference {refshape}')

            # Append input to stack
            stack.append(stack_input)
        
        # If specified, offset sub-stacks / frames to align baselines
        if self.align:
            stack = self.align_baselines(stack)
        
        # Convert stack to numpy array 
        if self.input_type == 'image':
            stack = np.stack(stack)
        else:
            stack = np.concatenate(stack)
        
        # Save stack as single file and return output filepath
        logger.info(f'generated {stack.shape[0]}-frames image stack')
        self.save_stack(output_fpath, stack, **kwargs)
        
        # Return
        if full_output:
            return output_fpath, stack.shape
        else: 
            return output_fpath


class TifStacker(ImageStacker):
    ''' TIF-specific image stacker. '''

    def load_image(self, fpath):
        with warnings.catch_warnings(record=True):
            image = loadtif(fpath, verbose=self.verbose)
        # Implement fix for first file that contains 10 frames (a mystery) -> we just take the last one.
        if image.ndim > 2:
            nframes = image.shape[0]
            logger.warning(f'image ("{os.path.basename(fpath)}") is corrupted (shape = {image.shape}) -> ommitting first {nframes - 1} frames')
            image = image[-1]
        return image
    
    def load_stack(self, fpath):
        return loadtif(fpath)
    
    def save_stack(self, fpath, stack, **kwargs):
        savetif(fpath, stack, **kwargs)


def stack_tifs(inputdir, input_key, output_key, pattern=P_TIFFILE, verbose=True, full_output=False, **kwargs):
    '''
    High-level function to merge individual TIF files into an TIF stack.

    :param inputdir: absolute path to directory containing the input images
    :param input_key: key from input path(s) to be replaced in output path(s)
    :param output_key: replacement key for output path(s)
    :param pattern: filename matching pattern 
    :param verbose: verbosity flag
    :param full_output: flag to return full output (stack and dimensions) instead of just the filepath
    :return: filepath to the created tif stack
    '''
    # Cast inputdir to absolute path
    inputdir = os.path.abspath(inputdir)
    # Get output file name
    pardir, dirname = os.path.split(inputdir)
    outdir = get_output_equivalent(pardir, input_key, output_key)
    output_fpath = os.path.join(outdir, f'{dirname}.tif')
    # Get tif files list
    try:
        fnames = get_sorted_filelist(inputdir, pattern=pattern)
    except ValueError:
        return None
    if full_output:
        # Exract parameters from each file
        pbyfile = pd.DataFrame([pd.Series(parse_experiment_parameters(fname)) for fname in fnames])
        # Compute trial and frame index from cycle and frame index
        pbyfile[Label.TRIAL] = (pbyfile[Label.CYCLE] - 1) // 2
        pbyfile[Label.FRAME] = pbyfile.groupby(Label.TRIAL).cumcount()
        ntrials, npertrial = pbyfile[Label.TRIAL].nunique(), pbyfile.groupby(Label.TRIAL).size().unique()
        if npertrial.size > 1:
            raise ValueError(f'Inconsistent number of frames per trial: {npertrial}')
        npertrial = npertrial[0]
    fpaths = [os.path.join(inputdir, fname) for fname in fnames]
    out = TifStacker(verbose=verbose).stack(fpaths, output_fpath, **kwargs)
    if full_output:
        return output_fpath, ntrials, npertrial
    else:
        return out


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