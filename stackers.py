# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-11 11:59:10
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-08-16 16:49:47

''' Collection of image stacking utilities. '''

import abc
import os
import warnings
import numpy as np

from constants import *
from logger import logger
from fileops import loadtif, savetif, get_sorted_filelist, get_output_equivalent, check_for_existence
from parsers import P_TIFFILE, P_TRIALFILE, P_RUNFILE_SUB


class ImageStacker(metaclass=abc.ABCMeta):
    ''' Generic interface to an image stacker. '''

    def __init__(self, *args, input_type='image', **kwargs):
        self.input_type = input_type
        super().__init__(*args, **kwargs)

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
    
    def check_stack_integrity(self, stack):
        ''' Stack integrity checking method. '''
        pass

    def stack(self, input_fpaths, output_fpath, overwrite='?', full_output=False):
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
        # Convert stack to numpy array 
        if self.input_type == 'image':
            stack = np.stack(stack)
        else:
            stack = np.concatenate(stack)
        # Check stack integrity
        self.check_stack_integrity(stack)
        # Save stack as single file and return output filepath
        logger.info(f'generated {stack.shape[0]}-frames image stack')
        self.save_stack(output_fpath, stack)
        # Return
        if full_output:
            return output_fpath, stack.shape
        else: 
            return output_fpath


class TifStacker(ImageStacker):
    ''' TIF-specific image stacker. '''

    def load_image(self, fpath):
        with warnings.catch_warnings(record=True):
            image = loadtif(fpath)
        # Implement fix for first file that contains 10 frames (a mystery) -> we just take the last one.
        if image.ndim > 2:
            nframes = image.shape[0]
            logger.warning(f'image ("{os.path.basename(fpath)}") is corrupted (shape = {image.shape}) -> ommitting first {nframes - 1} frames')
            image = image[-1]
        return image
    
    def load_stack(self, fpath):
        return loadtif(fpath)
    
    def save_stack(self, fpath, stack):
        savetif(fpath, stack)

    def check_stack_integrity(self, stack):
        # Check that final stack size is correct
        nframes, *_ = stack.shape
        if nframes != REF_NFRAMES:
            logger.warning(f'final stack size = {nframes} frames, seems suspicious...')


def stack_tifs(inputdir, pattern=P_TIFFILE, input_key='raw', **kwargs):
    '''
    high-level function to merge individual TIF files into an TIF stack.

    :param inputdir: absolute path to directory containing the input images
    :param pattern: filename matching pattern 
    :param input_key: input key for output path replacement
    :return: filepath to the created tif stack
    '''
    # Cast inputdir to absolute path
    inputdir = os.path.abspath(inputdir)
    # Get output file name
    pardir, dirname = os.path.split(inputdir)
    outdir = get_output_equivalent(pardir, input_key, 'stacked')
    output_fpath = os.path.join(outdir, f'{dirname}.tif')
    # Get tif files list
    try:
        fnames = get_sorted_filelist(inputdir, pattern=pattern)
    except ValueError:
        return None
    fpaths = [os.path.join(inputdir, fname) for fname in fnames]
    return TifStacker().stack(fpaths, output_fpath, **kwargs)


def group_by_run(files):
    '''
    Group a large file list into consecutive trial files for each run
    
    :param files: input files list
    :return: dictionary of files list per run index
    '''
    # Create output dictionary
    fbyrun = {}
    # For each file
    for fname in files:
        # Extract run and trial index from name
        mo = P_TRIALFILE.match(fname)
        *_, irun, itrial = mo.groups()
        irun, itrial = int(irun), int(itrial)
        # Create run list if not already there
        if irun not in fbyrun:
            # Get run filename
            run_fname = P_TRIALFILE.sub(P_RUNFILE_SUB, fname)
            fbyrun[irun] = [run_fname, []]
        # Add filename to appropriate run list
        fbyrun[irun][1].append(fname)
    # Return dictionary
    return fbyrun


def stack_trial_tifs(inputdir, pattern=P_TIFFILE, input_key='resampled', **kwargs):
    '''
    Stack TIFs of consecutive trials together for each run in a data folder
    
    :param inputdir: absolute path to directory containing the input stacks
    :param pattern: filename matching pattern 
    :param input_key: input key for output path replacement
    :return: filepaths to the created tif stacks per run
    '''
    # Get equivalent output directory
    outdir = get_output_equivalent(inputdir, input_key, 'stacked')
    output_fpaths = []
    # Get filelist by run
    flist = get_sorted_filelist(inputdir, pattern=pattern)
    flist_by_run = group_by_run(flist)
    # For each run
    for irun, (out_fname, files) in flist_by_run.items():
        # Get input and output filepaths
        input_fpaths = [os.path.join(inputdir, f) for f in files]
        output_fpath = os.path.join(outdir, out_fname)
        # Stack trial TIFs together
        TifStacker(input_type='stack').stack(input_fpaths, output_fpath, **kwargs)
        output_fpaths.append(output_fpath)
    # Return list of output filepaths
    return output_fpaths



def split_multichannel_tifs(inputdir, pattern=P_TIFFILE, input_key='stacked', **kwargs):
    '''
    Split channels Stack TIFs of consecutive trials together for each run in a data folder
    
    :param inputdir: absolute path to directory containing the input stacks
    :param pattern: filename matching pattern 
    :param input_key: input key for output path replacement
    :return: filepaths to the created tif stacks per run
    '''
    output_fpaths = []
    # Get input filelist
    flist = get_sorted_filelist(inputdir, pattern=pattern)
    # For each file
    for input_fname in flist:
        # Load input stack
        input_fpath = os.path.join(inputdir, input_fname)
        stack = loadtif(input_fpath)
        # If stack has more than 3 dimensions (i.e. multi-channel)
        if stack.ndim > 3:
            # Extract number of channels
            nchannels = stack.shape[1]
            # Loop through stack channels
            for i in range(nchannels):
                # Derive channel output filepath
                output_fpath = get_output_equivalent(
                    input_fpath, input_key, f'split/channel{i + 1}')
                # Save channel data to specific file
                savetif(output_fpath, stack[:, i], **kwargs)
                output_fpaths.append(output_fpath)
    # Return list of output filepaths
    return output_fpaths