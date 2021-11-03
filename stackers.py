# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-11 11:59:10
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-11-03 09:40:25

''' Collection of image stacking utilities. '''

import abc
import os
import warnings
import logging
import numpy as np

from constants import *
from logger import logger
from fileops import loadtif, savetif, get_sorted_filelist, get_output_equivalent, check_for_existence
from parsers import P_TIFFILE


class ImageStacker(metaclass=abc.ABCMeta):
    ''' Generic interface to an image stacker. '''

    @abc.abstractmethod
    def load_image(self, fpath):
        ''' Abstract image loading method. '''
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
            # Load corresponding image
            image = self.load_image(fpath)
            # Assign reference image shape or ensure match of current image with reference
            if refshape is None:
                refshape = image.shape
            else:
                assert image.shape == refshape, 'image {i} shape {image.shape} does not match reference {refshape}'
            # Append image to stack
            stack.append(image)
        # Convert stack to numpy array and check its integrity
        stack = np.stack(stack)
        self.check_stack_integrity(stack)
        logger.info(f'generated {stack.shape[0]}-frames image stack')
        # Save stack as single file and return output filepath
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

    def save_stack(self, fpath, stack):
        savetif(fpath, stack)

    def check_stack_integrity(self, stack):
        # Check that final stack size is correct
        nframes, *_ = stack.shape
        if nframes != REF_NFRAMES:
            logger.warning(f'final stack size = {nframes} frames, seems suspicious...')


def stack_tifs(inputdir, pattern=P_TIFFILE, **kwargs):
        '''
        high-level function to merge individual TIF files into an TIF stack.

        :param inputdir: absolute path to directory containing the input images
        :param overwrite: one of (True, False, '?') defining what to do if output stack file already exists
        :param pattern: filename matching pattern 
        :return: filepath to the created tif stack
        '''
        # Cast inputdir to absolute path
        inputdir = os.path.abspath(inputdir)
        # Get output file name
        pardir, dirname = os.path.split(inputdir)
        outdir = get_output_equivalent(pardir, 'raw', 'stacked')
        output_fpath = os.path.join(outdir, f'{dirname}.tif')
        # Get tif files list
        try:
            fnames = get_sorted_filelist(inputdir, pattern=pattern)
        except ValueError:
            return None
        fpaths = [os.path.join(inputdir, fname) for fname in fnames]
        return TifStacker().stack(fpaths, output_fpath, **kwargs)