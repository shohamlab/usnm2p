# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-11 11:59:10
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2024-08-07 15:59:54

''' Collection of image stacking utilities. '''

import os
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from .constants import *
from .logger import logger
from .utils import moving_average
from .fileops import StackProcessor, NoProcessor, process_and_save, load_acquisition_settings, save_acquisition_settings
from .parsers import P_RUNFILE, P_RUNFILE_SUB


class NoResamplerFilter(NoProcessor):
    ''' No resampler instance to substitute in the code in case no resampling is wanted '''
    pass


class StackResampler(StackProcessor):
    ''' Generic interface to an stack resampler. '''

    def __init__(self, fps_in, fps_out, *args, smooth=True, **kwargs):
        '''
        Constructor

        :param fps_in: sampling rate of the input files (Hz)        
        :param fps_out: target sampling rate for the output array (Hz)
        :param smooth: whether to apply pre-smoothing with moving average 
            to avoid sub-sampling outlier values 
        '''
        self.fps_in = fps_in
        self.fps_out = fps_out
        self.smooth = smooth
        super().__init__(*args, **kwargs)
    
    def __str__(self) -> str:
        return f'{self.__class__.__name__}(fps_in={self.fps_in}Hz, fps_out={self.fps_out}Hz, smooth={self.smooth})'
    
    @property
    def rootcode(self):
        return DataRoot.RESAMPLED

    @property
    def code(self):
        s = f'{self.fps_in}Hz_{self.fps_out}Hz'
        if self.smooth:
            s = f'{s}_smooth'
        return s

    @property
    def fps_in(self):
        return self._fps_in

    @fps_in.setter
    def fps_in(self, value):
        if value <= 0:
            raise ValueError('Input sampling rate must be strictly positive.')
        self._fps_in = value

    @property
    def fps_out(self):
        return self._fps_out

    @fps_out.setter
    def fps_out(self, value):
        if value <= 0:
            raise ValueError('Output sampling rate must be strictly positive.')
        if value > self.fps_in:
            raise ValueError('Output sampling rate must be lower than input sampling rate.')
        self._fps_out = value
    
    @property
    def fps_ratio(self):
        ''' Resampling factor (i.e., ratio between output and input sampling rates) '''
        return self.fps_out / self.fps_in

    def get_output_nframes(self, nin):
        '''
        Get the target number of frames for an given input stack size
        
        :param nframes: number of frames in the input stack
        :return: number of frames in the output stack
        '''
        return int(np.ceil(nin * self.fps_ratio))

    @property
    def smooth(self):
        return self._smooth

    @smooth.setter
    def smooth(self, value):
        if not isinstance(value, bool):
            raise ValueError('smooth must be a boolean')
        self._smooth = value
    
    @property
    def navg_smoothing(self):
        ''' Get the size (i.e., number of frames) of averaging window for pre-smoothing '''
        return int(np.round(self.fps_in / self.fps_out))
        
    def resample(self, x):
        '''
        Resample array to a specific sampling rate along first axis
        
        :param x: n-dimensional array
        :param fps_in: reference sampling rate of the input array (Hz)
        :param fps_out: target sampling rate for the output array (Hz)
        :return: resampled array
        '''
        # Log resampling operation
        s = f'resampling {x.shape} stack from {self.fps_in} Hz to {self.fps_out} Hz'
        if x.ndim > 1:
            s = f'{s} along axis 0'
        logger.info(f'{s} ...')
        
        # Create input and output time vectors
        nin = x.shape[0]
        nout = self.get_output_nframes(nin)
        tin = np.arange(nin) / self.fps_in  # s
        tout = np.linspace(tin[0], tin[-1], nout)
        
        # Interpolate each pixel along target time vector
        return interp1d(tin, x, axis=0)(tout)

    def run(self, stack):
        '''
        Pre-process, smooth and resample image stack.

        :param stack: input image stack
        :return: processed image stack
        '''
        ref_dtype = stack.dtype
        # If specified, smooth stack with moving average along time axis
        if self.smooth:
            stack = moving_average(stack, n=self.navg_smoothing)

        # Resample at target sampling rate
        res_stack = self.resample(stack)

        # Round, cast as input integer type and return
        return np.round(res_stack).astype(ref_dtype)
    
    def get_target_fname(self, fname):
        '''
        Get target file name upon resampling
        
        :param fname: input stack file name
        :return: output stack file name with adapted number of frames and sampling rate
        '''
        # Extract nframes from file name
        mo = P_RUNFILE.match(fname)
        if mo is None:
            raise ValueError(f'file "{fname}" does not fit {P_RUNFILE.pattern} pattern')
        nframes = int(mo.group(2))
  
        # Replace by new values in file name, and return
        return P_RUNFILE.sub(
            P_RUNFILE_SUB.format(
                nframes=self.get_output_nframes(nframes),
                fps=self.fps_out
            ), fname)
    
    def plot_comparative_frameavg(self, ref_stack, res_stack):
        ''' 
        Plot comparative time profiles of frame average
        
        :param ref_stack: reference stack
        :param res_stack: resampled stack
        :return: figure handle
        '''
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title('Temporal evolution of frame average')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('average frame intensity')

        # Compute frame average profiles for both stacks
        yref = ref_stack[:, 0].mean(axis=(1, 2))
        yres = res_stack[:, 0].mean(axis=(1, 2))

        # Plot profiles with appropriate time scales
        label_out = f'resampled @ {self.fps_out} Hz'
        if self.smooth:
            label_out = f'smoothed & {label_out}'
        ax.plot(np.arange(yref.size) / self.fps_in, yref, label=f'original ({self.fps_in} Hz)')
        ax.plot(np.arange(yres.size) / self.fps_out, yres, label=label_out)

        # Add legend
        ax.legend(frameon=False)

        # Return figure
        return fig


def resample_tifs(input_fpaths, input_key, fps_out, smooth=False, **kwargs):
    '''
    High-level stack resampling function

    :param input_fpaths: list of full paths to input TIF stacks
    :param input_root: input data root code
    :param fps_out: target sampling rate for the output files (Hz). The input sampling rate is
        extracted from the acquisition settings file found in the folder of the first file. 
    :param smooth: whether to apply pre-resampling smoothing
    :return: list of resampled TIF stacks
    '''
    logger.info(f'resampling {len(input_fpaths)} stacks to {fps_out} Hz')
    # Load acquisition settings
    try:
        dirname = os.path.dirname(input_fpaths[0])
        daq_settings = load_acquisition_settings(dirname)
    except FileNotFoundError:
        raise ValueError('acquisition settings file is not found')

    # Extract input sampling rate and number of channels from acquisition settings        
    logger.info('extracting input sampling rate and number of channels...')
    fps_in = 1. / daq_settings['framePeriod']
    if fps_in > 20.:
        fps_in = np.round(fps_in)
    try:
        nchannels = len(daq_settings['SI.hChannels.channelSave'])
    except KeyError:
        nchannels = 1

    # Create stack resampler object
    sr = StackResampler(fps_in, fps_out, smooth=smooth, nchannels=nchannels)

    # Resample each stack file
    output_fpaths = process_and_save(sr, input_fpaths, input_key, **kwargs)
    
    # Load DAQ metadata
    try:
        input_dir = os.path.dirname(input_fpaths[0])
        daq_settings = load_acquisition_settings(input_dir)
    except FileNotFoundError:
        raise ValueError('acquisition settings file is not found in input directory')
    
    # Change frameRate and nFramesPerTrial fields as a result of resampling
    daq_settings['framePeriod'] = 1. / fps_out
    npertrial_in = daq_settings['nFramesPerTrial']
    npertrial_out = sr.get_output_nframes(npertrial_in)
    daq_settings['nFramesPerTrial'] = npertrial_out

    # Save updated DAQ metadata to output directory
    output_dir = os.path.dirname(output_fpaths[0])
    save_acquisition_settings(output_dir, daq_settings)
    
    # Return list of output file paths
    return output_fpaths