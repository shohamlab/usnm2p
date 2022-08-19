# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-11 11:59:10
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-08-19 15:07:20

''' Collection of image stacking utilities. '''

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from constants import *
from logger import logger
from utils import moving_average
from fileops import StackProcessor, NoProcessor, process_and_save
from parsers import P_TRIALFILE, P_TRIALFILE_SUB


class NoResamplerFilter(NoProcessor):
    ''' No resampler instance to substitute in the code in case no resampling is wanted '''
    pass


class StackResampler(StackProcessor):
    ''' Generic interface to an stack resampler. '''

    def __init__(self, ref_sr, target_sr, *args, smooth=True, **kwargs):
        '''
        Constructor
        
        :param ref_sr: reference sampling rate of the input array (Hz)
        :param target_sr: target sampling rate for the output array (Hz)
        :param smooth: whether to apply pre-smoothing with moving average 
            to avoid sub-sampling outlier values 
        '''
        self.ref_sr = ref_sr
        self.target_sr = target_sr
        self.smooth = smooth
        super().__init__(*args, **kwargs)
    
    def __str__(self) -> str:
        return f'{self.__class__.__name__}(ref_sr={self.ref_sr}Hz, target_sr={self.target_sr}Hz, smooth={self.smooth})'
    
    @property
    def rootcode(self):
        return 'resampled'

    @property
    def code(self):
        s = f'sr_ref{self.ref_sr}_target{self.target_sr}'
        if self.smooth:
            s = f'{s}_smooth'
        return s

    @property
    def ref_sr(self):
        return self._ref_sr

    @ref_sr.setter
    def ref_sr(self, value):
        if value <= 0:
            raise ValueError('Reference sampling rate must be strictly positive.')
        self._ref_sr = value

    @property
    def target_sr(self):
        return self._target_sr

    @target_sr.setter
    def target_sr(self, value):
        if value <= 0:
            raise ValueError('Target sampling rate must be strictly positive.')
        if value > self.ref_sr:
            raise ValueError('Target sampling rate must be lower than reference sampling rate.')
        self._target_sr = value

    @property
    def smooth(self):
        return self._smooth

    @smooth.setter
    def smooth(self, value):
        if not isinstance(value, bool):
            raise ValueError('smooth must be a boolean')
        self._smooth = value
        
    def resample(self, x):
        '''
        Resample array to a specific sampling rate along first axis
        
        :param x: n-dimensional array
        :param ref_sr: reference sampling rate of the input array (Hz)
        :param target_sr: target sampling rate for the output array (Hz)
        :return: resampled array
        '''
        s = f'resampling {x.shape} stack from {self.ref_sr} Hz to {self.target_sr} Hz'
        if x.ndim > 1:
            s = f'{s} along axis 0'
        logger.info(f'{s} ...')
        # Compute sampling rate ratio
        sr_ratio = self.target_sr / self.ref_sr
        # Create reference and target time vectors
        tref = np.arange(x.shape[0]) / self.ref_sr  # s
        ntarget = int(np.ceil(tref.size * sr_ratio))
        ttarget = np.linspace(tref[0], tref[-1], ntarget)
        # Interpolate each pixel along target time vector
        return interp1d(tref, x, axis=0)(ttarget)

    def run(self, stack):
        '''
        Pre-process, smooth and resample image stack.

        :param stack: input image stack
        :return: processed image stack
        '''
        ref_dtype = stack.dtype
        # If specified, smooth stack with moving average along time axis
        if self.smooth:
            stack = moving_average(
                stack, n=int(np.round(self.ref_sr / self.target_sr)))
        # Resample at target sampling rate
        res_stack = self.resample(stack)
        # Round, cast as input integer type and return
        return np.round(res_stack).astype(ref_dtype)
    
    def get_target_nframes(self, nframes):
        ''' Get the target number of frames for an given input stack size '''
        return int(np.ceil(nframes * self.target_sr / self.ref_sr))
    
    def get_target_fname(self, fname):
        '''
        Get target file name upon resampling
        
        :param fname: input stack file name
        :return: output stack file name with adapted number of frames and sampling rate
        '''
        # Extract nframes from file name
        mo = P_TRIALFILE.match(fname)
        if mo is None:
            raise ValueError(f'file "{fname}" does not fit {P_TRIALFILE.pattern} pattern')
        nframes = int(mo.group(2))
        # Replace by new values in file name, and return
        return P_TRIALFILE.sub(
            P_TRIALFILE_SUB.format(nframes=self.get_target_nframes(nframes),
            sr=self.target_sr), fname)
    
    def plot_comparative_frameavg(self, ref_stack, res_stack):
        ''' Plot comparative time profiles of frame average '''
        yref = ref_stack[:, 0].mean(axis=(1, 2))
        yres = res_stack[:, 0].mean(axis=(1, 2))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title('Temporal evolution of frame average')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('average frame intensity')
        s = f'resampled @ {self.target_sr} Hz'
        if self.smooth:
            s = f'smoothed & {s}'
        ax.plot(np.arange(yref.size) / self.ref_sr, yref, label=f'original ({self.ref_sr} Hz)')
        ax.plot(np.arange(yres.size) / self.target_sr, yres, label=s)
        ax.legend(frameon=False)
        return fig


def resample_tifs(input_fpaths, ref_sr, target_sr, input_root='raw', **kwargs):
    '''
    High-level stack resampling function

    :param input_fpaths: list of full paths to input TIF stacks
    :param ref_sr: reference sampling rate of the input array (Hz)
    :param target_sr: target sampling rate for the output array (Hz)
    :return: list of resampled TIF stacks
    '''
    # Create stack resampler object
    sr = StackResampler(ref_sr=ref_sr, target_sr=target_sr)
    # Resample each stack file
    resampled_stack_fpaths = process_and_save(
        sr, input_fpaths, input_root, overwrite=False, **kwargs)
    return resampled_stack_fpaths