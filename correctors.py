# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-11 11:59:10
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-08-18 16:19:32

''' Collection of image stacking utilities. '''

from multiprocessing.dummy import Value
import numpy as np
from scipy.optimize import curve_fit

from constants import *
from logger import logger
from utils import StackProcessor, NoProcessor, expdecay, biexpdecay
from fileops import process_and_save


class NoCorrector(NoProcessor):
    ''' No corrector instance to substitute in the code in case no resampling is wanted '''
    pass


class ExponentialCorrector(StackProcessor):
    ''' Generic interface to an stack exponential decay corrector. '''

    def __init__(self, nexps=1, nsubs=0):
        '''
        Constructor
        
        :param nexps: number of exponentials in the decay function
        :param nsubs: number of frames to substitute after detrending 
        '''
        self.nexps = nexps
        self.nsubs = nsubs
    
    def __str__(self) -> str:        
        return f'{self.__class__.__name__}(nexps={self.nexps}, nsubs={self.nsubs})'
    
    @property
    def rootcode(self):
        return 'corrected'

    @property
    def code(self):
        nexps_str = {1: 'mono',2: 'bi'}[self.nexps]
        s = f'{nexps_str}expdecay'
        if self.nsubs > 0:
            s = f'{s}_{self.nsubs}sub'
        if self.nsubs > 1:
            s = f'{s}s'
        return s

    @property
    def nexps(self):
        return self._nexps

    @nexps.setter
    def nexps(self, value):
        if value not in (1, 2):
            raise ValueError('Number of exponentials must be one of (1, 2).')
        # Adjust function fo number of exponentials 
        if value == 1:
            self.decayfunc = expdecay
        elif value == 2:
            self.decayfunc = biexpdecay
        self._nexps = value

    @property
    def nsubs(self):
        return self._nsubs

    @nsubs.setter
    def nsubs(self, value):
        if value < 0:
            raise ValueError('number of initial substituted frames must be positive')
        self._nsubs = value
    
    def expdecayfit(self, y):
        ''' 
        Fit an exponential decay to a signal
        
        :param y: signal array
        :return: fitted decay array
        '''
        # Input vector
        x = np.arange(y.size)
        # Initial parameters guess
        H0 = y.min()
        A0 = 1
        x0 = 0
        # Parameters bounds
        Hbounds = (y.min(), np.median(y))
        Abounds = (0, 1e3)
        # x0bounds = (0, y.size // 4)
        x0bounds = (-np.inf, np.inf)
        # Adapt inputs to number of exponentials
        p0 = (H0,) + (A0,) * self.nexps + (x0,) * self.nexps
        pbounds = (Hbounds, *([Abounds] * self.nexps), *([x0bounds] * self.nexps))
        pbounds = tuple(zip(*pbounds))
        # Least-square fit
        popt, _ = curve_fit(self.decayfunc, x, y, p0=p0, bounds=pbounds)
        # Compute fitted profile
        yfit = self.decayfunc(x, *popt)
        # Compute rmse of fit
        rmse = np.sqrt(((y - yfit) ** 2).mean())
        # Compute ratio of rmse fo signal median
        rel_rmse = rmse / np.median(y)
        # Raise error if too high
        if rel_rmse > DECAY_FIT_MAX_REL_RMSE:
            raise ValueError(f'{self} fit quality too poor (relative RMSE = {rel_rmse:.2f})')
        return yfit

    def run(self, stack):
        '''
        Correct image stack for initial exponential decay.

        :param stack: input image stack
        :return: processed image stack
        '''
        # For multi-channel stack, process each channel separately
        if stack.ndim > 3:
            outstack = np.stack([
                self.run(stack[:, i]) for i in range(stack.shape[1])])
            return np.swapaxes(outstack, 0, 1)            
        # Save input data type
        ref_dtype = stack.dtype
        # Compute frame average over time
        y = stack.mean(axis=(1, 2))
        # Compute exponential decay fit on frame average profile
        yfit = self.expdecayfit(y)
        # Subtract mean-corrected fit from each pixel to detrend stack
        res_stack = (stack.T - yfit).T + yfit.mean()
        # Substitute first n frames to avoid false transients created by detrending
        res_stack[:self.nsubs] = res_stack[self.nsubs + 1]
        # Round, cast as input integer type and return
        return np.round(res_stack).astype(ref_dtype)


def correct_tifs(input_fpaths, nexps, nsubs, input_root='raw'):
    '''
    High-level stack detrending function

    :param input_fpaths: list of full paths to input TIF stacks
    :param nexps: number of exponentials for decay fit
    :param nsubs: number of frames to substitute after detrending 
    :return: list of detrended TIF stacks
    '''
    # Create stack corrector object
    sr = ExponentialCorrector(nexps=nexps, nsubs=nsubs)
    # Detrend each stack file
    corrected_stack_fpaths = process_and_save(
        sr, input_fpaths, input_root, overwrite=False)
    return corrected_stack_fpaths