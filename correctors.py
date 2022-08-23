# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-11 11:59:10
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-08-23 15:19:32

''' Collection of image stacking utilities. '''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from constants import *
from logger import logger
from utils import expdecay, biexpdecay
from fileops import StackProcessor, NoProcessor, process_and_save
from scipy.stats import linregress
import statsmodels.api as sm
from tqdm import tqdm


class NoCorrector(NoProcessor):
    ''' No corrector instance to substitute in the code in case no resampling is wanted '''
    pass


class Corrector(StackProcessor):

    @property
    def rootcode(self):
        return 'corrected'

    def run(self, stack):
        '''
        Correct image stack.

        :param stack: input image stack
        :return: processed image stack
        '''
        # For multi-channel stack, process each channel separately
        if stack.ndim > 3:
            outstack = []
            for i in range(stack.shape[1]):
                logger.info(f'working on channel {i + 1}...')
                outstack.append(self.run(stack[:, i]))
            outstack = np.stack(outstack)
            return np.swapaxes(outstack, 0, 1)            
        # Save input data type
        ref_dtype = stack.dtype
        # Cast as float64 and apply correction function
        res_stack = self.correct(stack.astype(np.float64))
        # Round, cast as input integer type and return
        return np.round(res_stack).astype(ref_dtype)
    
    def subtract_vector(self, stack, y):
        ''' Detrend stack using a frame-average subtraction vector '''
        assert y.size == stack.shape[0], f'inputs incompatibility:{y.size} and {stack.shape}'
        # Subtract mean from vector
        y -= y.mean()
        # Subtract mean-corrected vector from stack
        return (stack.T - y).T

    def plot(self, y, yfit):
        fig, ax = plt.subplots()
        ax.plot(y, label='data')
        ax.plot(yfit, label='fit')
        ax.plot(y - yfit + yfit.mean(), label='detrended')
        ax.legend()
        for sk in ['top', 'right']:
            ax.spines[sk].set_visible(False)
        plt.show()


class LinRegCorrector(Corrector):

    def __init__(self, iref, *args, **kwargs):
        self.iref = iref
        super().__init__(*args, **kwargs)
        
    def __str__(self) -> str:        
        return f'{self.__class__.__name__}(iref={self.iref})'
        
    @property
    def code(self):
        return f'linreg_iref_{self.iref.start}_{self.iref.stop}'

    def linreg(self, frame, ref_frame):
        '''
        Perform robust linear regression between a frame and a reference frame
        
        :param frame: frame 2D array
        :param ref_frame: reference frame 2D array
        :return: linear fit parameters (offset and slope, such that frame = slope * ref_frame + intercept) 
        '''
        res = linregress(ref_frame.ravel(), frame.ravel())
        slope, intercept = res.slope, res.intercept
        # x, y = ref_frame.ravel(), frame.ravel()
        # x = sm.add_constant(x)
        # model = sm.RLM(y, x, M=sm.robust.norms.HuberT())
        # fit = model.fit()
        # intercept, slope = fit.params
        return slope, intercept
    
    def plot_linreg_params(self, linreg_params):
        ''' Plot linear regression parameters over time '''
        fig, ax = plt.subplots()
        slopes, intercepts = list(zip(*linreg_params))
        iframes = np.arange(len(slopes))
        ax.plot(iframes, intercepts, label='intercept')
        ax.plot(iframes, slopes, label='slope')
        ax.legend()
        return fig
    
    def correct(self, stack):
        ''' Correct image stack with linear regresion to reference frame '''
        # Compute average reference frame within predefined frame range
        ref_frame = stack[self.iref].mean(axis=0)
        corrected_frames = []
        linreg_params = []
        # For each frame
        for frame in stack:
            # Compute linear regression to reference frame
            slope, intercept = self.linreg(frame, ref_frame)
            linreg_params.append([slope, intercept])
            # Correct frame accordingly: corrected_frame = (frame - intercept) / slope             
            corrected_frames.append((frame - intercept) / slope)
        # fig = self.plot_linreg_params(linreg_params)
        # plt.show()
        # Return corrected stack
        return np.stack(corrected_frames)


class MedianCorrector(Corrector):
        
    def __str__(self) -> str:        
        return self.__class__.__name__
        
    @property
    def code(self):
        return 'median'
        
    def correct(self, stack):
        '''
        Correct image stack for with median-subtraction.

        :param stack: input image stack
        :return: processed image stack
        '''
        logger.info(f'applying median correction to {stack.shape[0]}-frames stack...')
        # Compute frame median over time
        ymed = np.median(stack, axis=(1, 2))
        # Subtract median-corrected fit from each pixel to detrend stack
        return self.subtract_vector(stack, ymed)
    

class MeanCorrector(Corrector):
        
    def __str__(self) -> str:        
        return self.__class__.__name__
        
    @property
    def code(self):
        return 'mean'
        
    def correct(self, stack):
        '''
        Correct image stack for with mean-subtraction.

        :param stack: input image stack
        :return: processed image stack
        '''
        logger.info(f'applying mean correction to {stack.shape[0]}-frames stack...')
        # Compute frame mean over time
        ymean = np.mean(stack, axis=(1, 2))
        # Subtract mean-corrected fit from each pixel to detrend stack
        return self.subtract_vector(stack, ymean)


class ExponentialCorrector(Corrector):
    ''' Generic interface to an stack exponential decay corrector. '''

    def __init__(self, nexps=1, nfit=None, ncorrupted=0):
        '''
        Constructor
        
        :param nexps: number of exponentials in the decay function
        :param nfit: number of initial frames on which to perform exponential fit  
        :param ncorrupted: number of initial corrupted frames to substitute after detrending 
        '''
        self.nexps = nexps
        self.nfit = nfit
        self.ncorrupted = ncorrupted
    
    def __str__(self) -> str:        
        return f'{self.__class__.__name__}(nexps={self.nexps}, nfit={self.nfit}, ncorrupted={self.ncorrupted})'

    @property
    def code(self):
        nexps_str = {1: 'mono',2: 'bi'}[self.nexps]
        s = f'{nexps_str}expdecay'
        if self.nfit is not None:
            s = f'{s}_{self.nfit}fit'            
        if self.ncorrupted > 0:
            s = f'{s}_{self.ncorrupted}corrupted'
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
    def nfit(self):
        return self._nfit

    @nfit.setter
    def nfit(self, value):
        if value is not None and value < 0:
            raise ValueError('number of initial frames for fit must be positive')
        self._nfit = value

    @property
    def ncorrupted(self):
        return self._ncorrupted

    @ncorrupted.setter
    def ncorrupted(self, value):
        if value < 0:
            raise ValueError('number of initial corrupted frames must be positive')
        self._ncorrupted = value
    
    def expdecayfit(self, y):
        ''' 
        Fit an exponential decay to a signal
        
        :param y: signal array
        :return: fitted decay array
        '''
        # Get signal size
        nsamples = y.size
        # Determine number of samples on which to perform the fit
        if self.nfit is not None:
            nfit = self.nfit
        else:
            nfit = nsamples + 1
        # Reduce input signal to nfit samples
        y = y[:nfit]
        # Compute signal statistics
        ptp = np.ptp(y)
        ymed = np.median(y)
        ystd = y.std()
        # Initial parameters guess
        H0 = ymed  # vertical offset: signal median
        A0 = 1  # amplitude: 1
        tau0 = 1 # decay time constant: 1 sample
        x0 = 0  # horizontal offset: 0 sample
        # Parameters bounds
        Hbounds = (ymed - 0.1 * ptp, ymed + 0.1 * ptp)  # vertical offset: within median +/-10% of variation range 
        Abounds = (-1e3, 1e3)  # amplitude: within +/- 1000
        taubounds = (1e-3, nfit / 2) # decay time constant: 0.001 - half-signal length 
        x0bounds = (-nfit, nfit)  # horizontal offset: within +/- signal length
        # Adapt inputs to number of exponentials
        p0 = (H0,) + (A0,) * self.nexps + (tau0,) * self.nexps + (x0,) * self.nexps
        pbounds = (Hbounds, *([Abounds] * self.nexps), *([taubounds] * self.nexps), *([x0bounds] * self.nexps))
        pbounds = tuple(zip(*pbounds))
        # Least-square fit over restricted signal
        xfit = np.arange(nfit)
        popt, _ = curve_fit(self.decayfunc, xfit, y, p0=p0, bounds=pbounds, max_nfev=20000)
        logger.info(f'popt: {popt}')
        # Compute fitted profile
        yfit = self.decayfunc(xfit, *popt)
        # Compute rmse of fit
        rmse = np.sqrt(((y - yfit) ** 2).mean())
        # Compute ratio of rmse to signal standard deviation
        rel_rmse = rmse / ystd
        s = f'RMSE = {rmse:.2f}, STD = {ystd:.2f}, RMSE / STD = {rmse / ystd:.2f}'
        logger.info(s)
        # Raise error if ratio is too high
        if rel_rmse > DECAY_FIT_MAX_REL_RMSE:
            self.plot(y, yfit)
            raise ValueError(f'{self} fit quality too poor: {s}')
        # Return fit over entire signal
        xfull = np.arange(nsamples)
        return self.decayfunc(xfull, *popt)

    def correct(self, stack):
        '''
        Correct image stack for initial exponential decay.

        :param stack: input image stack
        :return: processed image stack
        '''
        logger.info(f'applying exponential detrending to {stack.shape[0]}-frames stack...')
        # Compute frame average over time
        y = stack.mean(axis=(1, 2))
        # Compute exponential decay fit on frame average profile beyond corrupted frames
        yfit = self.expdecayfit(y[self.ncorrupted:].copy())
        # Subtract fit from each pixel to detrend stack beyond corrupted frames
        stack[self.ncorrupted:] = self.subtract_vector(stack[self.ncorrupted:], yfit)
        # Substitute corrupted first n frames
        stack[:self.ncorrupted] = stack[self.ncorrupted]
        # Return stack
        return stack


def correct_tifs(input_fpaths, input_root='raw', **kwargs):
    '''
    High-level stack detrending function

    :param input_fpaths: list of full paths to input TIF stacks
    :return: list of detrended TIF stacks
    '''
    # Apply linear regression correction     
    lrc = LinRegCorrector(IREF_FRAMES_BERGAMO)
    corrected_stack_fpaths = process_and_save(
        lrc, input_fpaths, input_root, **kwargs)
    # # Apply median correction
    # mc = MedianCorrector()
    # median_corrected_stack_fpaths = process_and_save(
    #     mc, input_fpaths, input_root, overwrite=False, **kwargs)
    # # Apply exponential detrending
    # ec = ExponentialCorrector(
    #     nexps=NEXPS_DECAY_DETREND, nfit=NSAMPLES_DECAY_DETREND, ncorrupted=NCORRUPTED_BERGAMO)
    # corrected_stack_fpaths = process_and_save(
    #     ec, median_corrected_stack_fpaths, 'corrected', overwrite=False, **kwargs)
    # Return output filepaths     
    return corrected_stack_fpaths