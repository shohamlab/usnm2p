# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-11 11:59:10
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-10-18 11:36:56

''' Collection of image stacking utilities. '''

import numpy as np
from scipy.stats import skew
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# from scipy.optimize import curve_fit

from constants import *
from logger import logger
# from utils import expdecay, biexpdecay, is_within
from postpro import mylinregress
from fileops import StackProcessor, NoProcessor, process_and_save
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

    def __init__(self, robust=False, iref=None, qmin=0, qmax=1, **kwargs):
        '''
        Initialization

        :param robust: whether or not to use robust linear regression
        :param iref: frame index range to use to compute reference image (optional)
        :param qmin: minimum quantile to use for pixel selection (optional)
        :param qmax: maximum quantile to use for pixel selection (optional)
        '''
        # Assign input arguments as attributes
        self.iref = iref
        self.robust = robust
        self.qmin = qmin
        self.qmax = qmax

        # Initialize empty dictionary of cached reference images
        self.refimg_cache = {}
        
        # Call parent constructor
        super().__init__(**kwargs)
    
    @property
    def iref(self):
        return self._iref
    
    @iref.setter
    def iref(self, value):
        if value is not None and not isinstance(value, range):
            raise ValueError('iref must be a range object')
        self._iref = value
    
    @property
    def robust(self):
        return self._robust
    
    @robust.setter
    def robust(self, value):
        if not isinstance(value, bool):
            raise ValueError('robust must be a boolean')
        self._robust = value
    
    @property
    def qmin(self):
        return self._qmin
    
    @qmin.setter
    def qmin(self, value):
        if not 0 <= value < 1:
            raise ValueError('qmin must be between 0 and 1')
        if hasattr(self, 'qmax') and value >= self.qmax:
            raise ValueError(f'qmin must be smaller than qmax ({self.qmax})')    
        self._qmin = value
    
    @property
    def qmax(self):
        return self._qmax
    
    @qmax.setter
    def qmax(self, value):
        if not 0 < value <= 1:
            raise ValueError('qmax must be between 0 and 1')
        if hasattr(self, 'qmin') and value <= self.qmin:
            raise ValueError(f'qmax must be larger than qmin ({self.qmin})')
        self._qmax = value
        
    def __str__(self) -> str:        
        return f'{self.__class__.__name__}(robust={self.robust}, iref={self.iref}, qmin={self.qmin}, qmax={self.qmax})'
        
    @property
    def code(self):
        s = f'linreg'
        if self.robust:
            s = f'{s}_robust'
        if self.iref is not None:
            s = f'{s}_iref_{self.iref.start}_{self.iref.stop - 1}'
        if self.qmin > 0:
            s = f'{s}_qmin{self.qmin:.2f}'
        if self.qmax < 1:
            s = f'{s}_qmax{self.qmax:.2f}'
        return s
    
    def get_reference_frame(self, stack):
        ''' Get reference frame from stack '''
        # If stack ID found in cache, return corresponding reference image
        if id(stack) in self.refimg_cache:
            return self.refimg_cache[id(stack)]
        
        # Otherwise, compute reference image and add it to cache
        if self.iref is not None:
            ibounds = (self.iref.start, self.iref.stop - 1)
            stack = stack[self.iref]
        else:
            ibounds = (0, stack.shape[0] - 1)
        logger.info(
            f'computing ref. image as median of frames {ibounds[0]} - {ibounds[1]}')
        refimg = np.median(stack, axis=0)
        s = skew(refimg.ravel())
        logger.info(f'ref. image skewness: {s:.2f}')
        self.refimg_cache[id(stack)] = refimg

        # Return reference image
        return refimg

    def get_pixel_mask(self, img):
        ''' 
        Get selection mask for pixels within quantile range of interest in input image
        
        :param img: image 2D array
        :return: boolean mask of selected pixels that can be used to select pixels
            from an image array by simply using img[mask]
        '''
        # Compute bounding values corresponding to input quantiles 
        vbounds = np.quantile(img, [self.qmin, self.qmax])
        # Create boolean mask of pixels within quantile range
        mask = np.logical_and(img >= vbounds[0], img <= vbounds[1])
        # Log
        logger.info(f'selecting {mask.sum()}/{mask.size} pixels within quantile range {self.qmin} - {self.qmax}')
        # Return mask
        return mask
        
    def plot_frame(self, frame, ax=None, mode='img', **kwargs):
        ''' 
        Plot frame image / distribution
        
        :param img: image 2D array
        :param ax: axis to use for plotting (optional)
        :param mode: type of plot to use:
            - "img" for the frame image
            - "dist" for its distribution
            - "all" for both
        :return: figure handle
        '''
        # If mode is "all", create figure with two axes, and 
        # plot both image and its distribution
        if mode == 'all':
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            for mode, ax in zip(['img', 'dist'], axes):
                self.plot_frame(frame, ax=ax, mode=mode, **kwargs)
            fig.tight_layout()
            return fig
        
        # Create/retrieve figure and axis 
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        sns.despine(ax=ax)

        # Plot image or distribution
        if mode == 'dist':
            # Flatten image array and convert to pandas DataFrame
            dist = pd.Series(frame.ravel(), name='intensity').to_frame()
            dist['selected'] = True
            hue, hue_order = None, None
            # If quantiles are provided, materialize them on the histogram distribution
            if self.qmin > 0:
                vmin = np.quantile(frame, self.qmin)
                dist['selected'] = np.logical_and(dist['selected'], dist['intensity'] >= vmin)
                hue, hue_order = 'selected', [True, False]
                ax.axvline(vmin, color='k', ls='--')
            if self.qmax < 1:
                vmax = np.quantile(frame, self.qmax)
                dist['selected'] = np.logical_and(dist['selected'], dist['intensity'] <= vmax)
                ax.axvline(vmax, color='k', ls='--')
                hue, hue_order = 'selected', [True, False]
            # Plot histogram
            sns.histplot(data=dist, x='intensity', hue=hue, hue_order=hue_order, ax=ax, **kwargs)
        elif mode == 'img':
            # Plot image
            ax.imshow(frame, cmap='viridis', **kwargs)
            # If quantiles are provided, materialize corresponding selected pixels
            # on the image by making the others slightly transparent
            if self.qmin > 0 or self.qmax < 1:
                mask = self.get_pixel_mask(frame)
                masked = np.ma.masked_where(mask, mask)
                ax.imshow(masked, alpha=.5, cmap='gray_r')
        else:
            raise ValueError(f'unknown plotting mode: {mode}')

        # Return figure handle
        return fig
    
    def plot_codist(self, refimg, img, ax=None, kind='hist', marginals=False, regres=None,
                    color=None, label=None, height=4, qmax=None, verbose=True):
        ''' 
        Plot co-distribution of pixel intensity in reference and current image
        
        :param refimg: reference image 2D array
        :param img: current image 2D array
        :param ax: axis to use for plotting (optional)
        :param kind: type of plot to use ("hist" or "scatter")
        :param marginals: whether to plot marginal distributions (optional)
        :param regres: linear regression parameters (optional)
        :param height: height of figure (optional). Only used if ax is None.
        :param qmax: maximum quantile to use for plot limits (optional)
        :param verbose: whether or not to log progress (optional)
        :return: figure handle
        '''
        # If single axis provided and marginals are required, raise error
        if marginals and ax is not None:
            raise ValueError('cannot plot marginals on provided axis')
        
        # Create dataframe from flattened images
        df = pd.DataFrame({'reference frame': refimg.ravel(), 'current frame': img.ravel()})

        # Log, if required
        if verbose:
            logger.info(f'plotting {kind} intensity co-distribution of {len(df)} pixels')

        # Create/retrieve figure and ax(es)
        if ax is None:
            # If marginals, create joint grid and extract axes
            if marginals:
                g = sns.JointGrid(height=height)
                ax = g.ax_joint
                axmargx = g.ax_marg_x
                axmargy = g.ax_marg_y
            # Otherwise, create facet grid and extract single axis
            else:
                g = sns.FacetGrid(data=df, height=height)
                ax = g.ax
            # Retrieve figure handle
            fig = g.fig
        else:
            fig = ax.get_figure()
        
        # Determine plotting function
        pltkwargs = dict(color=color)
        if kind == 'hist':
            pltfunc = sns.histplot
        elif kind == 'scatter':
            pltfunc = sns.scatterplot
            pltkwargs.update(dict(s=1, alpha=0.1))
        else:
            raise ValueError(f'unknown plotting kind: {kind}')
        
        # Plot co-distribution
        pltfunc(x=df['reference frame'], y=df['current frame'], ax=ax, label=label, **pltkwargs)

        # Plot marginals, if required
        if marginals:
            sns.histplot(x=df['reference frame'], ax=axmargx)
            sns.histplot(y=df['current frame'], ax=axmargy)

        # Plot linear regression, if provided
        if regres is not None:
            xref = df['reference frame'].mean()
            yref = xref * regres['slope'] + regres['intercept']
            ax.axline((xref, yref), slope=regres['slope'], color=color)
        
        # # Make sure to include (0, 0) in plot limits
        # ax.set_xlim(left=0)
        # ax.set_ylim(bottom=0)

        # If maximum quantile is provided, set plot limits accordingly
        if qmax is not None:
            xmax = np.quantile(df['reference frame'], qmax)
            ymax = np.quantile(df['current frame'], qmax)
            ax.set_xlim(right=xmax)
            ax.set_ylim(top=ymax)
        
        # Add grid
        ax.grid(True)
                
        # Return figure handle
        return fig
    
    def plot_codists(self, stack, iframes, regres=None, height=3, col_wrap=4, **kwargs):
        ''' 
        Plot co-distributions of pixel intensity of several stack frames
        with the stack reference image

        :param stack: input image stack
        :param iframes: indices of frames for which to plot co-distributions
        :param regres: linear regression parameters dataframe (optional)
        :return: figure handle
        '''
        # Get reference frame from stack
        refimg = self.get_reference_frame(stack)

        # Select subset of pixels to use for plotting as mask
        mask = self.get_pixel_mask(refimg)

        # Apply mask to reference image
        refimg = refimg[mask]

        # If frame index provied as range object, convert to list
        if isinstance(iframes, range):
            iframes = list(iframes)
        
        # Create figure
        fig = sns.FacetGrid(
            pd.DataFrame({'frame': iframes}), 
            height=height, 
            col='frame', 
            col_wrap=col_wrap
        ).fig

        # Plot co-distributions for each frame of interest
        logger.info(f'plotting intensity co-distribution of {len(iframes)} frames')
        for i, (ax, iframe) in enumerate(zip(fig.axes, tqdm(iframes))):
            self.plot_codist(
                refimg, 
                stack[iframe][mask], 
                ax=ax, 
                regres=None if regres is None else regres.loc[iframe],
                verbose=False,
                **kwargs
            )

        # Add global title 
        fig.suptitle('intensity co-distributions', y=1.05)
        
        # Return figure handle
        return fig

    def regress_frame(self, frame, ref_frame, idxs=None):
        '''
        Perform robust linear regression between a frame and a reference frame
        
        :param frame: frame 2D array
        :param ref_frame: reference frame 2D array
        :param idxs: serialized indices of pixels to use for regression (optional)
        :return: linear fit parameters as pandas Series
        '''
        x, y = ref_frame.ravel(), frame.ravel()
        if idxs is not None:
            x, y = x[idxs], y[idxs]
        return mylinregress(x, y, robust=self.robust)
    
    def regress_frames(self, stack, npix=None):
        ''' 
        Correct image stack with linear regresion to reference frame
        
        :param stack: input image stack
        :param npix: number of pixels to use for regression (optional). 
            If None, all pixels are used.
        :return: dataframe of linear regression parameters
        '''
        # Get reference frame from stack
        ref_frame = self.get_reference_frame(stack)
        # Select subset of pixels to use for regression as mask
        mask = self.get_pixel_mask(ref_frame)
        logger.info(f'performing {"robust " if self.robust else ""}linear regression on {stack.shape[0]} frames')
        # If required, select random subset of pixels to use for regression 
        if npix is not None:
            logger.info(f'selecting random {npix} pixels for regression')
            idxs = np.random.choice(ref_frame.size, npix, replace=False)
        else:
            idxs = None
        regouts = []
        # For each frame
        for frame in tqdm(stack):
            regouts.append(self.regress_frame(
                frame[mask], ref_frame[mask], idxs=idxs))
        # Return dataframe of linear regression parameters
        return pd.concat(regouts, axis=1).T
    
    def plot_linreg_params(self, stack, params=None, axes=None, fps=None, delimiters=None):
        ''' 
        Plot linear regression parameters (along with median frame intensity) over time
        
        :param stack: input image stack
        :param params: dataframe of linear regression parameters (optional)
        :param axes: list of axes to use for plotting (optional)
        :param fps: frame rate (optional)
        :param delimiters: list indices to highlight (optional)
        '''
        # If regression parameters not provided, compute them
        if params is None:
            params = self.regress_frames(stack)
        
        # Create/retrieve figure and axes
        keys = params.columns
        naxes = len(keys) + 1
        if axes is None:
            fig, axes = plt.subplots(naxes, 1, figsize=(8, naxes))
            sns.despine(fig=fig)
        else:
            if len(axes) != naxes:
                raise ValueError(f'number of axes must match number of parameters + 1 {naxes}')
            fig = axes[0].get_figure()

        # Create time (or index) vector
        t = np.arange(len(params))
        if fps is not None:
            t  = t / fps
            xlabel = 'time (s)'
        else:
            xlabel = 'frame index'
        
        # Compute median frame (or frame subset) intensity over time 
        if self.qmin > 0 or self.qmax < 1:
            mask = self.get_pixel_mask(self.get_reference_frame(stack))
            substack = np.array([frame[mask] for frame in stack])
            ymed = np.median(substack, axis=1)
        else:
            ymed = np.median(stack, axis=(1, 2))

        # Compute and plot median frame intensity (of selected pixels) over time 
        axes[0].plot(t, ymed)
        axes[0].set_ylabel('med. I')

        # Plot linear regression parameters over time
        for k, ax in zip(params, axes[1:]):
            ax.plot(t, params[k])
            ax.set_ylabel(k)
        
        # Set x-axis label on last axis
        axes[-1].set_xlabel(xlabel)

        # Highlight delimiters, if provided
        if delimiters is not None:
            for ax in axes:
                for d in delimiters:
                    ax.axvline(d, color='k', ls='--')

        # Adjust layout
        fig.tight_layout()

        # Add global title
        fig.suptitle('linear regression parameters', y=1.05)

        # Return figure handle
        return fig
    
    def correct(self, stack, regparams=None):
        ''' Correct image stack with linear regresion to reference frame '''
        # Save input data type and cast as float64 for increased precision
        ref_dtype = stack.dtype
        stack = stack.astype(np.float64)

        # Compute linear regression parameters over time, if not provided
        if regparams is None:
            regparams = self.regress_frames(stack)
        else:
            if len(regparams) != stack.shape[0]:
                raise ValueError(
                    f'number of provided regression parameters ({len(regparams)}) does not match stack size ({stack.shape[0]})')
        # Extract slopes and intercepts, and reshape to 3D
        slopes = regparams['slope'].values[:, np.newaxis, np.newaxis]
        intercepts = regparams['intercept'].values[:, np.newaxis, np.newaxis]
        # Correct stack
        logger.info('correcting stack with linear regression parameters')
        corrected_stack = (stack - intercepts) / slopes

        # Cast from floating point to input type (usually 16-bit integer)
        corrected_stack = corrected_stack.astype(ref_dtype)

        # Return
        return corrected_stack


# class MedianCorrector(Corrector):
        
#     def __str__(self) -> str:        
#         return self.__class__.__name__
        
#     @property
#     def code(self):
#         return 'median'
        
#     def correct(self, stack):
#         '''
#         Correct image stack for with median-subtraction.

#         :param stack: input image stack
#         :return: processed image stack
#         '''
#         logger.info(f'applying median correction to {stack.shape[0]}-frames stack...')
#         # Compute frame median over time
#         ymed = np.median(stack, axis=(1, 2))
#         # Subtract median-corrected fit from each pixel to detrend stack
#         return self.subtract_vector(stack, ymed)
    

# class MeanCorrector(Corrector):
        
#     def __str__(self) -> str:        
#         return self.__class__.__name__
        
#     @property
#     def code(self):
#         return 'mean'
        
#     def correct(self, stack):
#         '''
#         Correct image stack for with mean-subtraction.

#         :param stack: input image stack
#         :return: processed image stack
#         '''
#         logger.info(f'applying mean correction to {stack.shape[0]}-frames stack...')
#         # Compute frame mean over time
#         ymean = np.mean(stack, axis=(1, 2))
#         # Subtract mean-corrected fit from each pixel to detrend stack
#         return self.subtract_vector(stack, ymean)


# class ExponentialCorrector(Corrector):
#     ''' Generic interface to an stack exponential decay corrector. '''

#     def __init__(self, nexps=1, nfit=None, ncorrupted=0):
#         '''
#         Constructor
        
#         :param nexps: number of exponentials in the decay function
#         :param nfit: number of initial frames on which to perform exponential fit  
#         :param ncorrupted: number of initial corrupted frames to substitute after detrending 
#         '''
#         self.nexps = nexps
#         self.nfit = nfit
#         self.ncorrupted = ncorrupted
    
#     def __str__(self) -> str:        
#         return f'{self.__class__.__name__}(nexps={self.nexps}, nfit={self.nfit}, ncorrupted={self.ncorrupted})'

#     @property
#     def code(self):
#         nexps_str = {1: 'mono',2: 'bi'}[self.nexps]
#         s = f'{nexps_str}expdecay'
#         if self.nfit is not None:
#             s = f'{s}_{self.nfit}fit'            
#         if self.ncorrupted > 0:
#             s = f'{s}_{self.ncorrupted}corrupted'
#         return s

#     @property
#     def nexps(self):
#         return self._nexps

#     @nexps.setter
#     def nexps(self, value):
#         if value not in (1, 2):
#             raise ValueError('Number of exponentials must be one of (1, 2).')
#         # Adjust function fo number of exponentials 
#         if value == 1:
#             self.decayfunc = expdecay
#         elif value == 2:
#             self.decayfunc = biexpdecay
#         self._nexps = value
    
#     @property
#     def nfit(self):
#         return self._nfit

#     @nfit.setter
#     def nfit(self, value):
#         if value is not None and value < 0:
#             raise ValueError('number of initial frames for fit must be positive')
#         self._nfit = value

#     @property
#     def ncorrupted(self):
#         return self._ncorrupted

#     @ncorrupted.setter
#     def ncorrupted(self, value):
#         if value < 0:
#             raise ValueError('number of initial corrupted frames must be positive')
#         self._ncorrupted = value
    
#     def expdecayfit(self, y):
#         ''' 
#         Fit an exponential decay to a signal
        
#         :param y: signal array
#         :return: fitted decay array
#         '''
#         # Get signal size
#         nsamples = y.size
#         # Determine number of samples on which to perform the fit
#         if self.nfit is not None:
#             nfit = self.nfit
#         else:
#             nfit = nsamples + 1
#         # Reduce input signal to nfit samples
#         y = y[:nfit]
#         # Compute signal statistics
#         ptp = np.ptp(y)
#         ymed = np.median(y)
#         ystd = y.std()
#         # Initial parameters guess
#         H0 = ymed  # vertical offset: signal median
#         A0 = 1  # amplitude: 1
#         tau0 = 1 # decay time constant: 1 sample
#         x0 = 0  # horizontal offset: 0 sample
#         # Parameters bounds
#         Hbounds = (ymed - 0.1 * ptp, ymed + 0.1 * ptp)  # vertical offset: within median +/-10% of variation range 
#         Abounds = (-1e3, 1e3)  # amplitude: within +/- 1000
#         taubounds = (1e-3, nfit / 2) # decay time constant: 0.001 - half-signal length 
#         x0bounds = (-nfit, nfit)  # horizontal offset: within +/- signal length
#         # Adapt inputs to number of exponentials
#         p0 = (H0,) + (A0,) * self.nexps + (tau0,) * self.nexps + (x0,) * self.nexps
#         pbounds = (Hbounds, *([Abounds] * self.nexps), *([taubounds] * self.nexps), *([x0bounds] * self.nexps))
#         pbounds = tuple(zip(*pbounds))
#         # Least-square fit over restricted signal
#         xfit = np.arange(nfit)
#         popt, _ = curve_fit(self.decayfunc, xfit, y, p0=p0, bounds=pbounds, max_nfev=20000)
#         logger.info(f'popt: {popt}')
#         # Compute fitted profile
#         yfit = self.decayfunc(xfit, *popt)
#         # Compute rmse of fit
#         rmse = np.sqrt(((y - yfit) ** 2).mean())
#         # Compute ratio of rmse to signal standard deviation
#         rel_rmse = rmse / ystd
#         s = f'RMSE = {rmse:.2f}, STD = {ystd:.2f}, RMSE / STD = {rmse / ystd:.2f}'
#         logger.info(s)
#         # Raise error if ratio is too high
#         if rel_rmse > DECAY_FIT_MAX_REL_RMSE:
#             self.plot(y, yfit)
#             raise ValueError(f'{self} fit quality too poor: {s}')
#         # Return fit over entire signal
#         xfull = np.arange(nsamples)
#         return self.decayfunc(xfull, *popt)

#     def correct(self, stack):
#         '''
#         Correct image stack for initial exponential decay.

#         :param stack: input image stack
#         :return: processed image stack
#         '''
#         logger.info(f'applying exponential detrending to {stack.shape[0]}-frames stack...')
#         # Compute frame average over time
#         y = stack.mean(axis=(1, 2))
#         # Compute exponential decay fit on frame average profile beyond corrupted frames
#         yfit = self.expdecayfit(y[self.ncorrupted:].copy())
#         # Subtract fit from each pixel to detrend stack beyond corrupted frames
#         stack[self.ncorrupted:] = self.subtract_vector(stack[self.ncorrupted:], yfit)
#         # Substitute corrupted first n frames
#         stack[:self.ncorrupted] = stack[self.ncorrupted]
#         # Return stack
#         return stack


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