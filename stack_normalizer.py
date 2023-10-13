# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-04 17:44:51
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-10-13 11:12:16

''' Collection of filtering utilities. '''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from logger import logger
from fileops import StackProcessor, NoProcessor
from constants import DataRoot


class NoNormalizer(NoProcessor):
    ''' No normalizer instance to substitute in the code in case no normalization is wanted '''
    pass


class GlobalStackNormalizer(StackProcessor):

    ''' Main interface to global stack normalizer '''

    def __init__(self, q=0.05, npix=100, projfunc=np.mean, **kwargs):
        '''
        Initialization

        :param q: target quantile (0-1).
        :param npix: number of pixels to select.
        :param projfunc: function used to project stack along time axis (default: np.mean)
        '''
        self.q = q
        self.npix = npix
        self.projfunc = projfunc
        super().__init__(**kwargs)

    def __str__(self) -> str:
        return f'{self.__class__.__name__}(q={self.q:.2f}, npix={self.npix})'

    @property
    def rootcode(self):
        return DataRoot.NORMALIZED

    @property
    def code(self):
        return f'gn_q{self.q:.2f}_npix{self.npix}'

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, value):
        if not 0 < value <= 1:
            raise ValueError('Quantile must be between 0 and 1.')
        self._q = value

    @property
    def npix(self):
        return self._npix

    @npix.setter
    def npix(self, value):
        if not isinstance(value, int) or value < 0:
            raise ValueError('npix must be a positive integer')
        self._npix = value
    
    @property
    def projfunc(self):
        return self._projfunc
    
    @projfunc.setter
    def projfunc(self, value):
        if not callable(value):
            raise ValueError('projfunc must be a callable')
        self._projfunc = value

    @staticmethod
    def select_quantile_pixels(stack, q, npix, projfunc=np.mean):
        '''
        Project stack along time axis and select pixels in stack that fall closest 
        to a given quantile on the intensity distribution of the projection image.
        
        :param stack: 3D image stack
        :param q: target quantile on the intensity distribution
        :param npix: number of pixels to select
        :param projfunc: function used to project stack along time axis (default: np.mean)
        :return: 2D array of selected pixels coordinates
        '''
        # Compute aggregated image
        logger.info(f'computing {projfunc.__name__} time-projection of {stack.shape[0]}-frames stack...')
        proj = projfunc(stack, axis=0)

        # Compute target value of aggregate pixel intensity corresponding to specified quantile
        vtarget = np.quantile(proj, q)
        logger.info(f'target intensity value based on {q * 1e2:.1f}-th percentile: {vtarget:.2f}')

        # Select pixel(s) with aggregate intensity closest to target value
        dists = np.abs(proj - vtarget)
        iclose = np.argpartition(dists.ravel(), npix-1)[:npix]
        xpix, ypix = np.unravel_index(iclose, proj.shape)
        logger.info(f'identified {xpix.size} pixels with aggregate intensity closest to target')

        # Return array of pixel coordinates
        return np.array([xpix, ypix])

    @classmethod
    def get_baseline_timecourse(cls, stack, q=0.05, npix=100, **kwargs):
        '''
        Extracts a baseline time course from a 3D image stack.

        :param stack: 3D image stack
        :param q: target quantile on the intensity distribution
        :param npix: number of pixels to select
        :return: 1D array of baseline time course
        '''
        # Select pixel(s) with aggregate intensity closest to target value
        xpix, ypix = cls.select_quantile_pixels(stack, q, npix, **kwargs)

        # Extract pixel(s) time course
        pixels_timecourse = np.array([stack[:, xp, yp] for xp, yp in zip(xpix, ypix)])
        
        # Average time course across pixels
        return pixels_timecourse.mean(axis=0).astype(float)

    def run(self, stack: np.array):
        '''
        Performs global normalization of a 3D image stack
    
        :param stack: 3D stack array (time, width, height)
        :return: normalized stack
        '''
        assert len(stack) != 0, 'Stack is empty.'
        assert len(stack) != 1, 'Stack must contain more than one element.'

        # Cast to float64 for increased precision
        input_type = stack.dtype
        stack = stack.astype(np.float64)
        
        # Extract baseline time course from stack
        vref = self.get_baseline_timecourse(
            stack, q=self.q, npix=self.npix, projfunc=self.projfunc)
        
        # Normalize stack by dividing each frame by its baseline value
        logger.info(f'normalizing {stack.shape[0]}-frames stack with {self}')
        normalized_stack = stack / vref[:, None, None]

        # Compute reference amplitude values (median of whole-image average over time)
        # for input and normalized stacks, and their ratio
        vref_in = np.median(stack.mean(axis=0))
        vref_out = np.median(normalized_stack.mean(axis=0))
        vratio = vref_in / vref_out
        logger.info(f'estimated in/out amplitude ratio: {vratio:.2f}')

        # Rescale normalized stack to match reference input amplitude
        # (and therefore hopefully avoid precision loss upon casting to integer)
        normalized_stack *= vratio

        # Extract maximal value in normalized stack and compare it to maximum 
        # allowed value of input type
        maxval = normalized_stack.max()
        input_ub = np.iinfo(input_type).max

        # If maxval exceeds input type max value, raise Error
        if maxval > input_ub:
            raise ValueError(
                f'maximal value in output stack ({maxval:.2f}) exceeds input type max value ({input_ub})')

        # Cast from floating point to input type (usually 16-bit integer)
        normalized_stack = normalized_stack.astype(input_type)

        # Return
        return normalized_stack

    def plot_projection_image(self, stack, cmap='viridis', ax=None):
        ''' 
        Compute and plot stack projection image.
        
        :param stack: 3D image stack
        :param cmap: colormap (default: viridis)
        :param ax: axis object (default: None)
        :return: figure object
        '''
        # Create/retrieve figure and axis
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        # Compute projection image
        proj_img = self.projfunc(stack, axis=0)

        # Select quantile pixels
        xypixs = self.select_quantile_pixels(stack, self.q, self.npix, self.projfunc)

        # Plot projection image
        ax.imshow(proj_img, cmap=cmap)

        # Return figure
        return fig
    
    def plot_select_pixels(self, ax, xypix, color='w'):
        '''
        Plot selected pixels on projection image.
        
        :param ax: axis object
        :param xypix: array of pixel coordinates
        :param color: pixel color (default: white)
        '''
        # Mark selected pixel on projection image
        ax.scatter(*xypix, s=30, ec=color, fc='none', lw=1)
    

        #     # Extract pixel(s) time course
        #     ypixs = np.array([stack[:, xp, yp] for xp, yp in zip(xpix, ypix)])

        #     # Rescale pixel(s) time course to show relative variation, if requested
        #     if yout in ('dFF', 'Z'):
        #         ybaselines = np.quantile(ypixs, 0.5, axis=1)[:, np.newaxis]
        #         ypixs = (ypixs - ybaselines) / ybaselines

        #     # Normalize pixel(s) time course by their noise level, if requested
        #     if yout == 'Z':
        #         munoise = np.mean(ypixs, axis=1)[:, np.newaxis]
        #         sigmanoise = np.mean(ypixs, axis=1)[:, np.newaxis]
        #         ypixs = (ypixs - munoise) / sigmanoise
            
        #     # Average time course across pixels
        #     ypix = ypixs.mean(axis=0).astype(float)

        #     # Plot pixel time course
        #     xvec = np.arange(ypix.size)
        #     if fps is not None:
        #         xvec = xvec / fps
        #     axes[1].plot(
        #         xvec, ypix, c=c, alpha=0.5 if fc is not None else 1, label=f'q={q:.2f}, raw')

        #     # Low-pass filter pixel time course, if cutoff frequency specified
        #     if fc is not None:
        #         if fps is None:
        #             raise ValueError('frame rate must be specified to filter pixel time course')
        #         order = 2
        #         nyq = 0.5 * fps
        #         sos = butter(order, fc / nyq, btype='low', output='sos')
        #         ypix_filt = sosfiltfilt(sos, ypix)
        #         axes[1].plot(xvec, ypix_filt, c=c, alpha=1)

        #     # Compute and plot power spectrum, if specified
        #     if add_spectrum:
        #         if np.any(np.isnan(ypix)):
        #             logger.warning('cannot compute spectrum: NaNs found in timeseries')
        #         else:
        #             spectrum = get_power_spectrum(ypix.copy(), fps, normalize=True)
        #             sns.lineplot(
        #                 data=spectrum.iloc[1:, :], x=Label.FREQ, y=Label.PSPECTRUM_DB, 
        #                 ax=axes[2], color=c)

        # # Add legend on time course plot if multiple quantiles are specified
        # if len(qs) > 1:
        #     axes[1].legend(frameon=False)

        # # Return figure
        # return fig