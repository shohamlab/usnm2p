# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-11 11:59:10
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2024-08-07 15:57:55

''' Collection of image stacking utilities. '''

import numpy as np
from scipy.stats import skew
from scipy.signal import butter, sosfiltfilt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns
from tqdm import tqdm
# from scipy.optimize import curve_fit

from .constants import *
from .logger import logger
from .utils import sigmoid #, expdecay, biexpdecay, is_within
from .postpro import mylinregress
from .fileops import StackProcessor, NoProcessor, process_and_save


class NoCorrector(NoProcessor):
    ''' No corrector instance to substitute in the code in case no resampling is wanted '''
    pass


class Corrector(StackProcessor):

    def __init__(self, refch=None, **kwargs):
        ''' 
        Initialization
        
        :param refch (optional): reference channel index (1-based) to use for correction (for multi-channel stacks only)
        '''
        super().__init__(**kwargs)
        self.refch = refch
    
    @property
    def refch(self):
        return self._refch

    @refch.setter
    def refch(self, value):
        if value is not None:
            if not isinstance(value, int):
                raise ValueError('reference channel must be an integer')
            if value not in range(1, self.nchannels + 1):
                raise ValueError(f'reference channel must be between 1 and {self.nchannels}')        
        self._refch = value

    @property
    def rootcode(self):
        return 'corrected'

    def run(self, stack):
        '''
        Correct image stack.

        :param stack: input image stack
        :return: processed image stack
        '''
        # Apply correction function, and return
        return self.correct(stack)
    
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

    def __init__(self, robust=False, intercept=True, iref=None, qmin=0, qmax=1, custom=False, wc=None, **kwargs):
        '''
        Initialization

        :param robust: whether or not to use robust linear regression
        :param intercept: whether or not to compute intercept during linear regression fit
        :param iref (optional): frame index range to use to compute reference image
        :param qmin (optional): minimum quantile to use for pixel selection (0-1 float, default: 0)
        :param qmax (optional): maximum quantile to use for pixel selection (0-1 float or "adaptive", default: 1)
        :param custom (optional): whether or not to use custom regressor
        :param wc (optional): normalized cutoff frequency for temporal low-pass filtering of regression parameters
        '''
        # Assign input arguments as attributes
        self.robust = robust
        self.intercept = intercept
        self.iref = iref
        self.qmin = qmin
        self.qmax = qmax
        self.custom = custom
        self.adaptive_qmax = None
        self.wc = wc

        # Initialize empty dictionary of cached reference images
        self.refimg_cache = {}
        
        # Call parent constructor
        super().__init__(**kwargs)
    
    @classmethod
    def from_string(cls, s, nchannels=1):
        ''' Instantiate class from string code '''

        # Check if string code is compatible with class
        if not s.startswith('linreg'):
            raise ValueError(f'invalid {cls.__name__} code: "{s}" does not start with "linreg"')
        
        # Split code by underscores
        s = s.split('_')[1:]

        # Define parameters dictionary
        params = {}

        # Extract parameters from code
        for item in s:
            if item == 'robust':
                params['robust'] = True
            elif item == 'nointercept':
                params['intercept'] = False
            elif item.startswith('iref'):
                params['iref'] = range(*[int(i) for i in item[4:].split('-')])
            elif item.startswith('qmin'):
                params['qmin'] = float(item[4:])
            elif item.startswith('qmax'):
                params['qmax'] = item[4:]
                if params['qmax'] != 'adaptive':
                    params['qmax'] = float(params['qmax'])
            elif item == 'custom':
                params['custom'] = True
            elif item.startswith('wc'):
                params['wc'] = float(item[2:])
            elif item.startswith('refch'):
                params['refch'] = int(item[5:])
            else:
                raise ValueError(f'unknown parameter: "{item}"')
        
        # Instantiate class with extracted parameters
        return cls(**params, nchannels=nchannels)
        
    @property
    def robust(self):
        return self._robust
    
    @robust.setter
    def robust(self, value):
        if not isinstance(value, bool):
            raise ValueError('robust must be a boolean')
        self._robust = value
    
    @property
    def intercept(self):
        return self._intercept
    
    @intercept.setter
    def intercept(self, value):
        if not isinstance(value, bool):
            raise ValueError('intercept must be a boolean')
        self._intercept = value

    @property
    def iref(self):
        return self._iref
    
    @iref.setter
    def iref(self, value):
        if value is not None and not isinstance(value, range):
            raise ValueError('iref must be a range object')
        self._iref = value
    
    @property
    def qmin(self):
        return self._qmin
    
    @qmin.setter
    def qmin(self, value):
        if not 0 <= value < 1:
            raise ValueError('qmin must be between 0 and 1')
        if hasattr(self, 'qmax') and isinstance(self.qmax, float) and value >= self.qmax:
            raise ValueError(f'qmin must be smaller than qmax ({self.qmax})')    
        self._qmin = value
    
    @property
    def qmax(self):
        return self._qmax
    
    @qmax.setter
    def qmax(self, value):
        if isinstance(value, str):
            if value != 'adaptive':
                raise ValueError(f'invalid qmax string code: "{value}"')
        else:
            if not 0 < value <= 1:
                raise ValueError('qmax must be between 0 and 1')
            if hasattr(self, 'qmin') and value <= self.qmin:
                raise ValueError(f'qmax must be larger than qmin ({self.qmin})')
        self._qmax = value
    
    @property
    def custom(self):
        return self._custom
    
    @custom.setter
    def custom(self, value):
        if not isinstance(value, bool):
            raise ValueError('custom must be a boolean')
        self._custom = value
    
    @property
    def wc(self):
        return self._wc

    @wc.setter
    def wc(self, value):
        if value is not None and (value <= 0 or value >= 1):
            raise ValueError('normalized cutoff frequency must be between 0 and 1')
        self._wc = value
        
    def __repr__(self) -> str:
        plist = [f'robust={self.robust}']
        if not self.intercept:
            plist.append('no intercept')
        if self.iref is not None:
            plist.append(f'iref={self.iref}')
        if self.qmin > 0:
            plist.append(f'qmin={self.qmin}')
        if self.qmax == 'adaptive' or self.qmax < 1:
            plist.append(f'qmax={self.qmax}')
        if self.custom:
            plist.append('custom')
        if self.wc is not None:
            plist.append(f'wc={self.wc}')
        if self.refch is not None:
            plist.append(f'refch={self.refch}')
        pstr = ', '.join(plist)
        return f'{self.__class__.__name__}({pstr})'
        
    @property
    def code(self):
        clist = []
        if self.robust:
            clist.append('robust')
        if not self.intercept:
            clist.append('nointercept')
        if self.iref is not None:
            clist.append(f'iref_{self.iref.start}_{self.iref.stop - 1}')
        if self.qmin > 0:
            clist.append(f'qmin{self.qmin:.2f}')
        if self.qmax == 'adaptive':
            clist.append('qmaxadaptive')
        elif self.qmax < 1:
            clist.append(f'qmax{self.qmax:.2f}')
        if self.custom:
            clist.append('custom')
        if self.wc is not None:
            clist.append(f'wc{self.wc:.2f}')
        if self.refch is not None:
            clist.append(f'refch{self.refch}')
        s = 'linreg'
        if len(clist) > 0:
            cstr = '_'.join(clist)
            s = f'{s}_{cstr}'
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

    @staticmethod
    def skew_to_qmax(s, zcrit=2, q0=.01, qinf=.99, sigma=1):
        '''
        Function mapping a distribution skewness value to a maximum selection quantile
        
        :param s: distribution skewness value.
        :param zcrit: critical skewness value (i.e. inflexion point of sigmoid)
        :param q0: selection quantile for zero skewness.
        :param qinf: selection quantile for infinite skewness.
        :param sigma: sigmoid steepness parameter.
        :return: maximum selection quantile.
        '''
        return sigmoid(s, x0=zcrit, sigma=sigma, A=qinf - q0, y0=q0)

    def get_qmax(self, frame):
        '''
        Parse qmax value to determine maximum quantile to use for pixel selection

        :param frame: image 2D array
        :return: maximum quantile to use for pixel selection
        '''
        if self.qmax == 'adaptive':
            # If adaptive qmax provided, use it 
            if self.adaptive_qmax is not None:
                return self.adaptive_qmax
            # Otherwise, compute it from reference image skewness
            return self.skew_to_qmax(skew(frame.ravel()))
        else:
            return self.qmax

    def get_pixel_mask(self, img):
        ''' 
        Get selection mask for pixels within quantile range of interest in input image
        
        :param img: image 2D array
        :return: boolean mask of selected pixels that can be used to select pixels
            from an image array by simply using img[mask]
        '''
        # Compute bounding values corresponding to input quantiles 
        qmax = self.get_qmax(img)

        # If entire image is selected, return full mask
        if self.qmin == 0 and qmax == 1:
            return np.ones(img.shape, dtype=bool)

        # Compute quantile bounds
        vbounds = np.quantile(img, [self.qmin, qmax])
        
        # Create boolean mask of pixels within quantile range
        mask = np.logical_and(img >= vbounds[0], img <= vbounds[1])
        
        # Log
        logger.info(f'selecting {mask.sum()}/{mask.size} pixels within quantile range {self.qmin:.3f} - {qmax:.3f}')
        
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
            hue, hue_order, palette = None, None, None
            # If quantiles are provided, materialize them on the histogram distribution
            if self.qmin > 0:
                vmin = np.quantile(frame, self.qmin)
                dist['selected'] = np.logical_and(dist['selected'], dist['intensity'] >= vmin)
                hue, hue_order = 'selected', [True, False]
                ax.axvline(vmin, color='k', ls='--')
            qmax = self.get_qmax(frame)
            if qmax < 1:
                vmax = np.quantile(frame, qmax)
                dist['selected'] = np.logical_and(dist['selected'], dist['intensity'] <= vmax)
                ax.axvline(vmax, color='k', ls='--')
            if 'selected' in dist:
                hue, hue_order, palette = 'selected', [True, False], {True: 'g', False: 'r'}
            # Plot histogram
            sns.histplot(
                ax=ax, 
                data=dist, 
                x='intensity', 
                hue=hue, 
                hue_order=hue_order,
                palette=palette, 
                bins=100,
                **kwargs
            )
        elif mode == 'img':
            # Plot image
            ax.imshow(frame, cmap='viridis', **kwargs)
            # If quantiles are provided, materialize corresponding selected pixels
            # on the image by making the others slightly transparent
            if self.qmin > 0 or self.get_qmax(frame) < 1:
                mask = self.get_pixel_mask(frame)
                masked = np.ma.masked_where(mask, mask)
                ax.imshow(masked, alpha=.5, cmap='gray_r')
        else:
            raise ValueError(f'unknown plotting mode: {mode}')

        # Return figure handle
        return fig    

    def get_legend_handle(self, kind, color, label):
        if kind == 'scatter':
            return Line2D(
                [0], [0], 
                label=label, 
                linestyle='',
                marker='o', 
                markersize=10, 
                markerfacecolor=color, 
                markeredgecolor='none',
            )
        elif kind == 'hist':
            return mpatches.Patch(color=color, label=label)
        else: 
            raise ValueError(f'unknown plotting kind: {kind}')
    
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
        pltfunc(x=df['reference frame'], y=df['current frame'], ax=ax, **pltkwargs)
        xref = df['reference frame'].mean()

        # Add unit diagonal line
        ax.axline((xref, xref), slope=1, color='k', ls='--')

        # Create and append legend handle, if provided
        if label is not None:
            handle = self.get_legend_handle(kind, color, label)
            if hasattr(ax, 'custom_legend_handles'):
                ax.custom_legend_handles.append(handle)
                ax.legend(handles=ax.custom_legend_handles)
            else:
                ax.custom_legend_handles = [handle]

        # Plot marginals, if required
        if marginals:
            sns.histplot(x=df['reference frame'], ax=axmargx, color=color)
            sns.histplot(y=df['current frame'], ax=axmargy, color=color)

        # Plot linear regression, if provided
        if regres is not None:
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

    def plot_codists(self, stack, iframes, regres=None, height=3, col_wrap=4, axes=None, **kwargs):
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
        
        # Create/extract figure and axes
        newfig = axes is None
        if axes is None:
            fig = sns.FacetGrid(
                pd.DataFrame({'frame': iframes}), 
                height=height, 
                col='frame', 
                col_wrap=col_wrap
            ).fig
            axes = fig.axes
        else:
            if len(axes) != len(iframes):
                raise ValueError(f'number of provided axes ({len(axes)}) must match number of evaluated frames ({len(iframes)})')
            fig = axes[0].get_figure()

        # Plot co-distributions for each frame of interest
        logger.info(f'plotting intensity co-distribution of {len(iframes)} frames')
        for i, (ax, iframe) in enumerate(zip(axes, tqdm(iframes))):
            self.plot_codist(
                refimg, 
                stack[iframe][mask], 
                ax=ax, 
                regres=None if regres is None else regres.loc[iframe],
                verbose=False,
                label=self.code if i == len(iframes) - 1 else None,
                **kwargs
            )

        # If not new figure, move legend
        if not newfig:
            sns.move_legend(axes[-1], bbox_to_anchor=(1.05, 1), loc='upper left')

        # Add global title 
        fig.suptitle('intensity co-distributions', y=1.05)
        
        # Return figure handle
        return fig
    
    def custom_fit(self, x, y):
        '''
        Fit custom regression model between two vectors

        :param x: reference vector
        :param y: current vector
        :return: regression fit parameters as pandas Series 
        '''
        # slope = ratio of standard deviations
        alpha = y.std() / x.std()
        # intercept = offset between means of rescaled reference frame and current frame 
        beta = y.mean() - alpha * x.mean()
        return pd.Series({
            'slope': alpha, 
            'intercept': beta
        })

    def fit_frame(self, frame, ref_frame, idxs=None):
        '''
        Fit linear regression model between a frame and a reference frame
        
        :param frame: frame 2D array
        :param ref_frame: reference frame 2D array
        :param idxs: serialized indices of pixels to use for regression (optional)
        :return: linear fit parameters as pandas Series
        '''
        x, y = ref_frame.ravel(), frame.ravel()
        if idxs is not None:
            x, y = x[idxs], y[idxs]
        if self.custom:
            return self.custom_fit(x, y)
        else:
            return mylinregress(x, y, robust=self.robust, intercept=self.intercept)
    
    def fit(self, stack, ref_frame=None, npix=None):
        ''' 
        Fit linear regression parameters w.r.t. reference frame for each frame in the stack
        
        :param stack: input image stack
        :param ref_frame: reference frame to use for regression (optional).
        :param npix: number of pixels to use for regression (optional). 
            If None, all pixels are used.
        :return: dataframe of fitted linear regression parameters
        '''
        # Get reference frame from stack, if not provided
        if ref_frame is None:
            ref_frame = self.get_reference_frame(stack)

        # Select subset of pixels to use for regression as mask
        mask = self.get_pixel_mask(ref_frame)

        # If required, select random subset of pixels to use for regression 
        if npix is not None:
            logger.info(f'selecting random {npix} pixels for regression')
            idxs = np.random.choice(ref_frame.size, npix, replace=False)
        else:
            idxs = None
        logger.info(f'performing {"robust " if self.robust else ""}linear regression on {stack.shape[0]} frames')
        
        # Perform fit for each frame
        res = []
        for frame in tqdm(stack):
            res.append(self.fit_frame(
                frame[mask], ref_frame[mask], idxs=idxs))
        
        # Concatenate results into dataframe
        df = pd.concat(res, axis=1).T
        
        # If required, apply temporal low-pass filtering to fit parameters
        if self.wc is not None:
            sos = butter(2, self.wc, btype='low', output='sos')
            for k in df:
                df[k] = sosfiltfilt(sos, df[k], axis=0)
        
        # Return dataframe
        return df
    
    def plot_fit(self, stack, params=None, keys=None, axes=None, periodicity=None, 
                           fps=None, delimiters=None, color=None, height=None, width=None):
        ''' 
        Plot linear regression parameters (along with median frame intensity) over time
        
        :param stack: input image stack
        :param params: dataframe of linear regression parameters (optional)
        :param keys: list of parameters to plot (optional)
        :param axes: list of axes to use for plotting (optional)
        :param periodicity: periodicity index used to aggregate data before plotting (optional)
        :param fps: frame rate (optional)
        :param delimiters: list indices to highlight (optional)
        '''
        # Adjust width and height based on periodicity flag
        if width is None:
            width = 8 if periodicity is None else 5
        if height is None:
            height = 2 if periodicity is None else 1
        
        # If regression parameters not provided, compute them. Otherwise, copy them
        if params is None:
            df = self.fit(stack)
        else:
            df = params.copy()
        
        # If keys provided, select corresponding columns
        if keys is not None:
            df = df[keys]
        
        # Compute median frame (or frame subset) intensity over time 
        if self.qmin > 0 or self.qmax == 'adaptive' or self.qmax < 1:
            mask = self.get_pixel_mask(self.get_reference_frame(stack))
            substack = np.array([frame[mask] for frame in stack])
            ymed = np.median(substack, axis=1)
        else:
            ymed = np.median(stack, axis=(1, 2))
        
        # Add median frame intensity to dataframe
        df.insert(0, 'med. I', ymed)
        
        # Create/retrieve figure and axes
        keys = df.columns
        naxes = len(keys)
        newfig = axes is None
        if axes is None:
            fig, axes = plt.subplots(naxes, 1, figsize=(width, naxes * height))
            sns.despine(fig=fig)
        else:
            if len(axes) != naxes:
                raise ValueError(f'number of axes must match number of parameters + 1 {naxes}')
            fig = axes[0].get_figure()

        # Create index vector
        df['frame'] = np.arange(len(params))
        xlabel = 'frame'

        # Wrap index around periodicity, if provided
        if periodicity is not None:
            df['frame'] = df['frame'] % periodicity
        
        # If frame rate provided, convert index to time 
        if fps is not None:
            df['time (s)'] = df['frame'] / fps
            xlabel = 'time (s)'

        # Plot each timeseries over time 
        for i, (k, ax) in enumerate(zip(keys, axes)):
            sns.lineplot(
                ax=ax, 
                data=df, 
                x=xlabel, 
                y=k, 
                color=color,
                label=self.code if i==0 else None
            )
        
        # Set x-axis label on last axis
        axes[-1].set_xlabel(xlabel)

        # Highlight delimiters, if provided
        if delimiters is not None:
            for ax in axes:
                for d in delimiters:
                    ax.axvline(d, color='k', ls='--')
        
        # If not new figure, create legend on last axis
        if not newfig:
            axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Adjust layout
        fig.tight_layout()

        # Add global title
        fig.suptitle('linear regression parameters', y=1.05)

        # Return figure handle
        return fig
    
    def _run(self, stack):
        '''
        Overwrite parent method to allow for processing of multi-channel stack
        with reference channel
        '''
        # If stack is 3D or no reference channel is given, call parent method 
        if stack.ndim == 3 or self.refch is None:
            return self.run(stack)
        
        # Verify that stack is 4D
        if stack.ndim != 4:
            raise ValueError(f'input stack has unsupported shape: {stack.shape}')
        
        # Run correction fit for reference channel
        logger.info(f'extracting fit parameters from channel {self.refch}...')
        params = self.fit(stack[:, self.refch - 1])

        # Process each channel separately and recombine
        chax = 1  # channel axis
        outstack = []
        for i in range(stack.shape[chax]):
            logger.info(f'correcting channel {i + 1} with extracted fit parameters...')
            outstack.append(self.correct(stack[:, i], regparams=params))
        return np.stack(outstack, axis=chax)
    
    def correct(self, stack, regparams=None):
        ''' Correct image stack with linear regresion to reference frame '''
        # Save input data type and cast as float64 for increased precision
        ref_dtype = stack.dtype
        stack = stack.astype(np.float64)

        # Compute linear regression parameters over time, if not provided
        if regparams is None:
            regparams = self.fit(stack)
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

        # If negative values are found, offset stack to obtain only positive values
        if stack.min() >=0 and corrected_stack.min() < 0:
            logger.warning('correction produced negative values -> offseting output stack to 1')
            corrected_stack = corrected_stack - corrected_stack.min() + 1

        # Adapt stack range to fit within input data type numerical bounds
        corrected_stack = self.adapt_stack_range(corrected_stack, ref_dtype)

        # Check that corrected stack is within input data type range
        self.check_stack_range(corrected_stack, ref_dtype)

        # If input was integer-typed, round corrected stack to nearest integers 
        if not np.issubdtype(ref_dtype, np.floating):
            logger.info(f'rounding corrected stack')
            corrected_stack = np.round(corrected_stack)
        
        # Cast back to input data type
        corrected_stack = corrected_stack.astype(ref_dtype)

        # Return
        return corrected_stack


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


def correct_tifs(input_fpaths, input_key, method, nchannels=1, **kwargs):
    '''
    High-level stack detrending function

    :param input_fpaths: list of full paths to input TIF stacks
    :param input_key: input key for output path replacement
    :param method: correction method to apply
    :param nchannels: number of channels in input TIFs
    :return: list of detrended TIF stacks
    '''
    # If 'linreg' in method, instantiate corrector from string
    if 'linreg' in method:
        corrector = LinRegCorrector.from_string(method, nchannels=nchannels)
    # Otherwise, instantiate corrector from method name
    elif method == 'median':
        corrector = MedianCorrector(nchannels=nchannels)
    elif method == 'mean':
        corrector = MeanCorrector(nchannels=nchannels)
    else:
        raise ValueError(f'unknown correction method: {method}')
    # # Apply exponential detrending
    # corrector = ExponentialCorrector(
    #     nexps=NEXPS_DECAY_DETREND, nfit=NSAMPLES_DECAY_DETREND, ncorrupted=NCORRUPTED_BERGAMO)
    
    # Correct all input TIFs     
    corrected_stack_fpaths = process_and_save(
        corrector, input_fpaths, input_key, **kwargs)

    # Return list of output filepaths     
    return corrected_stack_fpaths