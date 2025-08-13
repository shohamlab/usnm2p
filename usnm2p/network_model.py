# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2024-03-14 17:13:28
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-08-12 23:21:33

''' Network model utilities '''

# External packages
import numba
import textwrap
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize
from tqdm import tqdm
from statannotations.Annotator import Annotator
from scipy.integrate import simps

# Internal modules
from .solvers import ODESolver, EventDrivenSolver
from .logger import logger
from .utils import expconv, expconv_reciprocal, threshold_linear, fast_threshold_linear, as_iterable, generate_unique_id
from .postpro import mylinregress
from .constants import *


@numba.njit
def fast_compute_drive(W, srel, r, x):
    '''
    Compute total input drive to gain function

    :param W: connectivity matrix (2D array)
    :param srel: vector of relative stimulus sensitivities (1D array)
    :param r: activity vector (Hz)
    :param x: stimulus input
    :return: total input drive vector
    '''    
    return np.dot(W, r) + x * srel


@numba.njit
def vector_fast_threshold_linear(x, params):
    ''' 
    Wrapper around fast_threshold_linear that handles vector inputs

    :param x: n-sized vector if input
    :param params: n-by-m matrix of parameters 
    :return: n-size vector of outputs
    '''
    n = params.shape[0]
    y = np.empty(n)
    for i in range(n):
        y[i] = fast_threshold_linear(x[i], params[i, 0], params[i, 1])
    return y


class FGainCallable:
    ''' Interface for gain function callable(s) '''

    def __init__(self, fgain, params):
        '''
        Class constructor

        :param fgain: callable gain function
        :param params: dataframe of gain function parameters per population
        '''        
        # Assign gain function callable and parameters
        self.fgain = fgain
        self.params = params

        # If gain function set to "threshold_linear", construct a 
        # vectorized "fast call" alternative
        if fgain is threshold_linear and isinstance(params, pd.DataFrame):
            self.fast_fgain = vector_fast_threshold_linear
            self.pmat = params.values
            self.has_fast_alternative = True
        
        # Otherwise, set flag to False
        else:
            self.has_fast = False
    
    def __call__(self, x):
        ''' 
        Internal call method, calling the gain function with its parameters

        :param x: input vector as numpy array
        :param fast: whether to use call the "fast" alternative
        :return: array of gain function output values per population
        '''
        # If possible, call the "fast" alternative with
        if self.has_fast_alternative:
            return self.fast_fgain(x, self.pmat)

        # Otherwise, call normal gain function for parameters dictionary
        out = np.empty(self.params.shape[0])
        for i in range(out.size):
            out[i] = self.fgain(x[i], **self.params.iloc[i].to_dict())
        return out


@numba.njit
def fast_compute_derivatives(r, g, tau):
    '''
    Compute activity derivatives from activity state and gain output

    :param r: activity vector
    :param g: gain output vector
    :param tau: time constants vector
    :return: activity derivatives
    '''
    return (g - r) / tau


# Define custom error classes

class SimulationError(Exception):
    ''' Simulation error '''
    pass


class ModelError(Exception):
    ''' Model error '''
    pass


class MetricError(Exception):
    ''' Metric error '''
    pass


class NetworkModel:
    ''' Network model of the visual cortex micro-circuitry '''

    # Cell-type-specific color palette
    palette = {
        'E': 'C0',
        'PV': 'C1',
        'SST': 'r',
        'VIP': 'C2',
    }

    # Max allowed activity value
    MAX_RATE = 1e3

    # Default colormap for connectivity matrices
    W_CMAP = sns.diverging_palette(145, 300, s=60, as_cmap=True)

    # Reference coupling strength for E -> SST connections, 
    # used to rescale connectivity matrices
    WREF = ('E', 'SST', 12.)

    # Reference gain threshold and slope for E population, used to rescale gain functions
    XREF = ('E', 5.)
    GREF = ('E', 1.)

    def __init__(self, W=None, tau=None, fgain=None, fparams=None, b=None, srel=None):
        '''
        Initialize the network model

        :param W (optional): network connectivity matrix, provided as dataframe. If None, set to 0 for all populations
        :param tau (optional): time constants vector (in ms), provided as pandas series. 
            If None, set to 10 ms for all populations
        :param fgain (optional): gain function. If None, use threshold-linear
        :param fparams (optional): gain function parameters, either:
            - a (name: value) dictionary / pandas Series of parameters, if unique
            - a dataframe with parameters as columns and populations as rows, if population-specific
        :param b (optional): baseline inputs vector, provided as pandas series
        :param srel (optional): relative stimulus sensitivity vector, provided as pandas series
        '''
        # Extract keys from first non-None input
        for param in (W, tau, fparams, b):
            if param is not None:
                self.set_keys(param.index.values)
                break
        if not self.has_keys():
            raise ModelError('at least one of the following parameters must be provided: W, tau, fparams, b')
        
        # Rescale connectivity matrix and gain function parameters to match model scale
        if W is not None:
            W = self.rescale_W(W)
        if fparams is not None:
            fparams = self.rescale_fparams(fparams)

        # Set attributes
        self.W = W
        self.tau = tau
        self.fgain = fgain
        self.fparams = fparams
        self.b = b
        self.srel = srel

        # Log
        logger.info(f'initialized {self}')
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(' + ', '.join(self.keys) + ')'
    
    def copy(self):
        ''' Return copy of current model '''
        return self.__class__(
            W=self.W.copy(),
            tau=self.tau.copy(),
            fgain=self.fgain,
            fparams=self.fparams.copy() if self.fparams is not None else None,
            b=self.b.copy() if self.b is not None else None,
            srel=self.srel.copy() if self.srel is not None else None
        )
    
    @property
    def size(self):
        return len(self.keys)

    @property
    def idx(self):
        ''' Return model constituent population names as pandas index '''
        return pd.Index(self.keys, name='population')
    
    def has_keys(self):
        ''' Return whether the model keys have been set or not '''
        return hasattr(self, 'keys') and self.keys is not None

    def check_keys(self, keys):
        '''
        Check that the given keys match the current model keys

        :param keys: keys value
        '''
        # Cast keys to list
        keys = np.asarray(keys)
        # Check that keys match
        isdiff = keys != self.keys
        if isinstance(isdiff, np.ndarray):
            isdiff = isdiff.any()
        if isdiff:
            raise ModelError(f'input keys ({keys}) do not match current model keys ({self.keys})')
    
    def set_keys(self, keys):
        '''
        Set the model keys (if not already set) or check that the given keys match the current model keys

        :param keys: keys value
        '''
        if not self.has_keys():
            self.keys = np.asarray(keys)
        else:
            self.check_keys(keys)
    
    def remove_population(self, key):
        ''' Remove population from model '''
        if not self.has_keys():
            raise ModelError('model keys have not been set')
        if key not in self.keys:
            raise ModelError(f'population "{key}" not found in model keys')
        logger.info(f'{self}: removing {key} population')
        # Remove population from keys
        self.keys = np.asarray([k for k in self.keys if k != key])
        # Remove population from connectivity matrix
        if self.W is not None:
            self.W = self.W.drop(key, axis=0).drop(key, axis=1)
        # Remove population from time constants
        if self.tau is not None:
            self.tau = self.tau.drop(key)
        # Remove population from gain function parameters
        if self.fparams is not None:
            if isinstance(self.fparams, pd.DataFrame):
                self.fparams = self.fparams.drop(key, axis=0)
            elif isinstance(self.fparams, pd.Series):
                self.fparams = self.fparams.drop(key)
        # Remove population from baseline inputs
        if self.b is not None:
            self.b = self.b.drop(key)
        # Remove population from relative stimulus sensitivities
        if self.srel is not None:
            self.srel = self.srel.drop(key)
        logger.info(f'resulting model: {self}')
    
    def get_empty_W(self):
        ''' Return model-sized empty connectivity matrix '''
        return pd.DataFrame(
            index=pd.Index(self.keys, name='pre-synaptic'), 
            columns=pd.Index(self.keys, name='post-synaptic')
        )
    
    @property
    def W(self):
        return self._W
    
    @W.setter
    def W(self, W):
        '''
        Set network connectivity matrix

        :param W: 2D square dataframe where rows and columns represent pre-synaptic
         and post-synaptic elements, respectively.
        '''
        if W is None:
            W = self.get_empty_W().fillna(0.)

        # Check that input is a dataframe with no NaN values
        if not isinstance(W, pd.DataFrame):
            raise ModelError('connectivity matrix must be a pandas dataframe')
        if W.isna().any().any():
            raise ModelError('connectivity matrix cannot contain NaN values')
        
        # Cast as float
        W = W.astype(float)
        
        # Check that matrix is square and that rows and columns match
        Wrows, Wcols = W.index.values, W.columns.values
        if len(Wrows) != len(Wcols):
            raise ModelError('connectivity matrix must be square')
        if not np.all(Wrows == Wcols):
            raise ModelError(f'mismatch between matrix rows {Wrows} and columns {Wcols}')
        self.set_keys(Wrows)

        # Set connectivity matrix
        self._W = W

        # Extract underlying 2D array of connectivity matrix
        self.Wmat = W.values.T
    
    def disconnect(self, presyn_key=None):
        '''
        Disconnect population(s)
        
        :param presyn_key (optional): pre-synaptic population key to disconnect. 
            If None (or "all"), disconnect all populations
        :return: model instance with updated connectivity matrix
        '''
        if presyn_key == 'all':
            presyn_key = None
        if presyn_key is not None:
            if presyn_key not in self.keys:
                raise ModelError(f'"{presyn_key}" population not found in model keys')
            W = self.W.copy()
            W.loc[presyn_key] = 0.
        else:
            W = self.get_empty_W().fillna(0.)
        self.W = W
        return self
    
    @property
    def Wnames(self):
        ''' Return names of network connectivity matrix elements '''
        return self.W.stack().index.values
    
    @property
    def nW(self):
        ''' Return number of elements in the connectivity matrix '''
        return len(self.W.stack())
    
    @property
    def Ikeys(self):
        ''' 
        Return array of inhibitory populations in the model
        (identified from connectivity matrix)
        '''
        # Identify inhibitory populations as those having negative weights
        # in connectivity matrix
        is_inhibitory = (self.W < 0).any(axis=1)
        # Return inhibitory populations
        return np.asarray(is_inhibitory[is_inhibitory].index.values, dtype='U')
    
    @property
    def Ekey(self):
        ''' 
        Return key corresponding to excitatory population in the model
        (identified from connectivity matrix)
        '''
        # Identify excitatory population as that having positive weights
        # in connectivity matrix
        is_excitatory = (self.W > 0).any(axis=1)
        # If more than one identified, raise error
        Ekey = is_excitatory[is_excitatory].index
        if len(Ekey) > 1:
            raise ModelError(f'more than 1 excitatory population identified: {Ekey.values}')
        # Return excitatory population
        return Ekey[0]
    
    def check_vector_input(self, v):
        '''
        Check vector input

        :param v: vector input, provided as pandas Series
        '''
        if not isinstance(v, pd.Series):
            raise ModelError(f'input vector must be provided as pandas series')
        if len(v) != self.size:
            raise ModelError(f'input vector length ({len(v)}) does not match model size ({self.size})')
        if not np.all(v.index.values == self.keys):
            raise ModelError(f'input vector keys ({v.index.values}) do not match model keys ({self.keys})')
        if v.index.name != 'population':
            raise ModelError(f'input vector index name must be "population"')
    
    @property
    def tau(self):
        return self._tau
    
    @tau.setter
    def tau(self, tau):
        if tau is None:
            tau = pd.Series(10., index=self.idx, name='tau (s)')
        self.check_vector_input(tau)
        self._tau = tau.round(2)
        self.tauvec = self._tau.values
    
    @property
    def b(self):
        return self._b
    
    @b.setter
    def b(self, b):
        if b is not None:
            self.check_vector_input(b)
        self._b = b

    @property
    def bvec(self):
        ''' Return baseline inputs as numpy array '''
        return self.b.values if self.b is not None else None
    
    @property
    def srel(self):
        return self._srel

    @srel.setter
    def srel(self, srel):
        # If no input provided, set all values to 1
        if srel is None:
            srel = pd.Series(1., index=self.idx, name='stimulus sensitivity')
        # Check input validity
        self.check_vector_input(srel)
        if srel.min() < 0:
            raise ModelError('relative stimulus sensitivities must be positive numbers')        
        # Cast as float
        srel = srel.astype(float)
        # Set attribute
        self._srel = srel
        # Extract underlying array of relative stimulus sensitivity
        self.srel_vec = srel.values
    
    @property
    def params_table(self):
        ''' Return dataframe with model parameters per population ''' 
        cols = []
        for col in [self.tau, self.b, self.srel]:
            if col is not None:
                cols.append(col) 
        return pd.concat(cols, axis=1)
    
    @property
    def fgain(self):
        return self._fgain

    @fgain.setter
    def fgain(self, fgain):
        if fgain is None:
            fgain = threshold_linear
        if not callable(fgain):
            raise ModelError('gain function must be callable')
        self._fgain = fgain
    
    @property
    def fparams(self):
        return self._fparams
    
    @fparams.setter
    def fparams(self, params):
        '''
        :param params (optional): gain function parameters, either:
        - a (name: value) dictionary / pandas Series of parameters, if unique
        - a dataframe with parameters as columns and populations as rows, if population-specific
        '''
        # If None provided, set to empty dictionary
        if params is None:
            params = {}
        
        # If dataframe provided, check that (1) rows match and (2) columns do not match model population names
        elif isinstance(params, pd.DataFrame):
            self.check_keys(params.index.values)
            for c in params.columns:
                if c in self.keys:
                    raise ModelError(f'gain function parameter column {c} matches a population name')
            params = params.round(2)
        
        # Otherwise, raise error
        else:
            raise ModelError(f'invalid fparams type: {type(params)}')

        # Assign to class, and extract callable
        self._fparams = params
        self.fgain_callable = FGainCallable(self.fgain, params)
    
    def is_fgain_bounded(self):
        '''
        Assess wither gain function is bounded to 0-1 interval
        '''
        return np.isclose(self.fgain(1e3), 1)
    
    @classmethod
    def rescale_W(cls, W):
        ''' Rescale connectivity matrix to match reference E-SST coupling strength '''
        Wval = W.loc[cls.WREF[0], cls.WREF[1]]
        return W * cls.WREF[2] / Wval
    
    @classmethod
    def rescale_fparams(cls, fparams):
        ''' Rescale gain function parameters to match reference threshold and slope '''
        # Extract reference threshold and slope
        x0 = fparams.loc[cls.XREF[0], 'x0']
        g = fparams.loc[cls.GREF[0], 'A']

        # Create copy and rescale threshold and slope
        fparams_rescaled = fparams.copy()
        fparams_rescaled['x0'] *= cls.XREF[1] / x0
        fparams_rescaled['A'] *= cls.GREF[1] / g

        # Return rescaled parameters
        return fparams_rescaled
    
    def attrcodes(self):
        '''
        Construct dictionary of unique identifiers for model attributes
        :return: dictionary of attribute codes
        '''
        # Define attributes to include in the code
        attrs = ['keys', 'W', 'fgain', 'fparams', 'tau', 'b', 'srel']
        
        # Assemble unique identifiers for each set attribute
        attrids = {}
        for k in attrs:
            val = getattr(self, k)
            if val is not None:
                attrids[k] = generate_unique_id(val)
        
        # Return attribute codes 
        return attrids

    @classmethod
    def plot_connectivity_matrix(cls, W, Werr=None, norm=False, ax=None, cbar=True, height='auto', 
                                 vmin=None, vmax=None, title='connectivity matrix', colwrap=4, 
                                 agg=False, clabel='connection strength'):
        '''
        Plot connectivity matrix(ces)

        :param W: connectivity matrix(ces) to plot.
        :param Werr (optional): error matrix to use for ± annotations
        :param norm (optional): whether to normalize the matrix before plotting
        :param ax (optional): axis handle
        :param cbar (optional): whether to display colorbar
        :param height (optional): figure height
        :param vmin (optional): minimum value for color scale
        :param vmax (optional): maximum value for color scale
        :param title (optional): axis title
        :param colwrap (optional): number of columns to wrap connectivity matrices
        :param agg (optional): whether to aggregate multiple matrices into a single one
        :return: figure handle
        '''
        # If disctionary provided, concatenate into single dataframe
        if isinstance(W, dict):
            W = pd.concat(W, axis=0, names=['matrix']) 

        # If auto-height, adjust height to matrix size 
        if height == 'auto':
            height = len(W.columns)

        # If multiple connectivity matrices provided
        if isinstance(W.index, pd.MultiIndex) and W.index.nlevels > 1:

            # Extract grouping variable 
            gby = W.index.names[0]

            # If only one group, drop groupby level
            if len(W.index.get_level_values(gby).unique()) == 1:
                W = W.droplevel(gby)

            else:
                # If error matrix providedm raise error
                if Werr is not None:
                    raise ModelError('error matrix cannot be provided for multi-matrix input')

                # If aggregation flag ON, aggregate across groups and 
                # plot mean matrix with ±std annotations
                if agg:
                    aggby = W.index.names[-1]
                    groups = W.groupby(aggby)
                    Wmean, Wstd = groups.mean(), groups.std()
                    if ax is not None:
                        fig = ax.get_figure()
                    else:
                        fig, ax = plt.subplots(figsize=(4, 3))
                    cls.plot_connectivity_matrix(
                        Wmean, 
                        Werr=Wstd, 
                        norm=norm, 
                        ax=ax,
                        title=f'mean across {gby}' if title is None else title,
                        clabel=clabel,
                        cbar=True
                    )
                    # fig.subplots_adjust(wspace=0.5)
                    return fig

                # Otherwise, plot each matrix on separate axis
                else:

                    # If normalization not requested, ensure color codes ar at least consistent by setting global bounds
                    if not norm:
                        Wamax = W.abs().max().max()
                        Wamax = np.ceil(10 * Wamax) / 10
                        vmin, vmax = -Wamax, Wamax

                    groups = W.groupby(gby)
                    ninputs = len(groups)
                    if ax is not None:
                        axes = as_iterable(ax)
                        if len(axes) != ninputs:
                            raise ModelError(f'number of axes ({len(axes)}) does not correspond to number of connectivity matrices ({ninputs})')
                        fig = axes[0].get_figure()
                    else:
                        ncols = min(ninputs, colwrap)
                        nrows = ninputs // colwrap
                        if ninputs % colwrap != 0:
                            nrows += 1
                        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * height, nrows * height))
                        if nrows > 1:
                            fig.subplots_adjust(hspace=1)
                        axes = axes.flatten()
                        suptitle = title
                        if norm:
                            suptitle = f'normalized {suptitle}'
                        fig.suptitle(suptitle, fontsize=12, y=1 + .05 / nrows)
                    for ax, (k, w) in zip(axes, groups):
                        cls.plot_connectivity_matrix(
                            w.droplevel(gby),
                            norm=norm,
                            ax=ax,
                            title=f'{gby} {k}' if gby == 'run' else k,
                            cbar=not norm and ax is axes[ninputs - 1],
                            vmin=vmin, vmax=vmax,
                            clabel=clabel,
                        )
                    for ax in axes[ninputs:]:
                        ax.axis('off')
                    fig.subplots_adjust(bottom=.1)
                    return fig       

        # Create/retrieve figure and axis
        if ax is None:
            width = height * 1.33 if cbar else height
            fig, ax = plt.subplots(figsize=(width, height))
        else:
            fig = ax.get_figure()

        # Set axis title
        ax.set_title(title, pad=10)

        # Replace infinite values by NaN
        W = W.replace([np.inf, -np.inf], np.nan)
        if Werr is not None:
            Werr = Werr.replace(np.inf, np.nan)
            if any(Werr.stack().dropna() < 0):
                raise ModelError('error matrix cannot contain negative values')
        
        # If normalization requested, normalize matrix(es)
        if norm:
            Wamax = W.abs().max().max()
            W = W / Wamax
            if Werr is not None:
                Werr = Werr / Wamax

        # If no vmin/vmax provided, set to symmetric values
        Wamax = W.abs().max().max()
        Wamax = np.ceil(10 * Wamax) / 10
        if vmin is None:
            vmin = -Wamax
        if vmax is None:
            vmax = Wamax
            
        # Plot connectivity matrix
        sns.heatmap(
            W, 
            ax=ax, 
            square=True, 
            vmin=vmin, 
            vmax=vmax,
            center=0, 
            cmap=cls.W_CMAP, 
            cbar=cbar,
            cbar_kws={'label': clabel} if cbar else None,
        )

        # Add annotations
        for y in range(W.shape[0]):
            for x in range(W.shape[1]):
                dy = 0
                w = W.iloc[y, x]
                if not np.isnan(w):
                    txt = f'{w:.2g}'
                    ishigh = np.abs(w) > 0.5 * Wamax
                    color = 'w' if ishigh else 'k'
                    if Werr is not None:
                        werr = Werr.iloc[y, x]
                        errtxt = f'\n±{werr:.1g}'
                        dy = 0.1
                    ax.text(
                        x + 0.5, y + 0.5 - dy, txt,
                        ha='center', va='center',
                        fontsize=12, 
                        color=color, 
                        fontweight='bold' if Werr is not None else 'normal',
                    )
                    if Werr is not None:
                        ax.text(
                            x + 0.5, y + 0.5 + dy, errtxt,
                            ha='center', va='center',
                            fontsize=10, 
                            color=color,
                        )

        # Remove ticks
        ax.tick_params(axis='both', which='both', bottom=False, left=False)

        # If colorbar predent, restrict its ticks to [vmin, 0, vmax]
        if cbar:
            cax = ax.collections[0].colorbar.ax
            cax.set_yticks([vmin, 0, vmax])

        # Return figure handle
        return fig
    
    def plot_time_constants(self, tau=None, ax=None, suffix=None):
        '''
        Plot firing rate adaptation time constants per population

        :param tau: time constants vector, provided as pandas series. If None, use current model time constants
        :param ax (optional): axis handle
        :param suffix (optional): suffix to append to the axis title
        :return: figure handle
        '''
        # If no time constants provided, use current model time constants
        if tau is None:
            tau = self.tau
        
        # Create/retrieve figure and axis
        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 3))
        else:
            fig = ax.get_figure()
        
        # Set axis layout
        sns.despine(ax=ax)
        title = 'time constants'
        if suffix is not None:
            title = f'{title} ({suffix})'
        ax.set_title(title)
        ax.set_xlabel('population')
        
        # Plot time constants
        ax.bar(
            tau.index, 
            tau.values, 
            color=[self.palette.get(k, None) for k in tau.index])

        # Set y-axis label and adjust layout
        ax.set_ylabel('time constant (ms)')
        # fig.tight_layout()

        # Return figure handle
        return fig

    def plot_stimulus_sensitivity(self, srel=None, ax=None, title=None, norm=False, add_stats=True):
        '''
        Plot relative stimulus sensitivity per population

        :param srel: relative sensitivities vector, provided as pandas series. If None, use current model sensitivities
        :param ax (optional): axis handle
        :param title (optional): axis title
        :param add_stats: whether to add statistical comparisons between populations
            (only if extra dimensions are found)  
        :return: figure handle
        '''
        # If no sensitivities provided, use current model sensitivities
        if srel is None:
            srel = self.srel
        
        # Create/retrieve figure and axis
        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 3))
        else:
            fig = ax.get_figure()
        
        # Set axis layout
        sns.despine(ax=ax)
        if title is None:
            title = 'stimulus sensitivities'
        ax.set_title(title)
        ax.set_xlabel('population')

        if norm:
            maxsrel = srel.groupby('population').mean().max()
            srel = srel / maxsrel

        pltkwargs = dict(
            data=srel.reset_index(),
            x='population',
            y=srel.name,
        )
        
        # Plot sensitivities
        sns.barplot(
            ax=ax,
            **pltkwargs,
            palette=self.palette,
            errorbar='se'
        )

        # If extra dimension, perform statistical comparison across populations
        if srel.index.nlevels > 1 and add_stats:
            pairs = list(itertools.combinations(self.keys, 2))
            # Perform tests and add statistical annotations
            annotator = Annotator(
                ax=ax, pairs=pairs, **pltkwargs)
            annotator.configure(
                test='Mann-Whitney', 
                text_format='star',
                loc='outside'
            )
            annotator.apply_and_annotate()

        # Return figure handle
        return fig
    
    def plot_fgain(self, ax=None, title=None):
        '''
        Plot the gain function(s)

        :param ax (optional): axis handle
        :param title (optional): axis title
        :return: figure handle
        '''
        # Create.retrieve figure and axis
        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 3))
        else:
            fig = ax.get_figure()

        # Set axis layout
        sns.despine(ax=ax)
        ax.set_xlabel('input')
        ax.set_ylabel('output')

        # Create x values (up to 3 times the maximum threshold value)
        x0max = self.fparams['x0'].max()
        x = np.linspace(0, 3 * x0max, 100)

        # Determine title if not provided
        if title is None:
            title = 'gain functions'
        
        # Plot gain functions
        for k, params in self.fparams.iterrows(): 
            ax.plot(x, self.fgain(x, *params.values), lw=2, c=self.palette.get(k, None), label=k)
        ax.legend(loc='upper left', frameon=False)

        # Adjust layout
        ax.set_title(title)
        fig.tight_layout()

        # Return figure handle
        return fig

    def plot_summary(self, height=3, W=None, srel=None, axes=None, add_time_constants=False, 
                     add_axtitles=True, norm=False):
        '''
        Plot model summary, i.e. time constants, gain functions and connectivity matrix

        :param height (optional): figure height
        :param add_time_constants (optional): whether to plot time constants graph (defaults to False) 
        :param axes (optional): axes on which to plot
        :param add_axtitles (optional): whether to add titles to axes (defaults to True)
        :param norm: whether to normalize stimulus sensitivities and coupling weights 
        :return: figure handle
        '''
        # Determine number of axes
        naxes = 3
        if add_time_constants:
            naxes += 1
        
        # Create/retrieve figure and axes
        if axes is None:
            fig, axes = plt.subplots(1, naxes, figsize=(height * naxes, height))
            has_parent = False
        else:
            if len(axes) != naxes:
                raise ValueError(f'number of input axes ({len(axes)} does not match number of required axes ({naxes})')
            fig = axes[0].get_figure()
            has_parent = True
        axiter = iter(axes)

        # Construct uniform empty axis title or leave it to be decided
        axtitle = None if add_axtitles else ''

        # Plot connectivity matrix
        self.plot_connectivity_matrix(
            self.W if W is None else W, agg=W is not None, norm=norm,
            ax=next(axiter), title=axtitle)

        # Plot gain functions
        self.plot_fgain(ax=next(axiter), title=axtitle)

        # Plot stimulus sensitivities
        self.plot_stimulus_sensitivity(srel=srel, ax=next(axiter), title=axtitle, norm=norm)

        # If required, plot time constants
        if add_time_constants:
            self.plot_time_constants(ax=next(axiter), title=axtitle)
        
        # If no parent, adjust layout
        if not has_parent:
            fig.tight_layout()
            fig.suptitle(self, fontsize=12, y=1.05)

        # Return figure handle
        return fig
    
    def compute_drive(self, r, x=None, signed=True):
        '''
        Compute total input drive to gain function

        :param r: activity vector (Hz)
        :param x (optional): stimulus input (scalar or model-sized vector)
        :param signed (optional): whether to compute and sum signed presynaptic drives 
            (i.e. inhibitory drives are negative) or absolute values (all drives are positive).
            Defaults to True (i.e. signed drives)
        :return: total input drive vector
        '''
        # If input activity is a dataframe, apply function on each row and
        # return as dataframe
        if isinstance(r, pd.DataFrame):
            self.check_keys(r.columns)
            d = np.array([self.compute_drive(rr, x=x, signed=signed) for rr in r.values])
            if x is not None:
                d += x
            return pd.DataFrame(d, columns=self.idx, index=r.index)
        
        # Extract connectivity matrix, and take absolute values if signed=False
        W = self.W.values.T.copy()
        if not signed:
            W = np.abs(W)
        
        # Compute total synaptic drive
        drive = np.dot(W, r)
        
        # Add baseline inputs if present
        if self.bvec is not None:
            drive += self.bvec
        
        # Add stimulus-driven input, if present
        if x is not None:
            drive += x * self.srel_vec

        # Return total drive
        return drive
    
    def get_steady_state_objfunc(self, *args, **kwargs):
        '''
        Return objective function whose root is the steady state, i.e.
        the function that solves x = gain(x)

        :return: objective function
        '''        
        return lambda x: self.fgain_callable(self.compute_drive(x, *args, **kwargs)) - x
    
    def compute_steady_state(self, *args, **kwargs):
        '''
        Compute steady states

        :return: steady state activity vector, formatted as pandas series (Hz)
        '''
        # Get steady-state objective function callable
        objfunc = self.get_steady_state_objfunc(*args, **kwargs)
        # Create initial guess vector
        p0 = np.zeros(self.size)
        # Find root of objective function
        ss = optimize.fsolve(objfunc, p0)
        # ss = optimize.root(objfunc, p0).x
        # Cast output as series, and return
        return pd.Series(data=ss, index=self.idx, name='activity')
    
    def find_baseline_drive(self, pkey, r, *args, bkey=None, maxiter=100, rtol=1e-2, **kwargs):
        '''
        Perform, binary search to find baseline drive necessary to reach desired
        steady state activity on specific population.

        :param pkey: name of population of interest regarding steady state activity
        :param r: desired steady state activity (scalar)
        :param bkey: name of population targeted by baseline input (defaults to population of interest)
        :param maxiter (optional): maximum number of iterations
        :param rtol (optional): relative tolerance for convergence
        :return: baseline drive (scalar)
        '''
        # If baseline key not provided, set to population of interest
        if bkey is None:
            bkey = pkey

        logger.info(f'{self}: finding baseline {bkey} drive required to reach {pkey} = {r:.2f} steady state activity')

        # If baseline is already set, keep a copy
        bcopy = None
        if self.b is not None:
            bcopy = self.b.copy()
        
        # Set initial baseline input: 0 except for for targeted population
        self.b = pd.Series(0., index=self.idx)
        self.b.loc[bkey] = 1

        # Compute steady state activity
        ss = self.compute_steady_state(*args, **kwargs)[pkey]

        # Initialize parameters
        niter = 0  # iterations counter
        converged = False  # convergence flag

        # First pass: progressively increase baseline input until steady state activity is above target
        while ss < r:
            logger.debug(f'iter {niter}: {bkey}drive = {self.b.loc[bkey]} -> ss = {ss:.2f}')
            self.b.loc[bkey] *= 2
            ss = self.compute_steady_state(*args, **kwargs)[pkey]
            if self.b.loc[bkey] > 1000:
                logger.warning(
                    f'cannot reach targeted {pkey} steady state activity with reasonable {bkey}drive')
                return np.nan
            niter += 1
        logger.debug(f'iter {niter}: {bkey}drive = {self.b.loc[bkey]} -> ss = {ss:.2f}')
        niter += 1

        # Second pass: binary search to find baseline input that yields steady state activity closest to target
        bounds = [self.b.loc[bkey] / 2, self.b.loc[bkey]]
        while not converged and niter <= maxiter:
            # Set baseline input to mean of bounds, and compute steady state activity
            self.b.loc[bkey] = np.mean(bounds)
            ss = self.compute_steady_state(*args, **kwargs)[pkey]
            logger.debug(f'iter {niter}: {bkey}drive = {self.b.loc[bkey]} -> ss = {ss:.2f}')

            # If steady state activity is close enough to target, mark convergence
            if np.isclose(ss, r, rtol=rtol):
                converged = True
            # Otherwise, update bounds
            elif ss < r:
                bounds[0] = self.b.loc[bkey]
            else:
                bounds[1] = self.b.loc[bkey]

            # Increment iteration counter
            niter += 1
        
        # If maximum number of iterations reached, raise error
        if niter > maxiter:
            raise ModelError('maximum number of iterations reached')
        
        # Store final baseline input
        out = self.b.loc[bkey]

        # Reset baseline input to original value
        self.b = bcopy

        # Return target baseline input
        return out
    
    def find_baseline_drives(self, r, *args, bkey=None, **kwargs):
        '''
        Find baseline drives necessary to reach desired steady state activities

        :param r: desired steady state activity per population, provided as pandas series
        :param bkey (optional): name of population targeted by baseline input (defaults to "self" for each population)
        :return: baseline drives for the corresponding populations, formatted as pandas series
        '''
        # Initialize baseline drives vector
        b = pd.Series(0., index=self.idx)
        # Iterate over populations
        for pkey, rpop in r.items():
            b[pkey] = self.find_baseline_drive(pkey, rpop, *args, bkey=bkey, **kwargs)
        # Return baseline drives
        return b

    def derivatives(self, r, x=0.):
        '''
        Return activity derivatives

        :param r: activity vector (Hz)
        :param x: stimulus amplitude
        :return: activity derivatives (Hz/s)
        '''
        # Compute total input drive
        drive = fast_compute_drive(self.Wmat, self.srel_vec, r, x)
        # Compute gain function output
        g = self.fgain_callable(drive)
        # Subtract leak term, divide by time constants and return
        return fast_compute_derivatives(r, g, self.tauvec)

    def simulate(self, tstop=500., r0=None, A=None, tstart=100., tstim=200., tau_stim=None, dt=None, verbose=True, target_dt=None):
        '''
        Simulate the network model

        :param tstop: total simulation time (ms)
        :param r0: initial activity vector, provided as pandas series
        :param A: external stimulus amplitude
        :param tstart: stimulus start time (ms)
        :param tstim: stimulus duration (ms)
        :param tau_stim (optional): stimulus rise/decay time constant (ms)
        :param dt: simulation time step (ms)
        :param verbose (optional): whether to log simulation progress
        :param target_dt (optional): target time step used for post-simulation resampling
        :return: activity time series
        '''
        # Determine logging level
        flog = logger.info if verbose else logger.debug 

        # Extract underlying numpy objects for connectivity matrix
        # and stimulus sensitivity vector
        self.Wmat = self.W.values.T
        self.srel_vec = self.srel.values

        # If no initial conditions provided, set to steady state
        if r0 is None:
            r0 = self.compute_steady_state()
        # Otherwise, check initial conditions validity
        else: 
            if not isinstance(r0, pd.Series):
                raise ModelError('initial conditions must be provided as pandas series')
            self.check_keys(r0.index.values)
        
        # If no external inputs provided
        if A is None:
            # Initialize simple ODE solver
            solver = ODESolver(
                self.keys, 
                lambda t, *args: self.derivatives(*args),
                dt=dt
            )
            # Define solver arguments
            solver_args = [r0, tstop]

        # Otherwise
        else:            
            # Define events vector
            if tau_stim is not None:
                tconv = expconv_reciprocal(.999, tau=tau_stim)
                if tconv > tstim:
                    logger.warning('stimulus rise/decay time constant does not allow exponential convergence within stimulus duration')
                    tconv = tstim
                tvec = np.linspace(0, tconv, 100)
                tevents = np.hstack([tvec + tstart, tvec + tstart + tstim])
                xconv = expconv(tvec, tau=tau_stim)
                xevents = np.hstack([xconv, 1 - xconv])
            else:
                tevents = [tstart, tstart + tstim]
                xevents = [1, 0]
            events = [(0, 0)] + list(zip(tevents, xevents))

            # Define event-driven solver
            solver = EventDrivenSolver(
                lambda x: setattr(solver, 'stim', A * x),  # eventfunc
                self.keys, 
                lambda t, *args: self.derivatives(*args, x=solver.stim),  # dfunc
                event_params={'stim': float(A)},
                dt=dt,
            )
            # Define solver arguments
            solver_args = [r0, events, tstop]

        # Compute solution
        flog(f'{self}: running {tstop} s long simulation with A = {A}')
        sol = solver(*solver_args, target_dt=target_dt)

        # If solution diverged, raise error
        if sol[self.keys].max().max() > self.MAX_RATE:
            raise SimulationError('simulation diverged')

        # If stimulus modulation vector is present, scale by stimulus amplitude
        if 'x' in sol:
            sol['x'] *= A

        # Return output dataframe
        return sol

    @staticmethod
    def extract_stim_bounds(x):
        '''
        Extract stimulus bounds from stimulus modulation vector

        :param x: stimulus pandas series, indexed by time
        :return: stimulus start and end times
        '''
        dx = x.diff()
        # If no stimulus modulation, return bounds of time vector
        if all(dx.dropna() == 0.):
            return x.index.values[0], x.index.values[-1]
        istart, iend = np.where(dx > 0)[0][0], np.where(dx < 0)[0][0] - 1
        tstart, tend = x.index.values[istart], x.index.values[iend]
        return tstart, tend
    
    def get_secondhalf_slice(self, data, tbounds):
        '''
        Return slice corresponding to second half of a time interval 
        in a results dataframe 

        :param data: simulation results dataframe
        :param tbounds: time bounds tuple
        :return: second half-slice of input data
        '''
        # Unpack time bounds
        tstart, tend = tbounds

        # Derive closest point to mid-interval time
        tmid = (tstart + tend) / 2
        imid = np.argmin(np.abs(data.index.values - tmid))
        tmid = data.index.values[imid]

        # Return second half-slice of input data
        return data.loc[tmid:tend]
    
    def is_stable(self, data):
        '''
        Assess whether activity in simulation results is stable

        :param data: activity series (or activity-per-population dataframe), indexed by time (s)
        :return: whether activity is stable
        '''
        # If input is a dataframe, assess stability for each population
        if isinstance(data, pd.DataFrame) and len(data.columns) > 1:
            is_stable = data.apply(self.is_stable, axis=0)
            return is_stable.all()

        # If standard deviation is negligible, return True
        if data.std() < 1e-1:
            return True

        # If not enough data points, return True 
        if len(data) < 3:
            return True

        # Predict relative linear variation over interval
        t = data.index.values
        regres = mylinregress(t, data.values, robust=True)
        tbounds = np.array([t[0], t[-1]])
        ybounds = regres.loc['slope'] * tbounds + regres.loc['intercept']
        yreldiff = np.diff(ybounds)[0] / np.mean(ybounds)

        # Return whether linear variations are negligible
        return np.abs(yreldiff) < 1e-2
        # return regres['pval'] >= 0.05

    def extract_response_magnitude(self, data, window='stim', metric='ss', verbose=True):
        '''
        Extract magnitude of stimulus-evoked response from simulation results

        :param data: simulation results dataframe
        :param window (optional): window interval during which to extract response magnitude, one of:
            - "pre": pre-stimulus interval
            - "stim": stimulus interval
            - "post": post-stimulus interval
        :param metric (optional): metric to extract, one of:
            - "ss": steady-state activity
            - "peak": peak activity
            - "mean": mean of activity
        :return: response magnitude activity series
        '''
        # If index contains extra dimensions, run extraction recursively
        if isinstance(data.index, pd.MultiIndex) and len(data.index.names) > 1:
            # Identify extra dimensions
            gby = data.index.names[:-1]
            if len(gby) == 1:
                gby = gby[0]
            # Initialize empty steady-state dictionary
            ss = {}
            # For each group from extra-dimensions
            for k, v in data.groupby(gby):
                # Attempt to extract steady-state
                try:
                    ss[k] = self.extract_response_magnitude(v.droplevel(gby), window=window, metric=metric, verbose=verbose)
                # If metric error, log warning and set metric to NaN
                except MetricError as e:
                    if verbose:
                        logger.warning(f'{gby} = {k:.2g}: {e}')
                    ss[k] = pd.Series(index=self.keys, name='activity')
            return pd.concat(ss, axis=0, names=as_iterable(gby)).unstack(level='population')

        # Create copy of data to avoid in-place modification
        data = data.copy()
         
        # If not stimulus modulation present, raise error
        if 'x' not in data.columns:
            raise ModelError('no stimulus modulation present in simulation results')

        # Extract stimulus time bounds from stimulus modulation vector
        tstim = self.extract_stim_bounds(data.pop('x'))

        # Derive time bounds of interest based on interval type
        if window == 'pre':
            tbounds = (data.index.values[0], tstim[0])
        elif window == 'stim':
            tbounds = tstim
        elif window == 'post':
            tbounds = (tstim[1], data.index.values[-1])
        else:
            raise ModelError(f'unknown window interval type: {window}')
    
        # If window is stim, extract activity baseline as last values preceding stimulus onset
        if window == 'stim':
            baseline = data.loc[tstim[0]].iloc[0]
        # Otherwise, set baseline to zero
        else:
            baseline = pd.Series(0., index=self.keys)

        # If steady-state requested, extract data for second half of selected window and check stability  
        if metric == 'ss':
            data = self.get_secondhalf_slice(data, tbounds)
            if not self.is_stable(data):
                raise MetricError(
                    f'unstable activity in [{tbounds[0]:.2f}s - {tbounds[1]:.2f}s] time interval')
            resp = data.mean(axis=0)
        
        # Compute appropriate metric
        elif metric == 'mean':
            # Mean: compute area under curve, and normalize by window duration
            resp = {}
            for col in data.columns:
                resp[col] = simps(data[col], data.index)
            resp = pd.Series(resp) / (tbounds[1] - tbounds[0])
        elif metric == 'peak':
            resp = data.max(axis=0)
        else:
            raise ModelError(f'unknown response quantification metric: {metric}')

        # Compute relative response magnitude
        resp = resp - baseline
        
        # Return computed metric
        resp.index.name = 'population'
        return resp.rename('activity')
    
    def predict_response_profiles(self, amps):
        '''
        Run simulation sweep and extract response profiles per population

        :param amps: stimulus amplitudes vector
        :return: amplitude-indexed dataframe of response profiles per population
        '''
        # Run sweep of simulations with a range of stimulus amplitudes
        sweep_data = self.run_sweep(amps)
        # Extract responses vs amps
        return self.extract_response_magnitude(sweep_data)
    
    def plot_timeseries(self, sol, ss=None, plot_stimulus=True, add_synaptic_drive=False, title=None, axes=None):
        ''' 
        Plot timeseries from simulation results

        :param sol: simulation results dataframe
        :param ss (optional): steady-state values to add to activity timeseries
        :param plot_stimulus (optional): whether to plot stimulus time series, if present
        :param add_synaptic_drive (optional): whether to add synaptic input drives to timeseries
        :param title (optional): figure title
        :param axes (optional): axes handles
        :return: figure handle
        '''
        # Create copy of input data to avoid in-place modification
        sol = sol.copy()

        # Set initial number of axes
        naxes = 1

        # Extract stimulus bounds, if present
        tbounds, x = None, None
        if 'x' in sol.columns:
            x = sol.pop('x')
            # If extra dimensions, average across them
            if isinstance(x.index, pd.MultiIndex):
                x = x.groupby('t').mean()
            tbounds = self.extract_stim_bounds(x)
            if x.max() == 0:
                tbounds = None
            if plot_stimulus:
                naxes += 1

        # Set columns index name
        sol.columns.name = 'population'

        # Add synaptic inputs to data, if requested
        if add_synaptic_drive:
            syninputs = self.compute_drive(sol)
            naxes += 1
        
        # Create figure backbone / retrieve axes
        if axes is None:
            fig, axes = plt.subplots(naxes, 1, figsize=(7, 2 * naxes), sharex=True)
            axes = np.atleast_1d(axes)
            sns.despine(fig=fig)
        else:
            axes = as_iterable(axes)
            if len(axes) != naxes:
                raise ModelError(f'number of axes ({len(axes)}) does not match number of subplots ({naxes})')
            fig = axes[0].get_figure()
            for ax in axes:
                sns.despine(ax=ax)
        if title is None:
            title = f'{self} - simulation results'
        axes[0].set_title(title)
        axiter = iter(axes)
        ax = next(axiter)

        # Plot stimulus time series, if present
        if x is not None and plot_stimulus:
            ax.plot(x.index, x.values, c='k')
            ax.set_ylabel('stimulus')
            ax = next(axiter)
        
        # Plot activity time series
        sns.lineplot(
            ax=ax,
            data=sol.stack().rename('activity').reset_index(),
            x='t',
            y='activity',
            hue='population',
            errorbar='se',
            palette=self.palette,
        )
        if ss is not None:
            for k, v in sol.items():
                ax.axhline(ss.loc[k], ls='--', c=self.palette[k])
        ax.legend(loc='upper right', frameon=False)
        if self.is_fgain_bounded():
            ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel('activity')

        # If requested, plot synaptic inputs and harmonize y limits with stimulus axis
        if add_synaptic_drive:
            ax = next(axiter)
            for k, v in syninputs.items():
                v.plot(
                    ax=ax, 
                    c=self.palette.get(k, None),
                )
            ax.set_ylabel('synaptic inputs')
            axlist = [axes[0], ax]
            ylims = [ax.get_ylim() for ax in axlist]
            ymin, ymax = min([ymin for ymin, _ in ylims]), max([ymax for _, ymax in ylims])
            for ax in axlist:
                ax.set_ylim(ymin, ymax)
        
        # Set x-axis label on last axis
        axes[-1].set_xlabel('time (ms)')

        # Highlight stimulus period on all axes, if present
        if tbounds is not None:
            for ax in axes:
                ax.axvspan(*tbounds, fc='k', ec=None, alpha=0.1)

        # Adjust layout
        fig.tight_layout()

        # Return figure handle
        return fig

    def run_sweep(self, amps, verbose=True, on_error='abort', **kwargs):
        '''
        Run sweep of simulations with a range of stimulus amplitudes

        :param amps: stimulus amplitudes vector
        :param verbose (optional): whether to log sweep progress
        :param on_error (optional): behavior on simulation error, one of:
            - "abort": log error and abort sweep
            - "continue": log error and continue sweep
        '''
        # Initialize results dictionary
        sweep_data = {}

        # Log sweep start, if requested
        if verbose:
            logger.info(f'{self}: running {amps.size}-amplitudes stimulation sweep')
        
        # For each stimulus amplitude
        for A in amps:
            # Attempt to run simulation and extract results
            try:
                data = self.simulate(A=A, verbose=False, **kwargs)
                sweep_data[A] = data
            # If simulation fails, log error and abort loop if requested
            except SimulationError as err:
                if verbose:
                    logger.error(f'simulation for amplitude {A} failed: {err}')
                if on_error == 'abort':
                    break
        sweep_data = pd.concat(sweep_data, axis=0, names=['amplitude'])
        return sweep_data

    def plot_sweep_results(self, data, ax=None, xkey='amplitude', norm=None, hue='population', 
                           style=None, style_order=None, col=None, row=None, title=None, 
                           height=2, aspect=1, legend=True):
        '''
        Plot results of stimulus amplitude sweep

        :param data: sweep extracted metrics per population, provided as dataframe
        :param ax (optional): axis handle
        :param xkey (optional): x-axis key
        :param norm (optional): dimension(s) along which to normalize results per population before plotting, one of:
            - None: no normalization
            - 'style': normalize per style
            - 'hue': normalize per hue
            - 'ax': normalize per axis
            - 'grid': normalize across entire grid
        :param hue (optional): extra grouping dimension to use for coloring (default = 'population')
        :param style (optional): extra grouping dimension to use for styling
        :param col (optional): extra grouping dimension to use for columns
        :param row (optional): extra grouping dimension to use for rows
        :param title (optional): axis title
        :param height (optional): figure height (for facet grid only)
        :param aspect (optional): aspect ratio (for facet grid only)
        :return: figure handle
        '''
        if xkey not in data.index.names:
            raise ModelError(f'level {xkey} not found in index')
    
        # Check validity of norm setting
        if norm not in (None, 'style', 'hue', 'ax', 'grid'):
            if norm == True:
                if row is not None or col is not None:
                    norm = 'grid'
                else:
                    norm = 'ax'
            else:
                raise ModelError(f'unknown normalization type: {norm}')
            
        # If input is a dataframe with multi-index, run plotting recursively
        if isinstance(data.index, pd.MultiIndex) and len(data.index.names) > 1:
            # Extract extra dimensions
            gby = [k for k in data.index.names if k not in (xkey, 'run')]
            nextra = len(gby)
            
            # If more than 3 extra dimensions, raise error
            if nextra > 3:
                raise ModelError('cannot plot activation profiles with more than 2 extra levels')
            
            # Assign dimensions to style, col and row
            params = [col, row, style]
            for p in params:
                if p is not None:
                    if p not in gby:
                        raise ModelError(f'level {p} not found in multi-index')
                    del gby[gby.index(p)]
            for i, p in enumerate(params):
                if p is None:
                    params[i] = gby.pop(0) if gby else None
            col, row, style = params
            
            # Determine whether multi-axis grid is needed
            grid_gby = [k for k in [col, row] if k is not None]

            # If grid is needed, create it and loop through axes
            if len(grid_gby) > 0:

                # If single axis provided, raise error 
                if ax is not None:
                    raise ModelError('axis provided but axis grid is needed')
                
                # If grid normalization requested, normalize data across entire grid
                if norm == 'grid':
                    data = self.normalize_profiles(data)
                    ykey = f'normalized {ykey}'
                    norm = None
                
                # Adapt axis title template
                templates = []
                idx_dtypes = data.index.dtypes 
                if col is not None and isinstance(idx_dtypes.loc[col], np.float64):
                    templates.append(f'{col}={{col_name:.2g}}')
                if row is not None and isinstance(idx_dtypes.loc[row], np.float64):
                    templates.append(f'{row}={{row_name:.2g}}')
                template = ', '.join(templates)
                
                # Create grid
                fg = sns.FacetGrid(
                    data=data.reset_index(), 
                    col=col,
                    row=row,
                    aspect=aspect,
                    height=height,
                )
                if template:
                    fg.set_titles(template=template)
                fig = fg.figure

                # Add title to figure, if provided
                if title is not None:
                    fig.suptitle(title, y=1.05)
                
                # Loop through axes and plot
                for ax, (k, v) in zip(fig.axes, data.groupby(grid_gby, sort=False)):
                    legend = ax is fig.axes[-1]
                    self.plot_sweep_results(
                        v.droplevel(grid_gby), ax=ax, xkey=xkey, norm=norm, hue=hue,
                        style=style, style_order=style_order, legend=legend)                
                    if legend:
                        sns.move_legend(ax, bbox_to_anchor=(1, .5), loc='center left', frameon=False)

                return fig

        # Define y-axis label
        ykey = 'activity'

        # If normalization requested
        if norm is not None:
            # If requested, normalize by hue
            if norm == 'hue':
                if hue is None:
                    logger.warning('normalization by hue requested but no hue grouping provided, skipping normalization')
                elif hue not in data.index.names:
                    if data.columns.name == hue:
                        logger.info(f'{hue} hue level is the columns dimension -> normalizing across columns')
                        norm = 'ax'
                    else:
                        raise ModelError(f'level {hue} not found in index')
                else:
                    data = data.groupby(hue, sort=False).apply(self.normalize_profiles).droplevel(0)
    
            # If requested, normalize by style
            elif norm == 'style':
                if style is None:
                    logger.warning('normalization by style requested but no style grouping provided, skipping normalization')
                elif style not in data.index.names:
                    if data.columns.name == style:
                        logger.info(f'{style} sytle level is the columns dimension -> normalizing across columns')
                        norm = 'ax'
                    else:
                        raise ModelError(f'level {style} not found in index')
                else:
                    data = data.groupby(style, sort=False).apply(self.normalize_profiles).droplevel(0)
            
            # If requested, normalize across entire axis
            if norm == 'ax':
                data = self.normalize_profiles(data)
        
        # Create/retrieve figure and axis
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        else:
            fig = ax.get_figure()
        
        # Adjust axis layout
        sns.despine(ax=ax)

        # Set axis title
        if title is not None:
            ax.set_title(title)

        # Plot activity vs. stimulus amplitude
        sns.lineplot(
            ax=ax,
            data=data.stack().rename(ykey).reset_index(),
            x=xkey,
            y=ykey,
            hue=hue,
            palette=self.palette,
            style=style,
            style_order=style_order,
            legend=legend,
            errorbar='se',
        )

        # Return figure handle
        return fig
   
    def get_coupling_bounds(self, wmax=None, relwmax=None, return_key=False):
        '''
        Get exploration bounds for each element of the network connectivity matrix
        of the current model

        :param wmax: max absolute coupling strength across the network
        :param relwmax: max relative deviation of coupling strength from their reference values 
        :param return_key: whether to also return a key describing how bounds where computed 
        :return: dataframe of 2-tuples representing coupling bounds matrix, with optional descriptive key 
        '''
        # If both (or none of) absolute and relative max strength are provided, raise error
        errmsg = 'one of relative/absolute max coupling strength must be provided'
        if wmax is None and relwmax is None:
            raise ValueError(errmsg)
        if wmax is not None and relwmax is not None:
            raise ValueError(f'only {errmsg}')
        
        # Initialize empty bounds matrix
        Wbounds = self.get_empty_W()

        # If wmax provided
        if wmax is not None:
            # Construct descriptive key
            desc_key = f'wmax = {wmax}'

            # Check that is is a positive int or float
            if not isinstance(wmax, (int, float)) or wmax <= 0:
                raise ModelError('wmax must be a positive number')
            wmax = float(wmax)
        
            # For each pre-synaptic population
            for key in self.keys:
                # Get cell-type-specific coupling stength bounds
                bounds = (0., wmax) if key == 'E' else (-wmax, 0.)
                # Assign them to corresponding row in bounds matrix
                Wbounds.loc[key, :] = [bounds] * self.size
        
        # If relwmax provided
        if relwmax is not None:
            # Construct descriptive key
            desc_key = f'relwmax = {relwmax}'

            # Check that is is a positive int or float
            if not isinstance(relwmax, (int, float)) or relwmax <= 0:
                raise ModelError('relwmax must be a positive number')
            relwmax = float(relwmax)

            # Compute connectivity matrices resulting from max positive and negative relative changes 
            Wmin = (1 - relwmax) * self.W
            Wmax = (1 + relwmax) * self.W

            # Re-arrange to get lower and upper bounds matrices
            Wlb = Wmin.combine(Wmax, func=np.minimum).round(3)
            Wub = Wmin.combine(Wmax, func=np.maximum).round(3)

            # Populate bounds matrix
            for prekey in self.keys:
                for postkey in self.keys:
                    Wbounds.loc[prekey, postkey] = (Wlb.loc[prekey, postkey], Wub.loc[prekey, postkey])

        # Return matrix and optional descriptive key
        if return_key:
            return Wbounds, desc_key 
        else:
            return Wbounds
    
    def check_coupling_bounds(self, Wbounds):
        '''
        Check that connectivitty matrix exploration bounds is comptaible with current model
        
        :param Wbounds: 2D dataframe of exploration bounds for connectivity matrix
        '''
        # Check input validity
        if not isinstance(Wbounds, pd.DataFrame):
            raise ModelError('Wbounds must be a DataFrame')
        if Wbounds.shape != (self.size, self.size):
            raise ModelError(f'Wbounds shape ({Wbounds.shape}) does not match network size ({self.size})')
        if not Wbounds.index.equals(Wbounds.columns):
            raise ModelError('Wbounds must have identical row and column indices')
        if not Wbounds.index.equals(self.W.index):
            raise ModelError('Wbounds indices must match network keys')
        if not all(isinstance(x, tuple) for x in Wbounds.values.ravel()):
            raise ModelError('all Wbounds values must be tuples')
    
    def parse_srel_bounds(self, srel_bounds, uniform_srel=False):
        ''' 
        Parse relative stimulus sensitivity bounds

        :param srel_bounds: (lb, ub) tuple for stimulus sensitivity 
        :para 
        :return: series of (lb, ub) stimulus bounds
        '''
        # If uniform sensitivity required, generate singleton series
        if uniform_srel:
            return pd.Series(index=['all'], data=[srel_bounds])
        # Otherwise, broadcast to all populations 
        else:
            return pd.Series(index=self.keys, data=[srel_bounds] * self.size)
    
    def check_srel_bounds(self, srel_bounds, is_uniform=False):
        ''' Check that stimulus sensitivity exploration bounds are compatible with current model '''
        if not isinstance(srel_bounds, pd.Series):
            raise ModelError('srel_bounds must be a Series')
        for x in srel_bounds.values:
            if not isinstance(x, tuple) or len(x) != 2 or any(not isinstance(xx, float) for xx in x):
                raise ModelError('all srel_bounds values must be 2-tuples of floats')
        if not is_uniform and any(srel_bounds.index.values != self.keys):
            raise ModelError('srel_bounds indices must match network keys')
        if is_uniform and len(srel_bounds) != 1:
            raise ModelError('uniform srel_bounds required but srel_bounds has more than 1 value')
    
    @staticmethod
    def normalize_profiles(y, ythr=MIN_ACTIVITY_LEVEL, verbose=True):
        '''
        Custom activity profiles normalization function that zeros profiles with max activity below a specific
        threshold, to avoid generation of "falsly varying" profiles due to numerical approximation errors.

        :param y: Dataframe of activity profiles per population
        :param ythr: threshold maximum absolute activity level below which to zero profile
        :return: dataframe of normalized activity profiles
        '''
        # Compute maximum absolute values per population (across runs if present)
        if 'run' in y.index.names:
            yavg = y.groupby([k for k in y.index.names if k != 'run']).mean()
            ymax = yavg.abs().max(axis=0)
        else:
            ymax = y.abs().max(axis=0)

        # Normalize each profile by its absolute max
        ynorm = y / ymax

        # Zero profiles of populations that evolve below set activity threshold 
        inactive_pops = ymax.index[ymax < ythr]
        if len(inactive_pops) > 0:
            logfunc = logger.warning if verbose else logger.debug
            logfunc(f'no activity detected in {", ".join(inactive_pops.values)} population{"s" if len(inactive_pops) > 1 else ""}')
            ynorm[inactive_pops] = 0.

        # Return
        return ynorm
    
    def Wvec_to_Wmat(self, Wvec):
        '''
        Convert network connectivity matrix vector to matrix

        :param Wvec: network connectivity matrix vector(s) (size n^2, where n is the number of populations in the network)
        :return: n-by-n network connectivity matrix
        '''
        # If input is series, extract values
        if isinstance(Wvec, pd.Series):
            Wvec = Wvec.values

        # Make sure vector has length n^2
        if len(Wvec) != self.nW:
            raise ModelError('input vector length does not match network connectivity matrix size')
        
        # Assemble as matrix and return
        return pd.Series(
            Wvec, 
            index=pd.MultiIndex.from_tuples(self.Wnames, names=['pre-synaptic', 'post-synaptic'])
        ).unstack()
    
    def parse_input_vector(self, xvec):
        '''
        Parse input parameters

        :param xvec: vector of input parameters
        :return: parsed connectivity matrix and sensitivity series
        '''
        # If input is formatted as series, extract underlying vector
        if isinstance(xvec, pd.Series):
            if not hasattr(self, '_xnames'):
                self.xnames = xvec.index.to_list()
            xvec = xvec.values
        
        # Initialize vectors
        W, srel = None, None

        # Parse connectivity values
        if self.Widx is not None:
            W = self.Wvec_to_Wmat(xvec[self.Widx])
        
        # Parse sensitivity values
        if self.srelidx is not None:
            srel = xvec[self.srelidx]
            if len(srel) == 1:
                srel = srel[0]
            srel = pd.Series(srel, index=self.idx, name='stimulus sensitivity')
                
        # Return parsed outputs
        return W, srel

    def set_from_vector(self, xvec):
        ''' 
        Set model connectivity and sensitivity parameters taken from parameters vector

        :param xvec: vector of model parameters
        :return: model instance (useful for chained commands)
        '''
        # Parse connectivity matrix and sensitivity vector from inputs 
        W, srel = self.parse_input_vector(xvec)

        # Assign them to model (if not None)
        if W is not None:
            self.W = W
        if srel is not None:
            self.srel = srel
    
    @property
    def xnames(self):
        return self._xnames
    
    @xnames.setter
    def xnames(self, l):
        self._xnames = l
        if l is None:
            self.Widx, self.srelidx = None, None
        else:
            self.Widx = [i for i, k in enumerate(l) if k.startswith('W')]
            if len(self.Widx) == 0:
                self.Widx = None 
            self.srelidx = [i for i, k in enumerate(l) if k.startswith('srel')]
            if len(self.srelidx) == 0:
                self.srelidx = None
    
    def compute_stim_drive(self, amps):
        '''
        Compute steady-state stimulus drive per population, over a range of stimulus amplitudes
        
        :param amps: stimulus amplitude vector
        :return: stimulus-amplitude-indexed dataframe of stimulus drive amplitudes per population
        '''
        return pd.DataFrame(
            data=np.outer(amps, self.srel),
            index=pd.Index(amps, name='amplitude'),
            columns=self.idx,
        )
    
    def compute_stim_drive_fraction(self, resp_profiles):
        '''
        Compute relative contribution of stimulus drive to total input drive,
        for a range of stimulus amplitudes
        
        :param resp_profiles: stimulus amplitude indexed dataframe of steady-state
            evoked response magnitude per population
        :return: stimulus amplitude indexed dataframe of relative stimulus
            contribution to total drive per population
        '''
        # Compute absolute net steady-state presynaptic drive
        # onto each post-synaptic population
        presyn_drive = self.compute_drive(resp_profiles).abs()

        # Compute stimulus drive onto each population
        stim_drive = self.compute_stim_drive(resp_profiles.index.values)

        # Compute total drive over input amplitudes range
        total_drive = (presyn_drive + stim_drive)
        
        # Compute relative contribution of stimulus drive to total drive
        rel_stim_drive = stim_drive / total_drive

        # Restrict output ot instances where there is a response
        # (setting other elements to NaN)
        rel_stim_drive = rel_stim_drive.where(
            resp_profiles >= MIN_ACTIVITY_LEVEL, other=np.nan)
        
        # Return
        return rel_stim_drive
    
    def compute_detailed_presynaptic_drives(self, resp_profiles):
        '''
        Compute the contribution of each presynaptic population to the network drive
        of each postynaptic population, for a range of input amplitudes

        :param resp_profiles: stimulus amplitude indexed dataframe of steady-state
            evoked response magnitude per population
        :return: post-synaptic population and stimulus amplitude indexed dataframe of
            presynaptic drive amplitude per population
        '''  
        return pd.concat({
            pop: self.W.loc[:, pop] * resp_profiles for pop in self.keys
        }, axis=0, names=['postsyn pop'])
    
    def compute_I_to_E_drive_fractions(self, detailed_presyn_drives):
        '''
        Compute fraction of contribution of each inhibitory population to
        total inhibitory drive to excitatory population

        :param detailed_presyn_drives: post-synaptic population and stimulus amplitude-indexed dataframe of
            presynaptic drive amplitude per population
        :return: stimulus amplitude indexed dataframe of fractional inhibitory contribution of each population
            to total inhibitory drive to excitatory population
        '''
        # Extract inhibitory presynaptic inputs to excitatory population 
        I_to_E_drives = detailed_presyn_drives.loc[self.Ekey, self.Ikeys]

        # Normalize them to sum of inhibitory drives
        return I_to_E_drives.div(I_to_E_drives.sum(axis=1), axis=0)

    def compute_coinhibition_fractions(self, detailed_presyn_drives, pop1, pop2):
        '''
        Compute inhibitory drive from each inhibitory population in a mutual
        inhibition loop as a fraction of the total inhibitory drive in the loop.

        :param detailed_presyn_drives: post-synaptic population and stimulus amplitude-indexed dataframe of
            presynaptic drive amplitude per population
        :param pop1: key of first inhbitory population
        :param pop2: key of second inhbitory population
        :return: stimulus amplitude indexed dataframe of fractional inhibitory contribution of each population
            to total inhibitory drive in the inhibitiry loop
        '''
        # Make sure that they corresponding to inhibitory population
        for k in [pop1, pop2]:
            if k not in self.Ikeys:
                raise ModelError(f'"{k}" not found in model inhibitory populations ({self.Ikeys})') 

        # Extract inhibitory presynaptic inputs in mutual inhibition loop
        I_to_I_drives  = pd.concat({
            pop1: detailed_presyn_drives.loc[pop2, pop1],
            pop2: detailed_presyn_drives.loc[pop1, pop2],
        }, axis=1)
        I_to_I_drives.columns.name = 'population'

        # Normalize them to sum of inhibitory drives in the loop
        return I_to_I_drives.div(I_to_I_drives.sum(axis=1), axis=0)
    
    def compute_drive_contributions(self, amps):
        '''
        Compute various breakdowns of drive contributions over a range of stimulus amplitudes
        
        :param amps: stimulus amplitude vector
        :return: dictionary of drive contributions dataframes
        '''
        logger.info('computing drive contributions breakdown')
        # Predict response profiles per population
        predicted_profiles = self.predict_response_profiles(amps)

        # Compute relative contribution of stimulus to total drive
        stim_drive_fraction = self.compute_stim_drive_fraction(predicted_profiles)

        # Compute detailed presynaptic drives between each pre-post population pair
        detailed_presyn_drives = self.compute_detailed_presynaptic_drives(predicted_profiles) 

        # Compute fractional contributions to inhibitory drive to excitatory population 
        I_E_fractions = self.compute_I_to_E_drive_fractions(detailed_presyn_drives) 

        # Compute fractional pre-synaptic inhibitory contributions in SST-PV coninhibition
        I_I_fractions = self.compute_coinhibition_fractions(detailed_presyn_drives, 'PV', 'SST')

        # Return in global dictionary
        return {
            'stimulus contribution to total drive': stim_drive_fraction,
            'pre-synaptic contributions to inhibitory drive to E': I_E_fractions,
            'pre-synaptic strengths in SST-PV co-inhibition loop': I_I_fractions    
        }
    
    def plot_drive_contributions(self, data, title=None):
        '''
        Plot detailed breakdown of drive contributions
        
        :paramd data: dictionary of drive contributions dataframes
        :param title (optional): figure title
        :return: figure object
        '''
        logger.info('plotting drive contributions breakdown')
        # Concatenate data into single pandas series
        data = pd.concat(
            {k: v.stack() for k, v in data.items()},
            names=['kind']
        ).rename('fraction')

        # Extract stimulus amplitude vector
        amps = data.index.unique(level='amplitude')

        # Plot all contributions, by breakdown kind
        g = sns.relplot(
            data=data.reset_index(),
                kind='line',
                x='amplitude',
                y='fraction',
                col='kind',
                hue='population',
                errorbar='se',
                palette=self.palette,
                height=3,
                aspect=1.
        )
        g.set_titles('{col_name}')

        # Add reference line at 0.5
        g.refline(y=0.5, ls='--', c='k')

        # Adapt axes titles and limits
        for ax in g.axes.ravel():
            ax.set_title(textwrap.fill(ax.get_title(), width=27))
            ax.set_xlim(-0.05 * amps.max(), 1.05 * amps.max())
            ax.set_ylim(-0.05, 1.05)
        
        # Extract figure
        fig = g.figure

        # Add title, if specified
        if title is not None:
            fig.suptitle(title, y=1.1)
        
        # Return figure
        return fig