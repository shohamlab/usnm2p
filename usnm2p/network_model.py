# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2024-03-14 17:13:28
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2024-08-07 15:59:02

import time
import glob
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import seaborn as sns
from scipy import optimize
from tqdm import tqdm
import os
import csv
import hashlib
import lockfile
import multiprocessing as mp
from statannotations.Annotator import Annotator
from scipy.integrate import simps

from .solvers import ODESolver, EventDrivenSolver
from .logger import logger
from .utils import expconv, expconv_reciprocal, threshold_linear, as_iterable
from .postpro import mylinregress
from .batches import get_cpu_count


def generate_unique_id(obj, max_length=None):
    '''
    Generate unique identifier for a given object

    :param obj: object to identify
    :param max_length (optional): maximum length of the identifier
    :return: unique object identifier
    '''
    # Convert object to string and create corresponding hash object
    hash_object = hashlib.md5(str(obj).encode())
    # Compute the hexadecimal digest of the hash
    shash = hash_object.hexdigest()
    # If max length provided, truncate hash string
    if max_length is not None and max_length < len(shash):
        shash = shash[:max_length]
    # If series input, return serialized series string if shorter than hash
    if isinstance(obj, pd.Series):
        s = ''.join([f'{k}{v}' for k, v in obj.items()])
        if len(s) < len(shash):
            return s
    # If input is a callable with a name, return the name if shorter than hash
    if callable(obj) and hasattr(obj, '__name__'):
        if len(obj.__name__) < len(shash):
            return obj.__name__
    # If input is a list / tuple / 1D array, return serialized list string if shorter than hash
    if isinstance(obj, (list, tuple)) or (isinstance(obj, np.ndarray) and obj.ndim == 1):
        s = '-'.join([str(v) for v in obj])
        if len(s) < len(shash):
            return s
    # Return hash string
    return shash


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


class OptimizationError(Exception):
    ''' Optimization error '''
    pass


class FGainCallable:
    def __init__(self, fgain, params):
        self.fgain = fgain
        self.params = params
    
    def __call__(self, x):
        return self.fgain(x, **self.params)


class NetworkModel:
    ''' Network model of the visual cortex micro-circuitry '''

    # Cell-type-specific color palette
    palette = {
        'E': 'C0',
        'PV': 'C1',
        'SST': 'r'
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

    # Default coupling strength bounds for excitatory and inhibitory connections
    WMAX = 20.
    
    # CSV delimiter
    DELIMITER = ','

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

        # # If 2-population model with 1 excitatory and 1 inhibitory population,
        # # Check that network is not excitation dominated
        # if len(W) == 2 and 'E' in W.index.values:
        #     Ikey = list(set(W.index.values) - set(['E']))[0]
        #     Wi = self.get_net_inhibition(Ikey, W=W)
        #     We = self.get_net_excitation(Ikey, W=W)
        #     if Wi < We:
        #         raise ModelError(f'Wi ({Wi}) < We ({We}) -> E/I balance not met')

        # Set connectivity matrix
        self._W = W
    
    def disconnect(self):
        ''' Disconnect all populations '''
        self.W = self.get_empty_W().fillna(0.)
        return self
    
    @property
    def Wmat(self):
        ''' Return connectivity matrix a 2D numpy array '''
        return self.W.values.T
    
    @property
    def Wnames(self):
        ''' Return names of network connectivity matrix elements '''
        return self.W.stack().index.values
    
    @property
    def nW(self):
        ''' Return number of elements in the connectivity matrix '''
        return len(self.W.stack())

    def get_net_inhibition(self, Ikey, Ekey='E', W=None):
        ''' 
        Compute strength of the net inhibition between and E and I populations, 
        as the product of (E to I) and (I to E) coupling strengths

        :param Ikey: inhibitory population key
        :param Ekey: excitatory population key (default: 'E')
        :param W (optional): connectivity matrix. If None, use current model connectivity matrix
        :return: net inhibition strength
        '''
        if W is None:
            W = self.W
        Wei = W.loc[Ekey, Ikey]  # E -> I (> 0)
        Wie = W.loc[Ikey, Ekey]  # I -> E (< 0) 
        return Wei * np.abs(Wie)  # < 0
    
    def get_net_excitation(self, Ikey, Ekey='E', W=None):
        ''' 
        Compute strength of the net excitation between E and I populations, 
        as the product of (E to E) and (I to I) coupling strengths

        :param Ikey: inhibitory population key
        :param Ekey: excitatory population key (default: 'E')
        :param W (optional): connectivity matrix. If None, use current model connectivity matrix
        :return: net excitation strength
        '''
        if W is None:
            W = self.W
        Wii = W.loc[Ikey, Ikey]  # I -> I (< 0)
        Wee = W.loc[Ekey, Ekey]  # E -> E (> 0)
        return np.abs(Wii) * Wee  # > 0

    def get_critical_value(self, pre_key, post_key, W=None):
        ''' 
        Compute critical pre-post value that would allow a stable network

        :param pre_key: pre-synaptic population key
        :param post_key: post-synaptic population key
        :param W (optional): connectivity matrix. If None, use current model connectivity matrix
        :return: critical value
        '''
        if W is None:
            W = self.W
        if len(W) > 2:
            raise ModelError('critical value can only be computed for 2-population models')
        Ikey = list(set(W.index.values) - set(['E']))[0]
        Wi = self.get_net_inhibition(Ikey, W=W)
        We = self.get_net_excitation(Ikey, W=W)
        if pre_key == post_key:
            otherkey = list(set(W.index.values) - set([pre_key]))[0]
            return -Wi / W.loc[otherkey, otherkey]
        else:
            return -We / W.loc[post_key, pre_key]
    
    @classmethod
    def rescale_W(cls, W):
        ''' Rescale connectivity matrix to match reference coupling strength '''
        Wval = W.loc[cls.WREF[0], cls.WREF[1]]
        return W * cls.WREF[2] / Wval
    
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
    
    @property
    def tauvec(self):
        ''' Return time constants as numpy array '''
        return self.tau.values
    
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
    
    @property
    def srel_vec(self):
        ''' Return relative stimulus sensitivity vector as numpy array '''
        return self.srel.values
    
    @property
    def is_srel_uniform(self):
        ''' Return whether all populations have the same relative stimulus sensitivity '''
        return np.isclose(self.srel.max(), self.srel.min())
    
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
        
        # If dictionary or series provided, check that keys do not match model population names
        elif isinstance(params, (dict, pd.Series)):
            for k in params.keys():
                if k in self.keys:
                    raise ModelError(f'gain function parameter key {k} matches a population name')
        
        # If dataframe provided, check that (1) rows match and (2) columns do not match model population names
        elif isinstance(params, pd.DataFrame):
            self.check_keys(params.index.values)
            for c in params.columns:
                if c in self.keys:
                    raise ModelError(f'gain function parameter column {c} matches a population name')
        
        # If params provided, round to 2 decimals
        if isinstance(params, (pd.Series, pd.DataFrame)):
            params = params.round(2)
        
        self._fparams = params
        self.fgain_callable = self.get_fgain_callables(params)
    
    def is_fgain_unique(self):
        ''' Return whether gain function is unique or population-specific '''
        return self.fparams is None or isinstance(self.fparams, (dict, pd.Series))
        # return not isinstance(self.fgain_callable, dict)

    def get_fgain_callable(self, params):
        '''
        Return gain function callable with given parameters

        :param params: dictionary gain function parameters
        :return: gain function callable
        '''
        # return lambda x: self.fgain(x, **params)
        return FGainCallable(self.fgain, params)
    
    def get_fgain_callables(self, params):
        '''
        Return gain function callable(s)

        :param params: gain function parameters
        :return: gain function callable(s)
        '''
        # If no customization parameters provided, return "generic" gain function
        if params is None:
            return self.fgain
        # If unique gain function parameters provided, return single callable
        elif isinstance(params, (dict, pd.Series)):
            return self.get_fgain_callable(params)
        # If population-specific gain function parameters provided, return dictionary of callables
        else:
            return {k: self.get_fgain_callable(params.loc[k, :]) for k in self.keys}
    
    def is_fgain_bounded(self):
        '''
        Assess wither gain function is bounded to 0-1 interval
        '''
        return np.isclose(self.fgain(1e3), 1)
    
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
    
    def create_log_file(self, fpath, pnames):
        ''' 
        Create batch log file if it does not exist.
        
        :param fpath: path to log file
        :param pnames: list of input parameter names
        '''
        if not os.path.isfile(fpath):
            logger.info(f'creating batch log file: "{fpath}"')
            with open(fpath, 'w') as csvfile:
                writer = csv.writer(csvfile, delimiter=self.DELIMITER)
                writer.writerow(['iteration', *pnames, 'cost'])
        else:
            logger.debug(f'existing batch log file: "{fpath}"')
    
    def append_to_log_file(self, fpath, iteration, params, cost):
        ''' Append current batch iteration to log file '''
        logger.debug(f'appending iteration {iteration} to batch log file: "{fpath}"')
        with lockfile.FileLock(fpath):
            with open(fpath, 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter=self.DELIMITER)
                writer.writerow([iteration, *params, cost])

    @classmethod
    def plot_connectivity_matrix(cls, W, Werr=None, norm=False, ax=None, cbar=True, height=2.5, 
                                 vmin=None, vmax=None, title=None, colwrap=4, 
                                 agg=False, clabel=None):
        '''
        Plot connectivity matrix(ces)

        :param W: connectivity matrix(ces) to plot.
        :param Werr (optional): error matrix to use for +/- annotations
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
                # plot mean matrix with +/- std annotations
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
                        cbar=True
                    )
                    fig.subplots_adjust(wspace=0.5)
                    return fig

                # Otherwise, plot each matrix on separate axis
                else:
                    groups = W.groupby(gby)
                    ninputs = len(groups)
                    if ax is not None:
                        axes = as_iterable(ax)
                        if len(axes) != ninputs:
                            raise ModelError(f'number of axes ({len(axes)}) does not correspond to number of connectivity matrices ({ninputs})')
                        fig = axes[0].get_figure()
                    else:
                        nrows, ncols = ninputs // colwrap + 1, colwrap
                        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * height, nrows * height))
                        if nrows > 1:
                            fig.subplots_adjust(hspace=1)
                        axes = axes.flatten()
                        suptitle = 'connectivity matrices'
                        if norm:
                            suptitle = f'normalized {suptitle}'
                        fig.suptitle(suptitle, fontsize=12, y=1 + .05 / nrows)
                    for ax, (k, w) in zip(axes, groups):
                        cls.plot_connectivity_matrix(w.droplevel(gby), norm=norm, ax=ax, title=k, cbar=False)
                    for ax in axes[ninputs:]:
                        ax.axis('off')
                    return fig       

        # Create/retrieve figure and axis
        if ax is None:
            width = height * 1.33 if cbar else height
            fig, ax = plt.subplots(figsize=(width, height))
        else:
            fig = ax.get_figure()

        # Set axis title
        if title is None:
            title = 'connectivity matrix'
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
        
        clabel = 'connection strength'

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
                txt = f'{W.iloc[y, x]:.2g}'
                ishigh = np.abs(W.iloc[y, x]) > 0.5 * Wamax
                color = 'w' if ishigh else 'k'
                dy = 0
                if Werr is not None:
                    errtxt = f'\n±{Werr.iloc[y, x]:.1g}'
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

    def plot_stimulus_sensitivity(self, srel=None, ax=None, title=None):
        '''
        Plot relative stimulus sensitivity per population

        :param srel: relative sensitivities vector, provided as pandas series. If None, use current model sensitivities
        :param ax (optional): axis handle
        :param title (optional): axis title
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
        if srel.index.nlevels > 1:
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
        if self.is_fgain_unique():
            x0max = self.fparams['x0']
        else:
            x0max = self.fparams['x0'].max()
        x = np.linspace(0, 3 * x0max, 100)

        # Determine title if not provided
        if title is None:
            title = 'gain function'
            if not self.is_fgain_unique():
                title = f'{title}s'
        
        # Plot gain function(s)
        if self.is_fgain_unique():
            ax.plot(x, self.fgain_callable(x), lw=2, c='k')
        else:
            for k, fgain in self.fgain_callable.items():
                ax.plot(x, fgain(x), lw=2, c=self.palette.get(k, None), label=k)
            ax.legend(loc='upper left', frameon=False)

        # Adjust layout
        ax.set_title(title)
        fig.tight_layout()

        # Return figure handle
        return fig

    def plot_summary(self, height=3):
        '''
        Plot model summary, i.e. time constants, gain functions and connectivity matrix

        :param height (optional): figure height
        :return: figure handle
        '''
        # Create figure
        ncols = 4
        fig, axes = plt.subplots(1, ncols, figsize=(height * ncols, height))
        # Plot time constants, gain function and connectivity matrix
        self.plot_time_constants(ax=axes[0])
        self.plot_fgain(ax=axes[1])
        self.plot_stimulus_sensitivity(ax=axes[2])
        self.plot_connectivity_matrix(self.W, ax=axes[3])
        # Adjust layout
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
            return pd.DataFrame(d, columns=self.keys, index=r.index)
        
        # Extract connectivity matrix, and take absolute values if signed=False
        W = self.Wmat.copy()
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
        return drive
    
    def compute_gain_output(self, drive):
        ''' 
        Compute gain function output 
        
        :param drive: total input drive vector
        '''
        # If unique gain function, apply it on drive vector
        if self.is_fgain_unique():
            return self.fgain_callable(drive)
        # Otherwise, apply separate gain function on each population drive
        else:
            return np.array([fgain(d) for fgain, d in zip(self.fgain_callable.values(), drive)])
    
    def get_steady_state_objfunc(self, *args, **kwargs):
        '''
        Return objective function whose root is the steady state, i.e.
        the function that solves x = gain(x)

        :return: objective function
        '''        
        return lambda x: self.compute_gain_output(self.compute_drive(x, *args, **kwargs)) - x
    
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

        logger.info(f'finding baseline {bkey} drive required to reach {pkey} = {r:.2f} steady state activity')

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

    def derivatives(self, r, *args, **kwargs):
        '''
        Return activity derivatives

        :param r: activity vector (Hz)
        :return: activity derivatives (Hz/s)
        '''
        # Compute total input drive
        drive = self.compute_drive(r, *args, **kwargs)
        # Compute gain function output
        g = self.compute_gain_output(drive)
        # Subtract leak term, divide by time constants and return
        return (g - r) / self.tauvec
        
    def tderivatives(self, t, r, *args, **kwargs):
        ''' 
        Wrapper around derivatives that:
            - also takes time as first argument
            - checks for potential simulation divergence
        '''
        # If rates reach unresaonable values, throw error
        if np.any(r > self.MAX_RATE):
            raise SimulationError('simulation diverged')
        return self.derivatives(r, *args, **kwargs)

    def simulate(self, tstop=500., r0=None, A=None, tstart=100., tstim=200., tau_stim=None, dt=None, verbose=True):
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
        :return: activity time series
        '''
        # Determine logging level
        flog = logger.info if verbose else logger.debug 

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
                self.tderivatives,
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
                lambda *args: self.tderivatives(*args, x=solver.stim),  # dfunc
                event_params={'stim': float(A)},
                dt=dt
            )
            # Define solver arguments
            solver_args = [r0, events, tstop]

        # Compute solution
        flog(f'{self}: running {tstop} s long simulation with A = {A}')
        t0 = time.perf_counter()
        sol = solver(*solver_args)
        tcomp = time.perf_counter() - t0
        flog(f'simulation completed in {tcomp:.3f} s')

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
            tbounds = self.extract_stim_bounds(x)
            if x.max() == 0:
                tbounds = None
            if plot_stimulus:
                naxes += 1

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
        for k, v in sol.items():
            v.plot(
                ax=ax, 
                c=self.palette.get(k, None),
            )
            if ss is not None:
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

    def run_stim_sweep(self, amps, verbose=True, on_error='abort', **kwargs):
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
            logger.info(f'running stimulation sweep')
        
        # For each stimulus amplitude
        iterable = tqdm(amps) if verbose else amps
        for A in iterable:
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

    def plot_sweep_results(self, data, ax=None, xkey='amplitude', norm=None, hue='population', style=None, style_order=None, col=None, row=None, title=None):
        '''
        Plot results of stimulus amplitude sweep

        :param data: sweep extracted metrics per population, provided as dataframe
        :param ax (optional): axis handle
        :param xkey (optional): x-axis key
        :param norm (optional): dimension(s) along which to normalize results per population before plotting, one of:
            - None: no normalization
            - 'style': normalize per style
            - 'ax': normalize per axis
            - 'grid': normalize across entire grid
        :param hue (optional): extra grouping dimension to use for coloring (default = 'population')
        :param style (optional): extra grouping dimension to use for styling
        :param col (optional): extra grouping dimension to use for columns
        :param row (optional): extra grouping dimension to use for rows
        :param title (optional): axis title
        :return: figure handle
        '''
        if xkey not in data.index.names:
            raise ModelError(f'level {xkey} not found in index')
        
        # Define normalization function
        def normfunc(x):
            return x / x.abs().max(axis=0).replace(0, 1)
            
        # If input is a dataframe with multi-index, run plotting recursively
        if isinstance(data.index, pd.MultiIndex) and len(data.index.names) > 1:
            # Extract extra dimensions
            gby = [k for k in data.index.names if k != xkey]
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
            grid_gby = [gby[i] for i in [col, row] if i is not None]

            # If grid is needed, create it and loop through axes
            if len(grid_gby) > 0:

                # If single axis provided, raise error 
                if ax is not None:
                    raise ModelError('axis provided but axis grid is needed')
                
                # If grid normalization requested, normalize data across entire grid
                if norm == 'grid':
                    data = normfunc(data)
                    ykey = f'normalized {ykey}'
                    norm = None
                
                # Adapt axis title template
                templates = []
                if col is not None:
                    templates.append(f'{col}={{col_name:.2g}}')
                if row is not None:
                    templates.append(f'{row}={{row_name:.2g}}')
                template = ', '.join(templates)
                
                # Create grid
                fg = sns.FacetGrid(
                    data=data.reset_index(), 
                    col=col,
                    row=row,
                    aspect=1,
                    height=2,
                )
                fg.set_titles(template=template)
                fig = fg.figure
                
                # Loop through axes and plot
                for ax, (_, v) in zip(fig.axes, data.groupby(grid_gby)):
                    self.plot_sweep_results(
                        v.droplevel(grid_gby), ax=ax, xkey=xkey, norm=norm, hue=hue,
                        style=style, style_order=style_order)
                return fig

        # Define y-axis label
        ykey = 'activity'

        # If normalization requested, normalize activity vectors for each population
        if style is not None:
            if norm == 'style':
                data = data.groupby(style).apply(normfunc).droplevel(0)
            elif norm == 'ax':
                data = normfunc(data)
        else:
            if norm:
                data = normfunc(data)
        
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
        )

        # Return figure handle
        return fig
   
    def get_coupling_bounds(self, wmax=None):
        '''
        Get exploration bounds for each element of the network connectivity matrix
        of the current model

        :param wmax: max absolute coupling strength across the network
        '''
        # If max strength not provided, use default
        if wmax is None:
            wmax = self.WMAX
        # Otherwise, check that is is a positive int or float
        else:
            if not isinstance(wmax, (int, float)) or wmax <= 0:
                raise ModelError('wmax must be a positive number')
            wmax = float(wmax)

        # Initialize empty bounds matrix
        Wbounds = self.get_empty_W()

        # For each pre-synaptic population
        for key in self.keys:
            # Get cell-type-specific coupling stength bounds
            bounds = (0., wmax) if key == 'E' else (-wmax, 0.)
            # Assign them to corresponding row in bounds matrix
            Wbounds.loc[key, :] = [bounds] * self.size

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

    def evaluate_stim_sweep(self, ref_profiles, sweep_data, norm=False, disparity_cost_factor=0., invalid_cost=np.inf):
        '''
        Evaluate stimulation sweep results by (1) assessing its validity, (2) comparing it to
        reference acivtation profile, and (3) computing cost metric

        :param ref_profiles: reference activation profiles per population, provided as dataframe
        :param sweep_data: stimulus amplitude sweep output dataframe
        :param norm (optional): whether to normalize reference and output activation profiles before comparison
        :param invalid_cost (optional): return value in case of "invalid" sweep output (default: np.inf)
        :param disparity_cost_factor (optional): scaling factor to penalize disparity in activation levels
            (i.e., max ratio between max activation levels) across populations. If None, no disparity penalty is applied.
        :return: evaluated cost
        '''
        if 't' not in sweep_data.index.names:
            raise ModelError('sweep_data must contain a time index')

        # Extract stimulus amplitudes vector from reference activation profiles 
        amps = ref_profiles.index.values

        # If some simulations in sweep failed, return invalidity cost
        out_amps = sweep_data.index.unique('amplitude').values
        if len(out_amps) < len(amps):
            logger.warning('simulation divergence detected')
            return invalid_cost
        
        # Extract stimulus-evoked steady-states from sweep output
        stim_ss = self.extract_response_magnitude(sweep_data, window='stim', metric='ss', verbose=False)
        # If some steady-states could not be extracted (i.e. unstable behavior), return invalidity cost
        if stim_ss.isna().any().any():
            logger.warning('unstable stimulus-evoked steady-states detected')
            return invalid_cost
        
        # Extract final steady-states from sweep output
        final_ss = self.extract_response_magnitude(sweep_data, window='post', metric='ss', verbose=False)
        # If some steady-states could not be extracted (i.e. unstable behavior), return invalidity cost
        if final_ss.isna().any().any():
            logger.warning('unstable final steady-states detected')
            return invalid_cost
        # If some final steady-states did not return to baseline, return invalidity cost
        if (final_ss > 1e-1).any().any():
            logger.warning('non-zero final steady-states detected')
            return invalid_cost

        # Create copies of reference and output activation profiles for comparison
        yref, ypred = ref_profiles.copy(), stim_ss.copy()

        # If specicied, normalize profiles 
        if norm:
            yref = yref / yref.abs().max()
            ypred = ypred / ypred.abs().max()
        
        # Compute errors between profiles
        err = yref - ypred

        # Compute root mean squared errors per population
        rmse = np.sqrt((err**2).mean())

        # Sum root mean squared errors across populations
        cost = rmse.sum()

        # If requested, add penalty for disparity in activity levels across populations
        if disparity_cost_factor > 0:
            # Compute max absolute activation levels across sweep for each population
            maxlevels = stim_ss.abs().max().rename('max level')
            # Compute all pairwise ratios of max activation levels
            pairs = list(itertools.combinations(maxlevels.index, 2))
            ratios = pd.Series(
                [maxlevels.loc[p[0]] / maxlevels.loc[p[1]] for p in pairs],
                index=pairs,
                name='max level ratio'
            )
            # Transform ratios < 1 to reciprocal
            ratios = ratios.where(ratios > 1, other=1 / ratios)
            # Compute max ratio
            maxratio = ratios.max()
            # Add max ratio to cost, with appropriate scaling factor
            cost += maxratio * disparity_cost_factor

        # Return cost
        return cost
    
    def set_coupling_from_vec(self, Wvec):
        '''
        Assign network parameters from 1D vector (useful for optimization algorithms)
        
        :param Wvec: network parameters vector (must be of size n^2, where n is the number of populations in the network)
        '''
        # Make sure input vector matches network dimensions
        if len(Wvec) != self.nW:
            raise ModelError('input vector length does not match network connectivity matrix size') 

        # Assign parameters to network connectivity matrix
        for (prekey, postkey), w in zip(self.Wnames, Wvec):
            self.W.loc[prekey, postkey] = w
    
    def parse_input_vector(self, xvec):
        '''
        Parse input parameters

        :param xvec: vector of input parameters
        :return: parsed connectivity and sensitivity vectors
        '''
        # Compute vector size
        nx = len(xvec)
        # Initialize output vectors to None
        Wvec, srelvec = None, None
        # If vector is of size n (or 1), assign to sensitivity vector
        if nx == self.size or nx == 1:
            srelvec = xvec
        # If vector is of size n^2, assign to connectivity vector
        elif nx == self.nW:
            Wvec = xvec
        # If vector is of size n^2 + n (or n^2 + 1), assign to both 
        # connectivity and sensitivity vectors
        elif (nx == self.nW + self.size) or (nx == self.nW + 1):
            Wvec, srelvec = xvec[:self.nW], xvec[self.nW:]
        # Otherwise, raise error
        else:
            raise ModelError(f'{self}: cannot parse input vector of size {nx}')
        # Return parsed vectors
        return Wvec, srelvec
    
    def parse_optimum_vector(self, xvec):
        '''
        Parse model parameters from optimum output vector

        :param xvec: vector of optimum output parameters
        :return: dictonary of corresponding model parameter objects
        '''
        Wvec, srelvec = self.parse_input_vector(xvec)
        opt = {}
        if Wvec is not None:
            opt['W'] = self.Wvec_to_Wmat(Wvec)
        if srelvec is not None:
            if len(srelvec) == 1:
                srelvec = srelvec[0]
            opt['srel'] = pd.Series(srelvec, index=self.idx, name='stimulus sensitivity')
        return opt
    
    def set_run_and_evaluate(self, xvec, ref_profiles, norm=False, 
                             disparity_cost_factor=0., Wdev_cost_factor=0.,
                             invalid_cost=np.inf, **kwargs):
        '''
        Adjust specific model parameters, run stimulation sweep, 
        evaluate cost, and reset parameters to original values

        :param xvec: vector of input parameters
        :param ref_profiles: reference activation profiles per population, provided as dataframe
        :param norm (optional): whether to normalize reference and output activation profiles before comparison
        :param disparity_cost_factor (optional): scaling factor to penalize disparity in activation levels across populations
        :param Wdev_cost_factor (optional): scaling factor to penalize deviation from reference network connectivity matrix
        :param invalid_cost (optional): return value in case of "invalid" sweep output (default: np.inf)
        :param kwargs: additional keyword arguments to pass to run_stim_sweep
        :return: evaluated cost
        '''        
        # Parse input parameters
        Wvec, srelvec = self.parse_input_vector(xvec)

        # If input coupling weights provided, store reference and assign to model
        if Wvec is not None:
            Wref = self.W.copy()
            self.set_coupling_from_vec(np.array(Wvec))

        # If input stimulus sensitivities provided, store reference and assign to model
        if srelvec is not None:
            if len(srelvec) == 1:
                srelvec = srelvec[0]
            srel_ref = self.srel.copy()
            self.srel = pd.Series(srelvec, index=self.idx)

        # Run stimulation sweep and evaluate cost
        amps = ref_profiles.index.values
        sweep_data = self.run_stim_sweep(amps, verbose=False, **kwargs)
        cost = self.evaluate_stim_sweep(
            ref_profiles, sweep_data, 
            norm=norm, 
            disparity_cost_factor=disparity_cost_factor, 
            invalid_cost=invalid_cost
        )

        # If requested, add cost for relative deviation from reference network connectivity matrix
        if Wdev_cost_factor > 0:
            Wreldiff = (self.W - Wref) / Wref
            cost += Wdev_cost_factor * Wreldiff.abs().sum().sum()

        # Reset model parameters that have been modified
        if Wvec is not None:
            self.W = Wref
        if srelvec is not None:
            self.srel = srel_ref

        # Return cost
        return cost
    
    def Wvec_to_Wmat(self, Wvec):
        '''
        Convert network connectivity matrix vector to matrix

        :param Wvec: network connectivity matrix vector(s) (size n^2, where n is the number of populations in the network)
        :return: n-by-n network connectivity matrix
        '''
        # If input is a series, assume 1 vector per row and convert each one to matrix
        if isinstance(Wvec, pd.Series):
            Wdict = {k: self.Wvec_to_Wmat(v) for k, v in Wvec.items()}
            gname = Wvec.index.name if Wvec.index.name is not None else 'matrix'
            return pd.concat(Wdict, axis=0, names=[gname])
        if len(Wvec) != self.nW:
            raise ModelError('input vector length does not match network connectivity matrix size')
        Wmat = pd.Series(
            Wvec, 
            index=pd.MultiIndex.from_tuples(self.Wnames)
        ).unstack()
        Wmat.index.name = 'pre-synaptic'
        Wmat.columns.name = 'post-synaptic'
        return Wmat
    
    def feval(self, args):
        '''
        Model evaluation function

        :param args: input arguments
        '''
        # Unpack input arguments and log evaluation
        if self.nevals is not None:
            i, x = args
            logstr = f'evaluation {i + 1}/{self.nevals}'
        else:
            pid = os.getpid()
            p = mp.current_process()
            if len(p._identity) > 0:
                i = p._identity[0]
            else:
                i = self.ieval
                self.ieval += 1
            logstr = f'pid {pid}, evaluation {i}'
            x = args 

        # Call evaluation function with input vector
        cost = self.set_run_and_evaluate(
            x, *self.eval_args, **self.eval_kwargs)
        
        # Log ieration and output
        logstr = f'{logstr} - cost = {cost:.3f}'
        logger.info(logstr)

        # Log to file, if path provided
        if self.eval_fpath is not None:
            self.append_to_log_file(self.eval_fpath, i, x, cost)
        
        # Return cost
        return cost

    def setup_eval(self, *args, nevals=None, fpath=None, **kwargs):
        ''' Initialize attributes used in evaluation function '''
        logger.info(f'setting up {self} for evaluation')
        self.eval_args = args
        self.eval_kwargs = kwargs
        self.eval_fpath = fpath
        self.nevals = nevals
        self.ieval = 0
    
    def cleanup_eval(self):
        ''' Clean-up attributes used in evaluation function '''
        logger.info(f'cleaning up {self} after evaluation')
        self.eval_args = []
        self.eval_kwargs = {}
        self.eval_fpath = None
        self.nevals = None
        self.ieval = None

    def __call__(self, args):
        ''' Call model evaluation function '''
        # Call evaluation function
        return self.feval(args)


class ModelOptimizer:

    # Allowed global optimization methods
    GLOBAL_OPT_METHODS = ('diffev', 'annealing', 'shg', 'direct')

    @staticmethod
    def get_exploration_bounds(model, Wbounds, srel_bounds, uniform_srel=False):
        '''
        Assemble list of exploration bounds, given bounds for specific input parameters.

        :param model: model instance
        :param Wbounds: coupling bounds matrix
        :param srel_bounds: stimulus amplitude bounds
        :param uniform_srel (optional): whether to assume uniform stimulus sensitivity
            across populations (default: False)
        :return: series of exploration bounds per parameter
        '''
        # Initialize empty list of exploration bounds
        xbounds = []

        # If coupling bounds matrix provided
        if Wbounds is not None:
            # Check compatibility with model
            model.check_coupling_bounds(Wbounds)
            # Append serialized bounds to exploration bounds
            xbounds.append(Wbounds.stack().rename('bounds'))

        # If stimulus sensitivities bounds provided
        if srel_bounds is not None:
            # If single tuple provided, broadcast to all populations
            if isinstance(srel_bounds, tuple):
                if uniform_srel:
                    srel_bounds = pd.Series(index=['all'], data=[srel_bounds])
                else:
                    srel_bounds = pd.Series(index=model.keys, data=[srel_bounds] * model.size)
            # Otherwise, check that bounds are compatible with model
            else:
                if not isinstance(srel_bounds, pd.Series):
                    raise ModelError('srel_bounds must be a Series')
                if not all(isinstance(x, tuple) for x in srel_bounds.values):
                    raise ModelError('all srel_bounds values must be tuples')
                if not uniform_srel and not srel_bounds.index.equals(model.keys):
                    raise ModelError('srel_bounds indices must match network keys')
                if uniform_srel and len(srel_bounds) != 1:
                    raise ModelError('uniform_srel=True but srel_bounds has more than 1 value')
            
            # Add stimulus sensitivities bounds to exploration
            xbounds.append(srel_bounds)
        
        # If no exploration bounds provided, raise error
        if len(xbounds) == 0:
            raise ModelError('no exploration bounds provided')
        
        # Concatenate exploration bounds, and return
        return pd.concat(xbounds)
    
    @staticmethod
    def extract_optimum(cost):
        '''
        Extract optimum from cost series

        :param cost: exploration results as multi-indexed pandas series
        :return: vector of parameter values yielding optimal cost
        '''
        if isinstance(cost, pd.DataFrame):
            cost = cost['cost']
        if len(cost) == 0:
            raise OptimizationError('no exploration results available')
        elif len(cost) == 1:
            return cost.index[0]
        elif cost.index.names[0] == 'run':
            return cost.groupby('run').agg(lambda x: x.droplevel('run').idxmin())
        else:
            return cost.idxmin()

    @classmethod
    def global_optimization(cls, model, *args, Wbounds=None, srel_bounds=None, uniform_srel=False, method='diffev', mpi=False, **kwargs):
        '''
        Use global optimization algorithm to find set of model parameters that minimizes
        divergence with a reference set of activation profiles.

        :param model: model instance
        :param Wbounds (optional): network connectivity matrix bounds. If None, use default bounds
        :param srel_bounds (optional): stimulus sensitivities bounds. If None, do not explore.
        :param uniform_srel (optional): whether to assume uniform stimulus sensitivity across populations (default: False)
        :param method (optional): optimization algorithm to use, one of:
            - "diffev": differential evolution algorithm
            - "annealing": simulated annealing algorithm
            - "shg": SGH optimization algorithm
            - "direct": DIRECT optimization algorithm
        :param mpi (optional): whether to use multiprocessing (default: False)
        :return: optimized network connectivity matrix
        '''
        if method == 'diffev':
            optfunc = optimize.differential_evolution
        elif method == 'annealing':
            optfunc = optimize.dual_annealing
        elif method == 'shg':
            optfunc = optimize.shgo
        elif method == 'direct':
            optfunc = optimize.direct
        else:
            raise OptimizationError(f'unknown optimization algorithm: {method}')
        
        # Initialize empty dictionary for optimization keyword arguments
        optkwargs = {}

        # If multiprocessing requested
        if mpi:
            # If optimization method is not compatible with multiprocessing, raise error
            if method != 'diffev':
                raise OptimizationError('multiprocessing only supported for differential evolution algorithm')
            # Get the number of available CPUs
            ncpus = get_cpu_count()
            # If more than 1 CPU available, update keyword arguments
            if ncpus > 1:
                # Use as many workers as there are cores
                optkwargs.update({
                    'workers': ncpus,
                    'updating': 'deferred'
                })
        
        # Extract exploration bounds per parameter
        xbounds = cls.get_exploration_bounds(model, Wbounds, srel_bounds, uniform_srel=uniform_srel)
                 
        # Run optimization algorithm
        s = f'running {method} optimization algorithm'
        if 'workers' in optkwargs:
            s = f'{s} with {optkwargs["workers"]} parallel workers'
        logger.info(s)
        model.setup_eval(*args, **kwargs)
        optres = optfunc(model, xbounds.values.tolist(), **optkwargs)
        model.cleanup_eval()

        # If optimization failed, raise error
        if not optres.success:
            raise OptimizationError(f'optimization failed: {optres.message}')

        # Extract solution array
        sol = optres.x

        # Return appropriate model output
        return model.parse_optimum_vector(sol)
    
    @staticmethod
    def get_log_filename(model, ref_profiles, method, Wbounds, srel_bounds, uniform_srel, norm, disparity_cost_factor, Wdev_cost_factor):
        ''' Generate log filename for optimization input arguments '''
        # Gather dictionary of model attribute codes
        model_ids = model.attrcodes()
        
        # Create empty dictionary for optimization IDs
        opt_ids = {}
        
        # Add "targets" ID 
        opt_ids['targets'] = generate_unique_id(ref_profiles)
        
        # Add "Wbounds" ID, casted as tuple of floats (if not None)
        if Wbounds is not None:
            Wbounds = Wbounds.applymap(lambda x: tuple(map(float, x)))
            opt_ids['wbounds'] = generate_unique_id(Wbounds)
            del model_ids['W']
        
        # Add "srel_bounds" ID, casted as tuple of floats (if not None)
        if srel_bounds is not None:
            if isinstance(srel_bounds, tuple):
                srel_bounds = tuple(map(float, srel_bounds))
            else:
                srel_bounds = srel_bounds.apply(lambda x: tuple(map(float, x)))
            opt_ids['srelbounds'] = generate_unique_id(srel_bounds)
            del model_ids['srel']
        
        # Assemble optimization code from IDs
        opt_code = '_'.join([f'{k}{v}' for k, v in opt_ids.items()])
        
        # Add optimization algorithm
        opt_code = f'{opt_code}_{method}'
        
        # Add "norm" suffix if normalization specified
        if norm:
            opt_code = f'{opt_code}_norm'
        
        # Add disparity cost factor, if non-zero
        if disparity_cost_factor > 0:
            opt_code = f'{opt_code}_xdisp{disparity_cost_factor:.2g}'
        
        # Add Wdev cost factor, if non-zero and Wbounds provided
        if Wbounds is not None and Wdev_cost_factor > 0:
            opt_code = f'{opt_code}_xwdev{Wdev_cost_factor:.2g}'
        
        # If exploration assumes uniform sensitivity, add "unisrel" suffix
        if srel_bounds is not None and uniform_srel:
            opt_code = f'{opt_code}_unisrel'

        # If srel is uniformly unitary, remove "srel" attribute code if present
        if 'srel' in model_ids and np.isclose(model.srel, 1.).all():
            del model_ids['srel']

        # Replace "keys" and "fgain" attribute keys by empty strings
        exclude = ['keys', 'fgain']
        keys = [k if k not in exclude else '' for k in model_ids.keys()]
        
        # Assemble into attribute code
        model_code  = '_'.join([f'{k}{v}' for k, v in zip(keys, model_ids.values())])
        
        # Merge model code and optimization code
        code = f'{model_code}_{opt_code}'

        # Add extension and return
        return f'{code}.csv'
    
    @staticmethod
    def load_log_file(fpath):
        '''
        Load exploration results from CSV log file

        :param model: model instance
        :param fpath: path to log file
        '''
        logger.info(f'loading optimization results from {fpath}')
        # Parse log file
        df = pd.read_csv(fpath)
        # Discard 'iteration' column
        del df['iteration']
        # Set all columns other than "cost" to index
        df.set_index(list(df.columns[:-1]), inplace=True)
        # Return only 'cost' column (i.e. discard 'iteration' column)
        return df['cost']

    @classmethod
    def optimize(cls, model, *args, Wbounds=None, srel_bounds=None, uniform_srel=False, method='diffev', norm=False,
                 disparity_cost_factor=0., Wdev_cost_factor=0., logdir=None, force_rerun=False, **kwargs):
        '''
        Find network connectivity matrix that minimizes divergence with a reference
        set of activation profiles.

        :param model: model instance 
        :param Wbounds (optional): network connectivity matrix bounds. If None, use default bounds
        :param srel_bounds (optional): stimulus sensitivities bounds. If None, do not explore.
        :param uniform_srel (optional): whether to assume uniform stimulus sensitivity across populations (default: False)
        :param method (optional): optimization algorithm (default = "diffev")
        :param norm (optional): whether to normalize reference and output activation profiles before comparison
        :param disparity_cost_factor (optional): scaling factor to penalize disparity in activation levels across populations
        :param Wdev_cost_factor (optional): scaling factor to penalize deviation from reference network connectivity matrix
        :param logdir (optional): directory in which to create log file to save exploration results. If None, no log will be saved.
        :return: optimized network connectivity matrix
        '''
        # Check validity of optimization algorithm
        if method not in cls.GLOBAL_OPT_METHODS:
            raise OptimizationError(f'unknown optimization algorithm: {method}')
        
        # If log folder is provided, derive path to log file
        if logdir is not None:
            fname = cls.get_log_filename(
                model, *args, method, Wbounds, srel_bounds, uniform_srel, norm, disparity_cost_factor, Wdev_cost_factor)
            fpath = os.path.join(logdir, fname)
        else:
            fpath = None
        
        # If log file exists,
        if fpath is not None and os.path.isfile(fpath): 
            # If force_rerun flag not set, load optimization results 
            # and return appropriate output
            if not force_rerun:
                cost = cls.load_log_file(fpath)
                xvec = cls.extract_optimum(cost)
                return model.parse_optimum_vector(xvec)

            # Otherwise, re-run optimization and save in different file
            else:
                irerun = 1
                newfpath = fpath
                while os.path.isfile(newfpath):
                    fcode, fext = os.path.splitext(fpath)
                    newfpath = f'{fcode}_rerun{irerun}{fext}'
                    irerun += 1
                fpath = newfpath
                logger.warning(f're-running optimization and saving results in new log file {fpath}')
        
        # Create log file, if path provided
        if fpath is not None:
            xnames = cls.get_exploration_bounds(
                model, Wbounds, srel_bounds, uniform_srel=uniform_srel).index.values
            model.create_log_file(fpath, xnames)
        
        # Run optimization algorithm and return
        return cls.global_optimization(
            model, *args, Wbounds=Wbounds, srel_bounds=srel_bounds, uniform_srel=uniform_srel, 
            fpath=fpath, method=method, norm=norm, disparity_cost_factor=disparity_cost_factor, 
            Wdev_cost_factor=Wdev_cost_factor, **kwargs)

    @classmethod
    def load_optimization_history(cls, model, *args, Wbounds=None, srel_bounds=None, uniform_srel=False, 
                                  method='diffev', norm=False, disparity_cost_factor=0., Wdev_cost_factor=0., 
                                  logdir=None, **kwargs):
        ''' 
        Load optimization history from CSV log file
        
        :param model: model instance 
        :param Wbounds (optional): network connectivity matrix bounds. If None, use default bounds
        :param srel_bounds (optional): stimulus sensitivities bounds. If None, do not explore.
        :param uniform_srel (optional): whether to assume uniform stimulus sensitivity across populations (default: False)
        :param method (optional): optimization algorithm (default = "diffev")
        :param norm (optional): whether to normalize reference and output activation profiles before comparison
        :param disparity_cost_factor (optional): scaling factor to penalize disparity in activation levels across populations
        :param Wdev_cost_factor (optional): scaling factor to penalize deviation from reference network connectivity matrix
        :param logdir (optional): directory in which to create log file to save exploration results. If None, no log will be saved.
        :return: network connectivity matrix optimizaiton history
        '''
        # Check validity of optimization algorithm
        if method not in cls.GLOBAL_OPT_METHODS:
            raise OptimizationError(f'unknown optimization algorithm: {method}')
        
        # If log folder is not provided, raise error
        if logdir is None:
            raise OptimizationError('log directory must be provided')
        
        # Derive path to log file
        fname = cls.get_log_filename(
            model, *args, method, Wbounds, srel_bounds, uniform_srel, norm, disparity_cost_factor, Wdev_cost_factor)
        fpath = os.path.join(logdir, fname)

        # If log file does not exist, raise error
        if not os.path.isfile(fpath):
            raise OptimizationError(f'no optimization log file found at {fpath}')

        # Split into code and extension
        fcode, fext = os.path.splitext(fpath)

        # List all "rerun" files that match the code in the log directory
        rerun_fpaths = glob.glob(f'{fcode}_rerun*{fext}')

        # Concatenate all log files
        fpaths = [fpath, *rerun_fpaths]
        
        # Sort files by creation date
        fpaths = sorted(fpaths, key=os.path.getctime)
        
        # Load optimization history of each file, and return
        costs = []
        for fpath in fpaths:
            cost = cls.load_log_file(fpath)
            cost = cost.to_frame()
            cost['iteration'] = np.arange(len(cost)) + 1
            costs.append(cost)
        if len(costs) == 1:
            return costs[0]
        else:
            return pd.concat(costs, keys=range(len(costs)), names=['run'])
    
    @classmethod
    def has_optimization_converged(cls, cost):
        '''
        Check whether optimization has converged based on cost history

        :param cost: optimization cost history
        :return: whether optimization has converged
        '''
        if isinstance(cost, pd.DataFrame):
            cost = cost['cost']
        if cost.index.names[0] == 'run':
            return cost.groupby('run').agg(
                lambda x: cls.has_optimization_converged(x.droplevel('run')))
        idxmin = cost.idxmin()
        imin = np.where(cost.index == idxmin)[0][0]
        return imin > 0.9 * len(cost)
