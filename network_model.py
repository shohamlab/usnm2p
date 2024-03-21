# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2024-03-14 17:13:28
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2024-03-17 17:28:16

import time
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

from solvers import ODESolver, EventDrivenSolver
from logger import logger
from utils import expconv, expconv_reciprocal, threshold_linear, as_iterable
from postpro import mylinregress
from batches import get_cpu_count


def generate_unique_id(obj):
    '''
    Generate unique identifier for a given object

    :param obj: object to identify
    :return: unique object identifier
    '''
    # Convert object to a string representation
    s = str(obj)
    # Create a hash object using hashlib
    hash_object = hashlib.md5(s.encode())
    # Return the hexadecimal digest of the hash
    return hash_object.hexdigest()


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

    # Default coupling strength bounds for excitatory and inhibitory connections
    DEFAULT_WBOUNDS = {
        'E': (0, 20),
        'I': (-20, 0)
    }
    
    # CSV delimiter
    DELIMITER = ','

    def __init__(self, W=None, tau=None, fgain=None, fparams=None, b=None):
        '''
        Initialize the network model

        :param W (optional): network connectivity matrix, provided as dataframe. If None, set to 0 for all populations
        :param tau (optional): time constants vector, provided as pandas series. 
            If None, set to 10 ms for all populations
        :param fgain (optional): gain function. If None, use threshold-linear
        :param fparams (optional): gain function parameters, either:
            - a (name: value) dictionary / pandas Series of parameters, if unique
            - a dataframe with parameters as columns and populations as rows, if population-specific
        :param b (optional): baseline inputs vector, provided as pandas series
        '''
        # Extract keys from first non-None input
        for param in (W, tau, fparams, b):
            if param is not None:
                self.set_keys(param.index.values)
                break
        if not self.has_keys():
            raise ModelError('at least one of the following parameters must be provided: W, tau, fparams, b')

        # Get default values for required attributes, if not provided
        if W is None:
            W = pd.DataFrame(
                data=0., 
                index=pd.Index(self.keys, name='pre-synaptic'), 
                columns=pd.Index(self.keys, name='post-synaptic')
            )
        if tau is None:
            tau = pd.Series(0.01, index=self.keys, name='tau (s)')
        if fgain is None:
            fgain = threshold_linear

        # Set attributes
        self.W = W
        self.tau = tau
        self.fgain = fgain
        self.fparams = fparams
        self.b = b

        # Log
        logger.info(f'initialized {self}')
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(' + ', '.join(self.keys) + ')'
    
    def code(self, add_W=False):
        ''' 
        Return model unique identifier code based on its attributes
        
        :param add_W (optional): whether to include the connectivity matrix in the code
        :return: model code
        '''
        # Define attributes to include in the code
        attrs = ['fparams', 'tau', 'b']
        if add_W:
            attrs.append('W')
        
        # Assemble unique identifiers for each set attribute
        attrids = {}
        for k in attrs:
            if getattr(self, k) is not None:
                attrids[k] = generate_unique_id(getattr(self, k))
        
        # Assemble into attribute code
        attrcode  = '_'.join([f'{k}{v}' for k, v in attrids.items()])

        # Extract population and fgain codes
        fgaincode = f'fgain_{self.fgain.__name__}'
        popcode = '-'.join(self.keys)

        # Assemble and return
        return '_'.join([self.__class__.__name__, popcode, fgaincode, attrcode])
    
    def get_log_filename(self, suffix):
        '''
        Generate log file code with current date and time

        :param suffix: suffix to append to the log file code
        :return: log file code
        '''
        s = self.code()
        if suffix:
            s += f'_{suffix}'
        return f'{s}.csv'
    
    def create_log_file(self, fpath):
        ''' Create batch log file if it does not exist. '''
        if not os.path.isfile(fpath):
            logger.info(f'creating batch log file: "{fpath}"')
            with open(fpath, 'w') as csvfile:
                writer = csv.writer(csvfile, delimiter=self.DELIMITER)
                writer.writerow(['iteration', *self.Wnames, 'cost'])
        else:
            logger.debug(f'existing batch log file: "{fpath}"')
    
    def append_to_log_file(self, fpath, iteration, params, cost):
        ''' Append current batch iteration to log file '''
        logger.debug(f'appending iteration {iteration} to batch log file: "{fpath}"')
        with lockfile.FileLock(fpath):
            with open(fpath, 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter=self.DELIMITER)
                writer.writerow([iteration, *params, cost])
    
    @property
    def size(self):
        return len(self.keys)
    
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
        # Check that input is a dataframe with no NaN values
        if not isinstance(W, pd.DataFrame):
            raise ModelError('connectivity matrix must be a pandas dataframe')
        if W.isna().any().any():
            raise ModelError('connectivity matrix cannot contain NaN values')
        
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
    
    @property
    def Wmat(self):
        ''' Return connectivity matrix a2D numpy array '''
        return self.W.values.T
    
    @property
    def Wnames(self):
        ''' Return names of network connectivity matrix elements '''
        return self.W.stack().index.values

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
    
    def process_vector_input(self, v):
        '''
        Process vector input

        :param v: vector input, provided as pandas Series
        :return: processed vector
        '''
        if not isinstance(v, pd.Series):
            raise ModelError(f'input vector must be provided as pandas series')
        self.set_keys(v.index.values)
        return v.values
    
    @property
    def tau(self):
        return self._tau
    
    @tau.setter
    def tau(self, tau):
        self._tau = self.process_vector_input(tau)
    
    def get_param_table(self, name, v):
        '''
        Get table of model parameter across populations

        :param name: parameter name
        :param v: parameter values vector
        :return: pandas series
        '''
        if not self.has_keys():
            raise ModelError('model keys must be set to convert parameter to series')
        return pd.Series(v, index=pd.Index(self.keys, name='population'), name=name)
    
    @property
    def taustr(self):
        '''
        Return time constants as pandas Series
        '''
        return self.get_param_table('tau (s)', self.tau)

    @property
    def b(self):
        return self._b
    
    @b.setter
    def b(self, b):
        if b is not None:
            self._b = self.process_vector_input(b)
        else:
            self._b = None
    
    @property
    def bstr(self):
        '''
        Return baseline inputs as pandas Series
        '''
        if self.b is None:
            return None
        return self.get_param_table('baseline input (?)', self.b)
    
    @property
    def params_table(self):
        ''' Return dataframe with model parameters per population ''' 
        cols = []
        for col in [self.taustr, self.bstr]:
            if col is not None:
                cols.append(col) 
        return pd.concat(cols, axis=1)
    
    @property
    def fgain(self):
        return self._fgain

    @fgain.setter
    def fgain(self, fgain):
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
    def plot_connectivity_matrix(cls, W, norm=False, ax=None, cbar=True, height=2.5, 
                                 vmin=None, vmax=None, title=None, colwrap=4):
        '''
        Plot connectivity matrix(ces)

        :param W: connectivity matrix(ces) to plot.
        :param norm (optional): whether to normalize the matrix before plotting
        :param ax (optional): axis handle
        :param cbar (optional): whether to display colorbar
        :param height (optional): figure height
        :param vmin (optional): minimum value for color scale
        :param vmax (optional): maximum value for color scale
        :param title (optional): axis title
        :return: figure handle
        '''
        # If multiple connectivity matrices provided, plot each on separate axis
        if isinstance(W, dict):
            ninputs = len(W)
            if ax is not None:
                axes = as_iterable(ax)
                if len(axes) != ninputs:
                    raise ModelError(f'number of axes ({len(axes)}) does not correspond to number of connectivity matrices ({ninputs})')
            else:
                nrows, ncols = ninputs // colwrap + 1, colwrap
                fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * height, nrows * height))
                if nrows > 1:
                    fig.subplots_adjust(hspace=1)
                axes = axes.flatten()
                suptitle = 'connectivity matrices'
                if norm:
                    suptitle = f'normalized {suptitle}'
                fig.suptitle(suptitle, fontsize=12, y=1 + .2 / nrows)
            for ax, (k, w) in zip(axes, W.items()):
                cls.plot_connectivity_matrix(w, norm=norm, ax=ax, title=k, cbar=False)
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
        
        # If normalization requested, normalize matrix
        if norm:
            W = W / W.abs().max().max()

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
            cmap='coolwarm', 
            annot=True,
            fmt='.2g',
            cbar=cbar,
            cbar_kws={'label': 'connection strength'} if cbar else None,
        )

        # Set x label on top and remove ticks
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.tick_params(axis='both', which='both', top=False, left=False)

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
            tau = self.taustr
        
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
            tau.values * 1e3, 
            color=[self.palette.get(k, None) for k in tau.index])

        # Set y-axis label and adjust layout
        ax.set_ylabel('time constant (ms)')
        # fig.tight_layout()

        # Return figure handle
        return fig
    
    def plot_fgain(self, ax=None, suffix=None):
        '''
        Plot the gain function(s)

        :param ax (optional): axis handle
        :param suffix (optional): suffix to append to the axis title
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

        # Create x values
        x = np.linspace(0, 100, 100)

        # Plot gain function(s)
        title = 'gain function'
        if self.is_fgain_unique():
            ax.plot(x, self.fgain_callable(x), lw=2, c='k')
        else:
            title = f'{title}s'
            for k, fgain in self.fgain_callable.items():
                ax.plot(x, fgain(x), lw=2, c=self.palette.get(k, None), label=k)
            ax.legend(loc='upper left', frameon=False)

        # Adjust layout
        if suffix is not None:
            title = f'{title} ({suffix})'
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
        fig, axes = plt.subplots(1, 3, figsize=(height * 3, height))
        # Plot time constants, gain function and connectivity matrix
        self.plot_time_constants(ax=axes[0])
        self.plot_fgain(ax=axes[1])
        self.plot_connectivity_matrix(self.W, ax=axes[2])
        # Adjust layout
        fig.tight_layout()
        fig.suptitle(self, fontsize=12, y=1.05)
        # Return figure handle
        return fig
    
    def compute_drive(self, r, s=None):
        '''
        Compute total input drive to gain function

        :param r: activity vector (Hz)
        :param s (optional): stimulus inputs vector
        :return: total input drive vector
        '''
        # If input activity is a dataframe, apply function on each row and 
        # return as dataframe
        if isinstance(r, pd.DataFrame):
            self.check_keys(r.columns)
            d = np.array([self.compute_drive(rr) for rr in r.values])
            if s is not None:
                d += s
            return pd.DataFrame(d, columns=self.keys, index=r.index)
        
        # Compute total synaptic drive
        drive = np.dot(self.Wmat, r)
        # Add baseline inputs if present
        if self.b is not None:
            drive += self.b
        # Add stimulus inputs if present
        if s is not None:
            drive += s
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
        return pd.Series(data=ss, index=self.keys, name='activity')

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
        return (g - r) / self.tau
        
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

    def simulate(self, tstop=.5, r0=None, s=None, tstart=0.1, tstim=.2, tau_stim=None, dt=None, verbose=True):
        '''
        Simulate the network model

        :param tstop: total simulation time (s)
        :param r0: initial activity vector, provided as pandas series
        :param s: external input amplitude vector, provided as pandas series
        :param tstart: stimulus start time (s)
        :param tstim: stimulus duration (s)
        :param tau_stim (optional): stimulus rise/decay time constant (s)
        :param dt: simulation time step (s)
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
        if s is None:
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
            # Check external inputs validity
            if not isinstance(s, pd.Series):
                raise ModelError('external inputs must be provided as pandas series')
            self.check_keys(s.index.values)
            s = s.values
            
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
                lambda x: setattr(solver, 's', s * x),  # eventfunc
                self.keys, 
                lambda *args: self.tderivatives(*args, s=solver.s),  # dfunc
                event_params={'s': s.copy()},
                dt=dt
            )
            # Define solver arguments
            solver_args = [r0, events, tstop]

        # Compute solution
        flog(f'{self}: running {tstop} s long simulation')
        t0 = time.perf_counter()
        sol = solver(*solver_args)
        tcomp = time.perf_counter() - t0
        flog(f'simulation completed in {tcomp:.3f} s')

        # If solution diverged, raise error
        if sol[self.keys].max().max() > self.MAX_RATE:
            raise SimulationError('simulation diverged')

        # Extract activity time series 
        data = sol[self.keys]
        data.columns = pd.MultiIndex.from_product([['activity'], self.keys])

        # If external input modulation vector is present, 
        # add external input vectors to solution dataframe
        if 'x' in sol:
            # Construct external input modulation time series with "all" key
            xf = sol['x'].to_frame()
            xf.columns = pd.MultiIndex.from_product([['x'], ['all']])

            # Compute external input vectors for each population
            ext_inputs = pd.DataFrame(
                np.outer(sol['x'].values, s), 
                columns=self.keys, 
                index=data.index)
            ext_inputs.columns = pd.MultiIndex.from_product([['external inputs'], self.keys])

            # Concatenate in global solution dataframe
            data = pd.concat([xf, ext_inputs, data], axis=1)

        # Return output dataframe
        return data

    @property
    def no_result(self):
        '''
        Return empty results dataframe
        '''
        cols = [
            pd.MultiIndex.from_product([['x'], ['all']]),
            pd.MultiIndex.from_product([['external inputs'], self.keys]),
            pd.MultiIndex.from_product([['activity'], self.keys])
        ]
        df = pd.concat([pd.DataFrame(columns=c) for c in cols], axis=1)
        df.index.name = 't'
        return df

    @staticmethod
    def extract_stim_bounds(x):
        '''
        Extract stimulus bounds from stimulus modulation vector

        :param x: stimulus modulation pandas series, indexed by time (s)
        :return: stimulus start and end times (s)
        '''
        dx = x.diff()
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

        # Predict relative linear variation over interval
        t = data.index.values
        regres = mylinregress(t, data.values, robust=True)
        tbounds = np.array([t[0], t[-1]])
        ybounds = regres.loc['slope'] * tbounds + regres.loc['intercept']
        yreldiff = np.diff(ybounds)[0] / np.mean(ybounds)

        # Return whether linear variations are negligible
        return np.abs(yreldiff) < 1e-2
        # return regres['pval'] >= 0.05

    def extract_steady_state(self, data, kind='stim', verbose=True):
        '''
        Extract steady-state stimulus-evoked activity from simulation results

        :param data: simulation results dataframe
        :param kind (optional): interval type during which to extract steady-state activity, one of:
            - "pre": pre-stimulus interval
            - "stim": stimulus interval
            - "post": post-stimulus interval
        :return: steady-state activity series
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
                    ss[k] = self.extract_steady_state(v.droplevel(gby), kind=kind, verbose=verbose)
                # If metric error, log warning and set metric to NaN
                except MetricError as e:
                    if verbose:
                        logger.warning(f'{gby} = {k:.2g}: {e}')
                    ss[k] = pd.Series(index=self.keys, name='activity')
            return pd.concat(ss, axis=0, names=as_iterable(gby)).unstack(level='population')
        
        # Extract output data types
        dtypes = list(data.columns.levels[0])

        # If not stimulus modulation present, raise error
        if 'x' not in dtypes:
            raise ModelError('no stimulus modulation present in simulation results')

        # Extract stimulus time bounds from stimulus modulation vector
        tstim = self.extract_stim_bounds(data[('x', 'all')])

        # Derive time bounds of interest based on interval type
        if kind == 'pre':
            tbounds = (data.index.values[0], tstim[0])
        elif kind == 'stim':
            tbounds = tstim
        elif kind == 'post':
            tbounds = (tstim[1], data.index.values[-1])
        else:
            raise ModelError(f'unknown interval type: {kind}')

        # Get data slice for second half of interval of interest
        subdata = self.get_secondhalf_slice(data, tbounds)['activity']
        
        # If activity is unstable in that slice, raise error
        if not self.is_stable(subdata):
            raise MetricError(
                f'unstable activity in [{tbounds[0]:.2f}s - {tbounds[1]:.2f}s] time interval')
        
        # Return average activity in that slice 
        mu_act = subdata.mean(axis=0).rename('activity')
        mu_act.index.name = 'population'
        return mu_act
    
    def plot_timeseries(self, data, ss=None, add_synaptic_drive=False):
        ''' 
        Plot timeseries from simulation results

        :param data: simulation results dataframe
        :param ss (optional): steady-state values to add to activity timeseries
        :param add_synaptic_drive (optional): whether to add synaptic input drives to timeseries
        :return: figure handle
        '''
        # Create copy of input data to avoid in-place modification
        data = data.copy()

        # Add synaptic inputs to data, if requested
        if add_synaptic_drive:
            syninputs = self.compute_drive(data['activity'])
            syninputs.columns = pd.MultiIndex.from_product([['synaptic inputs'], syninputs.columns])
            data = pd.concat([data, syninputs], axis=1)

        # Extract output data types
        dtypes = data.columns.get_level_values(0).unique()

        # Extract stimulus bounds, if present
        tbounds = None
        if 'x' in dtypes:
            tbounds = self.extract_stim_bounds(data['x'])
            del data['x']
            dtypes = data.columns.get_level_values(0).unique()
        
        # Create figure backbone
        naxes = len(dtypes)
        fig, axes = plt.subplots(naxes, 1, figsize=(7, 2 * naxes), sharex=True)
        axes = np.atleast_1d(axes)
        sns.despine(fig=fig)        
        axes[0].set_title(f'{self} - simulation results')

        # For each type in output data
        axdict = {}
        for dtype, ax in zip(dtypes, axes):
            # Plot data type time series per population
            ax.set_ylabel(dtype)
            ncols = len(data[dtype].columns)
            for k, v in data[dtype].items():
                v.plot(
                    ax=ax, 
                    c=self.palette.get(k, None) if ncols > 1 else 'k',
                )
                if dtype == 'activity' and ss is not None:
                    ax.axhline(ss.loc[k], ls='--', c=self.palette[k])
            if ncols > 1:
                ax.legend(loc='upper right', frameon=False)
            if dtype == 'activity' and self.is_fgain_bounded():
                ax.set_ylim(-0.05, 1.05)
            axdict[dtype] = ax
        
        # If synaptic inputs axis exists, harmonize y limits with external inputs axis
        if add_synaptic_drive:
            axlist = [axdict['synaptic inputs'], axdict['external inputs']]
            ylims = [ax.get_ylim() for ax in axlist]
            ymin, ymax = min([ymin for ymin, _ in ylims]), max([ymax for _, ymax in ylims])
            for ax in axlist:
                ax.set_ylim(ymin, ymax)
        
        # Set x-axis label on last axis
        axes[-1].set_xlabel('time (s)')

        # Highlight stimulus period on all axes, if present
        if tbounds is not None:
            for ax in axes:
                ax.axvspan(*tbounds, fc='k', ec=None, alpha=0.1)

        # Adjust layout
        fig.tight_layout()

        # Return figure handle
        return fig

    def plot_trajectory(self, data, ax=None, ss=None):
        ''' 
        Plot 3D trajectory from simulation results

        :param data: simulation results dataframe
        :param ax (optional): axis handle
        :param ss (optional): steady-state values to add to activity trajectory
        :return: figure handle
        '''
        # Create/retrieve figure and 3D axis
        if ax is None:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(projection='3d')
        else:
            if not isinstance(ax, Axes3D):
                raise ModelError('input axis must be 3D')
            fig = ax.get_figure()

        # Select acivtity timeseries
        data = data['activity']

        # Plot trajectory
        xk, yk, zk = data.columns
        ax.set_xlabel(xk)
        ax.set_ylabel(yk)
        ax.set_zlabel(zk)
        if self.is_fgain_bounded():
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_zlim(-0.05, 1.05)
        x, y, z = data.values.T
        ax.plot(x, y, z, label='trajectory', c='k')
        ax.scatter(
            [x[0]], [y[0]], [z[0]], c='b', s=30, label='start')
        ax.legend()

        # Add steady-state values if present
        if ss is not None:
            ax.scatter(
                ss[xk], ss[yk], ss[zk], c='r', s=30, label='steady state')
            ax.legend()

        # Return figure handle
        return fig

    def run_stim_sweep(self, srel, amps, verbose=True, on_error='abort', **kwargs):
        '''
        Run sweep of simulations with a range of stimulus amplitudes

        :param srel: relative stimulus amplitude per population, provided as pandas series
        :param amps: stimulus amplitudes vector
        :param verbose (optional): whether to log sweep progress
        :param on_error (optional): behavior on simulation error, one of:
            - "abort": log error and abort sweep
            - "continue": log error and continue sweep
        '''
        # Make sure relative stimulus amplitudes are normalized
        srel = srel / srel.max()

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
                data = self.simulate(s=A * srel, verbose=False, **kwargs)
                sweep_data[A] = data
            # If simulation fails, log error and abort loop if requested
            except SimulationError as err:
                if verbose:
                    logger.error(f'simulation for amplitude {A} failed: {err}')
                if on_error == 'abort':
                    break
        sweep_data = pd.concat(sweep_data, axis=0, names=['amplitude'])
        return sweep_data

    def plot_sweep_results(self, data, ax=None, xkey='amplitude', norm=False, style=None, col=None, row=None, title=None):
        '''
        Plot results of stimulus amplitude sweep

        :param data: sweep extracted metrics per population, provided as dataframe
        :param ax (optional): axis handle
        :param norm (optional): whether to normalize results per population before plotting
        :param style (optional): extra grouping dimension to use for styling
        :param col (optional): extra grouping dimension to use for columns
        :param row (optional): extra grouping dimension to use for rows
        :param title (optional): axis title
        :return: figure handle
        '''
        if xkey not in data.index.names:
            raise ModelError(f'level {xkey} not found in index')
            
        # If input is a dataframe with multi-index, run plotting recursively
        if isinstance(data.index, pd.MultiIndex) and len(data.index.names) > 1:
            # Extract extra dimensions
            gby = [k for k in data.index.names if k != xkey]
            nextra = len(gby)
            # If more than 3 extra dimensions, raise error
            if nextra > 3:
                raise ModelError('cannot plot acitvation profiles with more than 2 extra levels')
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
            if len(grid_gby) > 0:
                if ax is not None:
                    raise ModelError('axis provided but axis grid is needed')
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
                    self.plot_sweep_results(v.droplevel(grid_gby), ax=ax, xkey=xkey, norm=norm, style=style)
                return fig

        # Define y-axis label
        ykey = 'activity'

        # If normalization requested, normalize activity vectors for each population
        if norm:
            normfunc = lambda x: x / x.abs().max(axis=0).replace(0, 1)
            if style is not None:
                data = data.groupby(style).apply(normfunc).droplevel(0)
            else:
                data = normfunc(data)
            ykey = f'normalized {ykey}'
        
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
            hue='population',
            palette=self.palette,
            style=style,
        )

        # Return figure handle
        return fig
    
    def run_W_sweep(self, v, srel, amps, pre_key=None, post_key=None, **kwargs):
        '''
        Run sweep in connectivity matrix parameter(s), and extract steady-state
        activity for each sweep value along a range of stimulus amplitudes.

        :param v: array of sweep values
        :param srel: relative stimulus amplitude per population, provided as pandas series
        :param amps: stimulus amplitudes vector
        :param pre_key (optional): pre-synaptic population key of W element to sweep
        :param post_key (optional): post-synaptic population key of W element to sweep
        :param kwargs: additional arguments to pass to
        '''
        # Construct matrix elements selector
        suffixes, pre_loc, post_loc = [], slice(None), slice(None)
        if pre_key is not None:
            pre_loc = pre_key
            suffixes.append(pre_key)
        else:
            suffixes.append(':')
        if post_key is not None:
            post_loc = post_key
            suffixes.append(post_key)
        else:
            suffixes.append(':')
        key = f'W[{",".join(suffixes)}]'
        
        # Make copies of model connectivity matrix for reference and sweep
        Wref = self.W.copy()
        Wsweep = Wref.copy()

        # Create empty results list
        sweep_rss = []

        # Run sweep
        logger.info(f'running {key} sweep ({len(v)} values)')
        for x in v:
            logger.info(f'setting {key} = {x:.2g}')
            if pre_key is not None and post_key is not None:
                Wsweep.loc[pre_loc, post_loc] = x
            else:
                Wsweep.loc[pre_loc, post_loc] = x * Wref.loc[pre_loc, post_loc]
            try: 
                self.W = Wsweep
            except ModelError as e:
                logger.error(e)
                continue
            sweep_data = self.run_stim_sweep(srel, amps)
            sweep_rss.append(self.extract_steady_state(sweep_data))
        
        # Concatenate sweep results into dataframe
        sweep_rss = pd.concat(sweep_rss, axis=0, keys=v, names=[key])
        
        # Restore original model connectivity matrix
        self.W = Wref

        # Return sweep results
        return sweep_rss
    
    def get_coupling_bounds(self, wbounds=None):
        '''
        Get exploration bounds for each element of the network connectivity matrix
        of the current model

        :param wbounds: dictionary of coupling strength bounds for excitatory and inhibitory connections
        '''
        # If bounds not provided, use defaults
        if wbounds is None:
            wbounds = self.DEFAULT_WBOUNDS
        # Otherwise, check that all required keys are present
        else:
            if not all(k in wbounds for k in ['E', 'I']):
                raise ModelError('wbounds must contain "E" and "I" keys')

        # Initialize empty bounds matrix
        Wbounds = pd.DataFrame(
            index=pd.Index(self.keys, name='pre-synaptic'), 
            columns=pd.Index(self.keys, name='post-synaptic') 
        )

        # For each pre-synaptic population
        for key in self.keys:
            # Get cell-type-specific coupling stength bounds
            bounds = wbounds['E'] if key == 'E' else wbounds['I']
            # Assign them to corresponding row in bounds matrix
            Wbounds.loc[key, :] = [bounds] * self.size

        return Wbounds

    def evaluate_stim_sweep(self, ref_profiles, sweep_data, norm=False, invalid_cost=np.inf):
        '''
        Evaluate stimulation sweep results by (1) assessing its validity, (2) comparing it to
        reference acivtation profile, and (3) computing cost metric

        :param ref_profiles: reference activation profiles per population, provided as dataframe
        :param sweep_data: stimulus amplitude sweep output dataframe
        :param norm (optional): whether to normalize reference and output activation profiles before comparison
        :param invalid_cost (optional): return value in case of "invalid" sweep output (default: np.inf)
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
        stim_ss = self.extract_steady_state(sweep_data, kind='stim', verbose=False)
        # If some steady-states could not be extracted (i.e. unstable behavior), return invalidity cost
        if stim_ss.isna().any().any():
            logger.warning('unstable stimulus-evoked steady-states detected')
            return invalid_cost
        
        # Extract final steady-states from sweep output
        final_ss = self.extract_steady_state(sweep_data, kind='post', verbose=False)
        # If some steady-states could not be extracted (i.e. unstable behavior), return invalidity cost
        if final_ss.isna().any().any():
            logger.warning('unstable final steady-states detected')
            return invalid_cost
        # If some final steady-states did not return to baseline, return invalidity cost
        if (final_ss > 1e-1).any().any():
            logger.warning('non-zero final steady-states detected')
            return invalid_cost

        # If specicied, normalize reference and output activation profiles 
        if norm:
            ref_profiles = ref_profiles / ref_profiles.abs().max()
            stim_ss = stim_ss / stim_ss.abs().max()
        
        # Compute errors between reference and output activation profiles
        err = ref_profiles - stim_ss

        # Compute root mean squared errors per population
        rmse = np.sqrt((err**2).mean())

        # Return sum of root mean squared errors
        return rmse.sum()
    
    def set_coupling_from_vec(self, x):
        '''
        Assign network parameters from 1D vector (useful for optimization algorithms)
        
        :param x: network parameters vector (must be of size n^2, where n is the number of populations in the network)
        '''
        # Make sure input vector matches network dimensions
        if len(x) != len(self.Wnames):
            raise ModelError('input vector length does not match network connectivity matrix size') 

        # Assign parameters to network connectivity matrix
        for (prekey, postkey), v in zip(self.Wnames, x):
            self.W.loc[prekey, postkey] = v
    
    def set_coupling_run_and_evaluate(self, x, srel, ref_profiles, norm=False, invalid_cost=np.inf, **kwargs):
        '''
        Assign coupling parameters from input vector, run stimulation sweep, 
        evaluate cost, and reset coupling parameters

        :param x: network parameters vector
        :param srel: relative stimulus amplitude per population, provided as pandas series
        :param ref_profiles: reference activation profiles per population, provided as dataframe
        :param norm (optional): whether to normalize reference and output activation profiles before comparison
        :param invalid_cost (optional): return value in case of "invalid" sweep output (default: np.inf)
        :param kwargs: additional keyword arguments to pass to run_stim_sweep
        :return: evaluated cost
        '''
        # Store copy of network connectivity matrix
        Wref = self.W.copy()

        # Assign network parameters from input vector
        self.set_coupling_from_vec(x)

        # Run stimulation sweep and evaluate cost
        amps = ref_profiles.index.values
        sweep_data = self.run_stim_sweep(srel, amps, verbose=False, **kwargs)
        cost = self.evaluate_stim_sweep(ref_profiles, sweep_data, norm=norm, invalid_cost=invalid_cost)

        # Reset network connectivity matrix to reference
        self.W = Wref

        # Return cost
        return cost
    
    def Wvec_to_Wmat(self, Wvec):
        '''
        Convert network connectivity matrix vector to matrix

        :param Wvec: network connectivity matrix vector (size n^2, where n is the number of populations in the network)
        :return: n-by-n network connectivity matrix
        '''
        if len(Wvec) != len(self.Wnames):
            raise ModelError('input vector length does not match network connectivity matrix size')
        Wmat = pd.Series(
            Wvec, 
            index=pd.MultiIndex.from_tuples(self.Wnames)
        ).unstack()
        Wmat.index.name = 'pre-synaptic'
        Wmat.columns.name = 'post-synaptic'
        return Wmat
    
    def load_log_file(self, fpath):
        '''
        Load exploration results from CSV log file
        '''
        logger.info(f'loading optimization results from {fpath}')
        # Infer number of W columns
        nWcols = len(self.Wnames)
        # Parse log file while setting W columns to index
        df = pd.read_csv(fpath, index_col=list(range(1, nWcols + 1)))
        # Return only 'cost' column (i.e. discard 'iteration' column)
        return df['cost']
    
    def feval(self, args):
        '''
        Model evaluation function

        :param args: input arguments
        '''
        # Unpack input arguments and log evaluation
        if self.nevals is not None:
            i, x = args
            logger.info(f'evaluation {i + 1}/{self.nevals}')
        else:
            pid = os.getpid()
            p = mp.current_process()
            if len(p._identity) > 0:
                i = p._identity[0]
            else:
                i = -1
            logger.info(f'pid {pid}, evaluation {i}')
            x = args 

        # Call evaluation function with input vector
        cost = self.set_coupling_run_and_evaluate(
            x, *self.eval_args, **self.eval_kwargs)

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

        # Create log file, if path provided
        if fpath is not None:
            self.create_log_file(fpath)
    
    def cleanup_eval(self):
        ''' Clean-up attributes used in evaluation function '''
        logger.info(f'cleaning up {self} after evaluation')
        self.eval_args = []
        self.eval_kwargs = {}
        self.eval_fpath = None
        self.nevals = None

    def __call__(self, args):
        ''' Call model evaluation function '''
        # Call evaluation function
        return self.feval(args)


class ModelOptimizer:

    # Allowed global optimization methods
    GLOBAL_OPT_METHODS = ['diffev', 'annealing', 'shg', 'direct']

    @staticmethod
    def explore(model, npersweep, *args, Wbounds=None, mpi=False, **kwargs):
        '''
        Explore divergence from reference activation profiles across a wide
        range of network connectivity parameters

        :param model: model instance
        :param npersweep: number of sweep values per parameter
        :param Wbounds (optional): network connectivity matrix bounds. If None, use default bounds
        :param mpi (optional): whether to use multiprocessing (default: False)
        :return: exploration results as multi-indexed pandas series
        '''
        # If no bounds provided, use default bounds
        if Wbounds is None:
            Wbounds = model.get_coupling_bounds()
        
        # Serialize bounds into series, if not already
        if not isinstance(Wbounds, pd.Series):
            Wbounds = Wbounds.stack().rename('bounds')
        
        # Define exploration values 
        logger.info('deriving exploration values')
        if npersweep > 1:
            Wexplore = (Wbounds
                .apply(lambda x: np.linspace(*x, npersweep).tolist())
                .rename('parameters')
            )
        else:
            Wexplore = (Wbounds
                .apply(lambda x: [np.mean(x)])
                .rename('parameters')
            )

        # Generate multi-index with all combinations of exploration values
        logger.info('assembling exploration queue')
        mux = pd.MultiIndex.from_product(Wexplore.values, names=model.Wnames)
        nevals = len(mux)

        # Get number of workers
        if mpi:
            # If multiprocessing requested, get minimum of number of available CPUs and number of evaluations
            nworkers = min(get_cpu_count(), nevals)
        else:
            # Otherwise, number of workers is 1
            nworkers = 1

        # Run exploration batch
        s = f'running {nevals} evaluations exploration'
        if nworkers > 1:
            s = f'{s} with {nworkers} parallel workers'
        logger.info(s)
        model.setup_eval(*args, invalid_cost=np.nan, nevals=nevals, **kwargs)
        if nworkers > 1:
            with mp.Pool(processes=nworkers) as pool:
                cost = pool.map(model, enumerate(mux))
        else:
            cost = list(map(model, enumerate(mux)))
        model.cleanup_eval()

        # Format output as multi-indexed series
        cost = pd.Series(cost, index=mux, name='cost')

        # Return
        return cost
    
    @staticmethod
    def extract_optimum(cost):
        '''
        Extract optimum from cost series

        :param cost: exploration results as multi-indexed pandas series
        :return: vector of parameter values yielding optimal cost
        '''
        if len(cost) == 0:
            raise OptimizationError('no exploration results available')
        elif len(cost) == 1:
            return cost.index[0]
        else:
            return cost.idxmin()

    @classmethod
    def brute_force(cls, model, *args, **kwargs):
        '''
        Optimize network connectivity matrix using brute-force algorithm

        :param model: model instance
        :return: optimal network connectivity matrix
        '''
        cost = cls.explore(model, *args, **kwargs)
        Wvec = cls.extract_optimum(cost)
        return model.Wvec_to_Wmat(Wvec)

    @classmethod
    def global_optimization(cls, model, *args, Wbounds=None, kind='diffev', mpi=False, **kwargs):
        '''
        Use global optimization algorithm to find network connectivity matrix that minimizes
        divergence with a reference set of activation profiles.

        :param model: model instance
        :param Wbounds (optional): network connectivity matrix bounds. If None, use default bounds
        :param kind (optional): optimization algorithm to use, one of:
            - "diffev": differential evolution algorithm
            - "annealing": simulated annealing algorithm
            - "shg": SGH optimization algorithm
            - "direct": DIRECT optimization algorithm
        :param mpi (optional): whether to use multiprocessing (default: False)
        :return: optimized network connectivity matrix
        '''
        # Check optimization algorithm validity and extract optimization function
        if kind not in cls.GLOBAL_OPT_METHODS:
            raise OptimizationError(f'unknown optimization algorithm: {kind}')
        if kind == 'diffev':
            optfunc = optimize.differential_evolution
        elif kind == 'annealing':
            optfunc = optimize.dual_annealing
        elif kind == 'shg':
            optfunc = optimize.shgo
        elif kind == 'direct':
            optfunc = optimize.direct
        else:
            raise OptimizationError(f'unknown optimization algorithm: {kind}')
        
        # Initialize empty dictionary for optimization keyword arguments
        optkwargs = {}

        # If multiprocessing requested
        if mpi:
            # If optimization method is not compatible with multiprocessing, raise error
            if kind != 'diffev':
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
        
        # If no bounds provided, use default bounds
        if Wbounds is None:
            Wbounds = model.get_coupling_bounds()
        
        # Serialize bounds into series, if not already
        if not isinstance(Wbounds, pd.Series):
            Wbounds = Wbounds.stack().rename('bounds')
                
        # Run optimization algorithm
        s = f'running {kind} optimization algorithm'
        if 'workers' in optkwargs:
            s = f'{s} with {optkwargs["workers"]} parallel workers'
        logger.info(s)
        model.setup_eval(*args, **kwargs) 
        optres = optfunc(model, Wbounds.values.tolist(), **optkwargs)
        model.cleanup_eval()

        # If optimization failed, raise error
        if not optres.success:
            raise OptimizationError(f'optimization failed: {optres.message}')

        # Extract solution array
        sol = optres.x

        # Re-assemble into connectivity matrix and return
        return model.Wvec_to_Wmat(sol)

    @classmethod
    def optimize(cls, model, *args, kind='diffev', npersweep=5, norm=False, logdir=None, **kwargs):
        '''
        Find network connectivity matrix that minimizes divergence with a reference
        set of activation profiles.

        :param model: model instance 
        :param kind (optional): optimization algorithm (default = "diffev")
        :param npersweep (optional): number of sweep values per parameter for brute-force algorithm (default: 5)
        :param norm (optional): whether to normalize reference and output activation profiles before comparison
        :param logdir (optional): directory in which to create log file to save exploration results. If None, no log will be saved.
        :return: optimized network connectivity matrix
        '''
        # Check validity of optimization algorithm
        if kind not in ['brute', *cls.GLOBAL_OPT_METHODS]:
            raise ModelError(f'unknown optimization algorithm: {kind}')
        
        # If log folder is provided, create log file
        if logdir is not None:
            opt_ids = {k: generate_unique_id(arg) for k, arg in zip(['srel', 'ref_profiles'], args)}
            opt_code = '_'.join([f'{k}{v}' for k, v in opt_ids.items()])
            suffix = f'optimize_{opt_code}_{kind}'
            if kind == 'brute':
                suffix = f'{suffix}_{npersweep}persweep'
            if norm:
                suffix = f'{suffix}_norm'
            fname = model.get_log_filename(suffix)
            fpath = os.path.join(logdir, fname)
        else:
            fpath = None       
        
        # If log file exists, load optimization results and return
        if fpath is not None and os.path.isfile(fpath): 
            cost = model.load_log_file(fpath)
            Wvec = cls.extract_optimum(cost)
            return model.Wvec_to_Wmat(Wvec)
        
        # Run optimization algorithm and return
        if kind == 'brute':
            return cls.brute_force(model, npersweep, *args, fpath=fpath, norm=norm, **kwargs)
        else:
            return cls.global_optimization(model, *args, fpath=fpath, kind=kind, norm=norm, **kwargs)
