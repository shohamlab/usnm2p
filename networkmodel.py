
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import seaborn as sns
from scipy.optimize import fsolve

from solvers import ODESolver, EventDrivenSolver
from logger import logger


class NetworkModel:
    ''' Network model of the visual cortex micro-circuitry '''

    # Cell-type-specific color palette
    palette = {
        'E': 'C0',
        'PV': 'C1',
        'SST': 'r'
    }

    def __init__(self, W, tau, fgain):
        '''
        Initialize the network model

        :param W: network connectivity matrix (2D array or dataframe)
        :param tau: time constants vector
        :param fgain: gain function
        '''
        self.W = W
        self.tau = tau
        self.fgain = fgain
    
    def __repr__(self) -> str:
        s = f'{self.__class__.__name__}('
        if self.has_keys():
            keysstr = '[' + ', '.join(self.keys) + ']'
            s = f'{s}{keysstr}'
        elif self.has_size():
            s = f'{s}{self.size}'
        return f'{s})'
    
    def has_size(self):
        ''' Return whether the model size has been set or not '''
        return hasattr(self, 'size') and self.size is not None
    
    def check_size(self, key, n):
        '''
        Check that the given size parameter matches the current model size

        :param n: size value
        '''
        if n != self.size:
            raise ValueError(f'{key} dimension ({n}) does not match current model size ({self.size})')
    
    def set_size(self, key, n):
        '''
        Set the model size (if not already set) or check that the given size parameter
        matches the current model size

        :param key: size parameter key
        :param n: size value
        '''
        if not self.has_size():
            self.size = n
        else:
            self.check_size(key, n)
    
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
            raise ValueError(f'input keys ({keys}) do not match current model keys ({self.keys})')
    
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

        :param W: n by n matrix (or dataframe) where the (i, j) element 
            is the connection strength from i to j
        '''
        if isinstance(W, pd.DataFrame):
            Wmat = W.values
            Wrows, Wcols = W.index.values, W.columns.values 
            if not np.all(Wrows == Wcols):
                raise ValueError(f'mismatch between matrix rows {Wrows} and columns {Wcols}')
            self.set_keys(Wrows)
        elif isinstance(W, np.ndarray):
            Wmat = W.copy()
        else:
            raise ValueError('connectivity matrix must be either a pandas dataframe or a numpy array')
        if Wmat.ndim != 2:
            raise ValueError('connectivity matrix must be 2D')
        nx, ny = W.shape
        if nx != ny:
            raise ValueError('connectivity matrix must be square')
        self.set_size('connectivity matrix', nx)
        self._W = Wmat
    
    @property
    def Wstr(self):
        ''' Return connectivity matrix as dataframe '''
        if not self.has_keys():
            raise ValueError('model keys must be set to convert connectivity matrix to dataframe')
        return pd.DataFrame(
            self.W, 
            index=pd.Index(self.keys, name='pre'), 
            columns=pd.Index(self.keys, name='post')
        )
    
    @property
    def tau(self):
        return self._tau
    
    @tau.setter
    def tau(self, tau):
        if isinstance(tau, pd.Series):
            self.set_keys(tau.index.values)
            tau = tau.values
        elif isinstance(tau, dict):
            self.set_keys(np.array(tau.keys()))
            tau = np.array(list(tau.values()))
        else:
            tau = np.asarray(tau)
        if tau.ndim != 1:
            raise ValueError('time constants must be 1D')
        self.set_size('time constants', len(tau))
        self._tau = tau
    
    @property
    def taustr(self):
        '''
        Return time constants as pandas Series
        '''
        if not self.has_keys():
            raise ValueError('model keys must be set to convert time constants to series')
        return pd.Series(self.tau, index=self.keys, name='tau (s)')
    
    @property
    def fgain(self):
        return self._fgain

    @fgain.setter
    def fgain(self, fgain):
        if not callable(fgain):
            raise ValueError('gain function must be callable')
        self._fgain = fgain
    
    def is_fgain_bounded(self):
        '''
        Assess wither gain function is bounded to 0-1 interval
        '''
        return np.isclose(self.fgain(1e3), 1)
    
    def plot_fgain(self, ax=None):
        '''
        Plot the gain function

        :param ax (optional): axis handle
        :return: figure handle
        '''
        # Create.retrieve figure and axis
        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 3))
        else:
            fig = ax.get_figure()

        # Set axis layout
        sns.despine(ax=ax)
        ax.set_title('gain function')
        ax.set_xlabel('input')
        ax.set_ylabel('output')

        # Create x values
        x = np.linspace(-5, 15, 100)

        # Plot gain function
        ax.plot(x, self.fgain(x), lw=2, c='k')

        # Adjust layout
        fig.tight_layout()

        # Return figure handle
        return fig
    
    def compute_drive(self, r, s=None):
        '''
        Compute total input drive to gain function

        :param r: activity vector (Hz)
        :param s (optional): external inputs vector
        :return: total input drive vector
        '''
        # Compute total synaptic drive
        drive = np.dot(self.W.T, r)
        # Add external inputs if present
        if s is not None:
            drive += s
        return drive
    
    def compute_gain_output(self, *args, **kwargs):
        ''' Compute gain function output '''
        return self.fgain(self.compute_drive(*args, **kwargs))
    
    def compute_steady_state(self, *args, **kwargs):
        '''
        Compute steady states

        :param s (optional): external inputs vector
        :return: steady state activity vector (Hz)
        '''
        # Initial guess
        p0 = np.ones(self.size)
        # Call fsolve to find the solution of x = gain(x)
        ss = fsolve(
            lambda x: self.compute_gain_output(x, *args, **kwargs) - x, p0)
        # If model has keys, cast output as series
        if self.has_keys():
            ss = pd.Series(data=ss, index=self.keys, name='activity')
        # Return output
        return ss
    
    def compute_nullplane(self, key, *args, **kwargs):
        '''
        Compute null-plane for a given population
        '''
        self.compute_steady_state(*args, **kwargs)
        # Create a grid of activity values
        r = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(r, r)
        # Compute the null-plane

    def derivatives(self, r, *args, **kwargs):
        '''
        Return activity derivatives

        :param r: activity vector (Hz)
        :param s (optional): external inputs vector
        :return: activity derivatives (Hz/s)
        '''
        # Compute gain function output
        g = self.compute_gain_output(r, *args, **kwargs)
        # Subtract leak term, divide by time constants and return
        return (g - r) / self.tau
        
    def tderivatives(self, t, *args, **kwargs):
        ''' Wrapper around derivatives that also takes time as first argument '''
        return self.derivatives(*args, **kwargs)

    def simulate(self, r0, tstop, s=None, tstim=.2, tonset=0.1, dt=None):
        '''
        Simulate the network model

        :param r0: dictionary / pandas Series of initial activity (Hz)
        :param tstop: total simulation time (s)
        :param s: external input amplitude vector
        :param tstim: stimulus duration (s)
        :param tonset: stimulus onset time (s)
        :param dt: simulation time step (s)
        :return: activity time series (Hz)
        '''
        # Check initial conditions validity
        if isinstance(r0, pd.Series):
            r0keys = r0.index.values
        elif isinstance(r0, dict):
            r0keys = np.array(list(r0.keys()))
        self.set_keys(r0keys)
        self.check_size('initial conditions', len(r0))
        
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
            # Check extenral inputs validity
            if isinstance(s, pd.Series):
                self.set_keys(s.index.values)
                s = s.values
            else:
                self.check_size('external inputs', len(s))

            # Define events vector
            events = [
                (0., 0.),
                (tonset, 1.), 
                (tonset + tstim, 0.)
            ]

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
            logger.info(f'{self}: running {tstop} s long simulation')
        t0 = time.perf_counter()
        sol = solver(*solver_args)
        tcomp = time.perf_counter() - t0
        logger.info(f'simulation completed in {tcomp:.3f} s')

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
    
    def plot_timeseries(self, data, ss=None):
        ''' 
        Plot timeseries from simulation results

        :param data: simulation results dataframe
        :param ss (optional): steady-state values to add to activity timeseries
        :return: figure handle
        '''
        # Extract output data types
        dtypes = list(data.columns.levels[0])
        naxes = len(dtypes)

        # Create figure backbone
        fig, axes = plt.subplots(naxes, 1, figsize=(7, 2 * naxes), sharex=True)
        axes = np.atleast_1d(axes)
        sns.despine(fig=fig)

        # Initialize axes iterator
        ax_iter = iter(axes)
        ax = next(ax_iter)
        ax.set_title('simulation results')

        # Plot stimulus if present
        if 'x' in dtypes:
            data = data.copy()
            x = data.pop('x')
            ax.plot(x.index, x.values, c='k')
            ax.set_ylabel('stimulus modulation')
            dx = x.diff()
            istart, iend = np.where(dx > 0)[0][0], np.where(dx < 0)[0][0] - 1
            tstart, tend = x.index.values[istart], x.index.values[iend]
            ax = next(ax_iter)
        else:
            tstart, tend = None, None
        
        # For each type in output data
        for dtype in data.columns.levels[0]:
            # Plot data type time series per population
            ax.set_ylabel(dtype)
            for k, v in data[dtype].items():
                v.plot(ax=ax, label=k, c=self.palette.get(k, None))
                if dtype == 'activity' and ss is not None:
                    ax.axhline(ss.loc[k], ls='--', c=self.palette[k])
            ax.legend(loc='upper right', frameon=False)
            if dtype == 'activity' and self.is_fgain_bounded():
                ax.set_ylim(-0.05, 1.05)
            try:
                ax = next(ax_iter)
            except StopIteration:
                break
        
        # Set x-axis label on last axis
        axes[-1].set_xlabel('time (s)')

        # Highlight stimulus period on all axes, if present
        if tstart is not None:
            for ax in axes:
                ax.axvspan(tstart, tend, fc='k', ec=None, alpha=0.1)

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
                raise ValueError('input axis must be 3D')
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




    