
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import seaborn as sns
from scipy import optimize

from solvers import ODESolver, EventDrivenSolver
from logger import logger
from utils import expconv, expconv_reciprocal


class NetworkModel:
    ''' Network model of the visual cortex micro-circuitry '''

    # Cell-type-specific color palette
    palette = {
        'E': 'C0',
        'PV': 'C1',
        'SST': 'r'
    }

    def __init__(self, W, tau, fgain, fgain_params=None, b=None):
        '''
        Initialize the network model

        :param W: network connectivity matrix, provided as dataframe
        :param tau: time constants vector, provided as pandas eries
        :param fgain: gain function
        :param fgain_params (optional): gain function parameters, either:
            - a (name: value) dictionary / pandas Series of parameters, if unique
            - a dataframe with parameters as columns and populations as rows, if population-specific
        :param b (optional): baseline inputs vector, provided as pandas series
        '''
        self.W = W
        self.tau = tau
        self.fgain = fgain
        self.fgain_params = fgain_params
        self.b = b
    
    def __repr__(self) -> str:
        s = f'{self.__class__.__name__}('
        if self.has_keys():
            keysstr = '[' + ', '.join(self.keys) + ']'
            s = f'{s}{keysstr}'
        elif self.has_size():
            s = f'{s}{self.size}'
        return f'{s})'
    
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

        :param W: 2D square dataframe where rows and columns represent pre-synaptic
         and post-synaptic elements, respectively.
        '''
        if not isinstance(W, pd.DataFrame):
            raise ValueError('connectivity matrix must be a pandas dataframe')
        Wmat = W.values
        Wrows, Wcols = W.index.values, W.columns.values 
        if not np.all(Wrows == Wcols):
            raise ValueError(f'mismatch between matrix rows {Wrows} and columns {Wcols}')
        self.set_keys(Wrows)
        nx, ny = W.shape
        if nx != ny:
            raise ValueError('connectivity matrix must be square')
        self._W = Wmat
    
    @property
    def Wtable(self):
        ''' Return connectivity matrix as dataframe table '''
        if not self.has_keys():
            raise ValueError('model keys must be set to convert connectivity matrix to dataframe')
        return pd.DataFrame(
            self.W, 
            index=pd.Index(self.keys, name='pre-synaptic'), 
            columns=pd.Index(self.keys, name='post-synaptic')
        )

    def get_net_inhibition(self, Ikey, Ekey='E'):
        ''' 
        Compute strength of the net inhibition between and E and I populations, 
        as the product of (E to I) and (I to E) coupling strengths

        :param Ikey: inhibitory population key
        :param Ekey: excitatory population key (default: 'E')
        :return: net inhibition strength
        '''
        Wei = self.Wtable.loc[Ekey, Ikey]  # E -> I (> 0)
        Wie = self.Wtable.loc[Ikey, Ekey]  # I -> E (< 0) 
        return Wei * np.abs(Wie)  # < 0
    
    def get_net_excitation(self, Ikey, Ekey='E'):
        ''' 
        Compute strength of the net excitation between E and I populations, 
        as the product of (E to E) and (I to I) coupling strengths

        :param Ikey: inhibitory population key
        :param Ekey: excitatory population key (default: 'E')
        :return: net excitation strength
        '''
        Wii = self.Wtable.loc[Ikey, Ikey]  # I -> I (< 0)
        Wee = self.Wtable.loc[Ekey, Ekey]  # E -> E (> 0)
        return np.abs(Wii) * Wee  # > 0
    
    def get_EI_balance(self, Ikey, Ekey='E'):
        '''
        Return the E/I balance between E and I populations 
        based on the connectivity matrix.
        
        :param Ikey: inhibitory population key
        :param Ekey: excitatory population key (default: 'E')
        :return: E/I balance
        '''
        # Compute net inhibition and excitation
        Wi = self.get_net_inhibition(Ikey, Ekey=Ekey)
        We = self.get_net_excitation(Ikey, Ekey=Ekey)

        # Return E/I balance
        return Wi - We
    
    def get_II_balance(self, Ikey1, Ikey2, Ekey='E'):
        '''
        Return the I/I balance between two inhibitory populations
        based on the connectivity matrix.
        
        :param Ikey1: first inhibitory population key
        :param Ikey2: second inhibitory population key
        :param Ekey: excitatory population key (default: 'E')
        :return: I/I balance
        '''
        # Compute net inhibition between E and I1
        Wi1 = self.get_net_inhibition(Ikey1, Ekey=Ekey)
        # Compute net inhibition between E and I2
        Wi2 = self.get_net_inhibition(Ikey2, Ekey=Ekey)
        
        # Return I/I balance
        return Wi1 - Wi2
    
    def process_vector_input(self, v):
        '''
        Process vector input

        :param v: vector input, provided as pandas Series
        :return: processed vector
        '''
        if not isinstance(v, pd.Series):
            raise ValueError(f'input vector must be provided as pandas series')
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
            raise ValueError('model keys must be set to convert parameter to series')
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
            raise ValueError('gain function must be callable')
        self._fgain = fgain
    
    @property
    def fgain_params(self):
        return self._fgain_params
    
    @fgain_params.setter
    def fgain_params(self, params):
        '''
        :param fgain_params (optional): gain function parameters, either:
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
                    raise ValueError(f'gain function parameter key {k} matches a population name')
        
        # If dataframe provided, check that (1) rows match and (2) columns do not match model population names
        elif isinstance(params, pd.DataFrame):
            self.check_keys(params.index.values)
            for c in params.columns:
                if c in self.keys:
                    raise ValueError(f'gain function parameter column {c} matches a population name')
        
        self._fgain_params = params
        self.fgain_callable = self.get_fgain_callables(params)
    
    def is_fgain_unique(self):
        ''' Return whether gain function is unique or population-specific '''
        return not isinstance(self.fgain_callable, dict)

    def get_fgain_callable(self, params):
        '''
        Return gain function callable with given parameters

        :param params: dictionary gain function parameters
        :return: gain function callable
        '''
        return lambda x: self.fgain(x, **params)
    
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
                                 vmin=None, vmax=None, title=None):
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
            if ax is not None:
                raise ValueError('cannot plot multiple connectivity matrices on single axis')
            fig, axes = plt.subplots(1, len(W), figsize=(len(W) * 1.3 * height, height))
            suptitle = 'connectivity matrices'
            if norm:
                suptitle = f'normalized {suptitle}'
            fig.suptitle(suptitle, fontsize=12, y=1.3)
            for ax, (k, w) in zip(axes, W.items()):
                cls.plot_connectivity_matrix(w, norm=norm, ax=ax, title=k, cbar=False)
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
    
    def plot_fgain(self, ax=None):
        '''
        Plot the gain function(s)

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
        x = np.linspace(0, 100, 100)

        # Plot gain function(s)
        if self.is_fgain_unique():
            ax.plot(x, self.fgain_callable(x), lw=2, c='k')
        else:
            for k, fgain in self.fgain_callable.items():
                ax.plot(x, fgain(x), lw=2, c=self.palette.get(k, None), label=k)
            ax.legend(loc='upper left', frameon=False)

        # Adjust layout
        fig.tight_layout()

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
        drive = np.dot(self.W.T, r)
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
        # d = self.compute_drive(np.zeros(self.size), *args, **kwargs)
        # p0 = self.compute_gain_output(d)
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
        
    def tderivatives(self, t, *args, **kwargs):
        ''' Wrapper around derivatives that also takes time as first argument '''
        return self.derivatives(*args, **kwargs)

    def simulate(self, tstop, r0=None, s=None, tstart=0.1, tstim=.2, tau_stim=None, dt=None, verbose=True):
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
                raise ValueError('initial conditions must be provided as pandas series')
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
                raise ValueError('external inputs must be provided as pandas series')
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

        # If solution siverged, raise error
        if sol[self.keys].max().max() > 1e3:
            raise ValueError('simulation diverged')

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

    def extract_stim_bounds(self, x):
        '''
        Extract stimulus bounds from stimulus modulation vector

        :param x: stimulus modulation pandas series, indexed by time (s)
        :return: stimulus start and end times (s)
        '''
        dx = x.diff()
        istart, iend = np.where(dx > 0)[0][0], np.where(dx < 0)[0][0] - 1
        tstart, tend = x.index.values[istart], x.index.values[iend]
        return tstart, tend

    def extract_steady_state(self, data):
        '''
        Extract steady-state stimulus-evoked activity from simulation results

        :param data: simulation results dataframe
        :return: steady-state activity series
        '''
        # Extract output data types
        dtypes = list(data.columns.levels[0])

        # If not stimulus modulation present, raise error
        if 'x' not in dtypes:
            raise ValueError('no stimulus modulation present in simulation results')

        # Extract stimulus time bounds from stimulus modulation vector
        tstart, tend = self.extract_stim_bounds(data[('x', 'all')])

        # Derive closest point to mid-stimulus time
        tmid = (tstart + tend) / 2
        imid = np.argmin(np.abs(data.index.values - tmid))
        tmid = data.index.values[imid]

        # Compute mean activity during second half of stimulus
        ss = data['activity'].loc[tmid:tend].mean()

        # Return steady-state activity
        ss.index.name = 'population'
        ss.name = 'activity'
        return ss
    
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
        naxes = len(dtypes)

        # Create figure backbone
        fig, axes = plt.subplots(naxes, 1, figsize=(7, 2 * naxes), sharex=True)
        axes = np.atleast_1d(axes)
        sns.despine(fig=fig)        
        axes[0].set_title('simulation results')

        # Extract stimulus bounds, if present
        tbounds = None
        if 'x' in dtypes:
            tbounds = self.extract_stim_bounds(data['x'])
        
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
            if dtype == 'x':
                ax.set_ylabel('stimulus modulation')
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




    