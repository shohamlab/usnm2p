# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2025-08-01 15:00:59
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-09-17 14:26:04

''' Model optimization utilities '''

import time
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize
import os

from .logger import logger
from .utils import as_iterable, generate_unique_id, swaplevels
from .batches import get_cpu_count
from .constants import *
from .network_model import NetworkModel


class OptimizationError(Exception):
    ''' Optimization error '''
    pass


class ModelOptimizer:

    # Dicationary of supported global optimization methods
    GLOBAL_OPT_METHODS = {
        'diffev': optimize.differential_evolution,
        'annealing': optimize.dual_annealing,
        'shg': optimize.shgo,
        'direct': optimize.direct,
    }

    # Keys for different cost types
    PRED_COST_KEY = 'prediction'
    CONN_COST_KEY = 'connectivity'
    DISP_COST_KEY = 'disparity'

    # Short keys for different cost types
    SHORT_COST_KEYS = {
        PRED_COST_KEY: 'y',
        CONN_COST_KEY: 'w',
        DISP_COST_KEY: 'd'
    }

    def __init__(self, model, ref_profiles, method=OPT_METHOD, norm=NORM_BEFORE_COMP, disparity_cost_factor=DISPARITY_COST_FACTOR, 
                 Wdev_cost_factor=WDEV_COST_FACTOR, invalid_cost=INVALID_COST):
        ''' 
        Constructor 

        :param model: NetworkModel instance
        :param ref_profiles: reference activation profiles per population, provided as dataframe
        :param method (optional): optimization algorithm to use, one of:
            - "diffev": differential evolution algorithm
            - "annealing": simulated annealing algorithm
            - "shg": SGH optimization algorithm
            - "direct": DIRECT optimization algorithm
            - "BO": Bayesian optimization algorithm
        :param norm: whether to normalize reference and output activation profiles before comparison
        :param disparity_cost_factor: scaling factor to penalize disparity in
            activation levels across populations
        :param Wdev_cost_factor: scaling factor to penalize deviation from
            reference network connectivity matrix
        :param invalid_cost: cost value assigned to "invalid" evaluations
        '''
        self.model = model
        self.ref_profiles = ref_profiles 
        self.opt_method = method
        self.norm = norm
        self.disparity_cost_factor = disparity_cost_factor
        self.Wdev_cost_factor = Wdev_cost_factor
        self.invalid_cost = invalid_cost
        logger.info(f'initialized {self}')
    
    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, model):
        if not isinstance(model, NetworkModel):
            raise OptimizationError(f'invalid model object: {model}')
        self._model = model
        # Keep copy of reference connecitivity matrix
        self.Wref = self._model.W.copy()
    
    @property
    def ref_profiles(self):
        return self._ref_profiles

    @ref_profiles.setter
    def ref_profiles(self, df):
        if not isinstance(df, pd.DataFrame):
            raise OptimizationError('ref_profiles must be provided as dataframe')
        for k in df.columns:
            if k not in self.model.keys:
                raise OptimizationError(f'population {k} not part of {self.model}')
        self._ref_profiles = df

    @property
    def opt_method(self):
        return self._opt_method
    
    @opt_method.setter
    def opt_method(self, method): 
        try:
            self.opt_func = self.GLOBAL_OPT_METHODS[method]
        except KeyError:
            raise OptimizationError(f'unknown optimization algorithm: {method}')
        self._opt_method = method
    
    def __repr__(self):
        pops = [f'{k}{"*" if k not in self.ref_profiles.columns else ""}' for k in self.model.keys]
        plist = [
            '[' + ', '.join(pops) + ']',
            self.opt_method,
        ]
        if self.norm:
            plist.append('norm')
        if self.disparity_cost_factor > 0: 
            plist.append(f'xdisp={self.disparity_cost_factor}')
        if self.Wdev_cost_factor is not None:
            plist.append(f'xwdev={self.Wdev_cost_factor}')
        pstr = ', '.join(plist)
        return f'{self.__class__.__name__}({pstr})'    
    
    @staticmethod
    def create_log_file(fpath, pnames):
        ''' 
        Create batch log file if it does not exist.
        
        :param fpath: path to log file
        :param pnames: list of input parameter names
        '''
        # If file exists, log and return
        if os.path.isfile(fpath):
            logger.debug(f'existing batch log file: "{fpath}"')
        
        # Log 
        logger.info(f'creating batch log file at {fpath}')

        # Assemble column names
        colnames = ['iteration', *pnames, 'cost']

        # Extract file extension
        ext = os.path.splitext(fpath)[1]

        # Create file
        df = pd.DataFrame({k: [-1] if k == 'iteration' else [0.] for k in colnames})
        df.to_hdf(fpath, OPTHISTORY_KEY, mode='w', format='table')
    
    @staticmethod
    def append_to_log_file(fpath, iteration, params, cost):
        '''
        Append current batch iteration to log file

        :param fpath: path to log file
        :param iteration: cost function evaluation number
        :param params: list of cost function input parameters
        :param cost: associated cost outputed by the cost function 
        '''
        # Log
        logger.debug(f'appending iteration {iteration} to history log file: "{fpath}"')

        # Assemble row data
        rowdata = [iteration, *params, cost]

        # Extract file extension
        ext = os.path.splitext(fpath)[1]

        # Append to file 
        colnames = pd.read_hdf(fpath, OPTHISTORY_KEY, stop=0).columns.tolist()
        df = pd.DataFrame([rowdata], columns=colnames)
        df.to_hdf(fpath, OPTHISTORY_KEY, mode='a', format='table', append=True)
    
    @classmethod
    def load_optimization_history(cls, fpath):
        '''
        Load optimization history from log file

        :param fpath: path to log file
        :return: dataframe of optimization history
        '''
        # Log
        logger.info(f'loading optimization history from {fpath}')

        # Load history from HDF5 file
        return (
            pd.read_hdf(fpath, OPTHISTORY_KEY)
            .iloc[1:]  # Remove dummy first row necessary for h5
            .set_index('iteration')
        )
    
    @classmethod
    def load_optimization_results(cls, fpath, verbose=True):
        '''
        Load optimization results from log file

        :param fpath: path to log file
        :return: 2-tuple with:
            - OptimizeResult instance
            - pandas Series of optimal parameters
        '''
        # Log
        logger.info(f'loading optimization results from {fpath}')

        # Set optimization result to None
        optres = None
        colnames = None

        # Load optimization results from file
        optres = pd.read_hdf(fpath, OPTRES_KEY)
        
        # Create OptimizeResult instance based on these results
        optres = optimize.OptimizeResult(**optres)
        if verbose:
            logger.info(f'optimization result:\n{optres}')

        # Assemble series of optimal parameters 
        colnames = (
            pd.read_hdf(fpath, OPTHISTORY_KEY, stop=0)
            .columns
            .tolist()
            [1:-1]
        )
        optparams = pd.Series(optres.x, index=colnames, name='optimum')

        # Return outputs
        return optres, optparams

    def evaluate_sweep(self, sweep_data, verbose=True):
        '''
        Evaluate model sweep results by (1) assessing its validity, (2) comparing it to
        reference acivtation profiles, and (3) computing cost metrics

        :param sweep_data: stimulus amplitude sweep output dataframe
        :return: dictionary of evaluated costs
        '''
        logwarnfunc = logger.warning if verbose else logger.debug
        if 't' not in sweep_data.index.names:
            raise OptimizationError('sweep_data must contain a time index')

        # Extract stimulus amplitudes vector from reference activation profiles 
        amps = self.ref_profiles.index.values

        # If some simulations in sweep failed, return invalidity cost
        out_amps = sweep_data.index.unique('amplitude').values
        if len(out_amps) < len(amps):
            logwarnfunc('simulation divergence detected')
            return {'divergent sim': self.invalid_cost}
        
        # Extract stimulus-evoked steady-states from sweep output
        stim_ss = self.model.extract_response_magnitude(
            sweep_data, window='stim', metric='ss', verbose=False)
        # If some steady-states could not be extracted (i.e. unstable behavior), return invalidity cost
        if stim_ss.isna().any().any():
            logwarnfunc('unstable stimulus-evoked steady-states detected')
            return {'unstable evoked ss': self.invalid_cost}
        
        # Extract final steady-states from sweep output
        final_ss = self.model.extract_response_magnitude(
            sweep_data, window='post', metric='ss', verbose=False)
        # If some steady-states could not be extracted (i.e. unstable behavior), return invalidity cost
        if final_ss.isna().any().any():
            logwarnfunc('unstable final steady-states detected')
            return {'unstable final ss': self.invalid_cost}

        # If some final steady-states did not return to baseline, return invalidity cost
        if (final_ss > 1e-1).any().any():
            logwarnfunc('non-zero final steady-states detected')
            return {'non-zero final ss': self.invalid_cost}
        
        # Initialize cost dictionary
        costs = {}

        # Create copies of reference and output activation profiles for comparison
        yref, ypred = self.ref_profiles.copy(), stim_ss.copy()

        # If specicied, normalize profiles 
        if self.norm:
            yref = self.model.normalize_profiles(yref, verbose=verbose)
            ypred = self.model.normalize_profiles(ypred, verbose=verbose)
        
        # Compute errors between profiles
        err = yref - ypred

        # Compute root mean squared errors per population
        rmse = np.sqrt((err**2).mean())

        # Compute average root mean squared errors across populations
        # (multiplied to ensure continuity with previous code where it was the sum, 
        # but average is better because the cost does not scale with model dimensions)
        costs[self.PRED_COST_KEY] = rmse.mean() * 3

        # If requested, add penalty for disparity in activity levels across populations
        if self.disparity_cost_factor > 0:
            # Compute max absolute activation levels across sweep for each population
            maxlevels = stim_ss.abs().max().rename('max level')

            # Identify populations with max activity above defined threshold 
            active_maxlevels = maxlevels[maxlevels > MIN_ACTIVITY_LEVEL]

            # If more than 1 active population
            if len(active_maxlevels) > 1:
                # Compute max/min ratio of max activation levels across active populations
                max_act_ratio = active_maxlevels.max() / active_maxlevels.min()
                # Convert to cost with appropriate scaling factor
                costs[self.DISP_COST_KEY] = max_act_ratio * self.disparity_cost_factor

            # Otherwise, set disparity cost to 0
            else:
                costs[self.DISP_COST_KEY] = 0.

            # # If one population has zero max activation, set infinity cost
            # if maxlevels.min() == 0.:
            #     costs[self.DISP_COST_KEY] = np.inf
            # # Otherwise, compute max/min ratio of max activation levels and 
            # # convert to cost with appropriate scaling factor
            # else:
            #     maxratio = maxlevels.max() / maxlevels.min()
            #     costs[self.DISP_COST_KEY] = maxratio * self.disparity_cost_factor

        # Return cost dictionary
        return costs

    @staticmethod
    def get_relative_change(x, x0):
        '''
        Compute relative change between a value and its referenc
        
        :param x: value object (scalar, array, dataframe, ...)
        :param x0: reference value object (same type ad value object) 
        :return: relative change object
        '''
        return (x - x0) / x0
    
    def compute_connectivity_cost(self, W, Wref):
        ''' 
        Compute cost associated with deviation from reference connectivity matrix

        :param W: current connectivity matrix
        :param Wref: reference connectivity matrix
        '''
        # Compute relative change in each coupling weight w.r.t. reference
        Wreldiff = self.get_relative_change(W, Wref)
        # Return mean of absolute values of relative changes, scaled by cost factor 
        return self.Wdev_cost_factor * Wreldiff.abs().mean().mean()

    def set_run_and_evaluate(self, xvec, verbose=True, **kwargs):
        '''
        Adjust specific model parameters, run model stimulation sweep, 
        evaluate costs, and reset parameters to original values

        :param xvec: vector of input parameters
        :param kwargs: additional keyword arguments to pass to model.run_sweep method
        :return: sum of evaluated costs
        '''
        # Parse input parameters
        W, srel = self.model.parse_input_vector(xvec)

        # If input connectivity matrix provided, store reference and assign to model
        if W is not None:
            Wref = self.model.W.copy()
            self.model.W.values[:] = W.values

        # If input stimulus sensitivities provided, store reference and assign to model
        if srel is not None:
            srel_ref = self.model.srel.copy()
            self.model.srel.values[:] = srel.values
        
        # Run stimulation sweep and evaluate cost
        amps = self.ref_profiles.index.values
        sweep_data = self.model.run_sweep(amps, verbose=False, **kwargs)
        costs = self.evaluate_sweep(sweep_data, verbose=verbose)

        # If requested, add cost for relative deviation from reference network connectivity matrix
        if W is not None:
            costs[self.CONN_COST_KEY] = self.compute_connectivity_cost(W, Wref)

        # Log costs
        if verbose:
            cost_str = ', '.join([f'{k} = {v:.3f}' for k, v in costs.items()])
            logger.info(f'{self} costs: {cost_str}')

        # Compute total cost
        cost = np.nansum(list(costs.values()))

        # Reset model parameters that have been modified
        if W is not None:
            self.W = Wref
        if srel is not None:
            self.srel = srel_ref

        # Bound cost to avoid errors in optimization algorithms
        if np.isnan(cost) or cost > MAX_COST:
            cost = MAX_COST

        # Return cost
        return cost
    
    def feval(self, x):
        '''
        Evaluation function

        :param x: input arguments vector
        '''
        return self.set_run_and_evaluate(x, verbose=False)
    
    def batch_feval(self, X, mpi=True):
        ''' 
        Call evaluation function on several sets of parameter inputs

        :param X: array of parameter inputs
        '''
        if mpi:
            with mp.Pool(processes=os.cpu_count()) as pool:
                res = pool.map(self.feval, X)
        else:
            res = list(map(self.feval, X))
        return res
    
    def test_eval(self, *args, **kwargs):
        ''' Run test call to evaluation function '''
        # Extract search bounds
        pbounds = self.get_exploration_bounds(*args, **kwargs)
        # Randomly select parameters within search bounds
        p = pbounds.apply(lambda x: np.random.uniform(*x))
        # Set model xnames to enable inputs parsing
        self.model.xnames = p.index.values
        # Call feval on parameters array
        out = self.feval(p.values)
        # Reset model xnames to None
        self.model.xnames = None
        # Return evaluation result
        return out

    def callback(self, xk):
        '''
        Callback function called at the end of every iteration during optimization

        :param xk: current best solution vector 
        '''
        # Evaluate cost for best candidate
        cost = self.feval(xk)

        # Log iteration number, computation time, and cost
        t = time.perf_counter()
        logger.info(f'{self} iteration {self.iter} completed in {t - self.tref:.1f} s, cost = {cost:.2e}')
        self.tref =t
        
        # Log to file, if path provided
        if self.logfpath is not None:
            self.append_to_log_file(self.logfpath, self.iter, xk, cost)
        
        # Increment iteration counter
        self.iter += 1

    def annealing_callback(self, x, f, context):
        self.callback(x)
    
    def diffev_callback(self, xk, convergence):
        self.callback(xk)
    
    def shg_callback(self, xk):
        self.callback(xk)
    
    def direct_callback(self, xk):
        self.callback(xk)

    def get_exploration_bounds(self, Wbounds, srel_bounds, uniform_srel=False):
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
            self.model.check_coupling_bounds(Wbounds)
            # Append serialized bounds to exploration bounds
            serialized_Wbounds = Wbounds.stack() 
            serialized_Wbounds.index = serialized_Wbounds.index.map(lambda x: f'W({x[0]}-{x[1]})')
            xbounds.append(serialized_Wbounds.rename('bounds'))

        # If stimulus sensitivities bounds provided
        if srel_bounds is not None:
            # If single tuple provided, parse it
            if isinstance(srel_bounds, tuple):
                srel_bounds = self.model.parse_srel_bounds(srel_bounds, uniform_srel=uniform_srel)
            # Otherwise, check that bounds are compatible with model
            else:
                self.model.check_srel_bounds(srel_bounds, is_uniform=uniform_srel)
            
            # Add stimulus sensitivities bounds to exploration
            xbounds.append(srel_bounds.add_prefix('srel '))
        
        # If no exploration bounds provided, raise error
        if len(xbounds) == 0:
            raise OptimizationError('no exploration bounds provided')
        
        # Concatenate exploration bounds, and return
        return pd.concat(xbounds)

    def global_optimization(self, Wbounds=None, srel_bounds=None, uniform_srel=False, mpi=False, fpath=None, **kwargs):
        '''
        Use global optimization algorithm to find set of model parameters that minimizes
        divergence with a reference set of activation profiles.

        :param Wbounds (optional): network connectivity matrix bounds. If None, use default bounds
        :param srel_bounds (optional): stimulus sensitivities bounds. If None, do not explore.
        :param uniform_srel (optional): whether to assume uniform stimulus sensitivity across populations (default: False)
        :param mpi (optional): whether to use multiprocessing (default: False)
        :param fpath (optional): path of file in which to log optimization process (inputs and costs) 
        :return: optimized network connectivity matrix
        '''
        # Initialize dictionary of optimization keyword arguments with appropriate callback function
        optkwargs = {
            **kwargs, 
            'callback': getattr(self, f'{self.opt_method}_callback')
        }
        
        # If multiprocessing requested
        if mpi:
            # If optimization method is not compatible with multiprocessing, turn it off
            if self.opt_method != 'diffev':
                logger.warning(f'multiprocessing not supported for {self.opt_method} optimization method -> turning off')
                mpi = False

            # Otherwise
            else:            
                # Get the number of available CPUs
                ncpus = get_cpu_count()

                # If only 1 CPU available, turn off multiprocessing  
                if ncpus == 1:
                    logger.warning('only 1 CPU available -> turning off multiprocessing')
                    mpi = False
                
                # Othwerise, update keyword arguments to use as 
                # many workers as there are cores
                else:
                    optkwargs.update({
                        'workers': ncpus,
                        'updating': 'deferred'
                    })
        
        # Extract exploration bounds per parameter
        xbounds = self.get_exploration_bounds(Wbounds, srel_bounds, uniform_srel=uniform_srel)
        xnames = xbounds.index.to_list()
        
        # Initialize evaluation counter and assign optional log file path 
        self.iter = 0
        self.logfpath = fpath

        # Run optimization algorithm
        s = f'running {len(xbounds)}D optimization'
        if 'workers' in optkwargs:
            s = f'{s} with {optkwargs["workers"]} parallel workers'
        xbounds_str = '\n'.join([f'   - {k}: {v}' for k, v in xbounds.items()])
        logger.info(f'{self}: {s} with parameter bounds:\n{xbounds_str}')
        self.tref = time.perf_counter()
        optres = self.opt_func(
            self.feval,
            xbounds.values.tolist(), 
            **optkwargs
        )
        self.tref = None

        # Log results to terminal
        logger.info(f'optimization result:\n{optres}')

        # If log file specified, attempt to save optimization results to file
        if fpath is not None:
            logger.info(f'saving optimization result in {fpath}')
            pd.Series(optres).to_hdf(fpath, OPTRES_KEY)

        # Reset evaluation counter and log file attributes
        self.iter = None
        self.logfpath = None

        # If optimization failed, log error
        if hasattr(optres, 'success') and not optres.success:
            logger.error(f'optimization failed: {optres.message}')

        # Return solution array parsed as pandas Series
        return pd.Series(optres.x, index=xnames, name='optimum')
    
    def get_log_filename(self, Wbounds, srel_bounds, uniform_srel, irun):
        ''' 
        Generate log filename from optimization input arguments

        :param Wbounds: network coupling weights bounds matrix.
        :param srel_bounds: stimulus sensitivities bounds vector.
        :param uniform_srel: whether to assume uniform stimulus sensitivity across populations
        :param irun: run number
        :return: name of log file
        '''
        # Gather dictionary of model attribute codes
        model_ids = self.model.attrcodes()
        
        # Create empty dictionary for optimization IDs
        opt_ids = {}
        
        # Add "targets" ID 
        opt_ids['targets'] = generate_unique_id(self.ref_profiles)
        
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
        opt_code = f'{opt_code}_{self.opt_method}'
        
        # Add "norm" suffix if normalization specified
        if self.norm:
            opt_code = f'{opt_code}_norm'
        
        # Add disparity cost factor, if non-zero
        if self.disparity_cost_factor > 0:
            opt_code = f'{opt_code}_xdisp{self.disparity_cost_factor:.2g}'
        
        # Add Wdev cost factor, if non-zero and Wbounds provided
        if Wbounds is not None and self.Wdev_cost_factor > 0:
            opt_code = f'{opt_code}_xwdev{self.Wdev_cost_factor:.2g}'
        
        # If exploration assumes uniform sensitivity, add "unisrel" suffix
        if srel_bounds is not None and uniform_srel:
            opt_code = f'{opt_code}_unisrel'

        # If srel is uniformly unitary, remove "srel" attribute code if present
        if 'srel' in model_ids and np.isclose(self.model.srel, 1.).all():
            del model_ids['srel']

        # Replace "keys" and "fgain" attribute keys by empty strings
        exclude = ['keys', 'fgain']
        keys = [k if k not in exclude else '' for k in model_ids.keys()]
        
        # Assemble into attribute code
        model_code  = '_'.join([f'{k}{v}' for k, v in zip(keys, model_ids.values())])
        
        # Merge model code, optimization code and run number
        code = f'{model_code}_{opt_code}_run{irun}'

        # Add extension and return
        return f'{code}.h5'
    
    def find_first_unlogged_run(self, logdir, *args):
        ''' 
        Find first run number that is not logged, for a set of input parameters
        '''
        # Initialize existence flag to True 
        log_exists = True

        # Initialize run index to -1
        irun = -1

        # While existence flag is true
        while log_exists:
            # Increment run number
            irun += 1

            # Assemble log file path 
            fname = self.get_log_filename(*args, irun)
            fpath = os.path.join(logdir, fname)

            # Update existence flag
            log_exists = os.path.isfile(fpath)
        
        # Return run number
        return irun
    
    @staticmethod
    def extract_optimum(data):
        '''
        Extract optimum from exploration results

        :param data: exploration results dataframe containing 'cost' column
        :return: pandas Series of parameter values yielding optimal cost
        '''
        # If no data available, raise error
        if len(data) == 0:
            raise OptimizationError('no exploration results available')
        
        # Find exploration entry yielding minimal cost 
        iopt = data['cost'].idxmin()
        opt = data.loc[iopt].rename('optimum')
        cost = opt.pop('cost')
        logger.info(f'found optimal entry (cost = {cost:.2e})')

        # Return
        return opt

    def optimize(self, *args, Wbounds=None, srel_bounds=None, uniform_srel=False, irun=0, logdir=None, nruns=1, **kwargs):
        '''
        Find network connectivity matrix that minimizes divergence with a reference
        set of activation profiles.

        :param model: model instance
        :param args: positional arguments passed to global optimization method .
        :param Wbounds (optional): network coupling weights bounds matrix. 
            If None, use default weights.
        :param srel_bounds (optional): stimulus sensitivities bounds vector. 
            If None, use default sensitivities.
        :param uniform_srel (optional): whether to assume uniform stimulus sensitivity 
            across populations (default: False)
        :param method (optional): optimization algorithm (default = "diffev")
        :param irun (optional): run number (defaults to 0). Use 'next' to force new run.
        :param logdir (optional): directory in which to save/load exploration results.
            If None, no log will be saved.
        :param nruns: number of runs to perform (defaults to 1)
        :param kwargs: keyword arguments passed to global optimization method.
        :return: optimized network connectivity matrix
        '''
        # If nruns set to "auto", set to number of corresponding log files found in log dir
        if nruns == 'auto':
            nruns = self.find_first_unlogged_run(
                logdir, *args, Wbounds, srel_bounds, uniform_srel)
            if nruns == 0:
                raise OptimizationError(f'{self}: no optimization log file found in {logdir}')
            else:
                logger.info(f'found {nruns} optimization log files in {logdir}')

        # If multiple runs requested, run them sequentially
        if nruns > 1:
            logger.info(f'{self}: running {nruns} optimization runs')
            opts = {}
            for irun in range(nruns):
                try:
                    opt = self.optimize(
                        *args,
                        Wbounds=Wbounds,
                        srel_bounds=srel_bounds,
                        uniform_srel=uniform_srel, 
                        irun=irun,
                        logdir=logdir,
                        nruns=1,
                        **kwargs
                    )
                except OptimizationError as e:
                    logger.error(e)
                    opt = pd.Series(np.nan, index=self.model.xnames, name='optimum')    
                opts[irun] = opt
            return pd.concat(opts, axis=1, names='run').T
        
        # If log folder is provided
        if logdir is not None:
            # If run number is "next", find first run index that is not logged 
            if irun == 'next':
                irun = self.find_first_unlogged_run(
                    logdir, *args, Wbounds, srel_bounds, uniform_srel)
            
            # Derive path to log file 
            fname = self.get_log_filename(
                *args, Wbounds, srel_bounds, uniform_srel, irun)
            fpath = os.path.join(logdir, fname)

        # Otherwise, set to None
        else:
            fpath = None
        
        # Set model parameter names from input exploration bounds 
        self.model.xnames = self.get_exploration_bounds(
            Wbounds, srel_bounds, uniform_srel=uniform_srel).index.values
        
        # If log file path provided
        if fpath is not None:
            # If log file exists
            if os.path.isfile(fpath):
                # Load optimization results from log
                optres, optparams = self.load_optimization_results(fpath, verbose=False)
                # If optimization failed, log error
                if hasattr(optres, 'success') and not optres.success:
                    logger.error(f'optimization failed: {optres.message}')
                # Return optimum solution
                return optparams
                
            # Otherwise, create log file, if path provided
            self.create_log_file(fpath, self.model.xnames)
        
        # Run optimization algorithm and return
        return self.global_optimization(
            Wbounds=Wbounds,
            srel_bounds=srel_bounds, 
            uniform_srel=uniform_srel,
            fpath=fpath,
            **kwargs
        )
    
    def load_optimization_histories(self, logdir, *args):
        '''
        Load optimization histories across runs for a specific of optimization parameters

        :param logdir: directory of log files
        :param args: list of argumetns passed on to "get_log_filename" method
        :return: (run, iteration)-indexed dataframe of explored parameters and associated costs
        '''
        # Extract number of runs
        nruns = self.find_first_unlogged_run(logdir, *args)
        # Lopp through runs and load optimization history
        opt_history = {}
        for irun in range(nruns):
            fname = self.get_log_filename(*args, irun)
            fpath = os.path.join(logdir, fname)
            opt_history[irun] = self.load_optimization_history(fpath)
        # Concatenate in multi-indexed dataframe, and return
        return pd.concat(opt_history, axis=0, names=['run'])
    
    def plot_optimization_history(self, history, ax=None, title=None):
        '''
        Plot optimization history

        :param history: iteration-indexed dataframe of explored parameters and associated costs.
            If other index dimensions are present, one line will be plotted for each combination
            of these higher dimensions.
        :param ax (optional): plotting axis
        :param title (optional): axis title
        :return: figure handle       
        '''
        # Create/retrieve axis
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        sns.despine(ax=ax)

        # Initialize plotting parameters dictionary
        pltkwargs = {}

        otherdims = [k for k in history.index.names if k != 'iteration'][::-1]
        if len(otherdims) > 2:
            raise ValueError('cannot plot histories with more than 2 extra dimensions')
        if len(otherdims) > 0:
            pltkwargs['hue'] = otherdims[0]
            pltkwargs['palette'] = 'Set2'
        else:
            pltkwargs['color'] = 'k'
        if len(otherdims) > 1:
            pltkwargs['style'] = otherdims[1]
        
        # Plot cost vs iteration, with grouping across higher dimensions
        sns.lineplot(
            ax=ax,
            data=history,
            x='iteration',
            y='cost',
            **pltkwargs
        )

        # # Add line materializing max number of iterations 
        # ax.axvline(OPT_MAXITER, c='k', ls='--')

        # Set log y-scale
        ax.set_yscale('log')
        ax.set_ylim(*COST_PLOT_RANGE)

        # Add title
        if title is not None:
            ax.set_title(title)

        # Return figure
        return fig

    def compare_prediction_to_reference(self):
        ''' 
        Extract dataframe comparing predicted vs reference evoked responses over
        a sweep of input stimulus amplitudes

        :return: multi-indexedex dataframe, and costs
        '''
        # Perform stimulus sweep and extract predicted responses profiles
        sweep_data = self.model.run_sweep(self.ref_profiles.index)
        predicted_profiles = self.model.extract_response_magnitude(sweep_data)
        
        # Assemble comparison dataframe
        df = pd.concat({
            'predicted': predicted_profiles,
            'reference': self.ref_profiles
        }, axis=0, names=['profile'])

        # Compare to reference profiles and extract evaluated costs
        costs = self.evaluate_sweep(sweep_data)

        # Return
        return df, costs
    
    def plot_costs(self, costs, ax=None, height=3, title=None, yscale='log', **kwargs):
        '''
        Plot costs per category

        :params costs: "kind"-indexed series of costs
        :param x (optional): plotting axis
        :param height (optional): figure height
        :param title (optional): figure title
        :param yscale: y-axis scale ('lin' or 'log')
        :return: figure handle        
        '''
        # If dataframe input, stack it
        if isinstance(costs, pd.DataFrame):
            costs = costs.stack()
        # If dict input, cast as series
        elif isinstance(costs, dict):
            costs = pd.Series(costs)
            costs.index.name = 'kind'
        
        # Extract number of costs categories
        nkinds = costs.groupby('kind').ngroups

        # Create retrieve figure and axis
        if ax is None:
            fig, ax = plt.subplots(figsize=(nkinds, height))
        else:
            fig = ax.get_figure()
        sns.despine(ax=ax)

        # Plot costs by category
        pltkwargs = dict(
            ax=ax,
            data=costs.rename('cost').reset_index(),
            x='kind',
            y='cost'
        )
        sns.stripplot(
            palette='Set2',
            **pltkwargs,
            **kwargs
        )
        sns.pointplot(
            color='k',
            linestyles='none',
            errorbar='se',
            **pltkwargs,
        )

        # Set log scale if requested
        if yscale == 'log':
            ax.set_yscale('log')
            ax.set_ylim(*COST_PLOT_RANGE)
        
        # Add title if provided
        if title is not None:
            ax.set_title(title)

        # Return figure
        return fig

    def plot_results(self, mparams=None, opt_history=None, axes=None, norm_params=False, norm_res='ax',
                     add_respthrs=False, add_axtitles=True, title=None, height=2.5, avg_across_runs=False,
                     return_costs=False, add_stats=False):
        '''
        Plot model parameters and predicted activity profiles against reference ones
        
        :param mparams: series of (or run-indexed dataframe of) serialized model parameters
        :param opt_history (optional): (run, iteration)-indexed dataframe of explored parameters and associated costs
        :param axes (optional): axes objects on which to plot
        :param norm_params (optional): whether to normalize model parameters prior to visualization (defaults to False)
        :param norm_res (optional): whether/how to normalize response profiles prior to visualization (defaults to 'ax', i.e. 1 normalization per axis)
        :param add_respthrs (optional): whether to add an axis on which to plot distributions
            of intrinic response thresholds (in stimulus units) per population 
        :param add_axtitles (optional): whether to add axes titles (defaults to True)
        :param title (optional): global figure title
        :param height: height per axis row
        :param avg_across_runs (optional): whether to average optimization results across runs (if any) prior to visualization (defaults to True). 
            If not, each run will be plotted on a separate axis row
        :param return_costs: whether to return also avaluated costs per run 
        :return: figure object, and optionally also evaluated costs per run
        '''
        # Get copy of matrix and srel vector
        Wcopy = self.model.W.copy()
        srelcopy = self.model.srel.copy()

        # Determine number of axes
        norm_res = as_iterable(norm_res)
        naxes = 3 + len(norm_res)
        if opt_history is not None:
            naxes += 1
        if add_respthrs:
            naxes += 1

        # Set title relative height 
        ytitle = 1.05

        # Set connectivitiy matrix and relative sensitivity vector to None
        W, srel = None, None

        # Set sweep comp data and error to None
        sweep_comp, error = None, None

        # If model parameters provided
        if mparams is not None:
            # If multi-run results
            if isinstance(mparams, pd.DataFrame):
                
                # Extract number of runs
                nruns = len(mparams)

                # If average specifiied
                if avg_across_runs:
                    logger.info(f'{self}: averaging optimal parameters across {nruns} runs')

                    # Parse parameters from vector for each run
                    parsed_mparams = {irun: self.model.parse_input_vector(v)
                        for irun, v in mparams.iterrows()}
                    
                    # Extract W and srel, if present
                    W = {k: v[0] for k, v in parsed_mparams.items()}
                    if any(w is None for w in W.values()):
                        W = None
                    else:
                        W = pd.concat(W, names=['run'])
                    srel = {k: v[1] for k, v in parsed_mparams.items()}
                    if any (s is None for s in srel.values()):
                        srel = None
                    else:
                        srel = pd.concat(srel, names=['run'])
                    
                    # Adjust model parameters, run sweep comparison and extract error for every run
                    sweep_comp = {}
                    costs = {}
                    for irun, v in mparams.iterrows():
                        self.model.set_from_vector(v)
                        sweep_comp[irun], costs[irun] = self.compare_prediction_to_reference()
                        costs[irun][self.CONN_COST_KEY] = self.compute_connectivity_cost(self.model.W, self.Wref)
                        costs[irun] = pd.Series(costs[irun])
                    sweep_comp = pd.concat(sweep_comp, axis=0, names=['run'])
                    costs = pd.concat(costs, names=['run']).unstack()
                    costs.columns.name = 'kind'

                    # Average model parameters across runs
                    mparams = mparams.mean(axis=0)

                    # Adapt title
                    suffix = f'average across {nruns} runs'
                    if title is None:
                        title = suffix
                    else:
                        title = f'{title} - {suffix}'

                else:
                    # Create figure with 2D axes grid
                    if axes is not None:
                        raise ValueError('cannot provide axes for multi-run entry')
                    nrows = nruns 
                    fig, axes = plt.subplots(
                        nrows, naxes, figsize=(height * naxes, height * nrows))
                    fig.subplots_adjust(wspace=.5, hspace=.5)

                    # Share x and y axes across each grid column
                    for axcol in axes.T:
                        for ax in axcol[1:]:
                            ax.get_shared_x_axes().join(axcol[0], ax)
                            ax.get_shared_y_axes().join(axcol[0], ax)
                    
                    # Call function recursively to plot optimization results for each run
                    if return_costs:
                        costs = {}
                    for axrow, (irun, v) in zip(axes, mparams.iterrows()):
                        out = self.plot_results(
                            mparams=v, 
                            opt_history=opt_history.loc[irun] if opt_history is not None else None,
                            axes=axrow, norm_params=norm_params, norm_res=norm_res,
                            add_respthrs=add_respthrs, add_axtitles=irun == 0, return_costs=return_costs)
                        if return_costs:
                            costs[irun] = out[1]
                        if title is not None:
                            fig.suptitle(title, y=1 - 0.015 * nrows)
                    if return_costs:
                        costs = pd.concat(costs, names=['run']).unstack()
                        costs.columns.name = 'kind'
                    
                    # Return figure (and optionally, evaluated costs)
                    if return_costs:
                        return fig, costs
                    else:
                        return fig

            # Adjust model parameters based on optimization results
            self.model.set_from_vector(mparams)

        # Create /retrieve figure and axes
        if axes is None:
            fig, axes = plt.subplots(1, naxes, figsize=(1.2 * height * naxes, height))
            fig.subplots_adjust(wspace=.5)
        else:
            if len(axes) != naxes:
                raise ValueError(f'number of input axes ({len(axes)} does not match number of required axes ({naxes})')
            fig = axes[0].get_figure()
        iax = 0

        # Extract reference vs predicted response dataframe over stimulus sweep
        if sweep_comp is None:
            sweep_comp, costs = self.compare_prediction_to_reference()
        
        # In case of single-run (cost dictionary), add connectivity cost if not present
        # and format as series
        if isinstance(costs, dict):
            if mparams is not None and self.CONN_COST_KEY not in costs:
                costs[self.CONN_COST_KEY] = self.compute_connectivity_cost(self.model.W, self.Wref)
            costs = pd.Series(costs)
            costs.index.name = 'kind'

        # If provided, plot optimization history
        if opt_history is not None:
            self.plot_optimization_history(opt_history, ax=axes[iax])
            iax += 1

        # Plot costs
        self.plot_costs(costs, ax=axes[iax])
        iax += 1

        # Plot connectivity matrix
        self.model.plot_connectivity_matrix(
            self.model.W if W is None else W,
            ax=axes[iax], agg=True,
            norm=norm_params, 
            title='')

        # If connectivity matrix provided, compute average magnitude of
        # relative change in each coupling weight
        if W is not None:
            dwwkey = 'avg |Δw/w|'
            if isinstance(W.index, pd.MultiIndex) and 'run' in W.index.names:
                Wreldev = W.groupby('run').apply(
                    lambda v: self.get_relative_change(v.droplevel('run'), self.Wref))
                print(Wreldev.abs().max().max())
                avg_abs_wreldev = Wreldev.groupby('run').apply(lambda df: df.abs().mean().mean())
                wreldev_str = f'{dwwkey} = {avg_abs_wreldev.mean() * 100:.1f}±{avg_abs_wreldev.std() * 100:.1f} %'
            else:
                Wreldev = self.get_relative_change(W, self.Wref)
                avg_abs_wreldev = Wreldev.abs().mean().mean()
                wreldev_str = f'{dwwkey} = {avg_abs_wreldev * 100:.1f} %'
            axes[iax].set_title(wreldev_str)
        iax += 1

        # Plot stimulus sensitivity
        self.model.plot_stimulus_sensitivity(
            srel=srel, ax=axes[iax], norm=norm_params, title='', add_stats=add_stats)
        iax += 1

        # If specified, compute and plot intrinsic response thresholds
        if add_respthrs:
            intrinsic_thrs = (self.model.fparams['x0'] / srel).rename('response threshold (stim. units)')
            if isinstance(srel.index, pd.MultiIndex):
                intrinsic_thrs_thr = intrinsic_thrs.groupby('population').agg(['mean', 'sem'])
            else:
                intrinsic_thrs_thr = intrinsic_thrs
            logger.info(f'intrinsic response thresholds (in stimulus units):\n{intrinsic_thrs_thr}')
            self.model.plot_stimulus_sensitivity(
                srel=intrinsic_thrs, ax=axes[iax], title='', add_stats=add_stats)
            iax += 1

        # Generate costs string
        if isinstance(costs, pd.Series):
            if any(k not in self.SHORT_COST_KEYS for k in costs.index):
                costs_str = costs.index[0]
            else:
                costs_str = {self.SHORT_COST_KEYS[k]: f'{v:.2g}' for k, v in costs.items()}
        else:
            costs_str = {self.SHORT_COST_KEYS[k]: f'{costs[k].mean():.2g} ± {costs[k].std():.2g}' for k in costs}
        if isinstance(costs_str, dict):
            costs_str = '\n'.join([f'ε{k} = {v}' for k, v in costs_str.items()])
            fontweight = 'normal'
        else:
            fontweight = 'bold'

        # Plot model sweep results and reference profiles, for each normalization type
        for ax, n in zip(axes[iax:], norm_res):
            self.model.plot_sweep_results(sweep_comp, norm=n, ax=ax, style='profile')
            if add_axtitles:
                ax.set_title(f'{f"{n}-normalized" if n else "absolute"} profiles')
            if ax is not axes[-1]:
                ax.get_legend().remove()
            if n == 'style':
                ax.text(
                    0.1, 1.0, costs_str, transform=ax.transAxes, ha='left', va='top', weight=fontweight)
        sns.move_legend(ax, bbox_to_anchor=(1, .5), loc='center left', frameon=False)

        # If requested, add figure title
        if title is not None:
            fig.suptitle(title, y=ytitle)

        # Reset matrix and srel vector
        self.model.W = Wcopy
        self.model.srel = srelcopy

        # Return output(s)
        if return_costs:
            return fig, costs
        else:
            return fig
        
    def compute_drive_contributions(self, optparams):
        '''
        Compute various breakdowns of drive contributions over reference range of
        stimulus amplitudes

        :param optparams: series (or run-indexed dataframe) of optimal model parameters
        :return: dictionary of (potentially run-indexed) dataframes of drive contributions
        '''
        # If multi-run input, compute function recursively for each run, and return
        # dictionary of run-indexed dataframes
        if isinstance(optparams, pd.DataFrame):
            drive_contributions = {irun: self.compute_drive_contributions(opt)
                                   for irun, opt in optparams.iterrows()}
            return {k: pd.concat(v, names=['run']) 
                    for k, v in swaplevels(drive_contributions).items()}
        
        # Store copy of model parameters
        W, srel = self.model.W, self.model.srel

        # Assign input parameters to model
        self.model.set_from_vector(optparams)

        # Compute all drive contributions
        drive_contributions = self.model.compute_drive_contributions(
            self.ref_profiles.index)
        
        # Restore model default parameters
        self.model.W = W
        self.model.srel = srel

        # Return drive contributions dictionary
        return drive_contributions
    
    def compute_timeseries(self, optparams, amps, norm=False):
        '''
        Plot detailed timeseries for a set of stimulus amplitudes

        :param optparams: series (or run-indexed dataframe) of optimal model parameters
        :param amps: amplitudes at which to compute solutions
        :param norm (optional): whether to normalize solutions by dividing each population trace
            by the maximal evoked response magnitude of that population along the full amplitude range  
        :return: amplitude and time (and potentially run-) indexed dataframe of solution timeseries
        '''
        # If multi-run input, compute function recursively for each run, and return
        # run-indexed timeseries
        if isinstance(optparams, pd.DataFrame):
            return pd.concat({irun: self.compute_timeseries(opt, amps, norm=norm)
                              for irun, opt in optparams.iterrows()}, names=['run'])
        
        # Store copy of model parameters
        W, srel = self.model.W, self.model.srel

        # Assign input parameters to model
        self.model.set_from_vector(optparams)

        # Run simulation sweep across input amplitudes, 
        # and resample to common time step
        sweep_data = self.model.run_sweep(amps, target_dt=1e0)

        # If requested, normalize by dividing by max response amplitudes evoked
        # along full amplitude range
        if norm:
            sweep_data[self.model.keys] /= self.model.predict_response_profiles(
                self.ref_profiles.index.values).max()
        
        # Restore model default parameters
        self.model.W = W
        self.model.srel = srel
        
        # Return
        return sweep_data

    def plot_timeseries(self, optparams, amps, norm=False, title=None):
        '''
        Plot detailed timeseries for a set of stimulus amplitudes

        :param optparams: series (or run-indexed dataframe) of optimal model parameters
        '''
        # Compute solution timeseries for input model parameters
        sweep_data = self.compute_timeseries(optparams, amps, norm=norm)
        
        # Plot detailed simulation results for a set of characteristic input amplitudes
        fig, axes = plt.subplots(
            amps.size, 1, figsize=(5, 1.5 * amps.size), 
            sharex=True, sharey=True)
        for (A, sol), ax in zip(sweep_data.groupby('amplitude'), axes):
            self.model.plot_timeseries(
                sol.droplevel('amplitude'), title=f'A = {A:.2f}', axes=ax, plot_stimulus=False)
    
        # If supplied, add title
        if title is not None:
            fig.suptitle(title, y=1.05)
        
        # Return figure
        return fig