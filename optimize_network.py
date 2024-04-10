# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2024-03-14 17:56:23
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2024-04-05 10:53:42

import numpy as np
import pandas as pd
from argparse import ArgumentParser
import os
import pandas as pd

from logger import logger
from network_model import *
from utils import threshold_linear
from fileops import get_data_root, get_output_equivalent

# Set up logging folder
logdir = get_output_equivalent(get_data_root(), 'raw', 'model')

# Time constants (from Romero-Sosa et al. 2020)
tau = pd.Series(
    data={
        'E': 0.010, 
        'PV': .004, 
        'SST': .006
    },
    name='tau (s)'
)
populations = tau.index

# Gain function
fgain = threshold_linear  

# Gain function parameters (from Romero-Sosa et al. 2020)
fparams = pd.DataFrame(
    data={
        'x0': [5, 30, 15],  # E, PV, SST
        'A': [1, 2.7, 1.6], # E, PV, SST
    },
    index=pd.Index(populations, name='population'),
)

# Null connectivity matrix
W = pd.DataFrame(
    data=np.zeros((len(populations), len(populations))),
    index=pd.Index(populations, name='pre-synaptic'), 
    columns=pd.Index(populations, name='post-synaptic')
)

# Relative input strength per population
srel = pd.Series(1., populations, name='external input')

# Define vector of amplitudes of external input w.r.t. reference value
Ithr = fparams.loc['E', 'x0']  # E activation threshold
rel_amps = np.linspace(0, 10, 25)
amps = rel_amps * Ithr

# Target activity profiles
ref_fpath = os.path.join(logdir, 'ref_profiles.csv')
ref_profiles = pd.read_csv(ref_fpath).set_index('amplitude')
ref_profiles.columns.name = 'population'

# Main
if __name__ == '__main__':

    # Parse command-line arguments
    parser = ArgumentParser(
        description='Script for network model connectivity matrix optimization')
    parser.add_argument(
        '--npops', type=int, choices=(2, 3), default=3, 
        help='Number of populations')
    parser.add_argument(
        '--wmax', type=float, default=NetworkModel.WMAX, help='Maximum absolute value for coupling weights')
    parser.add_argument(
        '--wbounds', metavar='KEY KEY VALUE VALUE', nargs='+', type=str, 
        help='List of coupling weight bounds to adjust search range')
    parser.add_argument(
        '--uniform-gain', action='store_true', help='Use uniform gain function for all populations')
    parser.add_argument(
        '-m', '--method', type=str, choices=('brute', 'diffev'), default='diffev', 
        help='Optimization method')
    parser.add_argument(
        '--npersweep', type=int, default=5, 
        help='Number of points per sweep (for brute-force method)')
    parser.add_argument(
        '--norm', action='store_true',
        help='Normalize profiles prior to comparison')
    parser.add_argument(
        '--pendisp', action='store_true', help='Penalize disparity in activity levels across populations')
    parser.add_argument(
        '--mpi', action='store_true', help='Run with MPI')
    parser.add_argument(
        '--nosave', action='store_true', help='Do not save results in log file')
    parser.add_argument(
        '--force-rerun', action='store_true', help='Enforce rerun of optimization')
    parser.add_argument(
        '--nruns', type=int, default=1, help='Number of optimization runs')
    
    args = parser.parse_args()
    npops = args.npops
    wmax = args.wmax
    if wmax == NetworkModel.WMAX:
        wmax = None
    uniform_gain = args.uniform_gain
    method = args.method
    npersweep = args.npersweep
    norm = args.norm
    penalize_disparity = args.pendisp
    mpi = args.mpi
    save = not args.nosave
    force_rerun = args.force_rerun
    nruns = args.nruns
    if nruns > 1:
        force_rerun = True

    # If nosave option, set logdir to None
    if not save:
        logdir = None

    # If 2 populations selected, remove PV related parameters
    if npops == 2:
        tau = tau.drop('PV')
        fparams = fparams.drop('PV')
        W = W.drop('PV', axis=0).drop('PV', axis=1)
        srel = srel.drop('PV')
        ref_profiles = ref_profiles.drop('PV', axis=1)
    
    # If uniform gain function requested, assign all populations to same values as E
    if uniform_gain:
        for k in fparams.index:
            fparams.loc[k] = fparams.loc['E']
    
    # Initialize model
    model = NetworkModel(W=W, tau=tau, fgain=fgain, fparams=fparams)

    # Parse weight bounds
    wbounds = None
    if args.wbounds is not None:
        if not len(args.wbounds) % 4 == 0:
            raise ValueError('Invalid number of arguments for wbounds (should be multiple of 4)')
        wquads = [args.wbounds[i:i+4] for i in range(0, len(args.wbounds), 4)]
        wbounds = {}
        for quad in wquads:
            keys, vals = quad[:2], quad[2:]
            for k in keys:
                if k not in model.keys:
                    raise ValueError(f'Invalid population key in wbounds: {k}')
            try:
                vals = list(map(float, vals))
            except ValueError:
                raise ValueError(f'Invalid weight bound tuple value in wbounds: {vals}')
            vals = tuple(sorted(vals))
            wbounds[tuple(keys)] = tuple(sorted(vals))

    # Initialize weight bounds matrix to None
    Wbounds = None

    # Adjust search range for all coupling weights if requested
    if wmax is not None:
        Wbounds = model.get_coupling_bounds(wmax=wmax)
        logger.info(f'default weight bounds:\n{Wbounds}')

    # Adjust search range for specific coupling weights if requested
    if wbounds is not None:
        if Wbounds is None:
            Wbounds = model.get_coupling_bounds()
        for (kpre, kpost), vals in wbounds.items():
            Wbounds.loc[kpre, kpost] = vals
            # Convert Wbounds to float tuples if not already
            Wbounds = Wbounds.applymap(lambda x: tuple(map(float, x)))
        logger.info(f'adjusted weight bounds:\n{Wbounds}')


    logger.info(f'target activity profiles:\n{ref_profiles}')

    # For each specified run
    convergence = []
    for i in range(nruns):
        logger.info(f'running {method} optimization for {model}')

        # Optimize connectivity matrix to minimize divergence with reference profiles
        try:
            Wopt = ModelOptimizer.optimize(
                model,
                srel,
                ref_profiles, 
                norm=norm,
                penalize_disparity=penalize_disparity,
                Wbounds=Wbounds,
                mpi=mpi,
                logdir=logdir,
                kind=method, 
                npersweep=npersweep,
                force_rerun=force_rerun
            )
            convergence.append(True)
        except OptimizationError as e:
            logger.error(e)
            convergence.append(False)
        
    # If one run failed, quit
    if not all(convergence):
        logger.error('At least one optimization run failed')
        quit()

    # Perform stimulus sweep with optimal connectivity matrix
    logger.info(f'optimal connectivity matrix:\n{Wopt}')
    model.W = Wopt
    sweep_data = model.run_stim_sweep(srel, amps)

    # Compare results to reference profiles
    rmse = model.evaluate_stim_sweep(
        ref_profiles, sweep_data, norm=norm, penalize_disparity=penalize_disparity)
    logger.info(f'RMSE = {rmse:.2f}')
