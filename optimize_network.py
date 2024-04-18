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
from constants import Label
from fileops import get_data_root, get_output_equivalent
from model_params import *

# Set up logging folder
logdir = get_output_equivalent(get_data_root(), 'raw', 'model')

# (key: name) sources dictionary for model parameters 
sources_dict = dict(zip([k.lower().split()[0] for k in tau_dict.keys()], tau_dict.keys()))

# Target activity profiles
ref_fpath = os.path.join(logdir, 'ref_profiles.csv')
ref_profiles = pd.read_csv(ref_fpath).set_index(Label.ISPTA)
ref_profiles.columns.name = 'population'
amps = ref_profiles.index.values

# Main
if __name__ == '__main__':

    # Define CLI parser
    parser = ArgumentParser(
        description='Script for network model optimization')
    parser.add_argument(
        '--source', type=str, choices=list(sources_dict.keys()), default='allen',
        help='Source of model parameters')
    parser.add_argument(
        '--npops', type=int, choices=(2, 3), default=3, 
        help='Number of populations')
    parser.add_argument(
        '--wmax', type=float, default=None, help='Maximum absolute coupling weight value (for exploration')
    parser.add_argument(
        '--wbounds', metavar='KEY KEY VALUE VALUE', nargs='+', type=str, 
        help='List of adjusted search bounds for specific coupling weights')
    parser.add_argument(
        '--srel', metavar='KEY VALUE', nargs='+', type=str, 
        help='List of stimulus sensitivity per populations')
    parser.add_argument(
        '--srel-max', type=float, default=None, help='Maximal stimulus sensitivity value (for exploration)')
    parser.add_argument(
        '--uniform-srel', action='store_true', help='Use uniform stimulus sensitivity for all populations')
    parser.add_argument(
        '--uniform-gain', action='store_true', help='Use uniform gain function for all populations')
    parser.add_argument(
        '-m', '--method', type=str, choices=ModelOptimizer.GLOBAL_OPT_METHODS, 
        default='diffev', help='Optimization method')
    parser.add_argument(
        '--norm', action='store_true',
        help='Normalize profiles prior to comparison')
    parser.add_argument(
        '--xdisp', type=float, default=0., 
        help='Cost scaling factor to penalize disparity in activity levels across populations')
    parser.add_argument(
        '--xwdev', type=float, default=0., 
        help='Cost scaling factor to penalize deviation from initial weights')
    parser.add_argument(
        '--mpi', action='store_true', help='Run with MPI')
    parser.add_argument(
        '--nosave', action='store_true', help='Do not save results in log file')
    parser.add_argument(
        '--force-rerun', action='store_true', help='Enforce rerun of optimization')
    parser.add_argument(
        '--nruns', type=int, default=1, help='Number of optimization runs')
    
    # Parse command-line arguments
    args = parser.parse_args()
    source_key = sources_dict[args.source]
    npops = args.npops
    wmax = args.wmax
    srelmax = args.srel_max
    uniform_srel = args.uniform_srel
    uniform_gain = args.uniform_gain
    method = args.method
    norm = args.norm
    disparity_cost_factor = args.xdisp
    Wdev_cost_factor = args.xwdev
    mpi = args.mpi
    save = not args.nosave
    force_rerun = args.force_rerun
    nruns = args.nruns
    if nruns > 1:
        force_rerun = True

    # If nosave option, set logdir to None
    if not save:
        logdir = None

    # Load model parameters
    tau = tau_dict[source_key].copy()
    fparams = fparams_dict[source_key].copy()
    W = W_dict[source_key].copy()

    # If 2 populations selected, remove PV related parameters
    if npops == 2:
        tau = tau.drop('PV')
        fparams = fparams.drop('PV')
        W = W.drop('PV', axis=0).drop('PV', axis=1)
        ref_profiles = ref_profiles.drop('PV', axis=1)
    
    # Extract populations list
    populations = tau.index.values.tolist()
    
    # If uniform gain function requested, assign all populations to same values as E
    if uniform_gain:
        for k in fparams.index:
            fparams.loc[k] = fparams.loc['E']
    
    # Parse stimulus sensitivities
    srel = None
    if args.srel is not None:
        if not len(args.srel) % 2 == 0:
            raise ValueError('Invalid number of arguments for srel (should be multiple of 2)')
        spairs = [args.srel[i:i + 2] for i in range(0, len(args.srel), 2)]
        srel = {}
        for pair in spairs:
            k, v = pair
            if k not in populations:
                raise ValueError(f'Invalid population key in srel: {k}')
            try:
                v = float(v)
            except ValueError:
                raise ValueError(f'Invalid relative stimulus sensitivity value in srel: {v}')
            srel[k] = v
        srel = pd.Series(srel, name='stimulus sensitivities')
        srel.index.name = 'population'
        srel = srel.reindex(populations)
    
    # Initialize model with rescaled parameters
    model = NetworkModel(
        W=NetworkModel.rescale_W(W),
        tau=tau, 
        fparams=NetworkModel.rescale_fparams(fparams),
        srel=srel
    )

    # If specified, parse coupling weight bounds
    Wbounds = None
    if wmax is not None:
        Wbounds = model.get_coupling_bounds(wmax=wmax)
        
        # Adjust search range for specific coupling weights if requested
        if args.wbounds is not None:
            if Wbounds is None:
                raise ValueError('Cannot specify specific wbounds if wmax is not set')
            if not len(args.wbounds) % 4 == 0:
                raise ValueError('Invalid number of arguments for wbounds (should be multiple of 4)')
            wquads = [args.wbounds[i:i + 4] for i in range(0, len(args.wbounds), 4)]
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

            for (kpre, kpost), vals in wbounds.items():
                Wbounds.loc[kpre, kpost] = vals
                # Convert Wbounds to float tuples if not already
                Wbounds = Wbounds.applymap(lambda x: tuple(map(float, x)))
        
        logger.info(f'coupling weight bounds:\n{Wbounds}')
    
    # If specified, parse, relative stimulus sensitivity bounds
    srel_bounds = None
    if srelmax is not None:
        srel_bounds = (0., srelmax)
        logger.info(f'stimulus sensitivity bounds: {srel_bounds}')

    # If Wbounds and srel_bounds are not specified, raise error
    if Wbounds is None and srel_bounds is None:
        raise ValueError('At least one of Wbounds or srel_bounds should be specified')
    
    logger.info(f'target activity profiles:\n{ref_profiles}')
    
    # Initialize convergence list
    convergence = []

    # For each specified run
    for i in range(nruns):
        logger.info(f'running {method} optimization for {model}')

        # Optimize connectivity matrix to minimize divergence with reference profiles
        try:
            opt = ModelOptimizer.optimize(
                model,
                ref_profiles, 
                norm=norm,
                disparity_cost_factor=disparity_cost_factor,
                Wdev_cost_factor=Wdev_cost_factor,
                Wbounds=Wbounds,
                srel_bounds=srel_bounds,
                uniform_srel=uniform_srel,
                mpi=mpi,
                logdir=logdir,
                kind=method, 
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
    
    # Extract optimal parameters from last optimization run and assign them to model
    for k, v in opt.items():
        logger.info(f'optimal {k}:\n{v}')
        setattr(model, k, v)

    # Perform stimulus sweep with optimal parameters
    sweep_data = model.run_stim_sweep(amps)

    # Compare results to reference profiles
    rmse = model.evaluate_stim_sweep(
        ref_profiles, sweep_data, norm=norm, 
        disparity_cost_factor=disparity_cost_factor)
    logger.info(f'RMSE = {rmse:.2f}')
