# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2024-03-14 17:56:23
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-08-01 17:18:02

import pandas as pd
from argparse import ArgumentParser
import os
import pandas as pd

from usnm2p.logger import logger
from usnm2p.network_model import *
from usnm2p.model_optimizer import *
from usnm2p.constants import Label, DataRoot
from usnm2p.fileops import get_data_root
from usnm2p.model_params import *

# Set up logging folder
logdir = get_data_root(kind=DataRoot.MODEL)

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
    
    # Source for reference connectivity matrix
    parser.add_argument(
        '--source', type=str, choices=list(sources_dict.keys()), default='allen',
        help='Source of model parameters')
    
    # Number of populations (2 = [E,SST], 3 = [E, SST, PV], 4 = [E, SST, PV, VIP]) 
    parser.add_argument(
        '--npops', type=int, choices=(2, 3, 4), default=4, help='Number of populations')
    
    # Coupling weights
    parser.add_argument(
        '--wmax', type=float, default=None, help='Maximum absolute coupling weight value during exploration')
    parser.add_argument(
        '--relwmax', type=float, default=None, help='Maximum relative deviation of coupling weights from their reference value during exploration')
    
    # Stimulus sensitivities
    parser.add_argument(
        '--srelmax', type=float, default=None, help='Maximal stimulus sensitivity value during exploration')
    parser.add_argument(
        '--uniform-srel', action='store_true', help='Use uniform stimulus sensitivity for all populations')
    
    # Gains
    parser.add_argument(
        '--uniform-gain', action='store_true', help='Use uniform gain function for all populations')
    
    # Optimization method 
    parser.add_argument(
        '-m', '--method', type=str, choices=list(ModelOptimizer.GLOBAL_OPT_METHODS.keys()), 
        default=OPT_METHOD, help='Optimization method')
    
    # Cost function parameters
    parser.add_argument(
        '--norm', type=bool, default=True,
        help='Normalize profiles prior to comparison')
    parser.add_argument(
        '--xdisp', type=float, default=DISPARITY_COST_FACTOR, 
        help='Cost scaling factor to penalize disparity in activity levels across populations')
    parser.add_argument(
        '--xwdev', type=float, default=WDEV_COST_FACTOR, 
        help='Cost scaling factor to penalize deviation from initial weights')
    
    # Execution parameters
    parser.add_argument(
        '--mpi', action='store_true', help='Run with MPI')
    parser.add_argument(
        '--nosave', action='store_true', help='Do not save results in log file')
    parser.add_argument(
        '--nruns', type=int, default=1, help='Number of optimization runs')
    
    # Parse command-line arguments
    args = parser.parse_args()
    source_key = sources_dict[args.source]
    npops = args.npops
    wmax = args.wmax
    relwmax = args.relwmax
    srelmax = args.srelmax
    uniform_srel = args.uniform_srel
    uniform_gain = args.uniform_gain
    method = args.method
    norm = args.norm
    disparity_cost_factor = args.xdisp
    Wdev_cost_factor = args.xwdev
    mpi = args.mpi
    save = not args.nosave
    nruns = args.nruns

    # If nosave option, set logdir to None
    if not save:
        logdir = None

    # Load model parameters
    tau = tau_dict[source_key].copy()
    fparams = fparams_dict[source_key].copy()
    W = W_dict[source_key].copy()

    # Determine populations to keep and remove
    toremove = []
    if npops == 2:
        toremove = ['PV', 'VIP'] 
    elif npops == 3:
        toremove = ['VIP']
    elif npops == 4:
        toremove = []
    else:
        logger.error(f'invalid number of populations: {npops}')

    # Remove inputs for discarded populations
    for k in toremove:
        if k in tau.index:
            tau = tau.drop(k)
        if k in fparams.index:
            fparams = fparams.drop(k)
        if k in W.index:
            W = W.drop(k, axis=0).drop(k, axis=1)
        if k in ref_profiles:
            ref_profiles = ref_profiles.drop(k, axis=1)
    
    # If uniform gain function requested, assign all populations to same values as E
    if uniform_gain:
        for k in fparams.index:
            fparams.loc[k] = fparams.loc['E']
    
    # Initialize model with rescaled parameters
    model = NetworkModel(
        W=W,
        tau=tau, 
        fparams=fparams,
    )

    # If specified, parse coupling weight bounds
    Wbounds = None
    if wmax is not None and relwmax is not None:
        raise ValueError('cannot specify both wmax and relwmax')    
    if wmax is not None:
        Wbounds = model.get_coupling_bounds(wmax=wmax)
    elif relwmax is not None:
        Wbounds = model.get_coupling_bounds(relwmax=relwmax)
    if Wbounds is not None:
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

    # Initialize optimizer object
    optimizer = ModelOptimizer(
        model,
        ref_profiles, 
        method=method,
        norm=norm,
        disparity_cost_factor=disparity_cost_factor, 
        Wdev_cost_factor=Wdev_cost_factor,
    )

    # Optimize model to minimize divergence with reference profiles
    opt = optimizer.optimize(
        Wbounds=Wbounds,
        srel_bounds=srel_bounds,
        uniform_srel=uniform_srel,
        mpi=mpi,
        logdir=logdir,
        nruns=nruns
    )
            
    logger.info('done')
