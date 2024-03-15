# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2024-03-14 17:56:23
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2024-03-14 17:57:53

import numpy as np
import pandas as pd
from argparse import ArgumentParser

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
ref_profiles = pd.DataFrame(
    data={
        'E': threshold_linear(amps, **fparams.loc['E', :]),
        'PV': threshold_linear(amps, **fparams.loc['PV', :]),
        'SST': threshold_linear(amps, **fparams.loc['E', :])
    },
    index=pd.Index(amps, name='amplitude')
)
ref_profiles.columns.name = 'population'


# Main
if __name__ == '__main__':

    # Parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument(
        '--npops', type=int, choices=(2, 3), default=2, 
        help='Number of populations')
    parser.add_argument(
        '-m', '--method', type=str, choices=('brute', 'diffev'), default='diffev', 
        help='Optimization method')
    parser.add_argument(
        '--npersweep', type=int, default=5, 
        help='Number of points per sweep (for brute-force method)')
    parser.add_argument(
        '--norm', action='store_true',
        help='Normalize profiles prior to comparison')
    args = parser.parse_args()
    npops = args.npops
    method = args.method
    npersweep = args.npersweep
    norm = args.norm

    # If 2 populations selected, remove PV related parameters
    if npops == 2:
        tau = tau.drop('PV')
        fparams = fparams.drop('PV')
        W = W.drop('PV', axis=0).drop('PV', axis=1)
        srel = srel.drop('PV')
        ref_profiles = ref_profiles.drop('PV', axis=1)
    
    # Initialize model
    model = NetworkModel(W=W, tau=tau, fgain=fgain, fparams=fparams)
    logger.info(f'running {method} optimization for {model}')

    # Optimize connectivity matrix to minimize divergence with reference profiles
    Wopt = model.optimize(
        srel,
        ref_profiles, 
        norm=norm,
        logdir=logdir,
        kind=method, 
        npersweep=npersweep
    )

    # Perform stimulus sweep with optimal connectivity matrix
    logger.info(f'optimal connectivity matrix:\n{Wopt}')
    model.W = Wopt
    sweep_data = model.run_stim_sweep(srel, amps)

    # Compare results to reference profiles
    rmse = model.evaluate_stim_sweep(ref_profiles, sweep_data, norm=norm)
    logger.info(f'RMSE = {rmse:.2f}')
