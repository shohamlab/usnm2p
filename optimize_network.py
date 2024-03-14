# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2024-03-14 17:56:23
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2024-03-14 17:57:53
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

from logger import logger
from networkmodel import *
from utils import *
from fileops import get_data_root, get_output_equivalent, save_figs_book

# Initialize empty figures dictionary
figs = {}

# Set up dataroot for IO operations
dataroot = get_output_equivalent(get_data_root(), 'raw', 'model')

# Time constants (from Romero-Sosa et al. 2020)
tau = pd.Series(
    data={
        'E': 0.010, 
        'SST': .006
    },
    name='tau (s)'
)
populations = tau.index

# Gain function
fgain = threshold_linear

# Gain function parameters (from Romero-Sosa et al. 2020)
fgain_params = pd.DataFrame(
    data={
        'x0': [
            5,  # E 
            15  # SST
        ],
        'A': [
            1,  # E  
            1.6  # SST
        ],
    },
    index=pd.Index(populations, name='population'),
)

# Connectivity matrix
W = pd.DataFrame(
    data=[
        [3, 12],   # E -> E, SST
        [-1, -2]   # SST -> E, SST
    ], 
    index=pd.Index(populations, name='pre-synaptic'), 
    columns=pd.Index(populations, name='post-synaptic')
)

# Initialize model
model = NetworkModel(W=W, tau=tau, fgain=fgain, fgain_params=fgain_params)

# Check balance of excitation vs inhibition
We = model.get_net_excitation('SST')
Wi = model.get_net_inhibition('SST')
if Wi < We:
    raise ValueError(f'net inhibition strength {Wi} < net excitation strength ({We})')

# Plot model summary
figs['2-pop model'] = model.plot_summary()

# External input
Ithr = fgain_params.loc['E', 'x0']  # E activation threshold
srel = pd.Series(1., index=model.keys, name='external input')  # relative input strength
s = 1.2 * Ithr * srel
logger.info(f'external input:\n{s}')

# Simulate model, and extract steady-state stimulus-evoked activity
data = model.simulate(s=s)
rss = model.extract_steady_state(data)

# Plot results
figs['2-pop timeseries'] = model.plot_timeseries(data, ss=rss, add_synaptic_drive=True)

# Define vector of amplitudes of external input w.r.t. reference value
rel_amps = np.linspace(0, 10, 25)
amps = rel_amps * Ithr

# Simulate model for each amplitude
sweep_data = model.run_stim_sweep(srel, amps)

# Extract steady-state stimulus-evoked activity for each relative amplitude
sweep_rss = model.extract_steady_state(sweep_data)
# Plot steady-state activity dependency on stimulus amplitude
figs['2-pop ss dep norm.'] = model.plot_sweep_results(sweep_rss)

# Define reference population-specific activation profiles 
ref_profiles = pd.DataFrame(
    data={
        'E': threshold_linear(amps, **fgain_params.loc['E', :]),
        'SST': threshold_linear(amps, **fgain_params.loc['E', :])
    },
    index=pd.Index(amps, name='amplitude')
)
ref_profiles.columns.name = 'population'

# Determine whether to normalize activation profiles
norm = True

# Determine loging folder
logdir = dataroot

# Explore/optimize and extract optimal connectivity matrix
# cost = model.explore(srel, ref_profiles, norm=norm, npersweep=3, logdir=logdir)
# Wopt = model.extract_optimal_W(cost)
Wopt = model.optimize(srel, ref_profiles, norm=True, logdir=logdir)

# Perform stimulus sweep with optimal connectivity matrix
logger.info(f'optimal connectivity matrix:\n{Wopt}')
model.W = Wopt
figs['2-pop Wopt'] = model.plot_connectivity_matrix(W=Wopt)
sweep_data = model.run_stim_sweep(srel, amps)
sweep_rss = model.extract_steady_state(sweep_data)

# Compare results to reference profiles
rmse = model.evaluate_stim_sweep(ref_profiles, sweep_data, norm=norm)
sweep_comp = pd.concat({
    'optimum': sweep_rss,
    'reference': ref_profiles
}, axis=0, names=['profile'])
figs['2-pop ss dep comp. norm.'] = model.plot_sweep_results(sweep_comp, norm=norm, style='profile')
ax = figs['2-pop ss dep comp. norm.'].axes[0]
sns.move_legend(ax, bbox_to_anchor=(1, .5), loc='center left', frameon=False)
ax.set_title(f'compariative profiles')
ax.text(0.1, 0.9, f'RMSE = {rmse:.2f}', transform=ax.transAxes, ha='left', va='top');

# Save figures
save_figs_book(figs, logdir, '2-pop model')
