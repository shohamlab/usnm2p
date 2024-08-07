# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2024-04-30 17:17:57
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2024-08-07 15:58:47

import os
import numpy as np
import pandas as pd

from .fileops import get_data_root, get_output_equivalent

''' Collection of model parameters gathered from the literature. '''

# ---------------------------- Load Allen data ----------------------------

# Define input data directory
datadir = get_output_equivalent(get_data_root(), 'raw', 'model')

# Parse ephys features (experimental, cortex V1 layer 2/3)
Allen_ephys_fpath = os.path.join(datadir, 'Allen_ephys_features.csv')
Allen_ephys_features = pd.read_csv(Allen_ephys_fpath).rename(columns={
    'cell type': 'population', 
    'membrane time constant (ms)': 'tau (ms)',
    'F/I threshold (pA)': 'x0',
    'F/I slope (Hz/pA)': 'A',
}).set_index('population')

# Parse coupling strength matrix (experimental, cortex V1 layer 2/3)
datadir = get_output_equivalent(get_data_root(), 'raw', 'model')
Wmat_fpath = os.path.join(datadir, 'Allen_Wmat.csv')
W_AllenInstitute = pd.read_csv(Wmat_fpath, index_col=0)
pops_mapper = {
    'L2/3 Pyr': 'E',
    'L2/3 Pv': 'PV',
    'L2/3 Sst': 'SST',
}
W_AllenInstitute.index = W_AllenInstitute.index.map(pops_mapper)
W_AllenInstitute.columns = W_AllenInstitute.columns.map(pops_mapper).rename('post-synaptic')

# -------------------------------- Populations -------------------------------   
POPULATIONS = ['E', 'PV', 'SST'] 

# ------------------------------ Time constants ------------------------------

tau_key = 'tau (ms)'

# Romero-Sosa et al. 2020, (experimental, cortex)
tau_RomeroSosa = pd.Series(
    data={
        'E': 10., 
        'PV': 4., 
        'SST': 6.,
    },
    name=tau_key
)
tau_RomeroSosa.index.name = 'population'

# Allen Institute database (experimental, cortex V1 layer 2/3)
tau_AllenInstitute = Allen_ephys_features[tau_key]

# Assemble into dictionary
tau_dict = {
    'Romero-Sosa 2020': tau_RomeroSosa,
    'Allen Institute': tau_AllenInstitute
}

# -------------------------- Gain function parameters --------------------------

fparams_idx = pd.Index(POPULATIONS, name='population')

# Romero-Sosa et al. 2020, (experimental, cortex)
fparams_RomeroSosa = pd.DataFrame(data={
    'x0': [5, 30, 15],  # E, PV, SST
    'A': [1, 2.7, 1.6], # E, PV, SST
}, index=fparams_idx)

# Allen Institute database (experimental, cortex V1 layer 2/3)
fparams_AllenInstitute = Allen_ephys_features[['x0', 'A']]

# Assemble into dictionary
fparams_dict = {
    'Romero-Sosa 2020': fparams_RomeroSosa,
    'Allen Institute': fparams_AllenInstitute,
}

# --------------------------- Connectivity matrices ---------------------------   

presyn_idx = pd.Index(POPULATIONS, name='pre-synaptic')
postsyn_idx = pd.Index(POPULATIONS, name='post-synaptic')
Wkwargs = dict(index=presyn_idx, columns=postsyn_idx)

# Pfeffer et al. 2013 (experimental, cortex V1, no E -> x data)
W_Pfeffer = pd.DataFrame(data=[
    [np.nan, np.nan, np.nan], # E -> E, PV, SST 
    [-1, -1.01, -0.03],  # PV -> E, PV, SST
    [-0.54, -0.33, -0.02]  # SST -> E, PV, SST
], **Wkwargs)

# Plaksin et al. 2016 (modeling, based on cortex S1 data, [RS, FS LTS] in lieu of [E, PV, SST])
W_Plaksin = pd.DataFrame(data=[
    [0.002, 0.04, 0.09],      # E -> E, PV, SST 
    [-0.015, -0.135, -0.86],  # PV -> E, PV, SST
    [-0.135, -0.02, 0]        # SST -> E, PV, SST
], **Wkwargs)

# Park et al. 2020 (auditory cortex, based on Pfeffer 2013 with additional E -> x data)
W_Park = pd.DataFrame(data=[
    [1.1, 1, 6],  # E -> E, PV, SST 
    [-2, -2, 0],  # PV -> E, PV, SST
    [-1, -2, 0]   # SST -> E, PV, SST
], **Wkwargs)

# Antonoudiou et al. 2020 (modeling, hippocampus)
W_Antonoudiou = pd.DataFrame(data=[
    [10, 30, 10],   # E -> E, PV, SST 
    [-15, -10, 0],  # PV -> E, PV, SST
    [-15, 0, -10]   # SST -> E, PV, SST
], **Wkwargs)

# Romero-Sosa et al. 2020, Figure 6A,B (modeling, cortex)
W_RomeroSosa = pd.DataFrame(data=[
    [7, 14, 12],   # E -> E, PV, SST 
    [-1.5, -1.5, -1],  # PV -> E, PV, SST
    [-0.5, -1, -2]   # SST -> E, PV, SST
], **Wkwargs)

# Richter et al. 2022 (modeling, cortex S1, with uniform gain and external inputs)
W_Richter = pd.DataFrame(data=[
    [0.1, 0.1, 0.1],   # E -> E, PV, SST 
    [-0.8, -0.8, 0],  # PV -> E, PV, SST
    [-1.6, -1.6, 0]   # SST -> E, PV, SST
], **Wkwargs)

# Assemble into dictionary
W_dict = {
    'Pfeffer 2013': W_Pfeffer,
    'Plaksin 2016': W_Plaksin,
    'Antonoudiou 2020': W_Antonoudiou,
    'Park 2020': W_Park,
    'Romero-Sosa 2020': W_RomeroSosa,
    'Richter 2022': W_Richter,
    'Allen Institute': W_AllenInstitute, 
}