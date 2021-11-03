# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-11-02 09:19:04
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-11-02 11:20:26

import os
import numpy as np
import pandas as pd

from constants import *
from postpro import add_cells_to_table, add_trial_indexes_to_table, add_signal_to_table, add_time_to_table

outdir = '/Users/tlemaire/Desktop/test_dataformat'

# Initial fake data
istep = 0
P = np.array([0.1, 0.2, 0.4, 0.4, 0.4])
DC = np.array([.5, .5, .2, .5, .8])
runIDs = np.arange(DC.size) + 753
np.random.shuffle(runIDs)
duration = 0.2
line = 'line3'
trial_length = 3
fps = 10.  # Hz
ntrials = 2
data = pd.DataFrame({
    LINE_LABEL: line,
    NPERTRIAL_LABEL: trial_length,
    DUR_LABEL: duration,
    FPS_LABEL: fps,
    P_LABEL: P,
    DC_LABEL: DC,
    RUN_LABEL: runIDs,
    NTRIALS_LABEL: ntrials
})
data.index.name = 'run'
data.to_csv(os.path.join(outdir, f'data{istep}.csv'))

# Add ROIs info
istep += 1
nROIs = 10
ROI_IDs = np.arange(nROIs)
data = add_cells_to_table(data, ROI_IDs)
data.to_csv(os.path.join(outdir, f'data{istep}.csv'))

# Add trials info
istep += 1
data = add_trial_indexes_to_table(data)
data.to_csv(os.path.join(outdir, f'data{istep}.csv'))

# Add fluorescence signal
istep += 1
Fdims = (nROIs, len(runIDs), ntrials, trial_length)
F_flattened = np.arange(np.prod(Fdims))
F = np.reshape(F_flattened, Fdims)
print('range:', F.min(), F.max())
print('along cells:', F[:, 0, 0, 0])
print('first cell, along runs:', F[0, :, 0, 0])
print('first cell, first run, along trials:', F[0, 0, :, 0])
print('first cell, first run, frist trial, along time:', F[0, 0, 0, :])
data = add_signal_to_table(data, F_LABEL, F)
assert np.all(data[F_LABEL].values == F_flattened), 'matrix serialization error'
data.to_csv(os.path.join(outdir, f'data{istep}.csv'))

# Add time signal
istep += 1
data = add_time_to_table(data, frame_offset=1)
data.to_csv(os.path.join(outdir, f'data{istep}.csv'))

# Add response type
istep += 1
resp_types = np.random.choice([-1, 0, 1], size=nROIs)
print('response types:', resp_types)
data[RESP_LABEL] = resp_types[data.index.get_level_values('cell')]
data.to_csv(os.path.join(outdir, f'data{istep}.csv'))

# Filter out some ROIs
ncells = 4
while ROI_IDs.size > ncells:
    idx = np.random.randint(ROI_IDs.size)
    ROI_IDs = np.delete(ROI_IDs, idx)