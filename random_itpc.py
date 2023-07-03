# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2023-07-03 16:03:49
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-07-03 16:39:33

''' Testing ITPC vs. number of trials for random phase values. '''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm


def itpc(y, axis=None):
    ''' Compute inter-trial phase coherence (ITPC) for a set of trials. '''
    # Apply Euler's formula to convert phase values to complex numbers
    c = np.exp(1j * y)
    # Compute complex mean across trials
    cmean = np.mean(c, axis=axis)
    # Compute and return ITPC
    return np.abs(cmean)


# Vector of trial number
ntrials = np.arange(1, 101)

# Number of repetitions for each number of trials
nreps = 1000

# Dataframe to hold ITPC values
itpc_vs_n = pd.DataFrame(columns=['mu', 'se'], index=ntrials, dtype=float)

# For each number of trials, compute ITPC
for nt in tqdm(ntrials):
    # For each repetition, generate a set of random phase values and compute ITPC
    itpc_vals = []
    for i in range(nreps):
        phases = np.random.rand(nt) * 2 * np.pi
        itpc_vals.append(itpc(phases))
    # Compute mean and std error across repetitions
    itpc_vs_n.loc[nt] = [np.mean(itpc_vals), np.std(itpc_vals) / np.sqrt(nreps)]

# Compute confidence interval upper and lower bounds
itpc_vs_n['lb'] = itpc_vs_n['mu'] - itpc_vs_n['se']
itpc_vs_n['ub'] = itpc_vs_n['mu'] + itpc_vs_n['se']

# Plot
fig, ax = plt.subplots()
sns.despine(fig=fig)
ax.set_xlabel('Number of trials')
ax.set_ylabel('ITPC')
ax.plot(itpc_vs_n.index, itpc_vs_n['mu'])
ax.fill_between(itpc_vs_n.index, itpc_vs_n['lb'], itpc_vs_n['ub'], alpha=0.5)
ax.set_title('ITPC vs. # trials')

# Compute IPDC for reference number of trials
ntrials_ref = 16
itpc_ref = itpc_vs_n.loc[ntrials_ref]
print(f'ITPC for {ntrials_ref} trials {itpc_ref["mu"]:.3f} +/- {itpc_ref["se"]:.3f}')
ax.axvline(ntrials_ref, c='k', ls='--')
ax.axhline(itpc_ref['mu'], c='k', ls='--')

plt.show()



