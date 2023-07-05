# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2023-07-05 17:26:30
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-07-05 17:41:19

''' Test circular correlation function for various types of dependent data. '''

import numpy as np
import matplotlib.pyplot as plt
from postpro import circ_corrcl
from scipy.stats import chi2

# Define functions to generate dependent data
funcs = {
    'rand[-1, 1]': lambda x: np.random.rand(x.size) * 2 - 1,
    'sin': lambda x: np.sin(x),
    'sin + rand[-1, 1]': lambda x: np.sin(x) + np.random.rand(x.size) * 2 - 1,
}

# Input parameters
nvec = np.logspace(0, 3.5, 30).round().astype(int)  # vector of sample size
nreps = 100  # number of repetitions per n

distfig, distax = plt.subplots(subplot_kw={'projection': 'polar'})
trendfig, (ax1, ax2) = plt.subplots(2, 1)
ax1.set_xscale('log')
ax2.set_xscale('log')
ax1.set_ylabel('rho')
ax2.set_ylabel('p')
ax2.set_xlabel('# samples')
ax2.set_ylim(-0.05, 1.05)

# For each function
for k, func in funcs.items():
    print(f'testing circular correlation with {k} dependent data')

    # Compute circular correlation for each sample size, averaged over repetitions
    rhovsn = np.zeros(nvec.size)
    pvsn = np.zeros(nvec.size)
    for i, n in enumerate(nvec):
        rhos = []
        for _ in range(nreps):
            # Generate random input phase vector
            x = np.random.rand(n) * 2 * np.pi - np.pi
            # Generate dependent output vector
            y = func(x)
            # Compute circular correlation
            rho, _ = circ_corrcl(x, y)
            # Append to list 
            rhos.append(rho)
        # Compute mean correlation over repetitions
        rhovsn[i] = np.mean(rhos)
        # Compute p-value
        pvsn[i] = 1 - chi2(2).cdf(n * rhovsn[i]**2)

    # Plot results
    ax1.plot(nvec, rhovsn, label=k)
    ax2.plot(nvec, pvsn, label=k)

    # Plot data distribution for largest sample size
    distax.scatter(x, y)

ax1.legend()

plt.show()