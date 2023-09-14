# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2023-09-13 10:57:27
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-09-13 11:24:49

''' Inter-trial phase clustering (ITPC) vs. sample size with random data.'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
from tqdm import tqdm
from scipy.optimize import curve_fit

from utils import complex_exponential

# Parse arguments
parser = ArgumentParser()
parser.add_argument('--ncrit', type=int, default=-1, help='critical sample size')
parser.add_argument('--nreps', type=int, default=200, help='number of repetitions')
parser.add_argument('--nmax', type=int, default=1000, help='maximum sample size')
args = parser.parse_args()
nreps = args.nreps
ncrit = args.ncrit
nmax = args.nmax

# Define vector of sample sizes
nsamples = np.unique(np.round(np.logspace(0, np.log10(nmax), 100))).astype(int)  

# Compute ITPC for each sample size and repetition
itpc = np.zeros((len(nsamples), nreps))
for i, n in enumerate(tqdm(nsamples)):
    for j in range(nreps):
        phi = np.random.rand(n) * 2 * np.pi - np.pi
        itpc[i, j] = np.abs(np.mean(complex_exponential(phi)))

# Compute summary statistics of ITPC for each sample size
mu_itpc, dev_itpc = np.mean(itpc, axis=1), np.std(itpc, axis=1)
lb_itpc, ub_itpc = mu_itpc - dev_itpc, mu_itpc + dev_itpc

# Compute ITPC vs. sample size linear regression in log-log space
def myfunc(x, a):
    return a * x
popt, _ = curve_fit(myfunc, np.log(nsamples), np.log(mu_itpc), p0=(-0.5))
pow_opt = popt[0]
print(pow_opt)

# Plot ITPC vs. sample size on log scale
fig, ax = plt.subplots()
sns.despine(ax=ax)
ax.set_xlabel('# samples')
ax.set_ylabel('ITPC')
ax.set_title('ITPC vs. sample size')
ax.plot(nsamples, mu_itpc, label='data')
ax.fill_between(nsamples, lb_itpc, ub_itpc, alpha=.5)
ax.plot(nsamples, 1 / np.sqrt(nsamples), ls='--', label='1/sqrt(n) predictor')
ax.plot(nsamples, np.power(nsamples, pow_opt), ls='--', label='fitted predictor')
ax.set_xscale('log')
ax.set_yscale('log')
if ncrit > 0:
    ycrit = np.interp(ncrit, nsamples, mu_itpc)
    print(f'ITPC({ncrit}) = {ycrit:.3f}')
    ax.axvline(ncrit, ls='--', c='k')
    ax.axhline(ycrit, ls='--', c='k', label=f'ITPC({ncrit})')
ax.legend()

plt.show()