# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2023-07-03 11:28:32
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-07-03 17:25:34

''' Testing spectrum package functionalities for spectral analysis. '''

import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
# import spectrum
import pandas as pd
from postpro import get_power_spectrum
from constants import Label
# from scipy.signal import welch, periodogram

# Create time vector
fsampling = 3
nsamples = 20000
t = np.arange(nsamples) / fsampling

# Create input signal made of two sinus and an additive gaussian noise
f1 = 0.305
f2 = 0.2
A1 = 1
A2 = 0.5
Anoise = 0.1
y = A1 * np.cos(2 * np.pi * f1 * t) + A2 * np.sin(2 * np.pi * f2 * t) + Anoise * np.random.rand(nsamples)

# Blank signal in defined window
# y[500:1000] = 0

# Create dataframe to hold power ratios
ratios = pd.DataFrame(columns=['power ratio', 'dB'])

# Compute ground-truth amplitude ratio
pratio = (A1 / A2)**2
ratios.loc['ground truth'] = [pratio, 10 * np.log10(pratio)]

# Convert signal to series
y = pd.Series(y)

# List of all methods included in wrapper
methods = [
    'welch',
    'fft',
    'periodogram',
    'pma',
    'pyule',
    'pburg',
    'pcovar',
    'pmodcovar',
    'pcorrelogram',
    'pminvar',
    'pmusic',
    'pev',
    'mtap'
]

# Compute PSDs for all methods
PSDs = {
    method: get_power_spectrum(y, fsampling, method=method, scaling='density')
    for method in methods
}

# For each PSD measure, estimate amplitude ratio between f1 and f2
for k, df in PSDs.items():
    psd1, psd2 = [np.interp(f, df[Label.FREQ], df[f'{Label.PSPECTRUM} density']) for f in [f1, f2]]
    psd_ratio = psd1 / psd2
    ratios.loc[k] = [psd_ratio, 10 * np.log10(psd_ratio)]

# Print ratios
print(ratios)

# Create figure
fig, axes = plt.subplots(2, 1, figsize=(8, 6))
sns.despine(fig=fig)

# Plot input signal
ax = axes[0]
ax.plot(t, y)
ax.set_title('Input signal')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')

# Plot all PSDs
colors = sns.color_palette('husl', len(PSDs))
if len(PSDs) > 6:
    styles = ['-', '--'] * (len(PSDs) // 2) + ['-']
else:
    styles = ['-'] * len(PSDs)
ax = axes[1]
ax.set_title('PSDs')
ax.grid(True)
for (key, df), color, ls in zip(PSDs.items(), colors, styles):
    freqs, psd = df[Label.FREQ], df[f'{Label.PSPECTRUM} density']
    psd[psd <= 0] = np.nan
    ax.plot(freqs, 10 * np.log10(psd), label=key, c=color, ls=ls)
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('PSD [dB]')
ax.set_ylim([-80, ax.get_ylim()[1]])
for f in [f1, f2]:
    ax.axvline(f, color='k', linestyle='--')
ax.legend()

# Adjust layout and show
sns.move_legend(ax, loc='upper right', bbox_to_anchor=(1.2, 1.0))
fig.tight_layout()
plt.show()