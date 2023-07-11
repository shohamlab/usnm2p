# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2023-07-10 16:18:26
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-07-10 16:19:30

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import complex_exponential


def plot_phase_dist(phases, title=None, binwidth=np.pi / 16, ax=None):
    '''
    Plot distribution of phases and mean resultant vector

    :param phases: array-like of phases
    :param title (optional): title for plot
    :param binwidth (optional): bin width for histogram
    :param ax (optional): axis to plot on
    :return: figure handle
    '''
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    else:
        fig = ax.figure
    sns.histplot(
        ax=ax,
        data=phases,
        binwidth=binwidth,
        ec=None,
        stat='density',
    )
    c = np.mean(complex_exponential(phases))
    theta, r = np.angle(c), np.abs(c)
    ax.plot([0, theta], [0, r], c='k', lw=2)
    ax.scatter(theta, r, c='k')
    ax.text(theta, r + .1, f'r = {r:.2f}', ha='left' if np.cos(theta) >= 0 else 'right', va='bottom')
    ax.plot(np.linspace(0, 2 * np.pi, 100), np.ones(100), c='dimgray', ls='--', lw=2)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_rlim(bottom=0, top=2)
    ax.set_ylabel('')
    if title is not None:
        ax.set_title(title)
    return fig

# Plot various phase distributions
n = 16
phasedict = {
    'random': np.random.rand(n) * 2 * np.pi,
    'coherent': np.random.normal(0, np.pi / 12, n),
}
fig, axes = plt.subplots(1, len(phasedict), figsize=(4 * len(phasedict), 4), subplot_kw={'projection': 'polar'})
for ax, (k, phases) in zip(axes, phasedict.items()):
    fig = plot_phase_dist(phases, title=k, ax=ax)

# Save figure
fig.savefig('/Users/tlemaire/Desktop/talk figures/phase_dist_examples.pdf')