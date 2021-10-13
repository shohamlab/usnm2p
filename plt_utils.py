import numpy as np
import matplotlib.pyplot as plt

from constants import *
from utils import *


''' Collection of plotting utilities. '''

def plot_stack_summary(stack, cmap='gray'):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.subplots_adjust(hspace=10.9)
    fig.patch.set_facecolor('w')
    titles = ['Median value', 'Standard deviation']
    funcs = [np.median, np.std]
    for ax, title, func in zip(axes, titles, funcs):
        ax.set_title(title)
        sm = ax.imshow(func(stack, axis=0), cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
        pos = ax.get_position()
        cbarax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.02, pos.height])
        cbar = plt.colorbar(sm, cax=cbarax)
    return fig


def timeseries_template(title=None, unit='s', extended=False):
    ''' Template for a timeseries figure.
    
        :param title: optional figure title
        :return: (figure handle, axis handle) tuple
    '''
    fig, ax = plt.subplots(figsize=(12, 4) if extended else None)
    for sk in ['top', 'right']:
        ax.spines[sk].set_visible(False)
    ax.set_xlabel(f'time ({unit})')
    if title is not None:
        ax.set_title(title)
    return fig, ax


def add_stim_marks(ax, fs):
    ''' Add vertical line on a timeseries plot indicating the stimulus onsets. '''
    ionsets = np.arange(NTRIALS_PER_RUN) * REF_NFRAMES // NTRIALS_PER_RUN + STIM_FRAME_INDEX
    tonsets = ionsets / fs
    for ton in tonsets:
        ax.axvline(ton, c='silver')


def plot_run(x, fs, ylabel, label=None, color=None, title=None, mark_stim=False):
    '''
    Plot the timecourse of a given variable along an single (entire) run.
    
    param x: 1D (nframes) or 2D (nconditions, nframes) signal array
    :param fs: sampling frequency (Hz)
    :param ylabel: variable name to use as y-label
    :param mark_stim: boolean indicating whether to indicate stimulus onsets
    :return: figure handle
    '''
    x = np.asarray(x)
    fig, ax = timeseries_template(title=f'full run - {title}', extended=True)
    ax.set_ylabel(ylabel)
    # Add stimulus marks if specified
    if mark_stim:
        add_stim_marks(ax, fs)
    # Plot signal timecourse
    if x.ndim == 1:
        ax.plot(np.arange(x.size) / fs, x, color=color, label=label)
    else:
        legend = True
        nsignals, nframes = x.shape
        if label is None:
            label = [None] * nsignals
            legend = False
        if color is None:
            color = [None] * nsignals
        for xx, l, c in zip(x, label, color):
            ax.plot(np.arange(nframes) / fs, xx, color=c, label=l)
        if legend:
            ax.legend(frameon=False)
    return fig


def plot_trial(dFF_pertrial, fs, title=None):
    '''
    Plot the relative fluorescence traces across trials for a given cell and condition.
    
    :param dFF_pertrial: 2D (ntrials, npertrial) fluorescence signal array
    :return: figure handle
    '''
    dFF_trialavg = np.mean(dFF_pertrial, axis=0)
    t = np.arange(dFF_trialavg.size) / fs  # in s
    fig, ax = plt.subplots()
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('dF/F0')
    for x in dFF_pertrial:
        ax.plot(t, x, color='gray')
    ax.plot(t, dFF_trialavg, color='k')
    return fig
