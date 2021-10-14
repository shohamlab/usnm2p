# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-13 11:41:52
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-10-14 19:32:51

from multiprocessing import Value
import numpy as np
import matplotlib.pyplot as plt

from constants import *
from utils import isIterable

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


def plot_x(ax, x, fs, color=None, label=None, alpha=1, avgmode=False):
    '''
    Plot the timecourse of a given variable across 1 / multiple conditions.
    
    :param ax: the axis handle on which to plot the signal(s)
    :param x: 1D (nsamples) or 2D (nconditions, nsamples) signal array
    :param fs: sampling frequency (Hz)
    :param label (optional): legend label (or label list) related to the signal(s)
    :param color (optional): color (or color list) with which to plot the signal(s)
    :param alpha (optional): line opacity (or list of opacities)
    :param avgmode (optional): whether to plot all signals in transparent gray and the average signal in black   
    :return: figure handle
    '''
    # 1D case -> redimension to 2D
    if x.ndim == 1:
        x = np.atleast_2d(x)
        legend = False
        if color is None:
            color = 'k'
    else:
        legend = True
    nsignals, nsamples = x.shape
    if avgmode:
        if nsamples == 1:
            raise ValueError('average mode is not available with 1D signals')
        else:
            color = 'silver'
            alpha = 0.5
            label = None
    # Adapt plotting parameters to 2D case
    if not isIterable(label):
        if label is None:
            legend = False
        label = [label] * nsignals
    if not isIterable(color):
        color = [color] * nsignals
    if not isIterable(alpha):
        alpha = [alpha] * nsignals
    # Plot signals timecourse
    t = np.arange(nsamples) / fs
    for xx, l, c, a in zip(x, label, color, alpha):
        ax.plot(t, xx, color=c, label=l, alpha=a)
    if avgmode:
        ax.plot(t, np.mean(x, axis=0), color='k')
    # Add legend if specified
    if legend:
        ax.legend(frameon=False)


def plot_run(x, fs, ylabel, title=None, mark_stim=False, **kwargs):
    '''
    Plot the timecourse of a given variable along an single (entire) run.
    
    :param x: signal array
    :param fs: sampling frequency (Hz)
    :param ylabel: variable name to use as y-label
    :param title (optional): figure title
    :param mark_stim (optional): boolean indicating whether to indicate stimulus onsets
    :return: figure handle
    '''
    x = np.asarray(x)
    fig, ax = timeseries_template(title=f'full run - {title}', extended=True)
    ax.set_ylabel(ylabel)
    # Add stimulus marks if specified
    if mark_stim:
        add_stim_marks(ax, fs)
    plot_x(ax, x, fs, **kwargs)
    return fig


def plot_response(x, fs, title=None, tbounds=(-5., 300.), tstim=None, **kwargs):
    '''
    Plot signal of individual response(s) to stimulus for a given cell and condition.
    :param x: 1D (npertrial) or 2D (ntrials, npertrial) relative fluorescence signal array
    :return: figure handle
    '''
    fig, ax = timeseries_template(title=f'response - {title}', unit='ms')
    ax.set_ylabel('dF/F0')

    # TODO: add stimulus marks
    plot_x(ax, x, fs * 1e-3, **kwargs)
    if tbounds is not None:
        ax.set_xlim(*tbounds)
    return fig    
