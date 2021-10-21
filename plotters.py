# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-13 11:41:52
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-10-20 22:10:22

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import pandas as pd
from colorsys import hsv_to_rgb

from constants import *
from utils import get_singleton, is_iterable

''' Collection of plotting utilities. '''

def hide_spines_top_right(ax):
    ''' Hide the top and right spines from an axis. '''
    for sk in ['top', 'right']:
        ax.spines[sk].set_visible(False)
    

def hide_ticks(ax, mode='xy'):
    '''
    Hide the ticks and tick labels of an axis.
    
    :param mode: one of 'x', 'y', or 'xy'
    '''
    if 'x' in mode:
        ax.set_xticks([])
    if 'y' in mode:
        ax.set_yticks([])


def plot_stack_summary(stack, cmap='gray'):
    plotfuncs = {
        'Median': np.median,
        'Mean': np.mean,
        'Standard deviation': np.std,
        'Max. projection': np.max
    }
    fig, axes = plt.subplots(1, len(plotfuncs), figsize=(5 * len(plotfuncs), 5))
    fig.subplots_adjust(hspace=10)
    fig.patch.set_facecolor('w')
    for ax, (title, func) in zip(axes, plotfuncs.items()):
        ax.set_title(title)
        sm = ax.imshow(func(stack, axis=0), cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
        pos = ax.get_position()
        # cbarax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.02, pos.height])
        # cbar = plt.colorbar(sm, cax=cbarax)
    return fig


def timeseries_template(title=None, unit='s', extended=False):
    ''' Template for a timeseries figure.
    
        :param title: optional figure title
        :return: (figure handle, axis handle) tuple
    '''
    fig, ax = plt.subplots(figsize=(12, 4) if extended else None)
    hide_spines_top_right(ax)
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


def plot_x(ax, x, fs, toffset=0., tbounds=None, color=None, label=None, alpha=1, avgmode=False):
    '''
    Plot the timecourse of a given variable across 1 / multiple conditions.
    
    :param ax: the axis handle on which to plot the signal(s)
    :param x: 1D (nsamples) or 2D (nconditions, nsamples) signal array
    :param fs: sampling frequency (Hz)
    :param toffset: time vector offset
    :param tbounds: optional time limits
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
    if not is_iterable(label):
        if label is None:
            legend = False
        label = [label] * nsignals
    if not is_iterable(color):
        color = [color] * nsignals
    if not is_iterable(alpha):
        alpha = [alpha] * nsignals
    # Define time vector
    t = np.arange(nsamples) / fs - toffset
    # Restrict vectors to time interval if specified
    if tbounds is not None:
        valid_indexes = np.logical_and(t > tbounds[0], t < tbounds[1])
        t = t[valid_indexes]
        x = x[:, valid_indexes]
        ax.set_xlim(*tbounds)
    # Plot signals timecourse
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


def plot_response2(x, fs, title=None, tbounds=(-2., 8.), ylabel='dF/F0', tstim=None, **kwargs):
    '''
    Plot signal of individual response(s) to stimulus for a given cell and condition.
    :param x: 1D (npertrial) or 2D (ntrials, npertrial) signal array
    :return: figure handle
    '''
    fig, ax = timeseries_template(title=f'response - {title}', unit='s')
    ax.set_ylabel(ylabel)
    if tstim is not None:
        ax.axvspan(0, tstim, ec=None, fc='C0', alpha=0.3)
    plot_x(ax, x, fs, toffset=STIM_FRAME_INDEX / fs, tbounds=tbounds, **kwargs)
    return fig


def plot_response(df, icell, irun, tbounds=(-2., 8.), key='dF/F0', **kwargs):
    '''
    Plot signal of individual response(s) to stimulus for a given cell and condition.
    :param x: 1D (npertrial) or 2D (ntrials, npertrial) signal array
    :return: figure handle
    '''
    fig, ax = timeseries_template(title=f'cell {icell}, run {irun}', unit='s')
    ax.set_ylabel(key)
    ax.axvspan(0, get_singleton(df, DUR_LABEL), ec=None, fc='C0', alpha=0.3)
    # Get cell - run combination, and generate pivot table of time x trials
    data = df.loc[(icell, irun)]
    if tbounds is not None:
        df = df[(df[TIME_LABEL] >= tbounds[0]) & (df[TIME_LABEL] <= tbounds[1])]
        ax.set_xlim(*tbounds)
    data = df.pivot_table(index=TIME_LABEL, columns='trial', values='dF/F0')
    for x in data:
        data[x].plot(ax=ax, c='silver')
    data.mean(axis=1).plot(ax=ax, c='k')
    # plot_x(ax, x, fs, toffset=STIM_FRAME_INDEX / fs, tbounds=tbounds, **kwargs)
    return fig


def plot_suite2p_registration_images(output_ops, title=None):
    ''' Plot summary registration images from suite2p processing output.

        :param output_ops: suite2p output
        :return: figure handle    
    '''    
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    if title is not None:
        fig.suptitle(title)
    
    # Reference image for registration 
    ax = axes[0]
    ax.imshow(output_ops['refImg'], cmap='gray')
    ax.set_title('Reference Image for Registration')
    
    # Maximum of recording over time
    ax = axes[1]
    ax.imshow(output_ops['max_proj'], cmap='gray')
    ax.set_title("Registered Image, Max Projection")
    
    # Mean registered image
    ax = axes[2]
    ax.imshow(output_ops['meanImg'], cmap='gray')
    ax.set_title("Mean registered image")
    
    # High-pass filtered mean regitered image
    ax = axes[3]
    ax.imshow(output_ops['meanImgE'], cmap='gray')
    ax.set_title("High-pass filtered Mean registered image")

    return fig


def plot_suite2p_registration_offsets(output_ops, title=None):
    ''' Plot registration offsets over time from suite2p processing output.

        :param output_ops: suite2p output
        :return: figure handle    
    '''    
    fig, axes = plt.subplots(4, 1, figsize=(18, 8))
    if title is not None:
        fig.suptitle(title)

    # Rigid y-offsets
    ax = axes[0]
    ax.plot(output_ops['yoff'][:1000])
    ax.set_ylabel('rigid y-offsets')

    # Rigid x-offsets
    ax = axes[1]
    ax.plot(output_ops['xoff'][:1000])
    ax.set_ylabel('rigid x-offsets')

    # Non-rigid y-offsets
    ax = axes[2]
    ax.plot(output_ops['yoff1'][:1000])
    ax.set_ylabel('nonrigid y-offsets')

    # Non-rigid x-offsets
    ax = axes[3]
    ax.plot(output_ops['xoff1'][:1000])
    ax.set_ylabel('nonrigid x-offsets')
    ax.set_xlabel('frames')

    for ax in axes:
        hide_spines_top_right(ax)

    return fig


def plot_suite2p_ROIs(data, output_ops, title=None):
    ''' Plot regions of interest identified by suite2p.

        :param data: data dictionary containing contents outputed by suite2p
        :param output_ops: dictionary of outputed suite2p options
        :return: figure handle
    '''
    iscell = data['iscell'][:, 0].astype(int)
    stats = data['stat']

    # Generate ncells random points
    h = np.random.rand(len(iscell))
    hsvs = np.zeros((2, REF_LY, REF_LX, 3), dtype=np.float32)
    
    # Assign a color to each ROIs coordinates
    for i, stat in enumerate(stats):
        # Get x, y pixels and associated mask values of ROI
        ypix, xpix, lam = stat['ypix'], stat['xpix'], stat['lam']
        hsvs[iscell[i], ypix, xpix, 0] = h[i]  # random hue
        hsvs[iscell[i], ypix, xpix, 1] = 1     # fully saturated
        hsvs[iscell[i], ypix, xpix, 2] = lam / lam.max()  # intensity depending on mask value

    # Convert HSV -> RGB space
    rgbs = np.array([hsv_to_rgb(*hsv) for hsv in hsvs.reshape(-1, 3)]).reshape(hsvs.shape)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    if title is not None:
        fig.suptitle(title)

    # Registered image (max projection)
    ax = axes[0]
    ax.imshow(output_ops['max_proj'], cmap='gray')
    ax.set_title('Registered Image, Max Projection')

    # Cells ROIs
    ax = axes[1]
    ax.imshow(rgbs[1])
    ax.set_title(f'Cell ROIs ({np.sum(iscell == 1)})')
    
    # Non-cell ROIs
    ax = axes[2]
    ax.imshow(rgbs[0])
    ax.set_title(f'Non-cell ROIs ({np.sum(iscell == 0)})')

    fig.tight_layout()
    return fig


def plot_zscore_distributions(zmin, zmax):
    '''
    Plot distribution of identified min and max z-scores per cell type.
    
    :param zmin: distribution of minimum z-score negative peak on average response trace per cell 
    :param zmax: distribution of maximum z-score positive peak on average response trace per cell 
    :return: figure handle
    '''
    fig, ax = plt.subplots()
    hide_spines_top_right(ax)
    ax.set_title('average trial response - peak z-scores distributions across cell and conditions')
    ax.set_xlabel('z-score')
    ax.set_ylabel('# cells')
    ax.hist(zmin, label='min peak', fc='C0', ec='k', alpha=0.5)
    ax.hist(zmax, label='max peak', color='C1', ec='k', alpha=0.5)
    ax.axvline(ZSCORE_THR_NEGATIVE, ls='--', c='C0', label='negative thr')
    ax.axvline(ZSCORE_THR_POSITIVE, ls='--', c='C1', label='positive thr')
    ax.set_xlim(-1.5, 1.5)
    ax.legend(frameon=False)
    return fig


def plot_cell_map(stats, df, title=None):
    ''' Plot spatial distribution of cells (per response type) on the recording plane.

        :param stats: suite2p output stats dictionary
        :param df: dataframe contanining all the info about the experiment.
        :param title (optional): figure title
        :return: figure handle
    '''
    # Initialize an RGB image matrix
    im = np.zeros((REF_LY, REF_LX, 3), dtype=np.float32)

    # Extract response types per cell
    resp_types = df.groupby('cell').first()[RESP_LABEL]
        
    # Assign response-type-dependent color to the pixels of each cell
    for i, (stat, rtype) in enumerate(zip(stats, resp_types)):
        im[stat['ypix'], stat['xpix'], :] = RGB_BY_TYPE[rtype]

    # Render image on figure
    fig, ax = plt.subplots()
    if title is not None:
        ax.set_title(title)
    ax.imshow(im)
    # hide_ticks(ax)

    # Add legend
    leg_items = [Line2D(
        [0], [0], label=f'{LABEL_BY_TYPE[k]} ({sum(resp_types == k)})',
        c='none', marker='o', mfc=v, mec='k', ms=10)
        for k, v in RGB_BY_TYPE.items()]
    ax.legend(handles=leg_items, bbox_to_anchor=(1, 1), loc='upper left', frameon=False)

    return fig


def plot_experiment_heatmap(df, key='dF/F0', title=None):
    '''
    Plot experiment heatmap (average response over time of each cell, culstered by similarity).
    
    :param df: dataframe contanining all the info about the experiment.
    :return: figure handle
    '''
    # Determine rows color labels from response types per cell
    resp_types = df.groupby('cell').first()[RESP_LABEL]
    row_colors = pd.Series([RGB_BY_TYPE[rtype] for rtype in resp_types], name=RESP_LABEL)
    # Generate 2D table of average dF/F0 response per cell, across runs and trials
    avg_resp_per_cell = df.pivot_table(
        index='cell', columns=TIME_LABEL, values=key, aggfunc=np.mean)
    # Generate cluster map of trial 
    # use Voor Hees (complete) algorithm to cluster based on max distance to force
    # cluster around peak of activity
    # Chebyshev distance just happens to give better resutls than euclidean
    cg = sns.clustermap(
        avg_resp_per_cell, method='complete', metric='chebyshev', cmap='viridis',
        row_cluster=True, col_cluster=False, row_colors=row_colors)
    # Hide dendogram axes
    cg.ax_row_dendrogram.set_visible(False)
    cg.ax_col_dendrogram.set_visible(False)
    # Set x-axis precision
    cg.ax_heatmap.set_xticklabels([
        f'{float(x.get_text()):.1f}' for x in cg.ax_heatmap.get_xticklabels()])
    # Remove right y-axis labels
    cg.ax_heatmap.set_yticks([])
    cg.ax_heatmap.set_ylabel('')
    # Set new y axis label on the left
    cg.ax_row_colors.set_ylabel('cells')
    # Move colobar directly on the right of the heatmap
    pos = cg.ax_heatmap.get_position()
    cg.ax_cbar.set_position([pos.x1 + .02, pos.y0, .05, pos.y1 - pos.y0])
    # Add colormap title
    cg.ax_cbar.set_title(key)
    # Add heatmap title, if any
    if title is not None:
        cg.ax_heatmap.set_title(title)
    return cg
