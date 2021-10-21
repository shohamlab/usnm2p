# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-13 11:41:52
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-10-21 17:55:05

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import seaborn as sns
from colorsys import hsv_to_rgb

from constants import *
from utils import get_singleton
from postpro import array_to_dataframe

''' Collection of plotting utilities. '''

def hide_spines(ax, mode='tr'):
    '''
    Hide specific spines from an axis.
    
    :param mode: string code specifying which spines to hide
    '''
    codes = {
        't': 'top',
        'r': 'right',
        'b': 'bottom',
        'l': 'left'
    }
    if mode == 'all':
        mode = 'trbl'
    for c in mode:
        ax.spines[codes[c]].set_visible(False)
    

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
    '''
    Plot summary imges from a TIF stack.
    :param stack: TIF stack
    :param cmap (optional): colormap
    :return: figure handle
    '''
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
        # pos = ax.get_position()
        # cbarax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.02, pos.height])
        # cbar = plt.colorbar(sm, cax=cbarax)
    return fig


def plot_raw_traces(F, title, delimiters=None, ylabel='F (a.u.)'):
    '''
    Plot raw fluorescence traces
    
    :param F: (ntraces, nframes) fluorescence matrix
    :param title: figure title
    :param delimiters (optional): temporal delimitations (shown as vertical lines)
    :param ylabel (optional): y axis label
    :return: figure handle
    '''
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4))
    hide_spines(ax)
    ax.set_title(f'raw fluorescence traces across {title} - all {F.shape[0]} cells')
    ax.set_xlabel('frames')
    ax.set_ylabel(ylabel)
    # Plot each trace
    for trace in F:
        ax.plot(trace)
    # Plot delimiters, if any
    if delimiters is not None:
        for iframe in delimiters:
            ax.axvline(iframe, color='k', linestyle='--')
    # Return
    return fig


def plot_trial_response(df, ax=None, title=None, tbounds=(-2., 8.), ykey='dF/F0', tstim=None, aggkeys=None):
    '''
    Plot signal of trial response(s) to stimulus for a given cell and condition.

    :param df: dataframe continaing the specific trials to plot (or optionally a 2D array).
    :param ax (optional): figure axis
    :param title (optional): axis title
    :param tbounds (optional): time limits
    :param ykey: key indicating the specific signals to plot on the y-axis
    :param tstim (optional): stimulus duration (s)
    :return: figure handle
    '''
    # If input is a 2D array -> create corresponding dataframe on the fly 
    if isinstance(df, np.ndarray):
        df = array_to_dataframe(df, ykey)
    if aggkeys is None:
        aggkeys = ['trial']
    # Create figure backbone
    if ax is not None:
        fig = ax.get_figure()
    else: 
        fig, ax = plt.subplots()
    hide_spines(ax)
    ax.set_ylabel(ykey)
    # Parse title
    title = '' if title is None else f'{title} - '
    # Add stimulus mark
    if tstim is None:
        tstim = get_singleton(df, DUR_LABEL)
    ax.axvspan(0, tstim, ec=None, fc='C0', alpha=0.3)
    # Restrict data to specific time interval (if provided)
    if tbounds is not None:
        df = df[(df[TIME_LABEL] >= tbounds[0]) & (df[TIME_LABEL] <= tbounds[1])]
        ax.set_xlim(*tbounds)
    # Generate (time, trials) table
    table = df.pivot_table(index=TIME_LABEL, columns=aggkeys, values=ykey)
    # Plot signals from each trial
    for i, x in enumerate(table):
        table[x].plot(ax=ax, c='silver')
    ax.set_title(f'{title}{i + 1} traces')
    # Plot average signal
    table.mean(axis=1).plot(ax=ax, c='k')
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
    # Hide spines and return
    for ax in axes:
        hide_spines(ax)
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
    # Create figure
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
    # Tighten and return
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
    hide_spines(ax)
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


def plot_cell_map(stats, resp_types, title=None):
    ''' Plot spatial distribution of cells (per response type) on the recording plane.

        :param stats: suite2p output stats dictionary
        :param resp_types: array of response types per cell.
        :param title (optional): figure title
        :return: figure handle
    '''
    # Initialize an RGB image matrix
    im = np.zeros((REF_LY, REF_LX, 3), dtype=np.float32)
    # Assign response-type-dependent color to the pixels of each cell
    for i, (stat, rtype) in enumerate(zip(stats, resp_types)):
        im[stat['ypix'], stat['xpix'], :] = RGB_BY_TYPE[rtype]
    # Render image on figure
    fig, ax = plt.subplots()
    if title is not None:
        ax.set_title(title)
    ax.imshow(im)
    # Add legend
    leg_items = [Line2D(
        [0], [0], label=f'{LABEL_BY_TYPE[k]} ({sum(resp_types == k)})',
        c='none', marker='o', mfc=v, mec='k', ms=10)
        for k, v in RGB_BY_TYPE.items()]
    ax.legend(handles=leg_items, bbox_to_anchor=(1, 1), loc='upper left', frameon=False)

    return fig


def plot_parameter_distributions(stats, pkeys, zthr=None):
    '''
    Plot distributions of several morphological parameters (extracted from suite2p output)
    across cells.
    
    :param stats: suite2p output stats dictionary
    :param pkeys: list of parameters to considers
    :param zthr: threshold z-score (number of standard deviations from the mean) used to identify outliers
    :return: figure handle, optionally with list of identified outliers indexes
    '''
    # Determine figure gird organization based on number of parameters
    ncols = min(len(pkeys), 4)
    nrows = int(np.ceil(len(pkeys) / ncols))
    # Create figure
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 4 * nrows))
    axes = axes.flatten()
    fig.suptitle('Morphological parameters - distribution across cells')
    # For each output stats parameter
    ioutliers = []
    for ax, pkey in zip(axes, pkeys):
        # Plot histogram distribution
        hide_spines(ax, mode='trl')
        ax.set_xlabel(pkey)
        ax.set_yticks([])
        d = np.array([x[pkey] for x in stats])
        ax.hist(d, bins=30)
        # If z/score threshold if provided, compute z-score distribution and identify outliers
        if zthr is not None:
            mu, std = d.mean(), d.std()
            lims = [mu + k * zthr * std for k in [-1, 1]]
            for l in lims:
                ax.axvline(l, ls='--', c='silver')
            is_outlier = np.logical_or(d < lims[0], d > lims[1])
            new_out = is_outlier.nonzero()[0].tolist()
            ioutliers += is_outlier.nonzero()[0].tolist()
    # Hide unused axes
    for ax in axes[len(pkeys):]:
        hide_spines(ax, mode='all')
        ax.set_xticks([])
        ax.set_yticks([])
    # Conditional return
    if zthr is None:
        return fig
    else:
        return fig, list(set(ioutliers))


def plot_types_sequence(rtypes):
    '''
    Plot a line showing color-coded response type per cell.
    
    :param rtypes: list of response types per cell.
    :return: figure handle
    '''
    fig, ax = plt.subplots(figsize=(12, 1))
    ax.imshow(
        np.array([rtypes]), interpolation='nearest',
        cmap=ListedColormap(list(RGB_BY_TYPE.values())))
    ax.set_aspect('auto')
    return fig


def plot_experiment_heatmap(df, key='dF/F0', title=None, ykey='roi', show_ylabel=True):
    '''
    Plot experiment heatmap (average response over time of each cell, culstered by similarity).
    
    :param df: dataframe contanining all the info about the experiment.
    :param ykey: one of ('roi', 'cell'), specifying which index to use on the yaxis
    :return: figure handle
    '''
    # Determine rows color labels from response types per cell
    resp_types = df.groupby(ykey).first()[RESP_LABEL].values
    row_colors = [RGB_BY_TYPE[rtype] for rtype in resp_types]
    # Generate 2D table of average dF/F0 response per cell (using roi as index),
    # across runs and trials
    avg_resp_per_cell = df.pivot_table(
        index=ykey, columns=TIME_LABEL, values=key, aggfunc=np.mean)
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
    # Hide y-labelling if specified
    if not show_ylabel:
        cg.ax_heatmap.set_ylabel('')
        cg.ax_heatmap.set_yticks([])
    # Set new y axis label on the left
    cg.ax_row_colors.set_ylabel(RESP_LABEL)
    # Move colobar directly on the right of the heatmap
    pos = cg.ax_heatmap.get_position()
    cg.ax_cbar.set_position([pos.x1 + .1, pos.y0, .05, pos.y1 - pos.y0])
    # Add colormap title
    cg.ax_cbar.set_title(key)
    # Add heatmap title, if any
    if title is not None:
        cg.ax_heatmap.set_title(title)
    return cg
