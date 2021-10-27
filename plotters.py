# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-13 11:41:52
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-10-27 16:03:25

''' Collection of plotting utilities. '''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import seaborn as sns
from colorsys import hsv_to_rgb

from logger import logger
from constants import *
from utils import get_singleton
from postpro import filter_data, get_response_types_per_cell
from viewers import get_stack_viewer


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


def plot_stack_summary(stack, cmap='viridis', title=None):
    '''
    Plot summary imges from a TIF stack.
    :param stack: TIF stack
    :param cmap (optional): colormap
    :return: figure handle
    '''
    plotfuncs = {
        # 'Median': np.median,
        'Mean': np.mean,
        'Standard deviation': np.std,
        'Max. projection': np.max
    }
    fig, axes = plt.subplots(1, len(plotfuncs), figsize=(5 * len(plotfuncs), 5))
    if title is not None:
        fig.suptitle(title)
    fig.subplots_adjust(hspace=10)
    fig.patch.set_facecolor('w')
    for ax, (title, func) in zip(axes, plotfuncs.items()):
        ax.set_title(title)
        ax.imshow(func(stack, axis=0), cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
    return fig


def plot_suite2p_registration_images(output_ops, title=None, cmap='viridis'):
    ''' Plot summary registration images from suite2p processing output.

        :param output_ops: suite2p output
        :return: figure handle    
    '''
    logger.info('plotting suite2p registered images...')    
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    if title is not None:
        fig.suptitle(title)
    # Reference image for registration 
    ax = axes[0]
    ax.imshow(output_ops['refImg'], cmap=cmap)
    ax.set_title('Reference Image for Registration')
    # Maximum of recording over time
    ax = axes[1]
    ax.imshow(output_ops['max_proj'], cmap=cmap)
    ax.set_title("Registered Image, Max Projection")
    # Mean registered image
    ax = axes[2]
    ax.imshow(output_ops['meanImg'], cmap=cmap)
    ax.set_title("Mean registered image")
    # High-pass filtered mean regitered image
    ax = axes[3]
    ax.imshow(output_ops['meanImgE'], cmap=cmap)
    ax.set_title("High-pass filtered Mean registered image")
    return fig


def plot_suite2p_registration_offsets(output_ops, title=None):
    ''' Plot registration offsets over time from suite2p processing output.

        :param output_ops: suite2p output
        :return: figure handle    
    '''    
    logger.info('plotting suite2p registration offsets...')
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
    logger.info('plotting suite2p identified ROIs...')
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
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = axes.flatten()
    fig.suptitle('Morphological parameters - distribution across cells')
    # For each output stats parameter
    is_outlier = np.zeros(len(stats)).astype(bool)
    for ax, pkey in zip(axes, pkeys):
        # Plot histogram distribution
        hide_spines(ax, mode='trl')
        ax.set_xlabel(pkey)
        ax.set_yticks([])
        d = np.array([x[pkey] for x in stats])
        ax.hist(d, bins=20, ec='k', alpha=0.7)
        # If z/score threshold if provided, compute z-score distribution and identify outliers
        if zthr is not None:
            mu, std = d.mean(), d.std()
            lims = [mu + k * zthr * std for k in [-1, 1]]
            for l in lims:
                ax.axvline(l, ls='--', c='silver')
            is_outlier += np.logical_or(d < lims[0], d > lims[1])
    # Hide unused axes
    for ax in axes[len(pkeys):]:
        hide_spines(ax, mode='all')
        ax.set_xticks([])
        ax.set_yticks([])
    # Conditional return
    if zthr is None:
        return fig
    else:
        return fig, is_outlier


def plot_raw_traces(F, title, delimiters=None, ylabel=F_LABEL):
    '''
    Simple function to plot fluorescence traces from a fluorescnece data matrix
    
    :param F: (ntraces, nframes) fluorescence matrix
    :param title: figure title
    :param delimiters (optional): temporal delimitations (shown as vertical lines)
    :param ylabel (optional): y axis label
    :return: figure handle
    '''
    logger.info(f'plotting {F.shape[0]} fluorescence traces...')
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4))
    hide_spines(ax)
    s = {F_LABEL: 'raw fluorescence', REL_F_CHANGE_LABEL: 'normalized fluorescence change'}[ylabel]
    ax.set_title(f'{s} traces across {title} - all {F.shape[0]} cells')
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
    

def plot_cell_map(data, s2p_data, title=None):
    ''' Plot spatial distribution of cells (per response type) on the recording plane.

        :param data: experiment dataframe.
        :param s2p_data: suite2p output dictionary
        :param resp_types: array of response types per cell.
        :param title (optional): figure title
        :return: figure handle
    '''
    rtypes = get_response_types_per_cell(data)
    logger.info('plotting cells map color-coded by response type...')
    # Initialize an RGB image matrix
    im = np.ones((REF_LY, REF_LX, 3), dtype=np.float32)
    # Assign response-type-dependent color to the pixels of each cell
    for i, (stat, rtype) in enumerate(zip(s2p_data['stat'], rtypes)):
        im[stat['ypix'], stat['xpix'], :] = RGB_BY_TYPE[rtype]
    # Render image on figure
    fig, ax = plt.subplots()
    if title is not None:
        ax.set_title(title)
    ax.imshow(im)
    # Add legend
    leg_items = [Line2D(
        [0], [0], label=f'{LABEL_BY_TYPE[k]} ({sum(rtypes == k)})',
        c='none', marker='o', mfc=v, mec='k', ms=10)
        for k, v in RGB_BY_TYPE.items()]
    ax.legend(handles=leg_items, bbox_to_anchor=(1, 1), loc='upper left', frameon=False)
    return fig


def plot_experiment_heatmap(data, key=REL_F_CHANGE_LABEL, title=None, ykey='roi', show_ylabel=True):
    '''
    Plot experiment heatmap (average response over time of each cell, culstered by similarity).
    
    :param data: experiment dataframe.
    :param ykey: one of ('roi', 'cell'), specifying which index to use on the yaxis
    :return: figure handle
    '''
    # Determine rows color labels from response types per cell
    rtypes = get_response_types_per_cell(data).values
    row_colors = [RGB_BY_TYPE[rtype] for rtype in rtypes]
    # Generate 2D table of average dF/F0 response per cell (using roi as index),
    # across runs and trials
    logger.info(f'generating ({ykey} x time) {key} pivot table...')
    avg_resp_per_cell = data.pivot_table(
        index=ykey, columns=TIME_LABEL, values=key, aggfunc=np.mean)
    # Generate cluster map of trial 
    # use Voor Hees (complete) algorithm to cluster based on max distance to force
    # cluster around peak of activity
    # Chebyshev distance just happens to give better resutls than euclidean
    logger.info(f'generating {key} cluster map...')
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


def plot_responses(data, tbounds=None, ykey=REL_F_CHANGE_LABEL, groupby=None, aggfunc='mean', ci=CI,
                   ax=None, mark_stim=True, title=None, **kwargs):
    ''' Plot trial responses of specific sub-datasets.
    
    :param data: experiment dataframe
    :param tbounds (optional): time limits for plot
    :param ykey (optional): key indicating the specific signals to plot on the y-axis
    :param groupby (optional): grouping variable that will produce lines with different colors.
    :param aggfunc (optional): method for aggregating across multiple observations within group. If None, all observations will be drawn.
    :param ci (optional): size of the confidence interval around mean traces (int, “sd” or None) 
    :param ax (optional): figure axis on which to plot
    :param mark_stim (optional): whether to add a stimulus mark on the plot
    :param title (optional): figure title (deduced if not provided)
    :param kwargs: keyword parameters that are passed to the filter_data function
    :return: figure handle
    '''
    # Quick fix for aggfunc
    if aggfunc == 'traces':
        aggfunc = None        
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    hide_spines(ax)
    # Filter data 
    filtered_data, filters = filter_data(data, full_output=True, tbounds=tbounds, **kwargs)
    if groupby is None and aggfunc is None:
        # If only 1 condition and all traces must be plotted -> use custom code
        # to plot all traces and mean
        logger.info('plotting...')
        ax.set_ylabel(ykey)
        aggkeys = list(filter(lambda x: x is not None and x != 'frame', filtered_data.index.names))
        table = filtered_data.pivot_table(
            index=TIME_LABEL,
            columns=aggkeys,
            values=ykey)
        for i, x in enumerate(table):
            table[x].plot(ax=ax, c='silver')
        table.mean(axis=1).plot(ax=ax, c='k')
    else:
        # Otherwise seaborn's lineplot function does the job pretty well
        # Log info on plot sub-processes
        s = []
        if groupby is not None:
            s.append(f'grouping by {groupby}')
        if aggfunc is not None:
            s.append('averaging')
        if ci is not None:
            s.append('estimating confidence intervals')
        s = f'{", ".join(s)} and ' if len(s) > 0 else ''
        logger.info(f'{s}plotting...')
        # Plot
        sns.lineplot(
            data=filtered_data,  # data
            x=TIME_LABEL,  # x-axis
            y=ykey,  # y-axis
            hue=groupby,  # grouping variable
            estimator=aggfunc,  # aggregating function
            ci=ci,  # confidence interval estimator
            ax=ax,  # axis object
            palette='flare',  # color palette
            sort=True,  # sort
            legend='full'  # legend
        )
    # Plot stimulus mark if specified
    if mark_stim:
        ax.axvspan(0, get_singleton(filtered_data, DUR_LABEL), ec=None, fc='C5', alpha=0.5)
    # Adjust time axis if specified
    if tbounds is not None:
        ax.set_xlim(*tbounds)
    # Add title and legend
    if title is None:
        if filters is None:
            filters = {'misc': 'all responses'}
        title = ' - '.join(filters.values())
    ax.set_title(title)
    if groupby is not None:
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, title=groupby, frameon=False)
    # Return figure
    return fig


def plot_mean_evolution(*args, **kwargs):
    ''' Plot the mean-corrected evolution of the average frame intensity '''
    norm = kwargs.pop('norm', True)
    cmap = kwargs.pop('cmap', 'viridis')
    bounds = kwargs.pop('bounds', None)
    ilabels = kwargs.pop('ilabels', None)
    ax = kwargs.pop('ax', None)
    viewer = get_stack_viewer(*args, **kwargs)
    viewer.init_render(norm=norm, cmap=cmap, bounds=bounds, ilabels=ilabels)
    if ax is None:
        fig, ax = plt.subplots()
        hide_spines(ax)
        ax.set_xlabel('frames')
        ax.set_ylabel('mean corrected mean frame intensity')
        leg = False
    else:
        fig = ax.get_figure()
        leg = True
    for header, fobj in zip(viewer.headers, viewer.fobjs):
        mu = viewer.get_mean_evolution(fobj, viewer.frange)
        ax.plot(mu - mu.mean(), label=header)
    if leg:
        ax.legend()
    return fig