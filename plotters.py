# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-13 11:41:52
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-11-05 17:08:15

''' Collection of plotting utilities. '''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import seaborn as sns
from colorsys import hsv_to_rgb
from seaborn.relational import lineplot
from tqdm import tqdm

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


def plot_stack_timecourse(stack, func='mean', ax=None, title=None, label=None):
    func_obj = {
        'mean': np.mean,
        'median': np.median,
        'min': np.min,
        'max': np.max
    }[func]
    return func_obj(stack, axis=(-2, -1))
    if ax is None:
        fig, ax = plt.subplots()
        title = '' if title is None else f'{title} - '
        ax.set_title(f'{title} timecourse')
        ax.set_xlabel('frames')
        ax.set_ylabel(f'{func} frame intensity')
        hide_spines(ax)
    else:
        fig = ax.get_figure()
    ax.plot(func_obj(stack, axis=(-2, -1)), label=label)
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
    hsvs = np.zeros((2, output_ops['Ly'], output_ops['Lx'], 3), dtype=np.float32)
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
    

def plot_parameter_distributions(data, pkeys, zthr=None):
    '''
    Plot distributions of several morphological parameters (extracted from suite2p output)
    across cells.
    
    :param data: suite2p output dictionary
    :param pkeys: list of parameters to considers
    :param zthr: threshold z-score (number of standard deviations from the mean) used to identify outliers
    :return: figure handle, optionally with a dataframe summarizing identified outliers
    '''
    # Determine figure gird organization based on number of parameters
    ncols = min(len(pkeys), 4)
    nrows = int(np.ceil(len(pkeys) / ncols))
    # Create figure
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = axes.flatten()
    fig.suptitle('Morphological parameters - distribution across cells')
    # Get cells IDs from s2p data
    is_cell = data['iscell'][:, 0]
    icells = is_cell.nonzero()[0]
    cellIDs = data[ROI_LABEL][icells]
    # Initialize outliers dataframe with cells ROI IDs
    df_outliers = pd.DataFrame({ROI_LABEL: cellIDs})
    # Fetch stats dictionary for cells only
    cell_stats = data['stat'][cellIDs]
    # For each output stats parameter
    for ax, pkey in zip(axes, pkeys):
        # Plot histogram distribution
        hide_spines(ax, mode='trl')
        ax.set_xlabel(pkey)
        ax.set_yticks([])
        d = np.array([x[pkey] for x in cell_stats])
        ax.hist(d, bins=20, ec='k', alpha=0.7)
        # If z-score threshold if provided, compute z-score distribution and identify outliers
        if zthr is not None:
            mu, std = d.mean(), d.std()
            lims = [mu + k * zthr * std for k in [-1, 1]]
            for l in lims:
                ax.axvline(l, ls='--', c='silver')
            df_outliers[pkey] = np.logical_or(d < lims[0], d > lims[1])
    # Hide unused axes
    for ax in axes[len(pkeys):]:
        hide_spines(ax, mode='all')
        ax.set_xticks([])
        ax.set_yticks([])
    # Conditional return
    if zthr is None:
        return fig
    else:
        # Set ROI IDs as dataframe index
        df_outliers = df_outliers.set_index(ROI_LABEL)
        # Reduce dataframe to only outliers cells
        df_outliers = df_outliers[df_outliers[pkeys].sum(axis=1) == 1]
        return fig, df_outliers


def plot_raw_traces(F, title, delimiters=None, ylabel=F_LABEL, labels=None, alpha=1., ybounds=None, cmap=None,
                    ionset=0):
    '''
    Simple function to plot fluorescence traces from a fluorescnece data matrix
    
    :param F: (ntraces, nframes) fluorescence matrix
    :param title: figure title
    :param delimiters (optional): temporal delimitations (shown as vertical lines)
    :param ylabel (optional): y axis label
    :param alpha (optional): opacity factor
    :param ybounds (optional): y-axis limits
    :return: figure handle
    '''
    if F.ndim == 1:
        F = np.atleast_2d(F)
    logger.info(f'plotting {F.shape[0]} fluorescence trace(s)...')
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4))
    hide_spines(ax)
    s = {
        F_LABEL: 'raw fluorescence',
        REL_F_CHANGE_LABEL: 'normalized fluorescence change',
        STACK_AVG_INT_LABEL: 'average stack intensity'
    }[ylabel]
    leg = True
    if labels is None:
        labels = [None] * F.shape[0]
        leg = False
        title = f'{title} - all {F.shape[0]} cells'
    ax.set_title(f'{s} trace(s) across {title}')
    ax.set_xlabel('frames')
    ax.set_ylabel(ylabel)
    # Determine traces colors
    if cmap is not None:
        colors = plt.get_cmap(cmap)(np.linspace(0, 1, F.shape[0]))
    else:
        colors = [None] * F.shape[0]
    # Plot each trace
    inds = np.arange(F.shape[-1]) + ionset
    for trace, label, color in zip(F, labels, colors):
        ax.plot(inds, trace, c=color, label=label, alpha=alpha)
    # Restrict y axis, if needed
    if ybounds is not None:
        ax.set_ylim(*ybounds)
    # Add legend if labels were provided
    if leg:
        ax.legend(frameon=False)
    # Plot delimiters, if any
    if delimiters is not None:
        logger.info(f'adding {len(delimiters)} delimiters')
        for iframe in delimiters:
            ax.axvline(iframe, color='k', linestyle='--')
    # Return
    return fig


def plot_zscore_distributions(zmin, zmax, rtypes, zthr=ZSCORE_THR):
    '''
    Plot distribution of identified min and max z-scores per cell type.
    
    :param zmin: distribution of minimum z-score negative peak on average response trace per cell 
    :param zmax: distribution of maximum z-score positive peak on average response trace per cell 
    :param rtypes: list of response type per cell
    :return: figure handle
    '''
    data = pd.DataFrame({
        'min z-score': zmin,
        'max z-score': zmax,
        'response type': [LABEL_BY_TYPE[rt] for rt in rtypes]
    })
    zabsmax = 1.05 * max(-zmin.min(), zmax.max())
    jgrid = sns.jointplot(
        data=data,
        x='min z-score',
        y='max z-score',
        hue='response type',
        palette={LABEL_BY_TYPE[k]: v for k, v in RGB_BY_TYPE.items()},
        xlim=[-zabsmax, 0], ylim=[0, zabsmax])
    jgrid.ax_joint.axhline(zthr, ls='--', c='k')
    jgrid.ax_joint.axvline(-zthr, ls='--', c='k')
    jgrid.ax_marg_x.axvline(-zthr, ls='--', c='k')
    jgrid.ax_marg_y.axhline(zthr, ls='--', c='k')
    return jgrid


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
    

def plot_cell_map(data, s2p_data, s2p_ops, title=None):
    ''' Plot spatial distribution of cells (per response type) on the recording plane.

        :param data: experiment dataframe.
        :param s2p_data: suite2p output data dictionary
        :param s2p_ops: suite2p output options dictionary
        :param resp_types: array of response types per cell.
        :param title (optional): figure title
        :return: figure handle
    '''
    rtypes = get_response_types_per_cell(data)
    logger.info('plotting cells map color-coded by response type...')
    # Initialize an RGB image matrix
    im = np.ones((s2p_ops['Ly'], s2p_ops['Lx'], 3), dtype=np.float32)
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


def plot_responses(data, tbounds=None, ykey=REL_F_CHANGE_LABEL, ybounds=None, aggfunc='mean', ci=CI, ax=None,
                   alltraces=False, hue=None, col=None, mark_stim=True, mark_response=True, title=None, **kwargs):
    ''' Plot trial responses of specific sub-datasets.
    
    :param data: experiment dataframe
    :param tbounds (optional): time limits for plot
    :param ykey (optional): key indicating the specific signals to plot on the y-axis
    :param ybounds (optional): y-axis limits for plot
    :param aggfunc (optional): method for aggregating across multiple observations within group. If None, all observations will be drawn.
    :param ci (optional): size of the confidence interval around mean traces (int, “sd” or None) 
    :param alltraces (optional): whether to plot all individual traces
    :param hue (optional): grouping variable that will produce lines with different colors.
    :param col (optional): grouping variable that will produce different axes.
    :param mark_stim (optional): whether to add a stimulus mark on the plot
    :param mark_response (optional): whether to add mark indicating the response analysis interval on the plot
    :param title (optional): figure title (deduced if not provided)
    :param kwargs: keyword parameters that are passed to the filter_data function
    :return: figure handle
    '''
    # Extract response interval (from unfiltered data) if requested 
    if mark_response:
        tresponse = [data[TIME_LABEL].values[i] for i in [I_RESPONSE.start, I_RESPONSE.stop]]
    else:
        tresponse = None

    # Filter data 
    filtered_data, filters = filter_data(data, full_output=True, tbounds=tbounds, **kwargs)

    # Use seaborn's lineplot function to do most of the plotting
    # Log info on plot sub-processes
    s = []
    # Determine figure aspect based on col parameters
    if col is not None:
        s.append(f'grouping by {col}')
        col_wrap = min(len(filtered_data.groupby(col)), 5)
        height = 5.
        if ax is not None:
            raise ValueError(f'cannot sweep over {col} with only 1 axis')
    else:
        col_wrap = None
        height = 4.           
    if hue is not None:
        s.append(f'grouping by {hue}')
        if hue == 'cell':
            del filters['cell']
    if aggfunc is not None:
        s.append('averaging')
    if ci is not None:
        s.append('estimating confidence intervals')
    s = f'{", ".join(s)} and ' if len(s) > 0 else ''
    logger.info(f'{s}plotting...')
    # Determine color palette depending on hue parameter
    palette = {
        None: None,
        P_LABEL: 'flare',
        DC_LABEL: 'crest',
        RESP_LABEL: RGB_BY_TYPE,
        # 'cell': TAB10[:len(filtered_data.index.unique('cell'))]
    }.get(hue, None)

    ###################### Plot ######################
    # Default plot arguments dictionary
    plot_kwargs = dict(
        data      = filtered_data, # data
        x         = TIME_LABEL,    # x-axis
        y         = ykey,          # y-axis
        hue       = hue,           # hue grouping variable
        estimator = aggfunc,       # aggregating function
        ci        = ci,            # confidence interval estimator
        palette   = palette,       # color palette
        legend    = 'full'         # use all hue entries in the legend
    )
    if ax is not None:
        # If axis object is provided -> add it to the dictionary and call axis-level plotting function
        plot_kwargs['ax'] = ax
        sns.lineplot(**plot_kwargs)
        axlist = [ax]
        fig = ax.get_figure()
    else:
        # Otherwise, add figure-level plotting arguments and call figure-level plotting function
        plot_kwargs.update(dict(
            kind     = 'line',   # kind of plot
            height   = height,   # figure height
            aspect   = 1.5,      # aspect ratio of the figure
            col_wrap = col_wrap, # how many axes per row
            col      = col,      # column (i.e. axis) grouping variable
        ))
        fg = sns.relplot(**plot_kwargs)
        axlist = fg.axes.flatten()
        fig = fg.figure
    
    # Remove right and top spines
    sns.despine()
    # Add individual traces if specified
    if alltraces:
        logger.info('plotting individual traces...')
        nconds = len(axlist) * len(axlist[0].get_lines())
        with tqdm(total=nconds - 1, position=0, leave=True) as pbar:
            # Aggregation keys = all index keys that are not "frame" 
            aggkeys = list(filter(lambda x: x is not None and x != 'frame', filtered_data.index.names))
            # Group data by col, if provided
            if col is not None:
                logger.debug(f'grouping by {col}')
                col_groups = filtered_data.groupby(col)
            else:
                col_groups = [('all', filtered_data)]
            # For each column group
            for (_, colgr), ax in zip(col_groups, axlist):
                # Group data by hue, if provided
                if hue is not None:
                    logger.debug(f'grouping by {hue}')
                    groups = colgr.groupby(hue)
                else:
                    groups = [('all', colgr)]
                # For each hue group
                for l, (_, gr) in zip(ax.get_lines(), groups):
                    color = l.get_color()
                    # Generate pivot table
                    table = gr.pivot_table(
                        index=TIME_LABEL,  # index = time
                        columns=aggkeys,  # each column = 1 line to plot 
                        values=ykey)  # values
                    # Plot a line for each entry in the pivot table
                    for i, x in enumerate(table):
                        ax.plot(table[x].index, table[x].values, c=color, alpha=0.2, zorder=-10)
                    pbar.update()

    # For each axis
    for ax in axlist:
        # Plot stimulus mark if specified
        if mark_stim:
            ax.axvspan(0, get_singleton(filtered_data, DUR_LABEL), ec=None, fc='C5', alpha=0.5)
        # Plot response interval if specified
        if tresponse is not None:
            for tr in tresponse:
                ax.axvline(tr, ls='--', c='k', lw=1.)
        # Adjust time axis if specified
        if tbounds is not None:
            ax.set_xlim(*tbounds)
        # Adjust y-axis if specified
        if ybounds is not None:
            ax.set_ylim(*ybounds)
    
    # Add title and legend
    if title is None:
        if filters is None:
            filters = {'misc': 'all responses'}
        title = ' - '.join(filters.values())
    if col is None: 
        # If only 1 axis (i.e. no column grouping) -> add to axis
        axlist[0].set_title(title)
    else:
        # Otherwise -> add as suptitle
        dy = 3  # inches
        height = fig.get_size_inches()[0]
        fig.subplots_adjust(top=1 - dy / height)
        fig.suptitle(title)
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