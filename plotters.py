# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-13 11:41:52
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-11-16 17:42:25

''' Collection of plotting utilities. '''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import seaborn as sns
from colorsys import hsv_to_rgb, rgb_to_hsv
from tqdm import tqdm

from logger import logger
from constants import *
from utils import get_singleton, plural
from postpro import filter_data, find_response_peak, get_response_types_per_ROI, get_trial_averaged
from viewers import get_stack_viewer
   

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
    if title is None:
        title = ''
    fig.suptitle(f'{title} - summary frames')
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
    sns.despine()
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
    
    :param data: cell-filtered suite2p output dictionary
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
    # Initialize outliers dictionary
    is_outlier = {}
    # Fetch stats dictionary
    stats = data['stat']
    # For each output stats parameter
    for ax, pkey in zip(axes, pkeys):
        # Plot histogram distribution
        sns.despine(ax=ax, left=True)
        ax.set_xlabel(pkey)
        ax.set_yticks([])
        d = np.array([x[pkey] for x in stats])
        ax.hist(d, bins=20, ec='k', alpha=0.7)
        # If z-score threshold if provided, compute z-score distribution and identify outliers
        if zthr is not None:
            mu, std = d.mean(), d.std()
            lims = [mu + k * zthr * std for k in [-1, 1]]
            for l in lims:
                ax.axvline(l, ls='--', c='silver')
            is_outlier[pkey] = np.logical_or(d < lims[0], d > lims[1])
    # Hide unused axes
    for ax in axes[len(pkeys):]:
        sns.despine(ax=ax, left=True, bottom=True)
        ax.set_xticks([])
        ax.set_yticks([])
    # Conditional return
    if zthr is None:
        return fig
    else:
        # return dataframe of outliers
        return fig, pd.DataFrame(is_outlier)


def plot_traces(data, iROI=None, irun=None, itrial=None, delimiters=None, ylabel=None,
                ybounds=None, cmap=None, title=None):
    '''
    Simple function to plot fluorescence traces from a fluorescnece data matrix
    
    :param data: fluorescence dataframe
    :param iROI (optional): ROI index
    :param irun (optional: run index
    :param itrial (optional): trial index
    :param delimiters (optional): temporal delimitations (shown as vertical lines)
    :param ylabel: name of the variable to plot
    :param ybounds (optional): y-axis limits
    :return: figure handle
    '''
    # Filter data based on selection criteria
    filtered_data, filters = filter_data(
        data, iROI=iROI, irun=irun, itrial=itrial, full_output=True)
    if filters is None:
        filters = {'misc': 'all responses'}

    # Get data dimensions
    npersignal, nsignals = filtered_data.shape

    # Get value of first frame index as the x-onset
    iframes = filtered_data.index.unique(level=FRAME_LABEL).values
    ionset = iframes[0]
    # If trials are marked in data -> adjust onset according to 1st trial index
    if TRIAL_LABEL in filtered_data.index.names:
        itrials = filtered_data.index.unique(level=TRIAL_LABEL).values
        ionset += itrials[0] * len(iframes)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.despine()
    ax.set_xlabel('frames')
    if ylabel is None:
        if nsignals > 1:
            logger.warning('ambiguous y-labeling for more than 1 signal')
        ylabel = filtered_data.columns[0]
    ax.set_ylabel(ylabel)
    parsed_title = ' - '.join(filters.values()) + f' trace{plural(nsignals)}'
    if title is not None:
        parsed_title = f'{parsed_title} ({title})' 
    ax.set_title(parsed_title)

    # Generate x-axis indexes
    xinds = np.arange(npersignal) + ionset

    # Determine traces colors
    if cmap is not None:
        colors = plt.get_cmap(cmap)(np.linspace(0, 1, nsignals))
    else:
        colors = [None] * nsignals

    # Plot traces
    logger.info(f'plotting {nsignals} fluorescence trace(s)...')
    for color, col in zip(colors, filtered_data):
        ax.plot(xinds, filtered_data[col], c=color, label=col)

    # Restrict y axis, if needed
    if ybounds is not None:
        ax.set_ylim(*ybounds)

    # Plot delimiters, if any
    if delimiters is not None:
        logger.info(f'adding {len(delimiters)} delimiters')
        for iframe in delimiters:
            ax.axvline(iframe, color='k', linestyle='--')

    # Add legend
    if nsignals > 1:
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
    

def plot_cell_map(data, s2p_data, s2p_ops, title=None):
    ''' Plot spatial distribution of cells (per response type) on the recording plane.

        :param data: experiment dataframe.
        :param s2p_data: suite2p output data dictionary
        :param s2p_ops: suite2p output options dictionary
        :param resp_types: array of response types per cell.
        :param title (optional): figure title
        :return: figure handle
    '''
    rtypes = get_response_types_per_ROI(data)
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


def plot_experiment_heatmap(data, key=DFF_LABEL, title=None, show_ylabel=True):
    '''
    Plot experiment heatmap (average response over time of each cell, culstered by similarity).
    
    :param data: experiment dataframe.
    :return: figure handle
    '''
    # Determine rows color labels from response types per cell
    rtypes = get_response_types_per_ROI(data).values
    row_colors = [RGB_BY_TYPE[rtype] for rtype in rtypes]
    # Generate 2D table of average dF/F0 response per cell (using roi as index),
    # across runs and trials
    logger.info(f'generating (ROI x time) {key} pivot table...')
    avg_resp_per_cell = data.pivot_table(
        index=ROI_LABEL, columns=TIME_LABEL, values=key, aggfunc=np.mean)
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
    cg.ax_row_colors.set_ylabel(ROI_RESP_TYPE_LABEL)
    # Move colobar directly on the right of the heatmap
    pos = cg.ax_heatmap.get_position()
    cg.ax_cbar.set_position([pos.x1 + .1, pos.y0, .05, pos.y1 - pos.y0])
    # Add colormap title
    cg.ax_cbar.set_title(key)
    # Add heatmap title, if any
    if title is not None:
        cg.ax_heatmap.set_title(title)
    return cg


def add_label_mark(ax, x, cmap=None, w=0.1):
    ''' Add a color marker in the top right corner of a plot '''
    if cmap is None:
        cmap = plt.get_cmap('viridis')
    if isinstance(cmap, dict):
        c = cmap[x]
        s = x
    else:
        c = cmap(x)
        s = f'{x:.2f}'
    brightness = rgb_to_hsv(*c[:3])[-1]
    tcolor = 'w' if brightness < 0.7 else 'k'
    ax.add_patch(Rectangle((1 - w,  1 - w), w, w, transform=ax.transAxes, fc=c, ec=None))
    ax.text(1 - w / 2,  1 - w / 2, s, transform=ax.transAxes, ha='center', va='center', c=tcolor)
    

def plot_responses(data, tbounds=None, ykey=DFF_LABEL, ybounds=None, aggfunc='mean', ci=CI, ax=None,
                   alltraces=False, hue=None, col=None,
                   mark_stim=True, mark_analysis_window=True, mark_peaks=False,
                   label=None, title=None, **kwargs):
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
    :param mark_analysis_window (optional): whether to add mark indicating the response analysis interval on the plot
    :param mark_peaks: whether to mark the peaks of each identified response
    :param label (optional): add a label indicating a specific field value on the plot (when possible)
    :param title (optional): figure title (deduced if not provided)
    :param kwargs: keyword parameters that are passed to the filter_data function
    :return: figure handle
    '''
    # Extract response interval (from unfiltered data) if requested 
    if mark_analysis_window:
        tresponse = [data[TIME_LABEL].values[i] for i in [I_RESPONSE.start, I_RESPONSE.stop]]
    else:
        tresponse = None

    ###################### Filtering ######################

    # Filter data based on selection criteria
    filtered_data, filters = filter_data(data, full_output=True, tbounds=tbounds, **kwargs)
    # Get number of ROIs in filtered data
    nROIs_filtered = len(filtered_data.index.unique(level=ROI_LABEL))

    ###################### Process log ######################
    s = []
    # Determine figure aspect based on col parameters

    # If col set to ROI and only ROI -> remove column assignment 
    if col == ROI_LABEL and nROIs_filtered == 1:
        col = None
    # If col set to ROI -> remove ROI filter info
    if col == ROI_LABEL and ROI_LABEL in filters:
        del filters[ROI_LABEL]

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
        if hue == ROI_LABEL and ROI_LABEL in filters:
            del filters[ROI_LABEL]
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
        ROI_RESP_TYPE_LABEL: RGB_BY_TYPE
    }.get(hue, None)

    ###################### Mean traces and CIs ######################

    # Default plot arguments dictionary
    plot_kwargs = dict(
        data      = filtered_data, # data
        x         = TIME_LABEL,    # x-axis
        y         = ykey,          # y-axis
        hue       = hue,           # hue grouping variable
        estimator = aggfunc,       # aggregating function
        ci        = ci,            # confidence interval estimator
        lw        = 2.0,           # line width
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

    ###################### Individual traces ######################
    
    if alltraces:
        logger.info('plotting individual traces...')
        nconds = len(axlist) * len(axlist[0].get_lines())
        with tqdm(total=nconds - 1, position=0, leave=True) as pbar:
            # Aggregation keys = all index keys that are not "frame" 
            aggkeys = list(filter(lambda x: x is not None and x != FRAME_LABEL, filtered_data.index.names))
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
                use_color_code = len(groups) == 1
                for l, (_, gr) in zip(ax.get_lines(), groups):
                    group_color = l.get_color()
                    group_alpha = 0.2
                    # Generate pivot table for the value
                    table = gr.pivot_table(
                        index=TIME_LABEL,  # index = time
                        columns=aggkeys,  # each column = 1 line to plot 
                        values=ykey)  # values
                    # Get response classification of each trace
                    is_resps = gr[IS_RESP_LABEL].groupby(aggkeys).first()                                        
                    # Plot a line for each entry in the pivot table
                    for i, (x, is_resp) in enumerate(zip(table, is_resps)):
                        if use_color_code:
                            color = {True: 'g', False: 'r'}[is_resp]
                            alpha = {True: .2, False: .2}[is_resp]
                        else:
                            color = group_color
                            alpha = group_alpha
                        ax.plot(table[x].index, table[x].values, c=color, alpha=alpha, zorder=-10)
                        # Add detected peak if specified
                        if mark_peaks:
                            window_trace = table[x][
                                (table[x].index >= tresponse[0]) &
                                (table[x].index <= tresponse[1])]
                            ipeak, ypeak = find_response_peak(
                                window_trace, return_index=True)
                            if not np.isnan(ipeak):
                                ax.scatter(
                                    window_trace.index[ipeak], ypeak, color=color, alpha=alpha)
                    pbar.update()

    ###################### Markers ######################

    # If indicator provided, check number of values per axis
    if label is not None:
        label_cmap = RGB_BY_TYPE if label == ROI_RESP_TYPE_LABEL else None
        if col is not None:
            label_values_per_ax = filtered_data.groupby(col)[label].unique().values
        else:
            label_values_per_ax = [filtered_data[label].unique()]
        
    # For each axis
    for iax, ax in enumerate(axlist):
        # Plot stimulus mark if specified
        if mark_stim:
            ax.axvspan(0, get_singleton(filtered_data, DUR_LABEL), ec=None, fc='C5', alpha=0.5)
        # Plot noise threshold level if key is z-score
        if ykey == ZSCORE_LABEL:
            ax.axhline(ZSCORE_THR, ls='--', c='k', lw=1.)
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
        # Add label for specific field value, if specified and possible
        if label is not None:
            label_values = label_values_per_ax[iax]
            if len(label_values) == 1:
                add_label_mark(ax, label_values[0], cmap=label_cmap)

    ###################### Title & legend ######################
    
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
    '''
    Plot the evolution of the average frame intensity over time

    :param correct (optional): whether to mean-correct the signal before plotting it  
    '''
    # Actual args of interest 
    ax = kwargs.pop('ax', None)  # axis
    correct = kwargs.pop('correct', False)  # whether to mean-correct the signal before plotting it
    title = kwargs.get('title', None)  # title
    # Viewer rendering args
    ilabels = kwargs.pop('ilabels', None)   # index of stimulation frames 
    norm = kwargs.pop('norm', True)  # normalize across frames before rendering
    cmap = kwargs.pop('cmap', 'viridis')  # colormap
    bounds = kwargs.pop('bounds', None)  # bounds

    # Initialize viewer and initialize its rendering
    viewer = get_stack_viewer(*args, **kwargs)
    viewer.init_render(norm=norm, cmap=cmap, bounds=bounds, ilabels=ilabels)

    if ax is None:
        # Create figure if not provided
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.despine()
        if title is None:
            title = ''
        ax.set_title(f'{title} spatial average time course')
        ax.set_xlabel('frames')
        ylabel = 'average frame intensity'
        if correct:
            ylabel = f'mean corrected {ylabel}'
        ax.set_ylabel(ylabel)
        # Plot delimiters, if any
        if ilabels is not None:
            logger.info(f'adding {len(ilabels)} delimiters')
            for iframe in ilabels:
                ax.axvline(iframe, color='k', linestyle='--')
    else:
        fig = ax.get_figure()
    
    # For each stack file-object provided
    for header, fobj in zip(viewer.headers, viewer.fobjs):
        # Get mean evolution of the stack
        mu = viewer.get_mean_evolution(fobj, viewer.frange)
        # Mean-correct the signal if needed
        if correct:
            mu -= mu.mean()
        # Plot the signal with the correct label
        ax.plot(mu, label=header)
    
    # Add/update legend
    ax.legend(frameon=False)
    return fig


def plot_stat_heatmap(data, key, expand=False, title=None, **kwargs):
    '''
    Plot ROI x run heatmap for some statistics
    
    :param data: multi-indexed (ROI x run x trial) statistics dataframe
    :param key: stat column key
    :param expand (optional): expand figure to clearly see each combination
    :return figure handle
    '''
    s = f'{key} per ROI & run'
    # Compute trial-averaged data
    trialavg_data, is_repeat = get_trial_averaged(data, key, full_output=True)
    # Determine whether stats is a repeated value or a real distribution
    if not is_repeat:
        s = f'trial-averaged {s}'
    logger.info(f'plotting {s}...')
    nROIs, nruns = [len(trialavg_data.index.unique(level=k)) for k in [ROI_LABEL, RUN_LABEL]]
    # Determine figure size
    figsize = (nruns / 2, nROIs / 5) if expand else None
    # Determine colormap center based on stat range
    center = 0 if trialavg_data.min() < 0 else None
    # Create figure and plot trial-averaged stat heatmap
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(trialavg_data.unstack(), center=center, **kwargs)
    # Add title
    if title is not None:
        s = f'{s} ({title})'
    ax.set_title(s)
    # Return
    return fig


def plot_stat_per_ROI(data, key, title=None):
    '''
    Plot the distribution of a stat per ROI over all experimental conditions
    
    :param data: multi-indexed (ROI x run x trial) statistics dataframe
    :return: figure handle
    '''
    s = key
    # Compute trial-averaged data
    trialavg_data, is_repeat = get_trial_averaged(data, key, full_output=True)
    # Determine whether stats is a repeated value or a real distribution
    if not is_repeat:
        s = f'trial-averaged {s}'
    # Group by ROI, get mean and std
    groups = trialavg_data.groupby(ROI_LABEL)    
    mu, sigma = groups.mean(), groups.std()
    x = np.arange(mu.size)
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlabel('ROIs')
    ax.set_ylabel(s)
    parsed_title = f'{s} per ROI'
    if title is not None:
        parsed_title = f'{parsed_title} ({title})'
    ax.set_title(parsed_title)
    # Plot mean trace with +/-std shaded area
    ax.plot(x, mu)
    ax.fill_between(x, mu - sigma, mu + sigma, alpha=0.2)
    sns.despine(ax=ax)
    return fig


def plot_stat_per_run(data, key, title=None):
    '''
    Plot the distribution of a stat per run over all ROIs
    
    :param data: multi-indexed (ROI x run x trial) statistics dataframe
    :return: figure handle
    '''
    s = key
    # Compute trial-averaged data
    trialavg_data, is_repeat = get_trial_averaged(data, key, full_output=True)
    # Determine whether stats is a repeated value or a real distribution
    if not is_repeat:
        s = f'trial-averaged {s}'
    # Create figure
    fig, ax = plt.subplots()
    parsed_title = f'{s} per run'
    if title is not None:
        parsed_title = f'{parsed_title} ({title})'
    ax.set_title(parsed_title)
    ax.set_xlabel('ROIs')
    ax.set_ylabel(s)
    # Bar plot with std error bars
    trialavg_data = trialavg_data.to_frame()
    trialavg_data = trialavg_data.reset_index(level=RUN_LABEL)
    sns.barplot(ax=ax, data=trialavg_data, x=RUN_LABEL, y=key, ci='sd')
    sns.despine(ax=ax)
    return fig


def plot_positive_runs_hist(n_positive_runs, resp_types, nruns, title=None):
    ''' Plot the histogram of the number of positive conditions for each ROI,
        per response type.
    '''
    fig, ax = plt.subplots()
    ax = sns.histplot(
        pd.DataFrame([resp_types, n_positive_runs]).T,
        x=NPOS_RUNS_LABEL, hue=ROI_RESP_TYPE_LABEL, bins=np.arange(nruns) + 0.5,
        ax=ax)
    sns.despine(ax=ax)
    ax.axvline(NPOS_CONDS_THR - .5, ls='--', c='k')
    labels = []
    for k, v in LABEL_BY_TYPE.items():
        nclass = sum(resp_types == k)
        if nclass > 0:
            labels.append(f'{v} (n = {nclass})')
    ax.legend(title=ROI_RESP_TYPE_LABEL, loc='upper right', labels=['threshold'] + labels[::-1])
    s = 'classification by # positive conditions'
    if title is not None:
        s = f'{s} ({title})'
    ax.set_title(s)
    return fig