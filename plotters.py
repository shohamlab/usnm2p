# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-13 11:41:52
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-02-01 15:57:24

''' Collection of plotting utilities. '''

from itertools import combinations
import random
from natsort import natsorted
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize, LogNorm, SymLogNorm, to_rgb
from matplotlib import cm
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.patches import Rectangle
import seaborn as sns
from statannotations.Annotator import Annotator
from colorsys import hsv_to_rgb, rgb_to_hsv
from tqdm import tqdm
from scipy.stats import normaltest

from logger import logger
from constants import *
from utils import *
from postpro import *
from viewers import get_stack_viewer
from fileops import loadtif
from parsers import get_info_table, parse_quantile

# Colormaps
rdgn = sns.diverging_palette(h_neg=130, h_pos=10, s=99, l=55, sep=3, as_cmap=True)
rdgn.set_bad('silver')
gnrd = sns.diverging_palette(h_neg=10, h_pos=130, s=99, l=55, sep=3, as_cmap=True)
gnrd.set_bad('silver')
nan_viridis = plt.get_cmap('viridis').copy()
nan_viridis.set_bad('silver')
rtype_cmap = LinearSegmentedColormap.from_list(
    'rtype', colors=list(Palette.RTYPE.values()))



def get_colors(cmap, N=None, use_index='auto'):
    '''
    Get colors based on a colormap
    
    :param cmap: colormap (string or matplotlib object). Can also be an iterable of colormaps
    :param N (optional): number of colors to output
    :param use_index (optional): whether to use the colormap index to sample colors
    :return: list of colors
    '''
    # Vectorize function
    if is_iterable(cmap):
        return np.vstack([get_colors(c) for c in cmap])
    
    # If cmap is string, get corresponding object 
    if isinstance(cmap, str):
        if use_index == 'auto':
            if cmap in ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']:
                use_index = True
            else:
                use_index = False
        cmap = plt.get_cmap(cmap)
    
    # Infer number of colors directly from colormap if not given
    if not N:
        N = cmap.N
    
    # Determine from colormap type if index should be used 
    if use_index == 'auto':
        if cmap.N > 100:
            use_index=False
        elif isinstance(cmap, LinearSegmentedColormap):
            use_index = False
        elif isinstance(cmap, ListedColormap):
            use_index = True
    
    # Extract colors from colormap
    if use_index:
        ind = np.arange(int(N)) % cmap.N
        colors = cmap(ind)
    else:
        colors = cmap(np.linspace(0, 1, N))
    
    # Return colors
    return colors


def get_color_cycle(*args, **kwargs):
    ''' Get color cycler based on a colormap '''
    return plt.cycler('color', get_colors(*args, **kwargs))


def to_binary_cmap(cmap):
    ''' Convert continous cmap to binary cmap with range extremities colors '''
    colors = [cmap(0.), cmap(1.)]
    return ListedColormap(colors=colors)


def harmonize_axes_limits(axes, axkey='y'):
    '''
    Harmonize x or y limits across a set of axes
    
    :param axes: list/array of axes
    :param axkey: axis key ("x" or "y"
    '''
    axes = np.asarray(axes)
    
    # Flatten axes array if needed
    if axes.ndim > 1:
        axes = axes.ravel()
    
    # Determine limits getter and setter functions for appropriate axis
    limgetter = lambda ax: getattr(ax, f'get_{axkey}lim')
    limsetter = lambda ax: getattr(ax, f'set_{axkey}lim')

    # Get limits, and extract min and max over axes
    lims = [limgetter(ax)() for ax in axes]
    mins, maxs = list(zip(*lims))
    bounds = min(mins), max(maxs)

    # Set as bounds for all axes
    for ax in axes.ravel():
        limsetter(ax)(*bounds)


def harmonize_jointplot_limits(jg):
    ''' Harmonize X and Y limits of joint plots '''
    lims = jg.ax_joint.get_xlim() + jg.ax_joint.get_ylim()
    lims = (min(lims), max(lims))
    for ax in [jg.ax_joint, jg.ax_marg_x]:
        ax.set_xlim(*lims)
    for ax in [jg.ax_joint, jg.ax_marg_y]:
        ax.set_ylim(*lims)


def add_jointplot_line(jg, val=0., mode='xy'):
    ''' Add reference line at some given value on joint plot'''
    if 'x' in mode:
        for ax in [jg.ax_joint, jg.ax_marg_x]:
            ax.axvline(val, ls='--', c='k')
    if 'y' in mode:
        for ax in [jg.ax_joint, jg.ax_marg_y]:
            ax.axhline(val, ls='--', c='k')


def plot_table(data, title=None, ax=None, fs=15, aspect=1):
    '''
    Plot a dictionary as a table

    :param data: dataframe
    :param title (optional): table title
    :return: figure handle
    '''
    if isinstance(data, dict):
        data = pd.DataFrame({'Parameter': data.keys(), 'Value': data.values()})

    nrows, ncols = data.shape
    # Initialize or retrieve figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, nrows * 0.5))
    else:
        fig = ax.get_figure()

    # Set title if provided
    if title is not None:
        ax.set_title(title, fontsize=fs + 2)
    
    # Remove axes rendering
    ax.axis('off')

    # Render dataframe content as table
    table = pd.plotting.table(
        data=data, ax=ax,
        rowColours=['silver'] * nrows,
        colColours=['silver'] * ncols,
        loc='center')
    table.set_fontsize(fs)
    table.scale(1, aspect)

    # Tighten figure layout
    fig.tight_layout()

    # Return figure
    return fig


def data_to_axis(ax, p):
    '''
    Convert data coordinates to axis coordinates
    
    :param ax: axis object
    :param p: (x, y) point in data coordinates
    :return: (x, y) point in axis coordinates
    '''
    # Transform from data to absolute coordinates
    trans = ax.transData.transform(p)
    # Transfrom from absolute to axis coordinates and return
    return ax.transAxes.inverted().transform(trans)


def set_normalizer(cmap, bounds, scale='lin'):
    norm = {
        'lin': Normalize,
        'log': LogNorm,
        'symlog': SymLogNorm
    }[scale](*bounds)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm._A = []
    return norm, sm


def plot_stack_histogram(stacks, title=None, yscale='log'):
    '''
    Plot summary histogram from TIF stacks.
    
    :param stacks: dictionary of TIF stacks
    :param title (optional): figure title
    :param yscale: scale type for y-axis (default = log) 
    :return: figure handle
    '''
    logger.info('plotting stack(s) histogram...')

    # Initialize figure
    fig, ax = plt.subplots()
    sns.despine(ax=ax)
    if title is None:
        title = ''
    else:
        title = f'{title} - '
    ax.set_title(f'{title}summary histogram')
    ax.set_xlabel('pixel intensity')
    ax.set_yscale(yscale)
    ax.set_ylabel('Count')

    # Plot histogram each stack entry 
    for k, v in stacks.items():
        ax.hist(v.ravel(), bins=50, label=k, ec='k', alpha=0.5)
    
    # Add legend 
    ax.legend()

    # Return figure handle
    return fig


def plot_frameavg_profiles(frameavgs, details=False, title=None):
    '''
    Plot frame-average profiles of a given acquisition
    
    :param frameavgs: array of frame-average profiles
    :param details: whether to plot individual traces
    :return: frame average profiles figure
    '''
    nacqs, nframes = frameavgs.shape[:2]
    # If multiple channels are present, re-arrange array to have channels on axis 0 
    if frameavgs.ndim > 2:
        nchannels = frameavgs.shape[2]
        frameavgs = np.moveaxis(frameavgs, 2, 0)
    # Otherwise, add dummy axis 0 for single channel
    else:
        nchannels = 1
        frameavgs = frameavgs[np.newaxis, :]

    # Create figure backbone
    fig, axes = plt.subplots(nchannels, figsize=(10, 3 * nchannels))
    if nchannels == 1:
        axes = [axes]
    stitle = f'{nacqs} acquisitions'
    if title is not None:
        stitle = f'{title} ({stitle})'
    axes[0].set_title(stitle)
    for i, ax in enumerate(axes):
        ax.set_xlabel('frames')
        ax.set_ylabel(f'channel {i + 1}')

    # Plot frame average profiles
    logger.info(f'plotting {nacqs} frame-average profiles...')
    iframes = np.arange(nframes)
    for ax, ychannel in zip(axes, frameavgs):
        sns.despine(ax=ax)
        # Calculate mean and SEM
        ymean = ychannel.mean(axis=0)
        ysem = ychannel.std(axis=0, ddof=1) / np.sqrt(ychannel.shape[0])
        # Plot mean profile
        ax.plot(iframes, ymean, label='avg', c='k')
        # Plot individual traces if specified
        if details:
            for i, yacq in enumerate(ychannel):
                ax.plot(iframes, yacq, label=f'acq {i + 1}', alpha=0.5, lw=1)
            if nacqs <= 10:
                ax.legend(loc='center right')
        # Otherwise, plot mean +/- sem shaded area
        else:
            ax.fill_between(
                iframes, ymean - ysem, ymean + ysem, fc='k', alpha=0.3)
    
    # Return figure handle
    return fig


def plot_stack_timecourse(*args, **kwargs):
    '''
    Plot the evolution of the average frame intensity over time, with shaded areas
    showing its standard deviation

    :param correct (optional): whether to mean-correct the signal before plotting it  
    '''
    # Extract args of interest 
    ax = kwargs.pop('ax', None)  # axis
    correct = kwargs.pop('correct', False)  # whether to mean-correct the signal before plotting it
    title = kwargs.get('title', None)  # title
    
    # Extract viewer rendering args
    ilabels = kwargs.pop('ilabels', None)   # index of stimulation frames 
    norm = kwargs.pop('norm', True)  # normalize across frames before rendering
    cmap = kwargs.pop('cmap', 'viridis')  # colormap
    bounds = kwargs.pop('bounds', None)  # bounds

    # Initialize viewer and initialize its rendering
    viewer = get_stack_viewer(*args, **kwargs)
    viewer.init_render(norm=norm, cmap=cmap, bounds=bounds, ilabels=ilabels)

    # Initialize figure if not provided
    if ax is None:
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
    for i, header in enumerate(viewer.headers):
        
        # Get evolution of frame average intensity and its standard deviation
        mu, sigma = viewer.get_frame_metric_evolution(
            viewer.fobjs[i], viewer.frange, func=lambda x: (x.mean(), x.std())).T
        
        # Mean-correct the signal if needed
        if correct:
            mu -= mu.mean()
        
        # Plot the signal with the correct label
        inds = np.arange(mu.size)
        ax.plot(inds, mu, label=header)
        ax.fill_between(inds, mu - sigma, mu + sigma, alpha=0.2)
    
    # Add/update legend
    ax.legend(frameon=False)
    
    # Return figure handle
    return fig


def plot_trialavg_stackavg_traces(fpaths, ntrials_per_run, title=None, tbounds=None,
                                  cmap=['tab10', 'Dark2', 'Accent'], iref=None, itrial=None):
    '''
    Plot trial-averaged, pixel-averaged intensity traces from a list of run stacks
    
    :param fpaths: list of paths to TIF stacks for the different runs
    :param ntrials_per_run: number of trials per run (used for trial-averaging)
    :param title (optional): figure title
    :param tbounds (optional): time (x) axis bounds
    :param cmap (optional): colormap from which to samples traces colors
    :param iref (optional): reference index at which to align traces on trhe y-axis. If none
        is provided, traces will be aligned according to a characteristic distribution quantile.
    :param itrial (optional): trial index(es) to average from. If none is provided, all trials
        are aggregated.
    :return: figure handle
    '''
    # Get runs sorted by I_SPTA
    df = get_info_table(fpaths, ntrials_per_run=ntrials_per_run)
    df = add_intensity_to_table(df)
    npertrial = df.loc[0, Label.NPERTRIAL]
    df = df.sort_values(by=Label.ISPTA)

    # Create figure backbone
    fig, ax = plt.subplots()
    sns.despine(ax=ax)
    ax.set_xlabel(Label.TIME)
    ax.set_ylabel(Label.DFF)
    if tbounds is not None:
        ax.set_xlim(*tbounds)
    if title is not None:
        ax.set_title(title)
    iframes = np.arange(npertrial)
    ax.axvline(0., c='k', ls='--')
    cycler = get_color_cycle(cmap, len(df))

    # Loop through runs of increasing intensity
    for c, (irun, run_info) in zip(cycler, df.iterrows()):
        
        # Extract run label and movide stack
        label = f'run {irun} ({run_info[Label.P]:.2f} MPa, {run_info[Label.DC]:.0f} % DC)'
        fpath = fpaths[irun]
        tplt = (iframes - FrameIndex.STIM) / run_info[Label.FPS]
        stack = loadtif(fpath, verbose=False)
        
        # Average across pixels and reshape as trials x frames
        stackavg_trace = stack.mean(axis=(-2, -1))
        stackavg_mat = stackavg_trace.reshape((-1, npertrial))
        
        # Remove specific trials if specified
        if itrial is not None:
            stackavg_mat = stackavg_mat[itrial, :]
        
        # Average across trials to get mean trace
        x = stackavg_mat.mean(axis=0)
        
        # Find baseline signal value 
        if isinstance(iref, int):
            xref = x[iref]  # specific index, if provided
        elif isinstance(iref, str) and iref.startswith('q'):
            q = parse_quantile(iref)
            xref = np.quantile(x, q)  # otherwise, low quantile
        else:
            raise ValueError(f'invalid reference index: {iref}')
        
        # Plot normalized mean trace
        ax.plot(tplt, (x - xref) / xref, c=c['color'], label=label)
    
    # Add legend
    ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Return figure handle
    return fig


def plot_stack_frequency_spectrum(stacks, fs, title=None, yscale='log'):
    '''
    Plot frequency spectrum of TIF stacks.
    
    :param stacks: dictionary of TIF stacks
    :param cmap (optional): colormap
    :return: figure handle
    '''
    # Compute stacks frequency power spectrums (using FFT) along time axis
    logger.info('computing stack(s) fft...')
    nframes = stacks[list(stacks.keys())[0]].shape[0]
    freqs = np.fft.rfftfreq(nframes, 1 / fs)
    ffts = {k: np.abs(np.fft.rfft(v, axis=0)) for k, v in stacks.items()}
    ps_avg = {k: np.array([(x**2).mean() for x in v]) for k, v in ffts.items()}

    # Initialize figure
    logger.info('plotting stack(s) frequency spectrum...')
    fig, ax = plt.subplots()
    if title is None:
        title = ''
    else:
        title = f'{title} - '
    ax.set_title(f'{title}frequency spectrum')
    sns.despine(ax=ax)
    ax.set_xlabel('frequency (Hz)')
    ax.set_yscale(yscale)
    ax.set_ylabel('power spectrum')

    # Plot all power spectrum profiles for each stack
    for k, v in ps_avg.items():
        ax.plot(freqs, v, label=k)
    
    # Add legend
    ax.legend()

    # Return figure handle
    return fig


def plot_stack_summary_frames(stack, cmap='viridis', title=None, um_per_px=None):
    '''
    Plot summary images from a TIF stack.
    
    :param stack: TIF stack
    :param cmap (optional): colormap used to plot summary frames
    :param title (optional): figure title
    :param um_per_px (optional): spatial resolution (um/pixel). If provided, ticks and tick labels
        on each image are replaced by a scale bar on the graph.
    :return: figure handle
    '''
    # Get projection functions dictionary
    projfuncs = {
        'Mean': np.mean,
        'Standard deviation': np.std,
        'Max. projection': np.max
    }

    # Initialize figure according to number of projections 
    fig, axes = plt.subplots(1, len(projfuncs), figsize=(5 * len(projfuncs), 5))
    if title is None:
        title = ''
    fig.suptitle(f'{title} - summary frames')
    fig.subplots_adjust(hspace=10)

    # Plot each projection function, project movie and plot resulting image
    for ax, (title, func) in zip(axes, projfuncs.items()):
        ax.set_title(title)
        ax.imshow(func(stack, axis=0), cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
        if um_per_px is not None:
            npx = stack.shape[-1]
            add_scale_bar(ax, npx, um_per_px, color='w')
    
    # Return figure handle
    return fig


def add_scale_bar(ax, npx, um_per_px, color='k'):
    '''
    Add a scale bar to a micrograph axis
    
    :param ax: axis object
    :param npx: number of pixels on each dimension of the axis image
    :param um_per_pixel: spatial scale (number of microns per pixel)
    :param color: color of the scale bar
    '''
    # Compute scale bar length (in microns)
    npx_bar_length = npx / 4   # starting point (in pxels): 1/4 of the image length
    um_bar_length = um_per_px * npx_bar_length  # conversion to microns

    # Round to appropriate power of ten factor
    pow10 = int(np.log10(um_bar_length))
    roundfactor = np.power(10, pow10)
    um_bar_length = np.round(um_bar_length / roundfactor) * roundfactor

    # Compute relative length (in axis units)
    npx_bar_length = um_bar_length / um_per_px
    rel_bar_length = npx_bar_length / npx

    # Define scale bar artist object
    scalebar = AnchoredSizeBar(ax.transAxes,
        rel_bar_length, 
        f'{um_bar_length:.0f} um',
        'lower right', 
        pad=0.1,
        color=color,
        frameon=False,
        size_vertical=.01)
    
    # Add scale bar to axis
    ax.add_artist(scalebar)


def plot_suite2p_registration_images(ops, title=None, cmap='viridis', um_per_px=None,
                                     full_mode=False):
    '''
    Plot summary registration images from suite2p processing output.

    :param ops: suite2p output options dictionary
    :param title (optional): figure title
    :param cmap (optional): colormap used to plot registration images
    :param um_per_px (optional): spatial resolution (um/pixel). If provided, ticks and tick labels
        on each image are replaced by a scale bar on the graph.
    :param full_mode: whether to plot images extracted from the registration process, or only
        a selected subset (default = False)
    :return: figure handle    
    '''
    logger.info('plotting suite2p registered images...')    
    
    # Gather images dictionary
    imkeys_dict = {
        'Max projection image': 'max_proj',
        'Cross-scale correlation map': 'Vcorr',
        'Mean image': 'meanImg'
    }
    if full_mode:
        imkeys_dict = {
            'Reference registration image': 'refImg',
            ** imkeys_dict,
            'Enhanced mean image (median-filtered)': 'meanImgE'
    }
    imgs_dict = {label: ops.get(key, None) for label, key in imkeys_dict.items()}
    imgs_dict = {k: v for k, v in imgs_dict.items() if v is not None}
    
    # Create figure
    nimgs = len(imgs_dict)
    fig, axes = plt.subplots(1, nimgs, figsize=(4 * nimgs, 4))
    if title is not None:
        fig.suptitle(title)
    
    # For each available image
    for ax, (label, img) in zip(axes, imgs_dict.items()):
        
        # Set image title and render image
        ax.set_title(label)
        ax.imshow(img, cmap=cmap)
        
        # Add scale bar if scale provided
        if um_per_px is not None:
            npx = img.shape[-1]
            for ax in axes:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                add_scale_bar(ax, npx, um_per_px, color='w')

    # Return figure
    return fig


def plot_suite2p_phase_corr_peak(ops):
    ''' 
    Plot peak of phase correlation with reference image over time
    from suite2p processing output.

    :param ops: suite2p output options dictionary
    :return: figure handle    
    '''
    # Check if registration metrics are present
    if 'corrXY' not in ops:
        logger.warning('looks like the data was not registered -> ignoring')
        return None
    
    # Initialize figure
    logger.info('plotting suite2p registration phase correlation peaks...')
    fig, ax = plt.subplots(figsize=(12, 3))
    sns.despine(ax=ax)
    ax.set_title('peak of phase correlation with ref. image over time')
    ax.set_xlabel('frames')
    ax.set_ylabel('phase correlation peak')

    # Plot rigid phase correlation peaks profiles
    ax.plot(ops['corrXY'], c='k', label='whole frame', zorder=5)

    # If available, plot non-rigid phase correlation peaks profiles for each sub-block
    if ops['nonrigid']:
        block_corrs = ops[f'corrXY1']
        for i, bc in enumerate(block_corrs.T):
            ax.plot(bc, label=f'block {i + 1}')
        ax.legend(bbox_to_anchor=(1, 0), loc='center left')
    
    # Return figure handle
    return fig


def plot_suite2p_registration_offsets(ops, fbounds=None, title=None):
    '''
    Plot registration offsets over time from suite2p processing output.

    :param ops: suite2p output options dictionary
    :param fbounds (optional): frames indexes bounds for which to plot the offsets. If none
        is provided, the entire recording is selected.
    :param title (optional): figure title
    :return: figure handle    
    '''
    # Check if registration metrics are present
    if 'yoff' not in ops:
        logger.warning('looks like the data was not registered -> ignoring')
        return None
    
    # Initialize figure
    logger.info('plotting suite2p registration offsets...')
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharey=True)
    if title is not None:
        fig.suptitle(title)
    for ax in axes[:-1]:
        ax.set_xticks([])
        sns.despine(ax=ax, bottom=True)
    axes[-1].set_xlabel('frames')
    sns.despine(ax=axes[-1], offset={'bottom': 10})

    # Determine frame index boundaries
    if fbounds is None:
        fbounds = [0, ops['nframes'] - 1]
    
    # For each X-Y dimension
    for ax, key in zip(axes, ['y', 'x']):
        
        # Extract and plot rigid offsets for that particular dimension
        offsets = ops[f'{key}off'][fbounds[0]:fbounds[1] + 1]
        ax.plot(offsets, c='k', label='whole frame', zorder=5)

        # If available, extract and plot non-rigid offsets for each sub-block
        if ops['nonrigid']:
            block_offsets = ops[f'{key}off1'][fbounds[0]:fbounds[1] + 1]
            for i, bo in enumerate(block_offsets.T):
                ax.plot(bo, label=f'block {i + 1}')
        
        # Set zero offset line and ylabel
        ax.axhline(0, c='silver', ls='--')
        ax.set_ylabel(key)

    # Move legend to side if it contains many labels
    if ops['nonrigid']: # and output_ops['xoff1'].shape[0] < 40:
        axes[0].legend(bbox_to_anchor=(1, 0), loc='center left')
    
    # Return figure handle
    return fig


def plot_suite2p_PCs(ops, nPCs=3, um_per_px=None):
    '''
    Plot average of top and bottom 500 frames for each PC across the movie
    
    :param ops: suite2p output options dictionary
    :param nPCs: number of principal components to plot (default = 3)
    :param um_per_pixel (optional): number of microns per pixel (for scale bar)
    :return: figure handle
    '''
    # Check if principal components metrics are present 
    if 'regPC' not in ops:
        logger.warning('looks like the data was not registered -> ignoring')
        return None
    
    # Extract principal components array for selected number of PCs 
    PCs = ops['regPC']
    if nPCs is not None:
        PCs = PCs[:, :nPCs]
    nPCs = PCs.shape[1]
    maxPCs, minPCs = PCs

    # Initialize figure    
    logger.info('plotting suite2p PCs average frames...')
    fig, axes = plt.subplots(nPCs, 3, figsize=(9, nPCs * 3))
    fig.suptitle(f'top {nPCs} PCs average images across movie')

    # Plot bottom and top projection images for each sleect PC
    for iPC, (axrow, minPC, maxPC) in enumerate(zip(axes, minPCs, maxPCs)):
        title = f'PC {iPC + 1}'
        axrow[0].imshow(minPC)
        axrow[0].set_title(f'{title}: bottom 500 frames')
        axrow[1].imshow(maxPC)
        axrow[1].set_title(f'{title}: top 500 frames')
        axrow[2].imshow(maxPC - minPC)
        axrow[2].set_title(f'{title}: difference')
    
    # Add scale bar if scale provided
    if um_per_px is not None:
        refnpx = max(maxPC.shape)
        for ax in axes.ravel():
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            add_scale_bar(ax, refnpx, um_per_px, color='w')
    
    # Return figure handle
    return fig


def plot_suite2p_PC_drifts(ops):
    '''
    Plot drifts of PCs w.r.t reference image
    
    :param ops: suite2p output options dictionary
    :return: figure handle
    '''
    # Check if principal components drift metrics are present 
    if 'regDX' not in ops:
        logger.warning('looks like the data was not registered -> ignoring')
        return None
    
    # Extract PC drifts as dictionary
    PCdrifts = ops['regDX'].T
    PCdrifts_dict = {
        'rigid': PCdrifts[0],
        'nonrigid avg': PCdrifts[1],
        'nonrigid max': PCdrifts[2]
    }

    # Initialize figure
    fig, ax = plt.subplots()
    sns.despine(ax=ax)
    ax.set_title('PC drifts w.r.t. reference image')
    ax.set_xlabel('# PC')
    ax.set_ylabel('absolute registration offset')

    # Plot drifts for each PC
    for k, v in PCdrifts_dict.items():
        ax.plot(v, label=k)
    
    # Adjust y-lims
    ylims = ax.get_ylim()
    ax.set_ylim(min(ylims[0], -0.1), max(ylims[1], 1.0))
    
    # Add legend
    ax.legend(frameon=False)

    # Return figure handle
    return fig
    

def plot_suite2p_sparse_maps(ops, um_per_px=None):
    ''' 
    Plot the maps of detected peaks generated at various downsampling factors of
    the sparse detection mode

    :param ops: suite2p output options dictionary
    :param um_per_pixel (optional): number of microns per pixel (for scale bar)
    :return: figure handle
    '''
    # Check if sparse mode maps are present 
    if not ops['sparse_mode']:
        logger.warning('looks like sparse mode was not turned on -> ignoring')
        return None
    
    logger.info('plotting suite2p sparse projection maps...')
    
    # Extract maps
    Vcorr, Vmaps = ops['Vcorr'], ops['Vmap']
    
    # Compute ratios
    refnpx = max(Vcorr.shape)
    npxs = np.array([max(x.shape) for x in Vmaps])  # get map dimensions
    ratios = npxs / refnpx  # get ratios to reference map
    ratios = np.power(2, np.round(np.log(ratios) / np.log(2)))  # round ratios to nearest power of 2
    
    # Find index of map with optimal scale for ROI detection 
    best_scale_px = ops['spatscale_pix']
    if isinstance(best_scale_px, np.ndarray):
        best_scale_px = best_scale_px[0]
    best_scale = np.log(best_scale_px / 3) / np.log(2)
    ibest_scale = np.where(1 / ratios == best_scale)[0][0]

    # Determine maps titles
    titles = [f'{npx} px (x 1/{1 / r:.0f}) map' for npx, r in zip(npxs, ratios)]

    # Add reference map
    titles = ['correlation map'] + titles
    Vmaps = [Vcorr] + Vmaps
    ibest_scale += 1

    # Create figure
    fig, axes = plt.subplots(2, len(Vmaps) // 2, figsize=(len(Vmaps) * 2, 8))
    fig.suptitle(f'Sparse detection maps (best ROI detection scale = {best_scale_px} px)')
    
    # Plot maps
    ths = []
    for ax, title, img in zip(axes.ravel(), titles, Vmaps):
        ths.append(ax.set_title(title))
        ax.imshow(img)
    
    # Flag map with best scale
    plt.setp(ths[ibest_scale], color='g')
    
    # Add scale bar if scale provided
    if um_per_px is not None:
        for ax in axes.ravel():
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            add_scale_bar(ax, refnpx, um_per_px, color='w')    
    
    # Return figure
    return fig


def get_image_and_cmap(ops, key, cmap, pad=True):
    '''
    Extract a reference image from the suite2p options dictionary, pad it if needed
    
    :param ops: suite2p output options dictionary
    :param key: key used to access image in options dictionary
    :param cmap: colormap (string) used to render image
    :param pad: whether to pad image boundaries truncated by registration (default = True)
    :return: image array and colormap used to render it
    '''
    # Get colormap
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad(color='silver')

    # Extract reference image
    refimg = ops[key]
    
    # If required, apply NaN padding to match original frame dimensions
    Lyc, Lxc = refimg.shape
    Ly, Lx = ops['Ly'], ops['Lx']
    if (Lxc < Lx) or (Lyc < Ly) and pad:
        dy, dx = (Ly - Lyc) // 2, (Lx - Lxc) // 2
        refimg = np.pad(refimg, ((dy, dy), (dx, dx)), constant_values=np.nan)
    
    # Return reference image and colormap as a tuple
    return refimg, cmap


def plot_suite2p_ROIs(data, ops, title=None, um_per_px=None, norm_mask=True,
                      superimpose=True, mode='contour', refkey='Vcorr', alpha_ROIs=1.,
                      cmap='viridis'):
    ''' Plot regions of interest identified by suite2p.

        :param data: data dictionary containing contents outputed by suite2p
        :param ops: suite2p output options dictionary
        :param title (optional): figure title
        :param um_per_pixel (optional): number of microns per pixel (for scale bar)
        :param norm_mask (default: True): whether to normalize mask values for each ROI
        :param superimpose (default: True): whether to superimpose ROIs on reference image
        :param mode (default: contour): ROIs render mode ('fill' or 'contour')
        :param refkey: key of reference image to fetch from options dictionary
        :param alpha_ROIs (default: 1): opacity value for ROIs rendering (only in 'fill' mode)
        :param cmap (default: viridis): colormap used to render reference image 
        :return: figure handle
    '''
    logger.info('plotting suite2p identified ROIs...')
    
    # Fetch parameters from data
    iscell = data['iscell'][:, 0].astype(int)
    stats = data['stat']
    Ly, Lx = ops['Ly'], ops['Lx']

    # Initialize pixel matrices
    Z = np.zeros((2, iscell.size, Ly, Lx), dtype=np.float32)
    if mode in ['fill', 'both']:
        # nROIs random hues
        hues = np.random.rand(len(iscell))
        hsvs = np.zeros((2, Ly, Lx, 3), dtype=np.float32)
    if mode in ['contour', 'both']:
        X, Y = np.meshgrid(np.arange(Lx), np.arange(Ly))
        contour_color = {0: 'tab:red', 1: 'tab:orange'}
    
    # Loop through each ROI coordinates
    for i, stat in enumerate(stats):
        
        # Get x, y pixels and associated mask values of ROI
        ypix, xpix, lam = stat['ypix'], stat['xpix'], stat['lam']
        
        # Set Z to ROI 1
        Z[iscell[i], i, ypix, xpix] = 1
        if mode in ['fill', 'both']:
            
            # Normalize mask values if specified
            if norm_mask:
                lam /= lam.max()
            
            # Assign HSV color
            hsvs[iscell[i], ypix, xpix, 0] = hues[i]  # Hue: from random hues array
            hsvs[iscell[i], ypix, xpix, 1] = 1        # Saturation: 1
            hsvs[iscell[i], ypix, xpix, 2] = lam      # Value: from mask
    
    if mode in ['fill', 'both']:
        # Convert HSV -> RGB space
        rgbs = np.array([hsv_to_rgb(*hsv) for hsv in hsvs.reshape(-1, 3)]).reshape(hsvs.shape)
        
        # Add transparency information to RGB matrices if in superimpose mode
        if superimpose:
            rgbs = np.append(rgbs, np.expand_dims(Z.max(axis=1) * alpha_ROIs, axis=-1), axis=-1)

    # Create figure
    if superimpose:
        naxes = 2
        iref, icell, inoncell = [0, 1], 0, 1
    else:
        naxes = 3
        iref, icell, inoncell = [1], 0, 2
    fig, axes = plt.subplots(1, naxes, figsize=(5 * naxes, 5))
    if title is not None:
        fig.suptitle(title)
    
    # Plot reference image 
    refimg, cmap = get_image_and_cmap(ops, refkey, cmap, pad=superimpose)
    for ax in axes[iref]:
        if not superimpose:
            ax.set_title('Reference image')
        ax.imshow(refimg, cmap=cmap)
    
    # Plot cell and non-cell ROIs
    for iax, iscell_bool, label in zip([icell, inoncell], [1, 0], ['Cell', 'Non-cell']):
        ax = axes[iax]
        ax.set_title(f'{label} ROIs ({np.sum(iscell == iscell_bool)})')
        if mode in ['contour', 'both']:  # "contour" mode
            for z in Z[iscell_bool]:
                if z.max() > 0:
                    ax.contour(X, Y, z, levels=[.5], colors=[contour_color[iscell_bool]])
            if not superimpose:
                ax.set_aspect(1.)
        if mode in ['fill', 'both']:  # "fill" mode
            ax.imshow(rgbs[iscell_bool])

    # Add scale bar if scale provided
    if um_per_px is not None:
        for ax in axes:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            add_scale_bar(ax, Lx, um_per_px, color='w')    

    # Tighten and return
    fig.tight_layout()
    return fig


def plot_suite2p_ROI_probs(iscell):
    '''
    Plot the histogram distribution of posterior probabilities of each ROIdistribution
    
    :param iscell: 2D array of (cell classification, cell probabilty) for each ROI
        identified by suite2p
    :return: figure handle
    '''
    # Transform iscell matrix into dataframe
    data = pd.DataFrame(data=iscell, columns=['cell?', 'probability'])

    # Remap the values of the dataframe
    codemap = {1: 'yes', 0: 'no'}
    data = data.replace({'cell?': codemap})
    
    # Create figure
    fig, ax = plt.subplots()
    ax.set_title('posterior cell probability distributions')
    sns.despine(ax=ax)
    
    # Plot histogram distribution of both classes 
    sns.histplot(data, x='probability', hue='cell?', hue_order=list(codemap.values()), bins=30)
    
    # Return figure handle
    return fig


def plot_npix_ratio_distribution(stats, thr=None):
    '''
    Plot the histogram distribution of number of pixels in soma vs whole ROI
    
    :param stats: suite2p stats dictionary
    :param thr (optional): threshold for outlier detection
    :return: figure handle and npix dataframe
    '''
    # Assemble npix and npix soma into dataframe
    data = pd.DataFrame({
        'npix': [x['npix'] for x in stats],
        'npix_soma': [x['npix_soma'] for x in stats]
    })

    # Compute npix ratio column
    data['npix_ratio'] = data['npix'] / data['npix_soma']

    # Initialize figure
    fig, ax = plt.subplots()
    ax.set_title('Ratios of (# pixels ROI) / (# pixels soma)')
    sns.despine(ax=ax)

    # If classification threshold was given, classify outliers
    hue = None
    if thr is not None:
        ax.axvline(thr, ls='--', c='silver')
        data['is_outlier'] = data['npix_ratio'] > thr
        hue = 'is_outlier'
    
    # Plot npix ratio histogram with optional outlier color code
    sns.histplot(data, x='npix_ratio', bins=30, ax=ax, hue=hue)

    # Return figure handle and npix dataframe
    return fig, data


def plot_ROI_traces(data, key=Label.F, xdelimiters=None, ydelimiters=None,
                    ntraces=None, stacked=False):
    '''
    Plot ROI traces for a particular variable

    :param data: fluorescence dataframe
    :param key: name of the column containing the variable of interest
    :param xdelimiters (optional): temporal delimiters at which to draw vertical lines
    :param ydelimiters (optional): vertical delimiters at which to draw horizontal lines
    :param ntraces (optional): number of traces to plot
    :param stacked (optional): whether to stack the traces vertically
    :return: figure handle
    '''
    # Reduce dataset to key of interest
    data = data[key]

    # Get ROIs indexes
    iROIs = data.index.unique(level=Label.ROI)

    # Check and adapt y-delimiters, if any
    if ydelimiters is not None:
        if ydelimiters.ndim  == 1:
            ydelimiters = np.atleast_2d(ydelimiters).T
        if ydelimiters.shape[0] != len(iROIs):
            raise ValueError(
                f'delimiters ({ydelimiters.shape}) do not match number of ROIs ({len(iROIs)})')

    # If number of traces specified, reduce to subset of traces (if applicable)
    if ntraces is not None and ntraces < len(iROIs):
        prefix  = ''
        idxs = random.sample(set(np.arange(iROIs.size)), ntraces)
        iROIs = iROIs[idxs]
        idx = get_mux_slice(data.index)
        idx[data.index.names.index(Label.ROI)] = iROIs
        data = data.loc[tuple(idx)]
        if ydelimiters is not None:
            ydelimiters = ydelimiters[idxs]
    else:
        prefix = 'all '
    nROIs = len(iROIs)
    logger.info(f'plotting {key} traces of {prefix}{nROIs} ROIs...')

    # Adjust height vertical delta if stacked
    if stacked:
        dy = data.groupby(Label.ROI).apply(np.ptp).quantile(.2)
        height = 0.5 * nROIs
    else:
        dy = 0
        height = 4

    # Initialize figure
    fig, ax = plt.subplots(figsize=(12, height))
    ax.set_title(f'{key} traces for {prefix}{nROIs} ROIs')
    sns.despine(ax=ax)
    ax.set_xlabel('frames')
    ax.set_ylabel(key)

    # Plot traces of all selected ROIs
    for i, (iROI, y) in enumerate(data.groupby(Label.ROI)):
        ax.plot(y.values + dy * i, lw=1, rasterized=True)
        if ydelimiters is not None:
            for yd in ydelimiters[i]:
                ax.axhline(yd + dy * i, c='k', ls='--', rasterized=True)
    ax.autoscale(enable=True, tight=True)
    
    # Plot delimiters, if any
    if xdelimiters is not None:
        logger.info(f'adding {len(xdelimiters)} delimiters')
        for iframe in xdelimiters:
            ax.axvline(iframe, color='k', linestyle='--')
    
    # Return figure
    return fig    


def plot_aggregate_traces(data, fps, ykey, aggfunc='mean', yref=None, hue=None, irun=None,
                          itrial=None, tbounds=None, icorrect=None, cmap='viridis',
                          groupbyROI=False, ci=None, ax=None, **kwargs):
    '''
    Plot ROI-aggregated traces across runs/trials or all dataset
    
    :param data: multi-indexed timeseries dataframe
    :param fps: sampling rate (frames / second) 
    :param ykey: name of column containing variable of interest
    :param aggfunc: aggregation function(s) (default: mean)
    :param yref (optional): vertical at which to draw a "reference" horizontal line
    :param hue (optional): variable/index dimension used to split the data before aggregating.
        If none is given, the entire dataset is aggregated
    :param irun (optonal): run index(es) to consider. If none is given, all runs are considered
    :param itrial (optonal): trial index(es) to consider. If none is given, all trials are considered      
    :param tbounds (optional): temporal bounds within the trial interval. If none is given, the entire
        trial interval is plotted 
    :param icorrect (optional): index at which to align the traces vertically. If none if given,
        traces are aligned according to a characteristic quantile of their distribution
    :param cmap (optional): colormap used to render the different traces
    :param groupbyROI (optional): whether to group data by ROI before aggregating
    :param ci: confidence interval used to render shaded areas aroudn traces. If none is given,
        only mean aggregate traces across ROIs are rendered
    :return: figure handle
    '''
    # Transform aggregation function and variable of interest to iterables
    aggfuncs = as_iterable(aggfunc)
    ykey = as_iterable(ykey)

    # Extract timeseries for variables of interest
    subkeys = ykey.copy()
    if hue is not None and hue not in data.index.names:
        subkeys.append(hue)
    plt_data = data[subkeys]

    # Define initial groupby dimensions and figure title
    groupby = [Label.RUN, Label.TRIAL]
    idx = get_mux_slice(plt_data.index)
    groupby = list(filter(lambda k: k in data.index.names, groupby))
    suptitle = []

    # Modify groupby dimensions, amend figure title and filter data if specific
    # runs/trials are specified
    if irun is not None:
        idx[plt_data.index.names.index(Label.RUN)] = irun
        if Label.RUN in groupby:
            groupby.remove(Label.RUN)
        suptitle.append(f'run {irun}')
    if itrial is not None:
        if Label.TRIAL not in plt_data.index.names:
            raise ValueError('trial subsets cannot be selected on trial-averaged data')
        idx[plt_data.index.names.index(Label.TRIAL)] = itrial
        if Label.TRIAL in groupby:
            groupby.remove(Label.TRIAL)
        suptitle.append(f'trial {itrial}')
    plt_data = plt_data.loc[tuple(idx), :]

    # If hue parameter is provided, change groupby to hue only
    if hue is not None:
        groupby = [hue]
    
    # Add frame dimension to groupby to get independent aggregation for each frame index
    groupby.append(Label.FRAME)

    # If specified, add ROIs to groupby to get independent aggretation for each ROI
    if groupbyROI:
        groupby.append(Label.ROI)
    
    # Group data and apply aggregation functions
    logger.info(f'grouping data by {groupby} and averaging...')
    plt_data = plt_data.groupby(groupby).agg(aggfuncs)

    # Add time column to dataframe
    plt_data[Label.FPS] = fps
    add_time_to_table(plt_data)
    
    # Initialize or retrieve figure
    if ax is not None:
        if len(aggfuncs) > 1:
            raise ValueError(
                'provided single axis incompatible with multiple aggregation functions')
        axes = [ax]
        fig = ax.get_figure()
        sns.despine(ax=ax)
    else:
        fig, axes = plt.subplots(1, len(aggfuncs), figsize=(6 * len(aggfuncs), 4))
        if len(suptitle) > 0:
            fig.suptitle(', '.join(suptitle))
        axes = np.atleast_1d(axes)
        for ax in axes:
            sns.despine(ax=ax)
    
    # If correction index was provided, transform it to multi-index
    if icorrect is not None and isinstance(icorrect, int):
        refidx = get_mux_slice(plt_data.index)
        refidx[-1] = icorrect
        refidx = tuple(refidx)
    
    # For each aggregation function
    for ax, k in zip(axes, aggfuncs):

        # If aggfunc is a callable object, extract its name
        if callable(k):
            k = k.__name__

        # Initialize axis
        ax.set_title(f'{k} traces')
        ax.set_xlabel(Label.TIME)
        ax.set_ylabel(ykey[0])
        if tbounds is not None:
            ax.set_xlim(tbounds)
        ax.axvline(0, c='k', ls='--')
        if yref is not None:
            ax.axhline(yref, c='k', ls='--')
        
        custom_lines = []

        # For each variable of interest
        for i, y in enumerate(ykey): 
                       
            # If some kind of vertival correction is specified 
            if icorrect is not None:

                # If integer, just correct according to defined frame index
                if isinstance(icorrect, int):
                    ycorrect = plt_data.loc[refidx, :][(y, k)].droplevel(Label.FRAME)
                
                # Otherwise, correct according to distribution quantile
                elif isinstance(icorrect, str) and icorrect.startswith('q'):
                    q = parse_quantile(icorrect)
                    if hue is not None:
                        ycorrect = plt_data[(y, k)].groupby(hue).quantile(q)
                    else:
                        ycorrect = plt_data[(y, k)].quantile(q)
                
                # Othwerwise, throw error
                else:
                    raise ValueError(f'unknown correction: {icorrect}')
                
                # Correct associated traces
                plt_data[(y, k)] = plt_data[(y, k)] - ycorrect
            
            # Plot aggregated traces
            sns.lineplot(
                data=plt_data, x=Label.TIME, y=(y, k), hue=hue, ci=ci,
                palette=cmap, legend='auto', ax=ax, **kwargs)
            if hue is None:
                custom_lines.append(
                    Line2D([0], [0], color=f'C{i}', lw=4))
    
    if len(custom_lines) > 1:
        ax.legend(custom_lines, ykey)

    # Tighten figure layout
    fig.tight_layout()

    # Return figurte handle
    return fig


def plot_multivar_traces(data, iROI=None, irun=None, itrial=None, delimiters=None,
                         ylabel=None, ybounds=None, cmap=None, title=None):
    '''
    Plot traces of one or multiple variables from a fluorescence timeseries dataframe
    
    :param data: fluorescence dataframe
    :param iROI (optional): ROI index
    :param irun (optional: run index
    :param itrial (optional): trial index
    :param delimiters (optional): temporal delimitations (shown as vertical lines)
    :param ylabel: name of the variable to plot
    :param ybounds (optional): y-axis limits
    :param cmap (optional): colormap used to render traces
    :param title (optional): figure title 
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
    iframes = filtered_data.index.unique(level=Label.FRAME).values
    ionset = iframes[0]
    
    # If trials are marked in data -> adjust onset according to 1st trial index
    if Label.TRIAL in filtered_data.index.names:
        itrials = filtered_data.index.unique(level=Label.TRIAL).values
        ionset += itrials[0] * len(iframes)

    # Create figure
    iROI = sorted(as_iterable(iROI))  # sort ROIs to ensure consistent looping order 
    nROIs = len(iROI)
    npersignal /= nROIs
    fig, axes = plt.subplots(nROIs, 1, figsize=(12, nROIs * 4))
    axes = as_iterable(axes)
    sns.despine()
    for ax in axes[:-1]:
        ax.set_xticks([])
    axes[-1].set_xlabel('frames')
    if ylabel is None:
        if nsignals > 1:
            logger.warning('ambiguous y-labeling for more than 1 signal')
        ylabel = filtered_data.columns[0]
    for ax in axes:
        ax.set_ylabel(ylabel)
    del filters[Label.ROI]
    parsed_title = ' - '.join(filters.values()) + f' trace{plural(nsignals)}'
    parsed_title = [parsed_title] * len(axes)
    if title is not None:
        if not is_iterable(title):
            title = [title] * len(axes)
        parsed_title = [f'{pt} ({t})' for pt, t in zip(parsed_title, title)]
    for ir, ax, pt in zip(iROI, axes, parsed_title):
        ax.set_title(f'ROI {ir} {pt}')

    # Generate x-axis indexes
    xinds = np.arange(npersignal) + ionset

    # Determine traces colors
    if cmap is not None:
        colors = plt.get_cmap(cmap)(np.linspace(0, 1, nsignals))
    else:
        colors = [None] * nsignals

    # Plot traces
    logger.info(f'plotting {nsignals} fluorescence trace(s)...')
    for (ir, subdata), ax in zip(filtered_data.groupby(Label.ROI), axes):
        for color, col in zip(colors, subdata):
            ax.plot(xinds, subdata[col], c=color, label=col)

    # Restrict y axis, if needed
    if ybounds is not None:
        for ax in axes:
            ax.set_ylim(*ybounds)

    # Plot delimiters, if any
    if delimiters is not None:
        logger.info(f'adding {len(delimiters)} delimiters')
        for ax in axes:
            for iframe in delimiters:
                ax.axvline(iframe, color='k', linestyle='--')

    # Add legend
    if nsignals > 1:
        for ax in axes:
            ax.legend(frameon=False)
    
    # Return figure handle
    return fig


def plot_linreg(data, iROI, x=Label.F_NEU, y=Label.F_ROI):
    '''
    Plot linear regression between neuropil and ROI fluorescence traces
    
    :param data: fluorescence timereseries data
    :param iROI: subset of ROIs to plot
    :param x: name of column containing neuropil traces
    :param y: name of column containing ROI traces
    :return: figure handle
    '''
    # Initialize figure
    fig, axes = plt.subplots(1, iROI.size, figsize=(iROI.size * 3.5, 4))
    fig.suptitle('Linear regressions F_ROI = A * F_NEU + B')
    for ax in axes:
        sns.despine(ax=ax)
        ax.set_aspect(1.)

    # For each selected ROI 
    for ax, ir in zip(axes, iROI):
        
        # Select sub-data
        logger.info(f'plotting F_ROI = f(F_NEU) with linear regression for ROI {ir}...')
        subdata = data.loc[pd.IndexSlice[ir, :, :, :]]
        ax.set_title(f'ROI {ir}')

        # Plot linear regression and add reference y=x line
        sns.regplot(
            data=subdata, x=x, y=y, ax=ax, label='data',
            robust=True, ci=None)
        ax.axline((0, 0), (1, 1), ls='--', color='k')

        # Perform robust linear regression and extract fitted parameters
        bopt, aopt = robust_linreg(subdata)
        logger.info(f'optimal fit: F_ROI = {aopt:.2f} * F_NEU + {bopt:.2f}')
        
        # Plot robust linear fit for comparison with standard one
        xvec = np.array([subdata[x].min(), subdata[x].max()])
        ax.plot(xvec, aopt * xvec + bopt, label='linear fit')

        # Add legend
        ax.legend()
    
    # Tighten figure
    fig.tight_layout()

    # Return figure handle
    return fig


def mark_trials(ax, mask, iROI, irun, npertrial, color='C1'):
    '''
    Mark trials on whole-run trace that meet specific stats condition.
    
    :param ax: plot axis
    :param mask: multi-indexed boolean mask series indicating which trial to mark
    :param iROI: ROI index
    :param irun: run index
    :param npertrial: number of frames per trial
    :param color (optional): color used to mark the trials
    '''
    # Reduce mask to ROI(s) and run(s) of interest 
    mask = mask.loc[pd.IndexSlice[iROI, irun, :]]

    # For each trial meeting the condition
    for (_, _, itrial) in mask[mask == 1].index:

        # Get trial start and end indexes in the run trace
        istart = npertrial * itrial + FrameIndex.STIM
        iend = istart + npertrial

        # Plot shaded area over trial interval 
        ax.axvspan(istart, iend, fc=color, ec=None, alpha=.3)
        

def plot_cell_map(ROI_masks, Fstats, ops, title=None, um_per_px=None, refkey='Vcorr',
                  mode='contour', cmap='viridis', hue=Label.ROI_RESP_TYPE, legend=True, alpha_ROIs=0.7, 
                  ax=None, verbose=True):
    '''
    Plot spatial distribution of cells (per response type) on the recording plane.

    :param ROI_masks: ROI-indexed dataframe of (x, y) coordinates and weights

    :param Fstats: statistics dataframe
    :param ops: suite2p output options dictionary
    :param title (optional): figure title
    :param um_per_px (optional): spatial resolution (um/pixel). If provided, ticks and tick labels
        on each image are replaced by a scale bar on the graph.
    :param refkey: key used to access the reference image in the output options dictionary
    :param mode (default: contour): ROIs render mode ('fill' or 'contour')
    :param cmap (default: viridis): colormap used to render reference image
    :param hue: hue parameter determining the color of each ROI
    :param alpha_ROIs (default: 1): opacity value for ROIs rendering (only in 'fill' mode)
    :return: figure handle
    '''
    # Fetch parameters from data
    slog = 'plotting cells map'
    Ly, Lx = ops['Ly'], ops['Lx']
    if hue == Label.ROI_RESP_TYPE:
        rtypes_per_ROI = get_response_types_per_ROI(Fstats, verbose=verbose)
        rtypes = get_default_rtypes()
        count_by_type = {k: (rtypes_per_ROI == k).sum() for k in rtypes}
        colors = Palette.RTYPE
        slog = f'{slog} color-coded by response type'
    else:
        iROIs = Fstats.index.unique(level=Label.ROI)
        rtypes_per_ROI = pd.Series(data=['notype'] * len(iROIs), index=iROIs) 
        rtypes = ['notype']
        count_by_type = {'notype': len(iROIs)}
        colors = {'notype': 'silver'}
        legend = False
    
    if verbose:
        logger.info(f'{slog}...')

    # Initialize pixels by cell matrix
    idx_by_type = dict(zip(rtypes, np.arange(len(rtypes))))
    Z = np.zeros((len(rtypes), rtypes_per_ROI.size, Ly, Lx), dtype=np.float32)

    # Compute mask per ROI & response type
    for i, (rtype, (_, ROI_mask)) in enumerate(zip(rtypes_per_ROI, ROI_masks.groupby(Label.ROI))):
        Z[idx_by_type[rtype], i, ROI_mask['ypix'], ROI_mask['xpix']] = 1
    
    if mode == 'contour':
        # Initialize pixel matrices
        X, Y = np.meshgrid(np.arange(Lx), np.arange(Ly))
    else:
        # Stack Z matrices along ROIs to get 1 mask per type
        masks = np.array([z.max(axis=0) for z in Z])
    
        # Assign color and transparency to each mask
        rgbs = np.zeros((*masks.shape, 4))
        for i, (c, mask) in enumerate(zip(colors.values(), masks)):
            if isinstance(c, str):
                c = to_rgb(c)
            rgbs[i][mask == 1] = [*c, alpha_ROIs]
    
    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()
    if title is not None:
        ax.set_title(title)
    
    # Plot reference image 
    refimg, cmap = get_image_and_cmap(ops, refkey, cmap, pad=True)
    ax.imshow(refimg, cmap=cmap)
    
    # Plot cell and non-cell ROIs
    for rtype, idx in idx_by_type.items():
        c = colors[rtype]
        if mode == 'contour':  # "contour" mode
            for z in Z[idx]:
                if z.max() > 0:
                    ax.contour(X, Y, z, levels=[.5], colors=[c])
        else:  # "fill" mode
            ax.imshow(rgbs[idx])

    # Add scale bar if scale provided
    if um_per_px is not None:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        add_scale_bar(ax, Lx, um_per_px, color='w')

    # Add legend
    if legend:
        labels = [f'{k} ({v})' for k, v in count_by_type.items()]
        if mode == 'contour':
            legfunc = lambda k: dict(c='none', marker='o', mfc='none', mec=colors[k], mew=2)
        else:
            legfunc = lambda k: dict(c='none', marker='o', mfc=colors[k], mec='none')
        leg_items = [
            Line2D([0], [0], label=l, ms=10, **legfunc(c))
            for c, l in zip(colors, labels)]
        ax.legend(handles=leg_items, bbox_to_anchor=(1, 1), loc='upper left', frameon=False)
    
    return fig


def plot_cell_maps(ROI_masks, stats, ops, title=None, colwrap=5, mode='contour',
                   hue=Label.ROI_RESP_TYPE, **kwargs):
    
    logger.info('plotting cell maps...')

    # Divide inputs per dataset
    masks_groups = dict(tuple(ROI_masks.groupby(Label.DATASET)))
    stats_groups = stats.groupby(Label.DATASET)
    ndatasets = stats_groups.ngroups

    # Create figure
    ncols = min(ndatasets, colwrap)
    nrows = int(np.ceil(ndatasets / colwrap))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))
    axes = axes.ravel()

    # Plot map for each dataset
    with tqdm(total=len(axes) - 1, position=0, leave=True) as pbar:
        for ax, (dataset_id, sgroup) in zip(axes, stats_groups):
            mgroup = masks_groups[dataset_id]
            ogroup = ops[dataset_id]
            plot_cell_map(
                mgroup, sgroup, ogroup, title=dataset_id, mode=mode, 
                um_per_px=ogroup['micronsPerPixel'], ax=ax, legend=False, hue=hue, 
                verbose=False, **kwargs)
            ax.set_aspect(1.)
            pbar.update()

    # Hide remaining axes
    for ax in axes[ndatasets:]:
        ax.set_visible(False)
    
    # Add title
    fig.suptitle(title)

    # Add legend if necessary
    if hue is not None:
        if ndatasets % colwrap > 0:
            fig.subplots_adjust(right=0.8)
        if mode == 'contour':
            legfunc = lambda color: dict(c='none', marker='o', mfc='none', mec=color, mew=2)
        else:
            legfunc = lambda color: dict(c='none', marker='o', mfc=color, mec='none')
        leg_items = [
            Line2D([0], [0], label=label, ms=10, **legfunc(c))
            for c, label in zip(Palette.RTYPE.values(), get_default_rtypes())]
        axes[ndatasets - 1].legend(
            handles=leg_items, bbox_to_anchor=(1, 1), loc='upper left', frameon=False)
    
    return fig


def plot_trial_heatmap(data, key, fps, irun=None, itrial=None, title=None, col=None,
                       colwrap=4, row=None, cmap=None, center=None, vmin=None, vmax=None,
                       quantile_bounds=(.01, .99), mark_stim=True, sort_ROIs=False,
                       col_order=None, col_labels=None, row_order=None, row_labels=None,
                       rect_markers=None, rasterized=False):
    '''
    Plot trial heatmap (average response over time of each cell within trial interval,
    culstered by similarity).
    
    :param data: multi-indexed timeseries dataframe.
    :param key: name of column containing variable of interest
    :param fps: sampling frequency (frames / second)
    :param irun (optional): run(s) subset
    :param itrial (optional): trial(s) subset
    :param title (optional): figure title
    :param col (optional): parameter/index dimension used to split the data on different axes.
        If none is given, the whole dataset is aggregated on to a single heatmap.
    :param colwrap: maximum of heatmaps per row 
    :param cmap: colormap used to render heatmap
    :param quantile_bounds (optional): distirbution quantiles used to set the colorbar limits.
        If none, the data bounds are taken.
    :param mark_stim: whether to mark the stimulus onset with a vertical line (default = True)
    :return: figure handle
    '''
    # Filter data according to selected run(s) & trial(s)
    idx = get_mux_slice(data.index)
    if irun is not None:
        idx[data.index.names.index(Label.RUN)] = irun
    if itrial is not None and Label.TRIAL in data.index.names:
        idx[data.index.names.index(Label.TRIAL)] = itrial
    data = data.loc[tuple(idx), :]

    # if row_order is not None:
    #     irow = list(data.index.names).index(row)
    #     rowmap = dict(zip(np.sort(row_order), row_order))
    #     def mapper(x):
    #         l = list(x)
    #         l[irow] = rowmap[l[irow]]
    #         return tuple(l)
    #     logger.info(f'sorting data {row}...')
    #     data.index.map(mapper)

    # Determine pivot index keys, number of rows per map, and resulting aspect ratio
    extra_pivot_index_keys = []
    if Label.DATASET in data.index.names:
        extra_pivot_index_keys.append(Label.DATASET)    
        nROIs_per_pivot = {}
        nROIs_per_pivot = pd.Series({
            k: len(tmp.index.unique(Label.ROI))
            for k, tmp in data.groupby(Label.DATASET)
        }).rename('ROI count')
    if row is not None:
        extra_pivot_index_keys.append(row)
        colwrap = data.groupby(col).ngroups
        nROIs = len(data.index.unique(Label.ROI))
        nROIs_per_pivot = pd.Series({
            k: nROIs for k, _ in data.groupby(row)
        }).rename('ROI count')
    if len(extra_pivot_index_keys) > 0:
        nROIs_per_pivot.index.names = extra_pivot_index_keys
        ysep_ends = nROIs_per_pivot.cumsum()
        ysep_starts = ysep_ends.shift(periods=1, fill_value=0.)
        ysep_mids = (ysep_starts + ysep_ends) / 2
        if row is not None:
            ysep_mids = ysep_mids.rename(f'{row} {{}}'.format) 
        pivot_index_keys = [Label.ROI] + extra_pivot_index_keys
    else:
        pivot_index_keys = Label.ROI
        ysep_ends = None
    nrowspermap = len(data.groupby(pivot_index_keys).first())
    aspect_ratio = nrowspermap / 100

    # Rectilinearize dataframe
    data = rectilinearize(data[key]).to_frame()

    # Determine colormap if required
    if cmap is None:
        if data[key].min() < 0.:
            cmap = 'icefire'
            center = 0
        else:
            cmap = 'viridis'

    # Extract plotting boundaries of variable of interest
    if quantile_bounds is not None:
        vmin_est, vmax_est = [data[key].quantile(x) for x in quantile_bounds]
    else:
        vmin_est, vmax_est = data[key].agg(['min', 'max'])
    if vmin is None:
        vmin = vmin_est
    if vmax is None:
        vmax = vmax_est

    # Add time column to dataframe
    data = add_time_to_table(data.copy(), fps=fps)
    
    # Group data according to col and/or row parameter(s)
    if col is not None:
        groups = data.groupby(col)
    else:
        groups = [('all', data)]

    # Initialize figure
    if col_order is not None:
        naxes = len(col_order)
    else:
        naxes = len(groups)
    nrows, ncols = int(np.ceil(naxes / colwrap)), min(colwrap, naxes)
    width = ncols * 2.5  # inches
    height = nrows * 2.5  # inches
    if nrows == 1:
        height *= aspect_ratio
    # Constrain figure height to fit letter aspect ratio
    height = min(height, width * 11 / 8.5)
    fig, axes = plt.subplots(
        nrows, ncols, 
        figsize=(width, height))
    if naxes == 1:
        axes = np.array([axes])
    fig.tight_layout()
    top = 0.9 if title is None else 0.8
    fig.subplots_adjust(bottom=0.1, right=0.8, top=top, hspace=.5)
    cbar_ax = fig.add_axes([0.85, 0.1, 0.05, top - 0.1])
    cbar_ax.set_title(key)

    # For each axis and data group
    logger.info(f'plotting {key} trial heatmap{"s" if naxes > 1 else ""}...')
    if col_order is None:
        col_order = np.arange(len(groups))
    else:
        col_order = np.asarray(col_order)
    add_cbar = True
    with tqdm(total=naxes - 1, position=0, leave=True) as pbar:
        for i, (glabel, gdata) in enumerate(groups):
            # Find axis position
            if i in col_order:
                iax = np.where(col_order == i)[0][0]
                ax = axes.ravel()[iax]
                # Generate 2D table of average traces per ROI
                table = gdata.pivot_table(
                    index=pivot_index_keys, columns=Label.TIME, values=key, aggfunc=np.mean)
                # Get row order
                if row_order is None:
                    row_order_exp = gdata.groupby(pivot_index_keys).first().index
                else:
                    row_order_exp = pd.MultiIndex.from_product([
                        gdata.index.unique(Label.ROI), row_order])
                table = table.reindex(row_order_exp, axis=0)

                if sort_ROIs:
                    # Compute metrics average in pre-stimulus and response windows for each ROI
                    ypre = apply_in_window(
                        gdata, key, FrameIndex.PRESTIM, verbose=False)
                    ypost = apply_in_window(
                        gdata, key, FrameIndex.RESPONSE, verbose=False)
                    ydiff = (ypost - ypre).rename('val')
                    # Remove column sorter from index, if present
                    if col is not None and col in ydiff.index.names:
                        ydiff = ydiff.droplevel(col)
                    # If additional pivot keys, group by them before sorting
                    sortby = []
                    if len(extra_pivot_index_keys) > 0:
                        sortby += extra_pivot_index_keys
                    # Average across remaining dimensions
                    ydiff = ydiff.groupby([Label.ROI] + sortby).mean()
                    # Sort by ascending differential metrics
                    sortby.append('val')
                    ydiff = ydiff.to_frame().sort_values(sortby)['val']
                    # Re-index table according to row order 
                    table = table.reindex(ydiff.index.values, axis=0)

                # Plot associated trial heatmap
                sns.heatmap(
                    data=table, ax=ax, vmin=vmin, vmax=vmax, 
                    cbar=add_cbar, cbar_ax=cbar_ax, center=center, cmap=cmap,
                    xticklabels=table.shape[1] - 1, # only render 2 labels at extremities
                    yticklabels=False, 
                    rasterized=rasterized)
                
                add_cbar = False
                
                # Set axis background color
                ax.set_facecolor('silver')

                # Correct x-axis label display
                ax.set_xticklabels([f'{float(x.get_text()):.1f}' for x in ax.get_xticklabels()])
                ax.set_xlabel(ax.get_xlabel(), labelpad=-10)

                # Add column title (only if informative)
                if glabel != 'all':
                    coltitle = f'{col} {glabel}'
                    if col_labels is not None:
                        coltitle = f'{coltitle} ({col_labels[iax]})'
                    ax.set_title(coltitle)
                
                # Add stimulus onset line, if specified
                if mark_stim:
                    istim = np.where(table.columns.values == 0.)[0][0]
                    ax.axvline(istim, c='w', ls='--', lw=1.)
                
                # Add dataset separators if available
                if ysep_ends is not None:
                    for y in ysep_ends:
                        ax.axhline(y, c='w', ls='--', lw=2.)
                    if i == 0:
                        ax.set_yticks(ysep_mids)
                        ax.set_yticklabels(
                            ysep_mids.index, rotation='vertical', va='center')
                        ax.tick_params(axis='y', left=False)

                # Add rectangular markers, if any 
                if rect_markers is not None:
                    if col in rect_markers.index.names:
                        refidx = get_mux_slice(rect_markers.index)
                        if glabel in rect_markers.index.unique(col):
                            refidx[rect_markers.index.names.index(col)] = glabel
                            submarks = rect_markers.loc[tuple(refidx)]
                            for dataset, color in submarks.iteritems():
                                yb, yt = ysep_starts.loc[dataset], ysep_ends.loc[dataset]
                                ax.add_patch(Rectangle(
                                    (ax.get_xlim()[0], yb), 
                                    ax.get_xlim()[1] - ax.get_xlim()[0], yt - yb,
                                    fc='none', ec=color, lw=10))

            pbar.update()
    
    # Hide remaining axes
    for ax in axes.ravel()[i + 1:]:
        ax.set_visible(False)
    
    # Add main figure title if specified
    if title is not None:
        fig.suptitle(title)

    # Return figure handle
    return fig


def add_label_mark(ax, x, cmap=None, w=0.1):
    '''
    Add a color marker in the top right corner of a plot
    
    :param ax: axis object
    :param x: label value
    :param cmap: colormap (or dictionary) used to determine label background color
    :param w: relative width of the marker rectangle w.r.t. axis
    '''
    # Extract colormap if not given
    if cmap is None:
        cmap = plt.get_cmap('viridis')
    
    # Derive string and background color for label value
    if isinstance(cmap, dict):
        c = cmap[x]
        s = x
    else:
        c = cmap(x)
        s = f'{x:.2f}'
    
    # Determine text color depending on background brightness
    brightness = rgb_to_hsv(*c[:3])[-1]
    tcolor = 'w' if brightness < 0.7 else 'k'

    # Add marker and label
    ax.add_patch(Rectangle((1 - w,  1 - w), w, w, transform=ax.transAxes, fc=c, ec=None))
    ax.text(1 - w / 2,  1 - w / 2, s, transform=ax.transAxes, ha='center', va='center', c=tcolor)


def plot_from_data(data, xkey, ykey, xbounds=None, ybounds=None, aggfunc='mean', weightby=None,
                   ci=CI, legend='full', err_style='band', ax=None, alltraces=False, kind='line',
                   nmaxtraces=None, hue=None, hue_order=None, col=None, col_order=None, 
                   label=None, title=None, dy_title=0.6, markerfunc=None, max_colwrap=5, ls='-', lw=2,
                   height=None, aspect=1.5, alpha=None, palette=None, marker=None, markersize=7,
                   hide_col_prefix=False, col_count_key=None, color=None, **filter_kwargs):
    ''' Generic function to draw line plots from the experiment dataframe.
    
    :param data: experiment dataframe
    :param xkey: key indicating the specific signals to plot on the x-axis
    :param ykey: key indicating the specific signals to plot on the y-axis
    :param xbounds (optional): x-axis limits for plot
    :param ybounds (optional): y-axis limits for plot
    :param aggfunc (optional): method for aggregating across multiple observations within group.
    :param weightby (optional): column used to weight observations upon aggregration.
    :param ci (optional): size of the confidence interval around mean traces (int, “sd” or None)
    :param err_style (“band” or “bars”): whether to draw the confidence intervals with translucent error bands or discrete error bars.
    :param alltraces (optional): whether to plot all individual traces
    :param nmaxtraces (optional): maximum number of traces that can be plot per group
    :param hue (optional): grouping variable that will produce lines with different colors.
    :param col (optional): grouping variable that will produce different axes.
    :param label (optional): add a label indicating a specific field value on the plot (when possible)
    :param title (optional): figure title (deduced if not provided)
    :param markerfunc (optional): function to draw additional markers for individual traces
    :param filter_kwargs: keyword parameters that are passed to the filter_data function
    :return: figure handle
    '''
    ###################### Filtering ######################

    # Filter data based on selection criteria
    filtered_data, filters = filter_data(data, full_output=True, **filter_kwargs)
    
    # Remove problematic trials (i.e. that contain NaN for the column of interest) 
    filtered_data = filtered_data.dropna(subset=[ykey])

    ###################### Process log ######################
    s = []

    # If col set to ROI
    if col == Label.ROI:
        # if only 1 ROI -> remove column assignment 
        if len(filtered_data.index.unique(level=Label.ROI)) == 1:
            col = None
    
    # If col set to ROI -> remove ROI filter info
    if col == Label.ROI and Label.ROI in filters:
        del filters[Label.ROI]
    
    # Adjust col_wrap and figure height if col was specified
    if col is not None:
        s.append(f'grouping by {col}')
        col_wrap = min(len(filtered_data.groupby(col)), max_colwrap)
        if height is None:
            height = 5.
        if ax is not None:
            raise ValueError(f'cannot sweep over {col} with only 1 axis')
    else:
        col_wrap = None
        if height is None:
            height = 4.      
    
    # Remove hue parameters from filters, if present 
    if hue is not None:
        s.append(f'grouping by {hue}')
        if hue == Label.ROI and Label.ROI in filters:
            del filters[Label.ROI]
    
    # Adjust filters if weightby is specified
    if weightby is not None:
        if weightby not in data:
            raise ValueError(f'weighting variable ({weightby}) not found in data')
        if aggfunc != 'mean':
            raise ValueError(f'cannot use {weightby}-weighting with {aggfunc} aggregation')
        s.append(f'weighting by {weightby}')
        filters['weight'] = f'weighted by {weightby}'
    
    # Add aggregation and confidence interval information to log, if specified
    if aggfunc is not None:
        s.append('averaging')
    if ci is not None:
        s.append('estimating confidence intervals')
    
    # Log
    s = f'{", ".join(s)} and ' if len(s) > 0 else ''
    logger.info(f'{s}plotting {aggfunc} {ykey} vs. {xkey} ...')
    
    # Determine color palette depending on hue parameter
    if palette is None:
        palette = {
            None: None,
            Label.P: Palette.P,
            Label.DC: Palette.DC,
            Label.ROI_RESP_TYPE: Palette.RTYPE
        }.get(hue, None)

    ###################### Aggregating function ######################

    # If weighting variable is specified, define aggregating function accordingly
    if weightby is not None:
        def aggfunc(x):
            if isinstance(x, pd.Series):
                # Series case (when computing aggregate) -> weight by "weightby" column values at same index
                weights = filtered_data.loc[x.index, weightby]
            else:
                # Array case case (when computing CI) -> uniform weighting
                weights = np.ones_like(x)
            # Return weighted average       
            return (x * weights).sum() / weights.sum()

    ###################### Mean traces and CIs ######################

    # Default plot arguments dictionary
    plot_kwargs = dict(
        data      = filtered_data, # data
        x         = xkey,          # x-axis
        y         = ykey,          # y-axis
        ls        = ls,            # line style
        marker    = marker,        # marker type
        markersize = markersize,   # marker size
        hue       = hue,           # hue grouping variable
        hue_order = hue_order,     # hue plotting order 
        estimator = aggfunc,       # aggregating function
        color     = color,         # plot color
        ci        = ci,            # confidence interval estimator
        err_style = err_style,     # error visualization style 
        lw        = lw,           # line width
        palette   = palette,       # color palette
        legend    = legend         # use all hue entries in the legend
    )

    # Adjust traces alpha if specified
    if alpha is not None:
        plot_kwargs['alpha'] = alpha

    # If axis object is provided -> add it to the dictionary and call 
    # axis-level plotting function
    if ax is not None:
        plot_kwargs['ax'] = ax
        if kind == 'line':
            sns.lineplot(**plot_kwargs)
        else:
            sns.scatterplot(**plot_kwargs)
        axlist = [ax]
        fig = ax.get_figure()
    
    # Otherwise, add figure-level plotting arguments and call figure-level plotting function
    else:
        plot_kwargs.update(dict(
            kind     = kind,     # kind of plot
            height   = height,   # figure height
            aspect   = aspect,   # width / height aspect ratio of each figure axis
            col_wrap = col_wrap, # how many axes per row
            col      = col,      # column (i.e. axis) grouping variable
            col_order = col_order
        ))
        fg = sns.relplot(**plot_kwargs)
        axlist = fg.axes.ravel()
        fig = fg.figure

    # If col is specified
    if col is not None:
        # Remove column prefix from axes titles, if specified
        if hide_col_prefix:
            for ax in fig.axes:
                ax.set_title(ax.get_title().replace(f'{col} = ', ''))

        # Reduce title size if too long
        for ax in fig.axes:
            s = ax.get_title()
            if len(s) > 20:
                ax.set_title(s, fontsize=10)

        # Add count per column to axes titles, if specified
        if col_count_key is not None:
            countfunc = lambda df: len(df.groupby(col_count_key).first())
            counts_per_col = filtered_data.groupby(col).apply(countfunc)            
            # If no column order provided, assume columns go by decreasing count
            if col_order is None:
                col_order = counts_per_col.sort_values(ascending=False).index.values
            for ax, k in zip(fig.axes, col_order):
                ax.set_title(f'{ax.get_title()} ({counts_per_col.get(k, 0)})')
        
    # Remove right and top spines
    sns.despine()

    ###################### Individual traces ######################
    
    if alltraces:
    
        # Aggregation keys = all index keys that are not "frame" 
        aggkeys = list(filter(
            lambda x: x is not None and x != Label.FRAME,
            filtered_data.index.names))
        
        # Remove run from aggregation keys if xkey is a run-dependent parameter
        if xkey in [Label.P, Label.DC] and Label.RUN in aggkeys:
            aggkeys.remove(Label.RUN)
        
        logger.info(f'plotting individual {ykey} vs. {xkey} traces...')

        # Get number of conditions to plot
        nconds = len(axlist) * len(axlist[0].get_lines())
        
        # Default opacity index of individual trace
        alpha_trace = 0.2
        
        with tqdm(total=nconds - 1, position=0, leave=True) as pbar:
            
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
                
                # Use color code for individual traces only if 1 group on the axis
                use_color_code = len(groups) == 1
                
                # For each hue group
                for l, (_, gr) in zip(ax.get_lines(), groups):
                    
                    # Get group color from aggregate trace 
                    group_color = l.get_color()
                    
                    # Generate pivot table for the value
                    table = gr.pivot_table(
                        index=xkey,  # index = xkey
                        columns=aggkeys,  # each column = 1 line to plot 
                        values=ykey)  # values
                    
                    # Randomly select n traces to plot, if more than specified max
                    _, ntraces = table.shape
                    itraces = np.arange(ntraces)
                    if nmaxtraces is not None and ntraces > nmaxtraces:
                        itraces = random.sample(set(itraces), nmaxtraces)
                    
                    # Get response classification of each trace (if available)
                    if Label.IS_RESP in gr:
                        is_resps = gr[Label.IS_RESP].groupby(aggkeys).first()
                    else:
                        is_resps = [None] * ntraces
                    
                    # Plot a line for each entry in the pivot table that is part of 
                    # the selected subset
                    for i, (x, is_resp) in enumerate(zip(table, is_resps)):
                        if i in itraces:
                            
                            # Determine trace color
                            if use_color_code:
                                if is_resp is not None:
                                    color = {True: 'g', False: 'r'}[is_resp]
                                else:
                                    color = None
                            else:
                                color = group_color
                            
                            # Plot trace in the background
                            plotfunc = ax.plot if kind == 'line' else ax.scatter
                            plotfunc(table[x].index, table[x].values, color=color,
                                     alpha=alpha_trace, zorder=-10)
                            
                            # Add individual trace markers if specified
                            if markerfunc is not None:
                                markerfunc(ax, table[x], color=color, alpha=alpha_trace)
                    
                    pbar.update()

    
    ###################### Axes limits ######################

    # For each axis
    for ax in axlist:

        # Adjust x-axis if specified
        if xbounds is not None:
            ax.set_xlim(*xbounds)
        
        # Adjust y-axis if specified
        if ybounds is not None:
            ylims = ax.get_ylim()
            ybounds_ax = [yb if yb is not None else yl for yb, yl in zip(ybounds, ylims)]
            ax.set_ylim(*ybounds_ax)

    ###################### Markers ######################

    # If indicator provided, check number of values per axis
    if label is not None:
        try:
            label_values = filtered_data[label]
        except KeyError:
            label_values = get_trial_aggregated(filtered_data)[label]
        if col is not None:
            label_values_per_ax = label_values.groupby(col).unique().values
        else:
            label_values_per_ax = [label_values.unique()]
        if label == Label.ROI_RESP_TYPE:
            label_cmap = dict(zip([x[0] for x in label_values_per_ax], sns.color_palette(Palette.RTYPE)))
        else:
            label_cmap = None
        
    # For each axis, add label for specific field value, if specified and possible
    for iax, ax in enumerate(axlist):
        if label is not None:
            label_values = label_values_per_ax[iax]
            if len(label_values) == 1:
                add_label_mark(ax, label_values[0], cmap=label_cmap)

    ###################### Title & legend ######################

    # Determine title 
    if title is None:
        if filters is None:
            filters = {'misc': 'all responses'}
        title = ' - '.join(filters.values())
    
    # Use appropriate title function depending on number of axes
    if col is None: 
        # If only 1 axis (i.e. no column grouping) -> add to axis
        axlist[0].set_title(title)
    else:
        # Otherwise -> add as suptitle
        height = fig.get_size_inches()[1]
        fig.subplots_adjust(top=1 - dy_title / height)
        fig.suptitle(title)

    # Return figure handle
    return fig


def mark_response_peak(ax, trace, tbounds=None, color='k', alpha=1.):
    '''
    Mark the peak of a response trace
    
    :param ax: axis object
    :param trace: pandas Series representing the trace values
    :param tbounds (optional): boundaries of the time window in which a peak must be detected.
        If none is given, the entire timseries is taken
    :param color (optional): color of the peak marker
    :param alpha (optional): transparency of the peak marker
    '''
    # Restrict trace to specific time interval (if specified)
    if tbounds is not None:
        trace = trace[(trace.index >= tbounds[0]) & (trace.index <= tbounds[1])]

    # Find peak on the response trace
    ipeak, ypeak = find_response_peak(
        trace, return_index=True)

    # If peak has been detected, plot it
    if not np.isnan(ipeak):
        ax.scatter(trace.index[ipeak], ypeak, color=color, alpha=alpha)


def plot_responses(data, tbounds=None, ykey=Label.DFF, mark_stim=True, mark_analysis_window=True,
                   mark_peaks=False, yref=None, **kwargs):
    '''
    Plot trial responses of specific sub-datasets.
    
    :param data: experiment dataframe
    :param tbounds (optional): time limits for plot
    :param ykey (optional): key indicating the specific signals to plot on the y-axis
    :param mark_stim (optional): whether to add a stimulus mark on the plot
    :param mark_analysis_window (optional): whether to add mark indicating the response analysis interval on the plot
    :param mark_peaks: whether to mark the peaks of each identified response
    :param yref (optional): vertical value at which to plot a horizontal reference line
    :param kwargs: keyword parameters that are passed to the generic plot_from_data function
    :return: figure handle
    '''
    # By default, no marker funtion is needed
    markerfunc = None

    # Extract response interval (from unfiltered data) if requested 
    if mark_analysis_window:
        tresponse = [data[Label.TIME].values[i] for i in [FrameIndex.RESPONSE.start, FrameIndex.RESPONSE.stop]]
        # Define marker function if mark_peaks is set to True
        if mark_peaks:
            markerfunc = lambda *args, **kwargs: mark_response_peak(
                *args, tbounds=tresponse, **kwargs)

    # Add tbounds to filtering criteria
    kwargs['tbounds'] = tbounds

    # Determine col order if column set to ROI response type
    if 'col' in kwargs and kwargs['col'] == Label.ROI_RESP_TYPE:
        kwargs['col_order'] = get_default_rtypes()

    # Plot with time on x-axis
    fig = plot_from_data(
        data, Label.TIME, ykey, xbounds=tbounds, markerfunc=markerfunc, **kwargs)
        
    # Add markers for each axis
    for ax in fig.axes:
        # Plot stimulus mark if specified
        if mark_stim:
            ax.axvspan(0, get_singleton(data, Label.DUR), ec=None, fc='C5', alpha=0.5)
        # Plot noise threshold level if key is z-score
        if yref is not None:
            ax.axhline(yref, ls='--', c='k', lw=1.)
        # Plot response interval if specified
        if tresponse is not None:
            for tr in tresponse:
                ax.axvline(tr, ls='--', c='k', lw=1.)

    # Return figure
    return fig


def plot_responses_across_datasets(data, ykey=Label.DFF, pkey=Label.P, avg=False, 
                                   groupby=None, **kwargs):
    '''
    Plot parameter-dependent response traces across datasets, for each response type
    
    :param data: multi-indexed dataframe containing timeseries of all datasets
    :param ykey: dependent variable of interest to plot
    :param pkey: independent parameter of interest (used as hue)
    :param avg: whether to average responses per category on each dataset.
    :return: figures dictionary or figure handle
    '''
    # Initialize propagated keyword arguments
    tracekwargs = dict(
        col = Label.DATASET if not avg else groupby, # 1 dataset/resp type on each axis
        hide_col_prefix = True,  # no column prefix
        max_colwrap = 4, # number of axes per line
        height = 2.3 if not avg else 3,  # height of each figure axis
        aspect = 1.,  # width / height aspect ratio of each axis
        ci = None,  # no error shading
    )

    # Determine dataset filter depending on parameter key
    if pkey == Label.P:
        tracekwargs['DC'] = DC_REF
    elif pkey == Label.DC:
        tracekwargs['P'] = P_REF
    else:
        raise ValueError(f'unknown parameter key: "{pkey}"')
    
    # Determine y-bounds depending on variable
    ybounds = {
        # Label.DFF: [-0.1, 0.15],
        # Label.ZSCORE: [-3., 6.],
        Label.EVENT_RATE: [0., 1 / MIN_EVENTS_DISTANCE]
    }.get(ykey, None)
    
    # Update with passed keyword arguments
    tracekwargs.update(kwargs)
    
    # Detailed mode: generate 1 figure per responder type
    if not avg:
        title = tracekwargs.pop('title', None)
        figdict = {}
        if groupby is not None:
            groups = data.groupby(groupby)
        else:
            groups = [('all', data)]
        for resptype, group in groups:
            logger.info(f'plotting {pkey} dependency curves for {resptype} responders...')
            nROIs_group = len(group.groupby([Label.DATASET, Label.ROI]).first())
            stitle = f'{resptype} responders ({nROIs_group} ROIs)'
            if title is not None:
                stitle = f'{title} - {stitle}'
            figdict[f'{resptype} {ykey} vs. {pkey}'] = plot_responses(
                group, ykey=ykey, hue=pkey, title=stitle, ybounds=ybounds, **tracekwargs)        
        return figdict
    
    # Average mode: generate a single figure with 1 axis per responder type
    else:
        fig = plot_responses(
            data, ykey=ykey, hue=pkey, ybounds=ybounds, **tracekwargs)
        return fig 


def add_numbers_on_legend_labels(leg, data, xkey, ykey, hue):
    '''
    Add sample size of each hue category on the plot
    
    :param leg: leg object
    :param data: dataframe used to plot graph from which legend was created
    :param xkey: name of the independent variable on the graph from which legend was created
    :param ykey: name of the dependent variable on the graph from which legend was created
    :param hue: hue parameter on the graph from which legend was created
    '''
    # Count number of samples per hue and x-axis value
    counts_by_hue = data.groupby([hue, xkey]).count().loc[:, ykey].unstack()

    # Keep only the max across all x-levels for each hue
    counts_by_hue = counts_by_hue.max(axis=1)

    # Extract counts index as string
    counts_by_hue.index = counts_by_hue.index.astype(str)
    
    # Map each legend entry to associated count, and enrich text 
    for t in leg.texts:
        s = t.get_text()
        if s in counts_by_hue:
            c = counts_by_hue.loc[s]
            cs = f'{c:.0f}'
        else:
            cs = '0'
        enriched_s = f'{s} (n = {cs})'
        t.set_text(enriched_s)


def plot_parameter_dependency(data, xkey=Label.P, ykey=None, yref=None, ax=None, hue=None,
                              avgprop=None, errprop='inter', marker=None, avgmarker='o', err_style='band',
                              add_leg_numbers=True, hue_alpha=None, ci=CI, legend='full', as_ispta=False,
                              stacked=False, fit=False, avglw=3, avgerr=True, **kwargs):
    ''' Plot parameter dependency of responses for specific sub-datasets.
    
    :param data: trial-averaged experiment dataframe
    :param xkey (optional): key indicating the independent variable of the x-axis
    :param ykey (optional): key indicating the dependent variable of the y-axis
    :param yref (optional): vertical at which to draw a "reference" horizontal line
    :param hue: hue grouping parameter
    :param avgprop: whether and how to propagate the data for global average reporting (None, "all", "hue" or "whue")
    :param errprop: how to propagate data for global standard error reporting ("inter or "intra")
    :param add_leg_numbers: whether to add sample counts for each legend entry  
    :param ci: confidence interval used to plot shaded areas around traces
        (default = 68 === SEM)
    :param kwargs: keyword parameters that are passed to the generic plot_from_data function
    :return: figure handle
    '''
    # If multi-dataset 
    if Label.DATASET in data.index.names:
        # If hue is not per dataset, call adapted function
        if hue != Label.DATASET:
            return plot_parameter_dependency_across_datasets(
                data, xkey=xkey, ykey=ykey, yref=yref, hue=hue, ax=ax, legend=legend,
                add_leg_numbers=add_leg_numbers, as_ispta=as_ispta, marker=marker,
                fit=fit, err_style=err_style, **kwargs)
        # Otherwise, offset values per dataset if specified
        else:
            if stacked:
                data[ykey] = offset_per_dataset(data[ykey])
    # Set plotting parameters
    hue_order = None
    hueplt = False
    hueerr_style = err_style
    if hue is None:
        avgprop = 'all'
    else:
        hueplt = True
        if hue == Label.ROI_RESP_TYPE:
            hue_order = get_default_rtypes()
    if avgprop is not None:
        hue_alpha_eff = 0.5
        hueerr_style = 'band'
    else:
        hue_alpha_eff = 1.

    if hue_alpha is None:
        hue_alpha = hue_alpha_eff

    # Get default ykey if needed
    if ykey is None:
        ykey = get_change_key(Label.DFF)

    # Restrict data based on xkey
    data = get_xdep_data(data, xkey)

    # Swicth xkey to ISPTA if specified
    if as_ispta:
        xkey = Label.ISPTA

    # Assemble common plotting arguments
    pltkwargs = dict(ax=ax, ci=ci, marker=marker, **kwargs)
    if hue_alpha == 0.:
        pltkwargs['ci'] = None

    # If hueplt specified
    if hueplt:
        fig = plot_from_data(
            data, xkey, ykey, hue=hue, hue_order=hue_order, alpha=hue_alpha,
            err_style=hueerr_style, legend=legend, **pltkwargs)
        # Get legend
        if ax is None:
            ax = fig.axes[0]
        leg = ax.get_legend()
        if leg is not None:
            # Add numbers on legend if needed
            if add_leg_numbers:
                add_numbers_on_legend_labels(leg, data, xkey, ykey, hue)
            # Move legend outside of plot if needed
            nhues = len(data.groupby(hue).first())
            if nhues > 3:
                bb = leg.get_bbox_to_anchor().transformed(ax.transAxes.inverted())
                xoff = 1 - bb.x0
                bb.x0 += xoff
                bb.x1 += xoff
                leg.set_bbox_to_anchor(bb, transform=ax.transAxes)
            leg.set(frame_on=False)

    # If propagated global average must be plotted
    if avgprop is not None:
        # If average trace relies on all traces
        if avgprop == 'all':
            # Compute average and standard error from entire dataset
            mean = data.groupby(xkey)[ykey].mean().rename('mean')
            sem = data.groupby(xkey)[ykey].sem().rename('sem')
        else:
            # If average trace relies on average across hue levels
            if avgprop == 'hue':
                # Generate uniform vector of counts per input and hue
                idx = data.groupby([xkey, hue]).first().index
                countsperhue = pd.Series([1] * len(idx), index=idx).rename('counts')
            # If average trace relies on cellcount-weighted average across hue levels
            elif avgprop == 'whue':
                # Count number of ROIs per hue and input level 
                celltypes = data.groupby([xkey, hue, Label.ROI]).first()
                countsperhue = celltypes.groupby([xkey, hue]).count().iloc[:, 0].rename('counts')
            else:
                raise ValueError(f'unknown average propagation mode: "{avgprop}"')
            # Compute weights vector based off of counts
            ntot = countsperhue.groupby(xkey).sum()
            weightsperhue = countsperhue / ntot
            # Compute mean for each input and hue
            means = data.groupby([xkey, hue])[ykey].mean().rename('mean')
            # Compute weighted mean for each input
            mean = (means * weightsperhue).groupby(xkey).sum().rename('wmean')
            # If global error must be propagated from hue ones
            if errprop == 'intra':
                # Compute standard error for each input and hue
                sems = data.groupby([xkey, hue])[ykey].sem().rename('sem')
                # Compute propagated standard error for each input
                sem = np.sqrt((weightsperhue * sems**2).groupby(xkey).sum()).rename('wsem')
            # If global error must be computed between hues
            elif errprop == 'inter':
                # Compute standard error between hues for each input
                sem = means.groupby(xkey).sem()
            else:
                raise ValueError(f'unknown error propagation mode: "{errprop}"')
        
        avg_kwargs = dict(c='k', marker=avgmarker, markersize=10, markeredgecolor='w')
        # Plot propagated global mean trace and standard error (as bars or band)
        sem = sem.fillna(0)
        if err_style == 'bars':
            if avgerr:
                ax.errorbar(
                    mean.index, mean.values, yerr=sem.values, 
                    lw=0 if fit else avglw, elinewidth=2 if fit else avglw,
                    **avg_kwargs)
            else:
                ax.plot(
                    mean.index, mean.values, 
                    lw=0 if fit else avglw, **avg_kwargs)
        else:
            ax.plot(
                mean.index, mean.values, 
                lw=0 if fit else avglw, **avg_kwargs)
            if avgerr:
                ax.fill_between(
                    mean.index, mean.values - sem.values, mean.values + sem.values, 
                    fc=avg_kwargs['c'], alpha=.3, ec=avg_kwargs['c'])

        if fit:
            ax.plot(
                *get_cubic_fit(mean.index, mean.values), 
                ls='--', lw=avglw, **avg_kwargs)

    # Add reference line(s) if specified
    if yref is not None:
        for y in as_iterable(yref):
            ax.axhline(y, c='k', ls='--')
    
    # Return figure handle
    return fig


def plot_parameter_dependency_across_datasets(data, xkey=Label.P, hue=None, ykey=None, ax=None,
                                              legend=True, yref=None, add_leg_numbers=True,
                                              marker='o', ls='-', as_ispta=False, title=None,
                                              weighted=True, fit=False, err_style='band', lw=1):
    '''
    Plot dependency of output metrics on a input parameter, using cell count-weighted
    averages and propagated standard errors from individual datasets
    
    :param data: multi-dataset trial-averaged experiment dataframe
    :param xkey (optional): key indicating the independent variable of the x-axis
    :param ykey (optional): key indicating the dependent variable of the y-axis
    :param yref (optional): vertical at which to draw a "reference" horizontal line
    '''
    # Get default ykey if needed
    if ykey is None:
        ykey = get_change_key(Label.DFF)

    # Reduce data to relevant input parameters
    data = get_xdep_data(data, xkey=xkey)

    # Swicth xkey to ISPTA if specified
    if as_ispta:
        xkey = Label.ISPTA

    # Initialize figure if needed
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    # Initialize axis
    sns.despine(ax=ax)
    ax.set_xlabel(xkey)
    ax.set_ylabel(ykey)
    
    # If hue not specified
    if hue is None:
        # Aggregate data with cell-count weighting
        aggdata = get_crossdataset_average(data, xkey, ykey=ykey, hue=hue, weighted=weighted)
        # Plot single weifghted average trace with propagated standard errors 
        if err_style == 'bars':
            ax.errorbar(
                aggdata[xkey], aggdata['mean'], yerr=aggdata['sem'], 
                marker=marker, ls=ls, c='k', lw=lw, 
                markeredgecolor='w', markersize=10)
        else:
            ax.plot(
                aggdata[xkey], aggdata['mean'], 
                marker=marker, ls=ls, c='k', lw=lw, 
                markeredgecolor='w', markersize=10)
            if err_style == 'band':
                ax.fill_between(
                    aggdata[xkey], 
                    aggdata['mean'] - aggdata['sem'], aggdata['mean'] + aggdata['sem'],
                    alpha=0.3, color='k')

    # Otherwise
    else:
        # For each hue value
        for htype, rdata in data.groupby(hue):
            # Aggregate data with cell-count weighting
            aggdata = get_crossdataset_average(rdata, xkey, ykey=ykey, hue=hue, weighted=weighted)
            if hue == Label.ROI_RESP_TYPE:
                color = Palette.RTYPE[htype]
            else:
                color = None
            
            if err_style == 'bars':
                ax.errorbar(
                    aggdata[xkey], aggdata['mean'], yerr=aggdata['sem'],
                    marker=marker, ls=ls, label=htype, color=color, 
                    linewidth=0 if fit else None, elinewidth=2 if fit else None)
            else:
                ax.plot(
                    aggdata[xkey], aggdata['mean'],
                    marker=marker, ls=ls, label=htype, color=color, 
                    linewidth=0 if fit else None)
                if err_style == 'band':
                    ax.fill_between(
                        aggdata[xkey], 
                        aggdata['mean'] - aggdata['sem'], aggdata['mean'] + aggdata['sem'],
                        alpha=0.3, color=color)

            # Add 3rd order polynomial fit if required
            if fit:
                ax.plot(
                    *get_cubic_fit(aggdata[xkey], aggdata['mean']),
                    ls='--', color=color)
        if legend:
            ax.legend(frameon=False)
            # Add numbers on legend if needed
            if add_leg_numbers:
                add_numbers_on_legend_labels(ax.get_legend(), data, xkey, ykey, hue)

    # Add reference line(s) if specified
    if yref is not None:
        for y in as_iterable(yref):
            ax.axhline(y, c='k', ls='--')

    if title is not None:
        ax.set_title(title)
    
    return fig


def plot_stimparams_dependency(data, ykey, title=None, axes=None, **kwargs):
    '''
    Plot dependency of a specific response metrics on stimulation parameters
    
    :param data: trial-averaged experiment dataframe
    :param ykey (optional): key indicating the dependent variable of the y-axis
    :param kwargs: keyword parameters that are passed to the plot_parameter_dependency function
    :return: figure handle
    '''
    # Initialize or retrieve figure        
    if axes is None:
        height = 4
        if kwargs.get('stacked', False):
            height = max(height, len(data.index.unique(Label.DATASET)))
        fig, axes = plt.subplots(1, 2, figsize=(10, height))
    else:
        if len(axes) != 2:
            raise ValueError('exactly 2 axes must be provided')
        fig = axes[0].get_figure()

    # Disable legend for all axes but last
    kwargs['legend'] = False
    # Plot dependencies on each parameter on separate axes
    for i, (xkey, ax) in enumerate(zip([Label.P, Label.DC], axes.T)):
        if i == len(axes) - 1:
            del kwargs['legend']
        plot_parameter_dependency(
            data, xkey=xkey, ax=ax, ykey=ykey, title=f'{xkey} dependency', **kwargs)
    
    # Harmonize axes limits
    harmonize_axes_limits(axes)

    if title is not None:
        fig.suptitle(title)

    # Return figure handle
    return fig


def plot_cellcounts(data, hue=Label.ROI_RESP_TYPE, count='pie', title=None):
    '''
    Plot a summary chart of the number of cells per response type and dataset
    
    :param data: multi-indexed stats dataframe with dataset as an extra index dimension
    :param hue: hue parameter (typically ROI responder type or dataset)
    :param count: total count per category reporting type (None, 'label', or 'pie')
    :param title (optional): figure title
    :return: figure handle
    '''
    # Restrict dataset to 1 element per ROI for each dataset
    celltypes = data.groupby([Label.DATASET, Label.ROI]).first()
    # Count total number of cells
    ntot = celltypes.count().iloc[0]

    # Figure out bar variable and plot orientation
    if hue is not None:
        groups = [Label.DATASET, Label.ROI_RESP_TYPE]
        bar = list(set(groups) - set([hue]))[0]
    else:
        bar = Label.DATASET
        count = None
    axdim = {Label.ROI_RESP_TYPE: 'x', Label.DATASET: 'y'}[bar]

    # Determine plotting order
    orders = {
        Label.ROI_RESP_TYPE: get_default_rtypes(),
        Label.DATASET: natsorted(data.index.unique(level=Label.DATASET).values.tolist())
    }
    bar2 = f'{bar} '
    if bar == Label.DATASET:
        barvals = celltypes.index.get_level_values(bar)
    else:
        barvals = celltypes[bar].values
    barvals = pd.Categorical(barvals, categories=orders[bar])
    celltypes[bar2] = barvals
    pltkwargs = {axdim: bar2}
    if hue is not None:
        pltkwargs['hue_order'] =  orders[hue]
        if hue == Label.ROI_RESP_TYPE:
            pltkwargs['palette'] = Palette.RTYPE

    # Plot stacked count bars
    fg = sns.displot(
        data=celltypes, multiple='stack', hue=hue, **pltkwargs)
    sns.despine()

    # Extract figure, axis and legend
    fig = fg.figure
    ax = fig.axes[0]
    leg = fg._legend
    fig.subplots_adjust(top=0.9)

    # Count number of cells of each bar and hue
    cellcounts = celltypes.groupby([Label.ROI_RESP_TYPE, Label.DATASET]).count().iloc[:, 0].rename('counts')

    # Sum counts per bar level
    cellcounts_per_bar = cellcounts.groupby(bar).sum()

    if count is not None:
        # If count label specified 
        if count == 'label':
            # If resp type is hue, add labels to legend
            if hue == Label.ROI_RESP_TYPE:
                nperhue = cellcounts.groupby(hue).sum().astype(int)
                for t in leg.texts:
                    s = t.get_text()
                    n = nperhue.loc[s]
                    t.set_text(f'{s} (n={n}, {n / ntot * 100:.0f}%)')
                leg.set_bbox_to_anchor([1.2, 0.5])

            # If resp type is bar, add labels on top of bars
            else:
                nperbar = cellcounts.groupby(bar).sum().astype(int)
                labels = [l.get_text() for l in ax.get_xticklabels()]
                offset = nperbar.max() * 0.02
                for i, label in enumerate(labels):
                    n = nperbar.loc[label]
                    ax.text(i, n + offset, f'{n} ({n / ntot * 100:.0f}%)', ha='center')
        
        # If count pie chart specified 
        elif count == 'pie':
            # Raise error if hue mode is incompatible with pie chart
            if hue != Label.ROI_RESP_TYPE:
                raise ValueError(
                    f'pie chart count report only available with "{Label.ROI_RESP_TYPE}" hue parameter')
            # Remove legend
            leg.remove()
            # Count cells by responder type
            counts_by_rtype = cellcounts.groupby(Label.ROI_RESP_TYPE, sort=False).sum()
            counts_by_rtype = counts_by_rtype.reindex(orders[Label.ROI_RESP_TYPE]).dropna()
            # Plot counts on pie chart
            ax2 = fig.add_axes([0.8, 0.1, 0.35, 0.8])
            counts_by_rtype.plot.pie(
                ax=ax2, ylabel='', autopct='%1.0f%%',
                colors=[Palette.RTYPE[k] for k in counts_by_rtype.index],
                startangle=90, textprops={'fontsize': 12}, 
                wedgeprops={'edgecolor': 'k', 'alpha': 0.7})
        else:
            raise ValueError(f'invalid count mode: "{count}"')

    # Add title
    datasets = list(celltypes.groupby(Label.DATASET).groups.keys())
    mice = sorted(list(set([x.split('_')[1] for x in datasets])))
    countsstr = f'{len(mice)} mice, {len(datasets)} regions, {ntot} ROIs'
    stitle = f'{countsstr}, avg = {cellcounts_per_bar.mean():.0f} +/- {cellcounts_per_bar.std():.0f}'
    if title is not None:
        stitle = f'{title} ({stitle})'
    fig.suptitle(stitle, fontsize=15)
    
    # Return figure handle
    return fig


def plot_P_DC_map(P, DC, fs=12, ax=None):
    ''' 
    Plot sonication protocol in the DC - pressure space
    
    :param P: array of peak pressure amplitudes (in MPa)
    :param DC: array of duty cycles (in %)
    :return: figure handle
    '''

    # Compute time average intensities over P - DC grid
    nperax = 100
    Prange = np.linspace(0, 1.25 * P.max(), nperax)  # MPa
    DCrange = np.linspace(0, 100, nperax)  # %
    Isppa = pressure_to_intensity(Prange / PA_TO_MPA) / M2_TO_CM2  # W/cm2
    Ispta = np.dot(np.atleast_2d(Isppa).T, np.atleast_2d(DCrange)) * 1e-2  # W/cm2

    # Create or retrieve figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.get_figure()
    ax.set_title(Label.ISPTA, fontsize=fs)
    ax.set_xlabel(Label.DC, fontsize=fs)
    ax.set_ylabel(Label.P, fontsize=fs)
    sns.despine(ax=ax)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)

    # Plot Ispta colormap over DC - DC space
    cmap = sns.color_palette('rocket', as_cmap=True).reversed()
    ax.pcolormesh(
        DCrange, Prange, Ispta, shading='gouraud', rasterized=True, cmap=cmap)

    # Plot contours for characteristic Ispta values
    Ispta_levels = np.array([.01, .2, 1, 2, 5, 10, 20])
    labels_DC = 80
    labels_Ispta = Ispta_levels / labels_DC * 1e2  # W/cm2
    labels_P = intensity_to_pressure(labels_Ispta * M2_TO_CM2) * PA_TO_MPA 
    labels_locs = [(labels_DC, p) for p in labels_P]
    CS = ax.contour(DCrange, Prange, Ispta, levels=Ispta_levels, colors='k')
    ax.clabel(CS, fontsize=fs, inline=True, fmt='%.2g', manual=labels_locs)

    # Plot sampled DC - P combinations
    ax.scatter(DC, P, c='deepskyblue', edgecolors='k', zorder=80)

    # Finalize figure layout
    fig.tight_layout()

    # Return figure
    return fig


def plot_stat_vs_offset_map(stats, xkey, ykey, outkey, interp=None, filters=None, title=None,
                            cmap='viridis', dx=0.5, dy=0.5, rmax=1., clevels=None):
    '''
    Plot map of output metrics as a function of XY offset
    
    :param stats: stats dataframe
    :param xkey: column name for X offset coordinates
    :param ykey: column name for Y offset coordinates
    :param outkey: column name for output metrics
    :param interp: type of interpolant used to geenrate XY map of output metrics (default=None)
    :param filters: potential filters to restrict dataset
    :param title: optional figure title
    :param cmap: colormap string
    :param dx: interpolation step size in x dimension
    :param dy: interpolation step size in y dimension 
    :return: 2-tuple with:
        - figure handle
        - offset coordinates of max response per dataset
    '''
    # Apply filters if provided
    if filters is not None:
        for k, v in filters.items():
            logger.info(f'restricting {k} to "{v}"')
            stats = stats[stats[k] == v]

    # Group stats by dataset
    logger.info('grouping stats by dataset...')
    groups = stats.groupby(Label.DATASET)

    # Get colormap object
    cmap = plt.get_cmap(cmap)
    cmap.set_bad(cmap(0))

    # Create figure backbone
    naxes = len(groups)
    fig, axes = plt.subplots(1, naxes, figsize=(5 * naxes, 4))
    stitle = 'normalized response strength vs. offset'
    if title is not None:
        stitle = f'{stitle} ({title})'
    for ax in axes:
        sns.despine(ax=ax)
        ax.set_aspect(1.)
        ax.set_xlabel(xkey)
        ax.set_ylabel(xkey)
        ax.axhline(0., c='k', ls='--', zorder=-10)
        ax.axvline(0., c='k', ls='--', zorder=-10)
    
    # Get colormap normalizer and mappable
    norm, sm = set_normalizer(cmap, [0, 1])

    # For each dataset
    logger.info(f'plotting map of "{outkey}" vs. XY offset...')
    xranges, yranges, interp_maps = [], [], []
    for (dataset_id, substats), ax in zip(groups, axes):
        ax.set_title(dataset_id)
        # Compute average output metrics for each XY offset value
        outkey_avg = substats[[xkey, ykey, outkey]].groupby(
            [xkey, ykey])[outkey].mean().reset_index()
        # Divide by max value to bring to [0, 1] range 
        outkey_avg[outkey] /= outkey_avg[outkey].max()
        # Plot output metrics across scanned XY offsets
        ax.scatter(
            outkey_avg[xkey], outkey_avg[ykey], s=80, c=outkey_avg[outkey], edgecolors='w', cmap=cmap)
        # If specified: plot interpolated 2D map of response strength vs offset location
        if interp is not None:
            # Get grid vectors and 2d map
            xrange, yrange, interp_map = interpolate_2d_map(
                outkey_avg, xkey, ykey, outkey, method=interp, dx=dx, dy=dy)
            # Normalize 2d map
            interp_map /= np.nanmax(interp_map)
            # Plot
            ax.pcolormesh(
                compute_mesh_edges(xrange), compute_mesh_edges(yrange), interp_map.T,
                zorder=-10, cmap=cmap, rasterized=True, norm=norm)
            # Append to global containers
            xranges.append(xrange)
            yranges.append(yrange)
            interp_maps.append(interp_map)
    
    # Add colorbar
    pos = ax.get_position()
    fig.subplots_adjust(right=0.93)
    cbar_ax = fig.add_axes([0.95, pos.y0, 0.02, pos.y1 - pos.y0])
    fig.colorbar(sm, cax=cbar_ax)

    # Add figure title
    if interp is not None:
        stitle = f'{stitle} - {interp} interpolant'
    fig.suptitle(stitle)

    # If more than 1 interpolation map was generated
    if len(xranges) > 1:
        dxs = np.unique(np.hstack([np.unique(np.diff(i)) for i in xranges]))
        dys = np.unique(np.hstack([np.unique(np.diff(i)) for i in yranges]))
        dxs, dys = [np.round(reduce_close_elements(xx), 2) for xx in [dxs, dys]]
        # If x and y step sizes are constant across maps
        if dxs.size == 1 and dys.size == 1:
            # Generate average 2D map (from normalized maps)
            dx, dy = dxs[0], dys[0]
            xmin, xmax = min([min(i) for i in xranges]), max([max(i) for i in xranges])
            ymin, ymax = min([min(i) for i in yranges]), max([max(i) for i in yranges])
            xrange, yrange = np.arange(xmin, xmax + dx / 2, dx), np.arange(ymin, ymax + dy / 2, dy)
            avgmap = np.full((len(xranges), xrange.size, yrange.size), np.nan)
            for i, (x, y, imap) in enumerate(zip(xranges, yranges, interp_maps)):
                ixshift = np.where(np.isclose(xrange, x.min()))[0][0]
                iyshift = np.where(np.isclose(yrange, y.min()))[0][0]
                avgmap[i, ixshift:ixshift + x.size, iyshift:iyshift + y.size] = imap
            avgmap = np.nanmean(avgmap, axis=0)

            # Plot average 2D map on new figure
            fs = 12
            newfig, newax = plt.subplots()
            newax.set_title('average map', fontsize=fs + 2)
            newax.set_xticks([])
            newax.set_yticks([])
            newax.set_aspect(1.)
            radii = np.arange(.5, rmax + .01, .5)
            lss = ['--'] * len(radii)
            lss[-1] = '-'
            wedges = [
                mpatches.Wedge(
                    (0., 0.), r, 180, 360, linestyle=ls, fc='none', ec='w')
                for r, ls in zip(radii, lss)]
            mesh = newax.pcolormesh(
                xrange, yrange, avgmap.T, 
                cmap=cmap, rasterized=True, shading='gouraud')
            for w in wedges:
                newax.add_patch(w)
            mesh.set_clip_path(wedges[-1])
            newax.set_xlim(-rmax, rmax)
            newax.set_ylim(-rmax, 0.25 * rmax)
            if clevels is not None:
                newax.contour(
                    xrange, yrange, avgmap.T, levels=as_iterable(clevels), colors=['w'])
            sns.despine(ax=newax, bottom=True, left=True)

            # Add scale bar
            scalex = np.ceil(rmax) / 2
            scalebar = AnchoredSizeBar(
                newax.transData,
                scalex, f'{scalex:.1f} mm', 'upper right', 
                color='k', frameon=False, label_top=True, size_vertical=.025,
                fontproperties={'size': fs})
            newax.add_artist(scalebar)

            # Add focus annotation
            newax.annotate(
                'focus', xy=(0., 0.),  xycoords='data',
                xytext=(0.5, .99), textcoords='axes fraction',
                arrowprops=dict(facecolor='black', shrink=0.01, width=2),
                horizontalalignment='center', verticalalignment='top', fontsize=fs)
            
            # Add colorbar
            pos = newax.get_position()
            newfig.subplots_adjust(right=0.85)
            newcbar_ax = newfig.add_axes([0.9, pos.y0, 0.05, pos.y1 - pos.y0])
            newfig.colorbar(sm, cax=newcbar_ax)

            # Return both figures
            return fig, newfig

    # Return figure
    return fig


def plot_comparative_metrics_across_datasets(data, ykey, compkey, groupby=Label.DATASET,
                                             kind='bar', add_stats=True, **kwargs):
    '''
    Plot comparative distributions of center vs fixed offset conditions

    :param data: stats dataframe
    :param ykey: output metrics key
    :param compkey: comparative conditions key
    :param groupby: goruping variable (default = dataset)
    :param kind: type of categorical plot (default = 'bar')
    :param add_stats: whther or not to add statistical comparisons
    '''
    nconds = data[compkey].nunique()
    # Set groupby as column if needed
    if groupby is not None:
        if groupby not in data.columns:
            if groupby in data.index.names:
                data[groupby] = data.index.get_level_values(level=groupby)
                data = data.droplevel(groupby)
            else:
                raise ValueError(f'"{groupby}" groupby parameter not found in data')

        nconds *= data[groupby].nunique()
    
    # Define plotting arguments
    pltkwargs = dict(
        data=data, 
        x=groupby,
        y=ykey,
        hue=compkey,
        dodge=True,
        legend_out=False,
        aspect=max(1.5, nconds * 0.1),
        kind=kind,
        **kwargs
    )
    # Render categorical plot
    fg = sns.catplot(**pltkwargs)
    # Adapt x-labels
    fig = fg.figure
    ax = fig.axes[0]
    ax.set_xticklabels(x.get_text().replace('_', '\n') for x in ax.get_xticklabels())
    ax.set_xlabel(ax.get_xlabel(), labelpad=10)

    # Add zero (and one) line if plot is not bar plot
    if kind != 'bar':
        ax.axhline(0, c='k', ls='--', zorder=-10)    
        if 'normalized' in ykey:
            ax.axhline(1, c='k', ls='--', zorder=-10)
            

    title_pad = 10
    comps_str = compkey
    if add_stats:
        # Determine groups for statistical tests
        groups = data.groupby([Label.DATASET, compkey])[ykey]
        # Determine which statistical test to apply
        isnotnormal = groups.agg(lambda s: normaltest(s)[1] < PTHR_DETECTION)
        test = 'Mann-Whitney' if any(isnotnormal) else 't-test_ind'
        # Extract conditions per dataset
        conds = groups.first().to_frame()
        conds['conds'] = conds.index.get_level_values(compkey)
        conds = conds.droplevel(compkey)['conds']
        # Extract pairs for statistical comparison
        combs = conds.groupby(Label.DATASET).apply(lambda s: list(combinations(s, 2)))
        maxcondsperdataset = max([len(x) for x in combs])
        combs = combs.explode()
        pairs = [[(dataset, item) for item in pair] for dataset, pair in combs.iteritems()]
        # Perform tests and add statistical annotations
        annotator = Annotator(
            ax=ax, pairs=pairs, **pltkwargs)
        annotator.configure(test=test, text_format='star', loc='outside')
        annotator.apply_and_annotate()
        title_pad += 30 * maxcondsperdataset
        comps_str = f'{comps_str} ({test} test)'

    # Add figure title
    stitle = f'response strength - comparison across {comps_str}'
    if 'normalized' in ykey:
        stitle = f'normalized {stitle}'
    ax.set_title(stitle, pad=title_pad)

    # Return figure
    return fig


def plot_comparative_metrics_across_conditions(data, ykey, condkey, kind='box', 
                                               test='t-test', paired=False, correct=False):
    '''
    Plot comparative distributions of center vs fixed offset conditions

    :param data: stats dataframe
    :param ykey: output metrics key
    :param conpkey: comparative conditions key
    :param groupby: goruping variable (default = dataset)
    :param kind: type of categorical plot (default = 'bar')
    :param add_stats: whther or not to add statistical comparisons
    '''
    # Aggregate output metrics across conditions and datasets 
    yagg = data.groupby([Label.DATASET, condkey]).mean()[ykey]
    # Establish pairs of conditions to compare
    pairs = list(combinations(yagg.unstack().columns, 2)) 
    
    # Define plot arguments
    pltkwargs = dict(
        data=yagg.reset_index(condkey),
        x=condkey,
        y=ykey
    )
    # Render categorical plot 
    fg = sns.catplot(
        kind=kind, 
        whis=False, 
        showfliers=False, 
        **pltkwargs
    )
    fig = fg.figure
    ax = fig.axes[0]
    # Show underlying data points
    sns.scatterplot(
        ax=ax, 
        hue=condkey, 
        s=100,
        zorder=20,
        legend=None,
        **pltkwargs)
    # If paired data
    if paired:
        # Set appropriate test
        if test is not None:
            test = 't-test_paired' 
        # Show links between points
        tmp = yagg.unstack().transpose()
        for s in tmp:
            ax.plot(tmp.index, tmp[s], c='k', zorder=10)
    # If test provided, apply and annotate
    if test is not None:
        annotator = Annotator(
            ax=ax, 
            pairs=pairs,
            **pltkwargs
        )
        annotator.configure(
            test=test, 
            text_format='star', 
            loc='outside',
            comparisons_correction='Bonferroni' if correct else None
        )
        annotator.apply_and_annotate()
    return fig


def plot_parameter_dependency_across_lines(data, xkey, ykey, yref=0.):
    '''
    Plot comparative parameter dependency curves (with error bars) across
    mouse lines, for each responder type

    :param data: responder-type-averaged multi-line statistics dataframe
    :param xkey: input parameter name
    :param ykey: output parameter name
    :param yref: reference vertical level to indicate with dashed line (optional)
    :return: figure
    '''
    depdata = get_xdep_data(data, xkey)
    fg = sns.relplot(
        data=depdata,
        kind='line',
        x=xkey, y=f'{ykey} - mean',
        hue=Label.LINE,
        col=Label.ROI_RESP_TYPE,
        col_order=Palette.RTYPE.keys(),
        marker='o',
        lw=2, markersize=10,
        palette=Palette.LINE
    )
    fig = fg.figure
    if yref is not None:
        for ax in fig.axes:
            ax.axhline(0., ls='--', c='k')
    for rtype, gdata in depdata.groupby(Label.ROI_RESP_TYPE):
        ax = fig.axes[list(Palette.RTYPE.keys()).index(rtype)]
        ax.set_ylabel(ykey)
        for line, ldata in gdata.groupby(Label.LINE):
            ldata = ldata.sort_values(xkey)
            ax.errorbar(
                ldata[xkey], ldata[f'{ykey} - mean'], ldata[f'{ykey} - sem'],
                ls='', c=Palette.LINE[line])
    fig.suptitle(f'{xkey} dependency', y=1.05)
    return fig


def plot_intensity_dependencies(data, ykey, ax=None, hue=Label.ROI_RESP_TYPE):
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_title(f'ISPTA dependency')
    else:
        fig = ax.get_figure()
    # Determine output metrics key
    logger.info(f'plotting {ykey} ISPTA dependency across responders...')
    # Plot ISPTA dependency profiles
    plot_parameter_dependency(
        data, xkey=Label.ISPTA, ykey=ykey, yref=0., hue=hue, ax=ax, 
        marker=None, as_ispta=True, ci=None if hue==Label.DATASET else CI,
        hue_alpha=1., avgprop='whue', avgmarker=None)
    # Indicate the "parameter sweep" origin of each data point
    # Plot dependencies on each parameter on same ISPTA axis
    for i, (xkey, marker) in enumerate(zip([Label.P, Label.DC], ['o', '^'])):
        plot_parameter_dependency(
            data, xkey=xkey, ykey=ykey, yref=0., ax=ax, hue=None, weighted=True, marker=marker,
            as_ispta=True, legend=False, add_leg_numbers=False, err_style=None, lw=0.)
    return fig


def plot_intensity_dependencies_across_lines(data, ykey):
    logger.info(f'plotting {ykey} ISPTA dependency across responders...')
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'ISPTA dependency', y=1.05)
    sns.despine(fig)
    axes[0].set_ylabel(ykey)
    # For each responder type
    for rtype, rdata in data.groupby(Label.ROI_RESP_TYPE):
        iax = list(Palette.RTYPE.keys()).index(rtype)
        ax = fig.axes[iax]
        ax.set_xlabel(Label.ISPTA)
        ax.axhline(0., ls='--', c='k')
        ax.set_title(f'{rtype} responders')
        # For each line
        for line, ldata in rdata.groupby(Label.LINE):
            # Plot dependencies on each parameter on same ISPTA axis
            for xkey, marker in zip([Label.P, Label.DC], ['o', '^']):
                depdata = get_xdep_data(ldata, xkey).sort_values(xkey)
                ax.errorbar(
                    depdata[Label.ISPTA], depdata[f'{ykey} - mean'], depdata[f'{ykey} - sem'],
                    marker=marker, ls='--', c=Palette.LINE[line], label=f'{line} - {xkey} dep')
            if iax == 0:
                ax.legend(frameon=False) 
    harmonize_axes_limits(axes)
    return fig


def plot_stat_heatmap(data, ykey, ax=None, run_order=None, aggfunc=None, **kwargs):
    ''' 
    Plot a heatmap of statistics across runs & trials

    :param data: multi-index statistics dataframe
    :param ykey: variable of interest
    :return: figure handle    
    '''
    # Create or retrieve figure
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ax.set_title(f'{ykey} across runs and trials')
    # Aggregate across ROIs for each run & trial
    if aggfunc is None:
        aggfunc = lambda x: x.mean()
    yagg = data[ykey].groupby([Label.RUN, Label.TRIAL]).agg(aggfunc)
    # Plot heatmap
    table = yagg.unstack()
    if run_order is not None:
        table = table.reindex(run_order, axis=0)
    sns.heatmap(table, ax=ax, center=0, **kwargs)
    # Return figure
    return fig


def plot_stat_graphs(data, ykey, run_order=None, irun_marker=None):
    ''' 
    Plot various representations of a particular statistics
    
    :param data: multi-index statistics dataframe
    :param ykey: name of statistics of interest 
    :return: figure handle
    '''
    # Order ROIs by increasing response strength
    ROI_order = (
        data[ykey]
        .groupby(Label.ROI)
        .mean()
        .sort_values()
        .index
    )

    # If run order not provided, provide identity order indexing
    if run_order is None:
        run_order = data.index.unique(Label.RUN)
        run_order = pd.Series(run_order, index=run_order)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax in axes:
        sns.despine(ax=ax)

    # Plot response strength histogram distribution
    sns.histplot(data, x=ykey, bins=100, ax=axes[0])

    # Plot response strength vs. run
    ax = axes[1]
    sns.barplot(
        ax=ax, data=data.reset_index(level=Label.RUN),
        x=Label.RUN, y=ykey, ci=68, 
        order=run_order.index.values)
    # Add zero line
    ax.axhline(0., c='k', lw=1)
    # Add marker for specific run, if specified
    if irun_marker is not None:
        irun_marker = run_order.index.get_loc(irun_marker)
        ax.scatter(irun_marker, .5 * ax.get_ylim()[1], marker='v', c='k')

    # Plot response strength heatmap per ROI & run
    ax = axes[2]
    ax.set_title(ykey)
    med = data[ykey].median()
    std = data[ykey].std()
    table = (
        data[ykey]
        .unstack()
        .reindex(ROI_order, axis=0)
        .reindex(run_order.index.values, axis=1)
    )
    sns.heatmap(table, center=0., ax=ax, vmax=med + 3 * std)

    # Return figure
    return fig


def plot_pct_responders(data, xkey, hue=Label.DATASET, xref=None, kind='line', 
                        avg_overlay=True, hue_highlight=None, **kwargs):
    ''' 
    Plot percentage of responder cells as a function of an input parameter

    :param data: statistics dataframe
    :param xkey: input parameter name
    :return: figure handle
    '''
    # Restrict data to input parameter dependency range
    data = get_xdep_data(data, xkey)
    # Count number of responses of each type, for each dataset and input value 
    groupby = [xkey]
    if hue is not None:
        groupby.append(hue)
    resp_counts = data.groupby(groupby)[Label.RESP_TYPE].value_counts().unstack()
    # Convert to proportions
    resp_counts['total'] = resp_counts.sum(axis=1)
    resp_props = resp_counts.div(resp_counts['total'], axis=0) * 100
    weights = resp_counts['total'].groupby(xkey).apply(lambda s: s / s.sum())
    weighted_resp_props = resp_props.multiply(weights, axis=0).groupby(xkey).sum()
    resp_props_sem = resp_props.groupby(xkey).sem()


    # Plot % responders profile(s) with appropriate function
    pltkwargs = dict(
        data=resp_props.reset_index(), 
        x=xkey, 
        y='positive', 
        **kwargs
    )
    if kind == 'line':
        pltfunc = sns.relplot
        if hue is None:
            pltkwargs.update(dict(
                marker = 'o',
                color='b',
                markersize=10,
                lw=3,
                markeredgecolor='w',
                ci=68
            ))
        else:
            # If specified, highlight specific hue value
            if hue_highlight is not None:
                huevals = resp_props.groupby(hue).first().index
                if hue_highlight not in huevals:
                    raise ValueError(f'"{hue_highlight}" not found in hue values')
                huecolors = ['silver'] * len(huevals)
                palette = dict(zip(huevals, huecolors))
                palette[hue_highlight] = 'r'
                pltkwargs['palette'] = palette

    elif kind in ['bar', 'box', 'boxen', 'violin']:
        if hue == Label.DATASET:
            raise ValueError('cannot plot distributions if split by dataset')
        pltfunc = sns.catplot
        pltkwargs['color'] = 'C0'
    else:
        raise ValueError(f'unknown plot type: "{kind}"')
    fg = pltfunc(
        kind=kind,
        hue=hue,
        height=4,
        **pltkwargs
    )
    # Extract figure and axis
    fig = fg.figure
    ax = fig.axes[0]
    # Post-process figure
    sns.despine(ax=ax)
    ax.set_ylim(0, 100)
    ax.set_ylabel('% responders')
    ax.axhline(PTHR_DETECTION * 100, c='k', ls='--')
    # Add average trace if specified and compatible
    if kind == 'line' and avg_overlay:
        wmean = weighted_resp_props['positive']
        wsem = resp_props_sem['positive']
        ax.plot(
            wmean.index, wmean.values, 
            color='k', marker='o', markersize=10, lw=3, markeredgecolor='w')
        ax.fill_between(
            wmean.index, wmean - wsem, wmean + wsem,
            fc='k', ec='k', alpha=0.3)

    # Apply statistical comparisons with reference input, if specified
    if xref is not None:
        xvals = sorted(resp_props.index.unique(level=xkey))
        if xref not in xvals:
            raise ValueError(
                f'reference input value {xref} not found in data (candidates are {xvals})')
        xpairs = [(xref, x) for x in xvals if x != xref]
        annotator = Annotator(pairs=xpairs, **pltkwargs)
        annotator.configure(
            test='t-test_ind', 
            text_format='star', 
            loc='outside',
            comparisons_correction='Bonferroni'
        )
        annotator.apply_and_annotate()

    return fig


def plot_classification_details(data, pthr=None, hue=None, avg_overlay=True):
    ''' 
    Plot details of cells classification as function of their response distribution
    for the current ensemble of datasets
    
    :param data: multi-dataset stats dataframe
    :param pthr (optional): threshold proportion of positive conditions, to be displayed
        on graph along with corresponding fraction of identified responders
    :return: figure handle
    '''
    # Count number of conditions per response type for each ROI
    cond_counts = (
        data[Label.RESP_TYPE]
        .groupby([Label.DATASET, Label.ROI])
        .value_counts()
        .unstack().fillna(0.)
    )
 
    # Translate to proportions
    cond_counts['total'] = cond_counts.sum(axis=1)
    cond_fracs = cond_counts.div(cond_counts['total'], axis=0)

    pltkwargs = dict(
        data=cond_fracs,
        x='positive',
        complementary=True,
    )

    # Plot inverse cumulative distribution to see how classification would 
    # vary as a function of threshold
    fg = sns.displot(
        kind='ecdf',
        hue=hue,
        height=4,
        **pltkwargs
    )
    fig = fg.figure
    ax = fig.axes[0]
    ax.set_xlabel('fraction of positive conditions')
    ax.set_ylabel('fraction of responders')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Add global average profile on top, if specified
    if hue is not None and avg_overlay:
        sns.ecdfplot(
            ax=ax,
            c='k',
            lw=2,
            **pltkwargs
        )

    # If threshold proportion of positive conditions specified,
    if pthr is not None:
        
        # Classify ROIs accordingly
        responder_types = (
            cond_fracs['positive'] >= pthr).astype(int).map(RTYPE_MAP)
        rtype_counts = responder_types.value_counts()
        prop_pos = rtype_counts['positive'] / rtype_counts.sum()
        logger.info(f'identified {prop_pos * 100:.1f}% of responders with {pthr} as threshold proportion of responding conditions')
        
        # Indicate threshold proportion and corresponding responders fraction on graph 
        ax.axvline(pthr, c='k', ls='--')
        ax.axhline(prop_pos, c='k', ls='--')

    # Return figure
    return fig