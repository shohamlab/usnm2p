# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-13 11:41:52
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-09-15 13:47:59

''' Collection of plotting utilities. '''

from itertools import combinations, chain
import random
import logging
import sys
from natsort import natsorted
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize, LogNorm, SymLogNorm, to_rgb
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.patches import Rectangle
from matplotlib.path import Path
from matplotlib.collections import PathCollection
from skimage.measure import find_contours
import seaborn as sns
from statannotations.Annotator import Annotator
from colorsys import hsv_to_rgb, rgb_to_hsv
from tqdm import tqdm
from scipy.stats import normaltest, binned_statistic_dd
from scipy.signal import spectrogram, sosfreqz, butter, sosfiltfilt
from scipy.interpolate import interp2d, griddata

from logger import logger
from constants import *
from utils import *
from postpro import *
from viewers import get_stack_viewer, extract_registered_frames
from fileops import loadtif
from parsers import get_info_table, parse_quantile

# Matplotlib parameters
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
if sys.platform != 'linux':
    matplotlib.rcParams['font.family'] = 'arial'

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


def adjust_xscale(ax, xscale=None):
    ''' Adjust x-scale if specified '''
    if xscale is not None:
        def signed_sqrt(x):
                if is_iterable(x):
                    return np.array([signed_sqrt(xi) for xi in x])
                return np.sqrt(x) if x >= 0 else -np.sqrt(-x)
        if xscale == 'sqrt':
            ax.set_xscale('function', functions=(signed_sqrt, np.square))
            xticks = list(filter(lambda x: x >= 0, ax.get_xticks()))
            if 0 not in xticks:
                xticks = [0] + xticks
            ax.set_xticks(xticks)
            xlims = ax.get_xlim()
            ax.set_xlim(-.001 * xlims[1], xlims[1])
            ax.spines['left'].set_position(('outward', 10))
        else:
            ax.set_xscale(xscale)


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


def plot_registered_frames(ops, irun, itrial, iframes, ntrials_per_run=None, fps=None, norm=True,
                           cmap='viridis', fs=15, height=3, axes=None, overlay_label=True, aggtrials=False, 
                           colwrap=5, add_cbar=False, qmin=None, qmax=None, **kwargs):
    '''
    Plot a series of frames from registered movie for a given run and trial

    :param ops: suite2p output options dictionary
    :param irun: run index
    :param itrial: trial(s) index
    :param iframes: list of frame indexes to plot
    :param ntrials_per_run (optional): number of trials per run. If none provided,
        inferred from REF_NFRAMES.
    :param fps (optional): frame rate (in Hz) used to infer the time of extracted frames
    :param norm (optional): whether to normalize frames to a common color scale (default = True)
    :param cmap (optional): colormap (default = 'gray')
    :param fs (optional): fontsize (default = 12)
    :param height (optional): figure height (default = 3)
    :param colwrap (optional): number of axes per row for single trial or trial-aggregated (default = 5)
    :param add_cbar (optional): whether to add a colorbar (default = False)
    :param qmin (optional): minimum quantile used to bound colormap for image rendering (default = None)
    :param qmax (optional): maximum quantile used to bound colormap for image rendering (default = None)
    :return: figure handle
    '''
    # Check quantiles validity if provided
    if qmin is not None and not is_within(qmin, (0, 1)):
        raise ValueError(f'invalid lower bound quantile: {qmin}')
    if qmax is not None and not is_within(qmax, (0, 1)):
        raise ValueError(f'invalid upper bound quantile: {qmax}')
    if qmin is not None and qmax is not None and qmin >= qmax:
        raise ValueError(f'invalid quantiles: {qmin} >= {qmax}')

    # Cast frames list to array
    iframes = np.atleast_1d(np.asarray(iframes))
    nframes = iframes.size

    # Compute number of trials per run if not specified
    if ntrials_per_run is None:
        ntrials_per_run = REF_NFRAMES // NFRAMES_PER_TRIAL

    # If no trial index provided, gather frames from all trials
    if itrial is None:
        itrial = np.arange(ntrials_per_run)

    # If several trial indexes provided, call function recursively
    if is_iterable(itrial) and not aggtrials:
        fig, axes = plt.subplots(
            len(itrial), len(iframes), figsize=(len(iframes) * height, len(itrial) * height),
            facecolor='white')
        logger.info(f'plotting frames {iframes} from trials {itrial}...')

        loglvl = logger.getEffectiveLevel()
        logger.setLevel(logging.WARNING)
        for i, it in enumerate(tqdm(itrial)):
            plot_registered_frames(
                ops, irun, it, iframes, ntrials_per_run=ntrials_per_run, fps=fps, norm=norm,
                cmap=cmap, fs=fs, height=height, axes=axes[i], overlay_label=overlay_label, 
                qmin=qmin, qmax=qmax)
            axes[i, 0].set_ylabel(f'trial {it}')
        logger.setLevel(loglvl)

        fig.subplots_adjust(top=0.97)
        fig.suptitle(f'run {irun}', fontsize=fs, y=.98)
        return fig

    # Reduce trial index to scalar if single-index iterable
    if is_iterable(itrial) and len(itrial) == 1:
        itrial = itrial[0]
    
    # Extract frames stack
    frames = extract_registered_frames(
        ops, irun, itrial, iframes, ntrials_per_run=ntrials_per_run, aggtrials=aggtrials, **kwargs)

    # Assess whether trial-aggregation is specified and adapt title
    if isinstance(itrial, (int, np.int64)):
        trial_str = f'trial {itrial}'
    else:
        if itrial.data.contiguous:
            trial_str = f'trials {itrial[0]} - {itrial[-1]}'
        else:
            trial_str = f'trials {itrial}'
        if aggtrials:
            trial_str = f'aggregate across {trial_str}'

    # Normalize frames to common color scale (and optional maximal bound) if requested
    if norm:
        vmin, vmax = frames.min(), frames.max()
        if qmax is not None:
            vmax = np.quantile(frames, qmax)
        if qmin is not None:
            vmin = np.quantile(frames, qmin)
        sig_bounds = (vmin, vmax)
        logger.info(f'normalizing frames to common {sig_bounds} interval')
        norm, sm = set_normalizer(cmap, sig_bounds)
    else:
        norm, sm = None, None

    # Create / retrieve figure and axes
    if axes is None:
        nrows, ncols = int(np.ceil(nframes / colwrap)), min(nframes, colwrap)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * height, nrows * height), facecolor='white')
        axes = np.atleast_1d(axes).ravel()
    else:
        fig = axes[0].get_figure()
        if len(axes) != nframes:
            raise ValueError(f'number of axis objects ({len(axes)}) does not match number of frames ({nframes})')
    
    # Plot frames
    for iax, (ax, iframe, frame) in enumerate(zip(axes, iframes, frames)):
        ax.imshow(frame, cmap=cmap, norm=norm, interpolation='none')
        sns.despine(ax=ax, bottom=True, left=True)
        ax.tick_params(
            left=False, labelleft=False, bottom=False, labelbottom=False)
        iframe_diff = iframe - FrameIndex.STIM
        if fps is None:
            title = 'frame = stim'
            if iframe_diff < 0:
                title = f'{title} - {-iframe_diff}'
            elif iframe_diff > 0:
                title = f'{title} + {iframe_diff}'
        else:
            title = 't = stim'
            if iframe_diff < 0:
                title = f'{title} - {-iframe_diff / fps:.2f} s'
            elif iframe_diff > 0:
                title = f'{title} + {iframe_diff / fps:.2f} s'
        if overlay_label:
            ax.text(.5, .9, title, color='w', va='center', ha='center', fontsize=fs, transform=ax.transAxes)
        else:
            ax.set_title(title, fontsize=fs)

    # Hide unused axes 
    for ax in axes[iax + 1:]:
        ax.axis('off')
        
    # Add colorbar, if specified
    if add_cbar:
        if norm is None:
            raise ValueError('colorbar cannot be added if frames are not normalized')
        yb, yt = axes[-1].get_position().y0, axes[0].get_position().y1
        fig.subplots_adjust(right=0.93)
        cbar_ax = fig.add_axes([0.95, yb, 0.02, yt - yb])
        cbar_ax.set_title('F (a.u.)', fontsize=fs)
        fig.colorbar(sm, cax=cbar_ax, ticks=sig_bounds)
        cbar_ax.tick_params(labelsize=fs-2)

    # Adjust figure layout 
    fig.subplots_adjust(top=0.95)
    fig.suptitle(f'run {irun}, {trial_str}', fontsize=fs, y=.98)

    # Return figure handle
    return fig


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
    fbounds = kwargs.pop('fbounds', None)  # frame index bounds

    # Initialize viewer and initialize its rendering
    viewer = get_stack_viewer(*args, **kwargs)
    viewer.init_render(norm=norm, cmap=cmap, fbounds=fbounds, ilabels=ilabels)

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


def plot_pixel_timecourse(fpath, projfunc=np.mean, q=0, fps=None, fc=None):
    '''
    Select pixel based on specific quantile in aggregate intensity,
    and plot its time course

    :param fpath: path to TIF file
    :param projfunc: projection function (default: np.mean)
    :param q: quantile of aggregate pixel intensity from which to select pixel (default: 0)
    :param fps: frame rate (in Hz) used to infer the time of extracted frames
    :param fc: cutoff frequency (in Hz) for low-pass filtering (default: None)
    :return: figure handle
    '''
    # Load stack
    stack = loadtif(fpath)

    # Compute projection image
    proj = projfunc(stack, axis=0)

    # Compute target value of aggregate pixel intensity corresponding to specified quantile
    vtarget = np.quantile(proj, q)
    logger.info(f'target intensity value based on {q * 1e2:.1f}-th percentile: {vtarget:.2f}')

    # Select pixel with aggregate intensity closest to target value
    xypix = np.unravel_index(np.argmin(np.abs(proj - vtarget)), proj.shape)
    vpix = proj[xypix]
    logger.info(f'pixel with aggregate intensity closest to target: P{xypix}={vpix:.2f}')

    # Extract pixel time course
    ypix = stack[:, xypix[0], xypix[1]]

    # Low-pass filter pixel time course, if cutoff frequency specified
    ypix_filt = None
    if fc is not None:
        if fps is None:
            raise ValueError('frame rate must be specified to filter pixel time course')
        order = 2
        nyq = 0.5 * fps
        sos = butter(order, fc / nyq, btype='low', output='sos')
        ypix_filt = sosfiltfilt(sos, ypix)

    # Plot projection image, and mark selected pixel
    fig, axes = plt.subplots(1, 2, figsize=(11, 3), width_ratios=(1, 2))
    ax = axes[0]
    ax.imshow(proj, cmap='gray')
    ax.scatter(*xypix[::-1], s=50, ec='r', fc='none', lw=2)

    # Plot pixel time course
    ax = axes[1]
    sns.despine(ax=ax)
    xvec = np.arange(ypix.size)
    if fps is not None:
        xvec = xvec / fps
        ax.set_xlabel('time (s)')
    else:
        ax.set_xlabel('frame index')
    ax.set_ylabel('pixel intensity')
    ax.plot(xvec, ypix, label='raw')
    if ypix_filt is not None:
        ax.plot(xvec, ypix_filt, label=f'low-pass filtered (fc={fc} Hz)')

    # Return
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
    freqs = np.fft.rfftfreq(nframes, 1 / fs)  # extract frequencies >= 0
    ffts = {k: np.abs(np.fft.rfft(v, axis=0)) for k, v in stacks.items()}  # compute FFT for each pixel along time axis
    ps_avg = {k: np.array([(x**2).mean() for x in v]) for k, v in ffts.items()}  # compute time-average FFT power spectrum for each pixel

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


def add_scale_bar(ax, npx, um_per_px, color='k', fs=None):
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

    # Set up fontproperties dict
    fontproperties = None
    if fs is not None:
        fontproperties = {'size': fs}
    
    # Define scale bar artist object
    scalebar = AnchoredSizeBar(ax.transAxes,
        rel_bar_length, 
        f'{um_bar_length:.0f} um',
        'lower right', 
        pad=0.1,
        color=color,
        frameon=False,
        size_vertical=.01,
        fontproperties=fontproperties
    )
    
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
    if not ops.get('sparse_mode', True):
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
                          groupbyROI=False, errorbar=None, ax=None, **kwargs):
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
    :param errorbar: errorbar method used to render shaded areas around traces. If none is given,
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
                data=plt_data, x=Label.TIME, y=(y, k), 
                errorbar=errorbar,
                hue=hue, palette=None if hue is None else cmap, 
                legend='auto', ax=ax, **kwargs)
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
            robust=True, errorbar=None)
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


def get_hue_per_ROI(Fstats, hue, verbose=False):
    '''
    Determine hue per ROI based on hue type.

    :param Fstats: fluorescence stats dataframe
    :param hue: hue type
    :param verbose (optional): verbosity level
    :return: 3-tuple with:
        - hue per ROI
        - list of hue values
        - dictionary of hue values to colors
    '''
    log_suffix = ''
    # If no hue is provided, all cells are plotted in gray
    if hue is None:
        iROIs = Fstats.index.unique(level=Label.ROI)
        hue_per_ROI = pd.Series(data=['notype'] * len(iROIs), index=iROIs) 
        hues = ['notype']
        colors = {'notype': 'silver'}
    
    # If hue is responder type, use discrete color-coding
    elif hue == Label.ROI_RESP_TYPE:
        hue_per_ROI = get_response_types_per_ROI(Fstats, verbose=verbose)
        hues = get_default_rtypes()
        colors = Palette.RTYPE
        log_suffix = ' color-coded by response type'
    
    # If hue is fraction of positive responses, use continuous color-coding
    elif hue == 'positive':
        hue_per_ROI = Fstats.groupby(Label.ROI)[hue].first()
        hues = np.sort(np.unique(hue_per_ROI))
        ROI_cmap = sns.color_palette('rocket', as_cmap=True)
        colors = {x: ROI_cmap(x) for x in hues}
        log_suffix = ' color-coded by fraction of positive responses'
    
    # If hue is not recognized, raise error
    else:
        raise ValueError(f'invalid hue parameter: "{hue}"')

    # Log process if specified
    if verbose:
        logger.info(f'generating cells map{log_suffix}')

    # Return hue per ROI, hues, and colors    
    return hue_per_ROI, hues, colors


def get_cells_mplobjs(ROI_masks, Fstats, dims, mode='contour', hue=None, verbose=False, alpha=1):
    '''
    Get matplotlib objects to plot the spatial distribution of cells in the field of view.
    
    :param ROI_masks (optional): ROI-indexed dataframe of (x, y) coordinates and weights
    :param Fstats (optional): statistics dataframe
    :param dims: (Ly, Lx) dimensions of the image
    :param mode (optional): 'contour' or 'mask'
    :param hue (optional): name of column in Fstats to use for cells color-coding
    :param verbose (optional): verbosity level
    :return: 3-tuple with:
        - mplobjs: dictionary of [matplotlib objects to plot] for each hue level
        - colors: dictionary of colors per hue level
        - lgdlabels: legend labels per hue level
    '''
    # Check inputs
    if mode not in ['contour', 'mask']:
        raise ValueError(f'unknown mode {mode} (can be "contour" or "mask")')

    # Get hue information
    hue_per_ROI, hues, colors = get_hue_per_ROI(Fstats, hue, verbose=verbose)
    
    # Get number of ROIs
    nROIs = len(hue_per_ROI)

    # Generate legend labels
    count_by_hue = {k: (hue_per_ROI == k).sum() for k in hues}
    if hue == 'positive':
        lgd_labels = [f'{k:.2f} ({v})' for k, v in count_by_hue.items()]
    else:
        lgd_labels = [f'{k} ({v})' for k, v in count_by_hue.items()]

    # Initialize empty 3D mask-per-cell matrix for each hue level
    Z = {k: np.zeros((nROIs, *dims), dtype=np.float32) for k in hues}
    
    # Compute mask per ROI & response type
    for i, (hue, (_, ROI_mask)) in enumerate(zip(hue_per_ROI, ROI_masks.groupby(Label.ROI))):
        Z[hue][i, ROI_mask['ypix'], ROI_mask['xpix']] = 1
    
    # Mask "contour" mode
    if mode == 'contour':
        # Extract contours for each hue
        contours = {k: list(chain.from_iterable(map(find_contours, z))) for k, z in Z.items()}
        
        # Invert x and y coordinates for compatibility with imshow
        contours = {k: [c[:, ::-1] for c in clist] for k, clist in contours.items()}
        
        # Assign contours as output objects
        mplobjs = contours

    # Full "mask" mode
    else:
        # Stack Z matrices along ROIs to get 1 mask matrix per hue
        masks = {k: z.max(axis=0) for k, z in Z.items()}

        # Assign color and transparency to each mask
        rgbmasks = {k: np.zeros((*m.shape, 4)) for k, m in masks.items()}
        for k, mask in masks.items():
            c = colors[k]
            if isinstance(c, str):
                c = to_rgb(c)
            rgbmasks[k][mask == 1] = [*c, alpha]
        
        # Assign masks as output objects
        mplobjs = rgbmasks
    
    # Return mpl objects, colors, and legend labels
    return mplobjs, colors, lgd_labels 


def plot_field_of_view(ops, ROI_masks=None, Fstats=None, title=None, um_per_px=None,
                       refkey='Vcorr', mode='contour', cmap='viridis', 
                       hue=None, legend=True, alpha_ROIs=0.7, ax=None, 
                       verbose=True, qmin=None, qmax=None):
    '''
    Plot field of view, with optional overlay of cells.

    :param ops: suite2p output options dictionary, containing various projection images
    :param ROI_masks (optional): ROI-indexed dataframe of (x, y) coordinates and weights
    :param Fstats (optional): statistics dataframe
    :param title (optional): figure title
    :param um_per_px (optional): spatial resolution (um/pixel). If provided, ticks and tick labels
        on each image are replaced by a scale bar on the graph.
    :param refkey (default: Vcorr): key used to access the specified projection image in the
        output options dictionary
    :param mode (default: contour): ROIs render mode ('fill' or 'contour')
    :param cmap (default: viridis): colormap used to render reference image
    :param hue (optional): hue parameter determining the color of each ROI
    :param legend (default: True): whether to add an ROI classification legend to the figure
    :param alpha_ROIs (default: 1): opacity value for ROIs rendering (only in 'fill' mode)
    :param ax (optional): axis on which to plot the figure
    :param verbose (default: True): verbosity level
    :param qmin (optional): lower quantile used to clip reference image
    :param qmax (optional): upper quantile used to clip reference image
    :return: figure handle
    '''
    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()
    if title is not None:
        ax.set_title(title, fontsize=10)
    
    # Plot reference image with optional color bounding
    refimg, cmap = get_image_and_cmap(ops, refkey, cmap, pad=True)
    vmin, vmax = None, None
    if qmin is not None:
        vmin = np.nanquantile(refimg, qmin)
    if qmax is not None:
        vmax = np.nanquantile(refimg, qmax)
    ax.imshow(refimg, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)

    # Add scale bar if scale provided
    if um_per_px is not None:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        add_scale_bar(ax, ops['Lx'], um_per_px, color='w')
    
    # If ROI masks and Fstats provided
    if ROI_masks is not None and Fstats is not None:
        
        # Get corresponding matplotlib objects to plot cells
        pltobjs, colors, pltlabels = get_cells_mplobjs(
            ROI_masks, Fstats, (ops['Ly'], ops['Lx']), 
            mode=mode, hue=hue, verbose=verbose, alpha=alpha_ROIs)

        # Plot cell and non-cell ROIs
        for hue, pp in pltobjs.items():
            # "contour" mode: add path collection defining contours for each relevant ROI
            if mode == 'contour':
                ax.add_collection(PathCollection(
                    [Path(ctr) for ctr in pp],
                    fc='none', ec=colors[hue], lw=1,
                ))
            # "fill" mode: add mask of ROIs union
            else:
                ax.imshow(pp, origin='lower')

        # Add legend
        if legend:
            if mode == 'contour':
                legfunc = lambda k: dict(c='none', marker='o', mfc='none', mec=colors[k], mew=2)
            else:
                legfunc = lambda k: dict(c='none', marker='o', mfc=colors[k], mec='none')
            leg_items = [
                Line2D([0], [0], label=l, ms=10, **legfunc(c))
                for c, l in zip(colors, pltlabels)]
            ax.legend(handles=leg_items, bbox_to_anchor=(1, 1), loc='upper left', frameon=False)

    return fig


def plot_fields_of_view(ops, ROI_masks=None, Fstats=None, title=None, colwrap=4, 
                        mode='contour', hue=None, outliers=None, **kwargs):
    '''
    Plot field of view (with optional overlaid cells) for each dataset in the options dictionary.
    '''
    logger.info('plotting cell maps...')

    if outliers is None:
        outliers = []

    # Divide inputs per dataset
    datasets = list(ops.keys())
    if ROI_masks is not None:
        masks_groups = dict(tuple(ROI_masks.groupby(Label.DATASET)))
    if Fstats is not None:
        stats_groups = dict(tuple(Fstats.groupby(Label.DATASET)))
        datasets = list(stats_groups.keys())
    ndatasets = len(datasets)

    # Create figure
    ncols = min(ndatasets, colwrap)
    nrows = int(np.ceil(ndatasets / colwrap))
    factor = 3
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * factor, nrows * factor),
        facecolor='w')
    axes = axes.ravel()

    # Plot map for each dataset
    with tqdm(total=ndatasets, position=0, leave=True) as pbar:
        for ax, dataset_id in zip(axes, datasets):
            ogroup = ops[dataset_id]
            mgroup = masks_groups[dataset_id] if ROI_masks is not None else None
            sgroup = stats_groups[dataset_id] if Fstats is not None else None
            plot_field_of_view(
                ogroup, ROI_masks=mgroup, Fstats=sgroup, title=dataset_id, mode=mode, 
                um_per_px=ogroup['micronsPerPixel'], ax=ax, legend=False, hue=hue, 
                verbose=False, **kwargs)
            ax.set_aspect(1.)
            ax.set_xticks([])
            ax.set_yticks([])
            if dataset_id in outliers:
                for sk in ax.spines:
                    ax.spines[sk].set_color('r')
                    ax.spines[sk].set(linewidth=3)
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
        hues = Fstats[hue].unique()
        if hue == Label.ROI_RESP_TYPE:
            leg_palette = {k: v for k, v in Palette.RTYPE.items() if k in hues}
        elif hue == 'positive':
            hues = np.linspace(0, 1, 5)
            ROI_cmap = sns.color_palette('rocket', as_cmap=True)
            leg_palette = {f'{x:.2f}': ROI_cmap(x) for x in hues}
        leg_items = [Line2D([0], [0], label=l, ms=10, **legfunc(c)) for l, c in leg_palette.items()]
        axes[ndatasets - 1].legend(
            handles=leg_items, bbox_to_anchor=(1, 1), loc='upper left', frameon=False)
    
    return fig


def plot_response_map(ROI_masks, Fstats, ops, hue='positive', title=None, um_per_px=None,
                      cmap='viridis', ax=None, fs=15, add_cbar=True, mark_ROIs=False,
                      ncells_per_ax=5, interp_method='cubic', alpha_map=None):
    '''
    Plot spatial distribution of population responsiveness on the recording plane.

    :param ROI_masks: ROI-indexed dataframe of (x, y) coordinates and weights
    :param Fstats: statistics dataframe
    :param ops: suite2p output options dictionary
    :param um_per_px (optional): spatial resolution (um/pixel). If provided, ticks and tick labels
        on each image are replaced by a scale bar on the graph.
    :param title (optional): figure title
    :param cmap (default: viridis): colormap used to render reference image
    :param hue: hue parameter used to draw heatmap
    :param ax (optional): axis handle
    :param fs (default: 15): font size
    :param add_cbar (default: True): whether to add a colorbar
    :param mark_ROIs (default: False): whether to mark ROIs on the heatmap
    :return: figure handle
    '''
    # If multiple datasets, plot each on a separate axis
    if Label.DATASET in Fstats.index.names: 
        fg = sns.FacetGrid(Fstats.reset_index(), col=Label.DATASET, col_wrap=4, height=2.5)
        fig = fg.figure
        axes = fig.axes
        for i, (ax, (dataset_id, gstats)) in enumerate(zip(axes, Fstats.groupby(Label.DATASET))):
            logger.info(f'dataset: {dataset_id}')
            plot_response_map(
                ROI_masks.loc[dataset_id], gstats.droplevel(Label.DATASET),
                ops[dataset_id], ax=ax, title=dataset_id, um_per_px=um_per_px,
                add_cbar=i == len(axes) - 1, fs=fs, cmap=cmap,
                mark_ROIs=mark_ROIs, ncells_per_ax=ncells_per_ax, 
                interp_method=interp_method, alpha_map=alpha_map)
        return fig

    # If no scale given, try to fetch it from ops 
    if um_per_px is None:
        um_per_px = ops.get('micronsPerPixel', None)

    # Compute location (i.e. mask center of) mass for each ROI
    ROIstats = ROI_masks[['xpix', 'ypix']].groupby(Label.ROI).mean()

    # Compute hue metrics per ROI
    ROIstats[hue] = Fstats.groupby(Label.ROI)[hue].first()

    # Remove ROIs with no hue value
    ROIstats = ROIstats[~ROIstats[hue].isna()]

    # Create 2D meshgrid covering FOV
    Ly, Lx = ops['Ly'], ops['Lx']
    x = np.linspace(0, Lx, ncells_per_ax + 1)
    y = np.linspace(0, Ly, ncells_per_ax + 1)

    # Compute average response score in each grid cell
    logger.info(f'Computing average response scores over {x.size}-by-{y.size} grid')
    z, *_ = binned_statistic_dd(
        ROIstats[['xpix', 'ypix']].values,
        ROIstats[hue].values,
        statistic='mean', bins=(x, y)
    )

    # If interpolation required
    if interp_method is not None:
        # Create coordinate vectors for grid cell centers
        xmids = (x[:-1] + x[1:]) / 2
        ymids = (y[:-1] + y[1:]) / 2

        # Create dense interpolation grid
        x = np.arange(Lx)
        y = np.arange(Ly)

        # If all grid cells containg a valid aggregate, use structured interpolation
        if np.all(~np.isnan(z)):
            logger.info(f'applying {interp_method} interpolation over {x.size}-by-{y.size} evaluation grid')
            finterp = interp2d(xmids, ymids, z.T, kind=interp_method)
            z = finterp(x, y).T
        
        # Otherwise, use unstructured interpolation
        else:
            # Create 2D meshgrid of grid cell centers
            X, Y = np.meshgrid(xmids, ymids, indexing='ij')

            # Serialize, merge with z-values, and remove invalid cells
            xyz_grid = np.array([X.ravel(), Y.ravel(), z.ravel()]).T
            isvalid = np.all(~np.isnan(xyz_grid), axis=1)
            xyz_grid = xyz_grid[isvalid]
            logger.info(f'defining unstructured {interp_method} interpolator from {xyz_grid.shape[0]}/{z.size} grid points')

            # Interpolate response map over dense grid to generate denser heatmap
            logger.info(f'applying interpolator over {x.size}-by-{y.size} evaluation grid')
            X, Y = np.meshgrid(x, y, indexing='ij')
            xyeval = np.array([x.ravel() for x in [X, Y]])
            z = griddata(
                xyz_grid[:, :2], xyz_grid[:, 2], xyeval.T, 
                method=interp_method, fill_value=np.nan
            )
            z = z.reshape(X.shape)

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.get_figure()
    
    # Add title if specified
    if title is not None:
        ax.set_title(title, fontsize=fs)

    # Prepare axis
    ax.set_aspect(1.)
    for k in ax.spines.keys():
        ax.spines[k].set_visible(True)

    # Create normalizer and associated mappable objects
    norm = plt.Normalize(0, 1)

    # Plot heatmap
    if alpha_map is None:
        alpha_map = 0.5 if mark_ROIs else 1
    sm = ax.pcolormesh(
        x, y, z.T, norm=norm, cmap=cmap, alpha=alpha_map)

    # Mark ROIs if specified
    if mark_ROIs:
        logger.info('adding ROI markers')
        ax.scatter(
            ROIstats['xpix'], ROIstats['ypix'], s=15,
            c=ROIstats[hue], cmap=cmap, marker='o', norm=norm)

    # Plot contours
    levels = [.5]
    ax.contour(x, y, z.T, levels=levels, colors='k')

    # Add colorbar
    if add_cbar:
        pos = ax.get_position()
        fig.subplots_adjust(right=0.93)
        cbar_ax = fig.add_axes([0.95, pos.y0, 0.02, pos.y1 - pos.y0])
        cbar_ax.set_title(hue, fontsize=fs)
        fig.colorbar(sm, cax=cbar_ax)
        cbar_ax.tick_params(labelsize=fs-2)

    # Add scale bar if scale provided
    if um_per_px is not None:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        add_scale_bar(ax, Lx, um_per_px, color='k', fs=fs-2)

    # Return figure handle
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
                   errorbar='se', legend='full', err_style='band', ax=None, alltraces=False, kind='line',
                   nmaxtraces=None, hue=None, hue_order=None, col=None, col_order=None, fs=None,
                   label=None, title=None, dy_title=0.6, markerfunc=None, max_colwrap=5, ls='-', lw=2,
                   height=None, aspect=1.5, alpha=None, palette=None, marker=None, markersize=5,
                   hide_col_prefix=False, col_count_key=None, color=None, markeredgecolor='k', 
                   **filter_kwargs):
    ''' Generic function to draw line plots from the experiment dataframe.
    
    :param data: experiment dataframe
    :param xkey: key indicating the specific signals to plot on the x-axis
    :param ykey: key indicating the specific signals to plot on the y-axis
    :param xbounds (optional): x-axis limits for plot
    :param ybounds (optional): y-axis limits for plot
    :param aggfunc (optional): method for aggregating across multiple observations within group.
    :param weightby (optional): column used to weight observations upon aggregration.
    :param errorbar (optional): errorbar method to plot shaded area around mean traces
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
    if errorbar is not None:
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
        markeredgecolor = markeredgecolor,  # marker edge color
        hue       = hue,           # hue grouping variable
        hue_order = hue_order,     # hue plotting order 
        estimator = aggfunc,       # aggregating function
        color     = color,         # plot color
        errorbar  = errorbar,      # errorbar estimation method
        err_style = err_style,     # error visualization style 
        lw        = lw,            # line width
        palette   = palette,       # color palette
        legend    = legend,        # use all hue entries in the legend
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

    ###################### XY Label size ######################

    if fs is not None:
        for ax in axlist:
            if ax.get_xlabel() is not None:
                ax.set_xlabel(ax.get_xlabel(), fontsize=fs)
            if ax.get_ylabel() is not None:
                ax.set_ylabel(ax.get_ylabel(), fontsize=fs)
            for item in ax.get_xticklabels() + ax.get_yticklabels():
                item.set_fontsize(fs)

    ###################### Title & legend ######################

    # Adjust legend font size if specified
    if fs is not None:
        for ax in axlist:
            leg = ax.get_legend()
            if leg is not None:
                for item in leg.texts:
                    item.set_fontsize(fs)
                if leg.get_title() is not None:
                    leg.set_title(leg.get_title().get_text(), prop={'size': fs})

    # Determine title 
    if title is None:
        if filters is None:
            filters = {'misc': 'all responses'}
        title = ' - '.join(filters.values())
    
    # Use appropriate title function depending on number of axes
    if col is None: 
        # If only 1 axis (i.e. no column grouping) -> add to axis
        axlist[0].set_title(title, fontsize=fs)
    else:
        # Otherwise -> add as suptitle
        height = fig.get_size_inches()[1]
        fig.subplots_adjust(top=1 - dy_title / height)
        fig.suptitle(title, fontsize=fs)

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
        errorbar = None,  # no error shading
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
    logger.info(f'adding sample counts per {hue} on legend labels...')

    # Count number of samples per hue and x-axis value
    counts_data = data
    if Label.ROI in data.index.names:
        counts_data = data.groupby([hue, xkey, Label.ROI]).first()
    counts_by_hue = counts_data.groupby([hue, xkey]).count().loc[:, ykey].unstack()

    # Keep only the max across all x-levels for each hue
    counts_by_hue = counts_by_hue.max(axis=1).astype(int)

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


def plot_parameter_dependency(data, xkey=Label.P, ykey=None, yref=0., ax=None, hue=None,
                              avgprop=None, errprop='inter', marker=None, avg_color='k',
                              err_style='band', add_leg_numbers=True, hue_alpha=1., errorbar='se',
                              legend='full', as_ispta=False, stacked=False, fit=None,
                              lw=1.5, avgmarker=None, avgmarkersize=5, avglw=3, avgerr=True, palette=None, outliers=None,
                              xscale='linear', **kwargs):
    ''' Plot parameter dependency of responses for specific sub-datasets.
    
    :param data: trial-averaged experiment dataframe
    :param xkey (optional): key indicating the independent variable of the x-axis (default = P)
    :param ykey (optional): key indicating the dependent variable of the y-axis (default: evoked DFF)
    :param yref (optional): vertical at which to draw a "reference" horizontal line (default: 0)
    :param hue (optional): hue grouping parameter (default: None)
    :param avgprop: whether and how to propagate the data for global average reporting (None, "all", "hue" or "whue")
    :param errprop: how to propagate data for global standard error reporting ("inter or "intra")
    :param marker (optional): marker to use for data points (if any)
    :param avg_color (optional): color to use for global average trend (default: "k")
    :param legend (optional): whether to plot a legend for each hue level (default: "full")
    :param add_leg_numbers: whether to add sample counts for each legend entry (default = False)
    :pramhue_alpha (optional): opacity level of indidvidual hue traces (default = 1)
    :param errorbar: errorbar method to plot shaded areas around traces (default = 'se' == SEM)
    :param as_ispta (optional): whether to project input parameter to ISPTA space (default: False)
    :param stacked (optional): whether to offset each hue trend vertically (default: False)
    :param fit (optional): tuple of (objective fit function, initial fit parameters function) to used to fit a function to the average dependency profile
    :param lw (optional): line width for individual traces
    :param avglw (optional): line width for global average trace
    :param avgerr: whether to plot the standard error for the globsl average (default: True)
    :param palette (optional): color palette for hue levels
    :param outliers (optional): specific hue categories that should be materialized as outliers (default = None)
    :param xscale (optional): scale of the x-axis (default = 'linear')
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
                avg_color=avg_color, fit=fit, err_style=err_style, lw=lw, **kwargs)
        # Otherwise, make copy and offset values per dataset if specified
        else:
            if stacked:
                data = data.copy()
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
        hueerr_style = 'band'

    # Get default ykey if needed
    if ykey is None:
        ykey = get_change_key(Label.DFF)

    # Restrict data based on xkey
    data = get_xdep_data(data, xkey)

    # Switch xkey to ISPTA if specified
    if as_ispta:
        xkey = Label.ISPTA

    # Create uniform palette, if requested
    if palette == 'uniform':
        palette = get_uniform_palette(data)
        legend = False
    if outliers is not None:
        palette = get_binary_palette(data, outliers)
        legend = False
    
    # Adapt average color to mouse line, if requested
    if avg_color == 'line':
        avg_color = Palette.LINE[data[Label.LINE].unique()[0]]
    
    # Assemble common plotting arguments
    pltkwargs = dict(
        ax=ax, 
        errorbar=errorbar, 
        marker=marker,
        palette=palette, 
        **kwargs
    )
    if hue_alpha == 0.:
        pltkwargs['errorbar'] = None

    # If hueplt specified
    if hueplt:
        fig = plot_from_data(
            data, xkey, ykey, hue=hue, hue_order=hue_order, alpha=hue_alpha,
            err_style=hueerr_style, legend=legend, lw=lw, **pltkwargs)
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
            # Compute weights vector based on counts
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
        
        avg_kwargs = dict(
            c=avg_color,
            markersize=avgmarkersize, 
            # markeredgecolor='w'
        )
        # Plot propagated global mean trace and standard error (as bars or band)
        sem = sem.fillna(0)
        if err_style == 'bars':
            if avgerr:
                ax.errorbar(
                    mean.index, mean.values, yerr=sem.values, 
                    lw=0 if fit else avglw, elinewidth=2 if fit else avglw,
                    marker=avgmarker, **avg_kwargs)
            else:
                ax.plot(
                    mean.index, mean.values, 
                    lw=0 if fit else avglw,
                    marker=avgmarker, **avg_kwargs)
        elif err_style == 'band':
            ax.plot(
                mean.index, mean.values, 
                lw=0 if fit else avglw, **avg_kwargs)
            if avgerr:
                ax.fill_between(
                    mean.index, mean.values - sem.values, mean.values + sem.values, 
                    fc=avg_kwargs['c'], alpha=.3, ec=None)
        else:
            raise ValueError(f'invalid error style: {err_style}')

        # If fit objects provided, compute and add to axis
        if fit:
            objfunc, initfunc = fit
            pltkwargs = dict(ls='--', lw=avglw, **avg_kwargs)
            compute_and_add_fit(
                ax, mean.index.values, mean.values, objfunc, initfunc=initfunc, **pltkwargs)

    # Add reference line(s) if specified
    if yref is not None:
        for y in as_iterable(yref):
            ax.axhline(y, c='k', ls='--')
    

    # Adjust x-scale
    adjust_xscale(ax, xscale=xscale)
    
    # Return figure handle
    return fig


def compute_and_add_fit(ax, xdata, ydata, objfunc, initfunc=None, **pltkwargs):
    '''
    Fit a function to and X-Y profile, compute the goodness of fit (R-squared)
    and plot the resulting function across the X-data range

    :param ax: axis object
    :param xdata: x data range
    :param ydata: y data range
    :param objfunc: function to fit to data
    :param initfunc (optional): function to estimate initial fit parameters
    :param pltkwargs: plotting keyword arguments
    '''
    # Call function to estimate initial parameters, if provided
    if initfunc is not None:
        logger.info(f'estimating initial fit parameters for {objfunc.__name__} function')
        p0 = initfunc(xdata, ydata)
    else:
        p0 = None
    
    # If specified, perform polynomial fit
    if isinstance(objfunc, str) and objfunc.startswith('poly'):
        order = int(objfunc[4])
        fitname = objfunc
        logger.info(f'fitting order {order} polynomial to data')
        objfunc = np.poly1d(np.polyfit(xdata, ydata, order))
        popt = []

    # Otherwise, perform classic fit
    else:
        fitname = objfunc.__name__
        logger.debug(f'computing fit with {fitname} function: p0 = {p0}')
        popt, _ = curve_fit(objfunc, xdata, ydata, p0, maxfev=10000)
    
    # Compute output value with fit predictor
    yfit = objfunc(xdata, *popt)

    # Compute error between fit prediction and data
    r2 = rsquared(ydata, yfit)
    logger.info(f'fitting results: popt = {popt}, R2 = {r2:.2f}')

    # Plot fit profile
    xdense = np.linspace(*bounds(xdata), 100)
    yfitdense = objfunc(xdense, *popt)
    ax.plot(xdense, yfitdense, **pltkwargs)

    # Display fit results on graph
    ax.text(
        .5, .9, f'{fitname} fit: R2 = {r2:.2f}', ha='center', va='top', 
        transform=ax.transAxes, fontsize=12)



def plot_parameter_dependency_across_datasets(data, xkey=Label.P, hue=None, ykey=None, ax=None,
                                              legend=True, yref=None, add_leg_numbers=True,
                                              marker='o', avg_color='k', ls='-', as_ispta=False, title=None,
                                              weighted=True, fit=None, err_style='band', errprop='inter', 
                                              lw=1, markersize=6):
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

    # Define propagated average and standard error keys
    mu_key, sem_key = f'{ykey} - mean', f'{ykey} - sem'

    # Adapt average color to mouse line, if requested
    if avg_color == 'line':
        avg_color = Palette.LINE[data[Label.LINE].unique()[0]]
    
    # If hue not specified
    if hue is None:
        # Aggregate data with cell-count weighting
        aggdata = get_crossdataset_average(
            data, xkey, ykey=ykey, hue=hue, weighted=weighted, errprop=errprop)

        # Plot single weighted average trace with propagated standard errors 
        if err_style == 'bars':
            ax.errorbar(
                aggdata[xkey], aggdata[mu_key], yerr=aggdata[sem_key], 
                marker=marker, ls=ls, c=avg_color, markeredgecolor='none',
                lw=lw, elinewidth=2, markersize=markersize)
        else:
            ax.plot(
                aggdata[xkey], aggdata[mu_key], 
                marker=marker, ls=ls, c=avg_color, lw=lw, 
                markeredgecolor='none', markersize=markersize)
            if err_style == 'band':
                ax.fill_between(
                    aggdata[xkey], 
                    aggdata[mu_key] - aggdata[sem_key], aggdata[mu_key] + aggdata[sem_key],
                    alpha=0.3, fc=avg_color, ec=None)
        
        # Add fit if required
        if fit is not None:
            objfunc, initfunc = fit
            compute_and_add_fit(
                ax, aggdata[xkey], aggdata[mu_key], objfunc, initfunc=initfunc,
                ls='--', color=avg_color)

    # Otherwise
    else:
        # For each hue value
        for htype, rdata in data.groupby(hue):
            # Aggregate data with cell-count weighting
            aggdata = get_crossdataset_average(
                rdata, xkey, ykey=ykey, hue=hue, weighted=weighted, errprop=errprop)
            if hue == Label.ROI_RESP_TYPE:
                color = Palette.RTYPE[htype]
            else:
                color = None
            if err_style == 'bars':
                ax.errorbar(
                    aggdata[xkey], aggdata[mu_key], yerr=aggdata[sem_key],
                    marker=marker, ls=ls, label=htype, color=color, 
                    linewidth=0 if fit else None, elinewidth=2 if fit else None)
            else:
                ax.plot(
                    aggdata[xkey], aggdata[mu_key],
                    marker=marker, ls=ls, label=htype, color=color, 
                    linewidth=0 if fit else None)
                if err_style == 'band':
                    ax.fill_between(
                        aggdata[xkey], 
                        aggdata[mu_key] - aggdata[sem_key], aggdata[mu_key] + aggdata[sem_key],
                        alpha=0.3, color=color)

            # Add fit if required
            if fit is not None:
                objfunc, initfunc = fit
                compute_and_add_fit(
                    ax, aggdata[xkey], aggdata[mu_key], objfunc, initfunc=initfunc,
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


def plot_stimparams_dependency(data, ykey, title=None, axes=None, xkeys=None, **kwargs):
    '''
    Plot dependency of a specific response metrics on stimulation parameters
    
    :param data: trial-averaged experiment dataframe
    :param ykey (optional): key indicating the dependent variable of the y-axis
    :param kwargs: keyword parameters that are passed to the plot_parameter_dependency function
    :return: figure handle
    '''
    if xkeys is None:
        xkeys = [Label.P, Label.DC]
    
    # Initialize or retrieve figure
    if axes is None:
        parent = True
        height = 3
        factor = height
        if kwargs.get('stacked', False):
            height = max(height, len(data.index.unique(Label.DATASET)))
        fig, axes = plt.subplots(1, len(xkeys), figsize=(len(xkeys) * factor, height))
    else:
        parent = False
        if len(axes) != len(xkeys):
            raise ValueError(f'exactly {len(xkeys)} axes must be provided')
        fig = axes[0].get_figure()

    # Disable legend for all axes but last
    legend = kwargs.get('legend', True)
    kwargs['legend'] = False
    tightened = False
    # Plot dependencies on each parameter on separate axes
    for i, (xkey, ax) in enumerate(zip(xkeys, axes.T)):
        if i == len(axes) - 1 and legend:
            del kwargs['legend']
        plot_parameter_dependency(
            data, xkey=xkey, ax=ax, ykey=ykey, title=f'{xkey} dependency', 
            xscale='sqrt' if xkey==Label.ISPTA else 'linear', **kwargs)   
        if parent and not tightened:
            fig.tight_layout()
            tightened = True

        axleg = ax.get_legend()
        if axleg is not None:
            nentries = len(axleg.get_lines())
            sns.move_legend(
                ax, 'upper left', bbox_to_anchor=(1, 1), frameon=False, 
                ncol=int(np.ceil(nentries / 15))
            )
    
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
    axdim = 'y' if bar == Label.DATASET else 'x'

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
                ax=ax2, 
                ylabel='', 
                autopct='%1.0f%%',
                colors=[Palette.RTYPE[k] for k in counts_by_rtype.index],
                startangle=90, 
                textprops={'fontsize': 12}, 
                wedgeprops={'edgecolor': 'k', 'alpha': 0.7}
            )
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


def plot_P_DC_map(P=None, DC=None, dose_key=None, cmap='rocket_r', color='silver', s=30,
                  ncontours=4, contour_labels=False, fs=12, ax=None):
    ''' 
    Plot sonication protocol in the DC - pressure space
    
    :param P (optional): array of peak pressure amplitudes (in MPa)
    :param DC (optional): array of duty cycles (in %)
    :param dose_key (optional): dose metrics to overlay as 2D colormap
    :param cmap: colormap to use for dose metrics plotting (default: 'rocket')
    :param s: size of sampled points (default: 30)
    :param color: color used to plot sampled points (default: 'silver')
    :param ncontours: number of contour levels to plot (default: 4)
    :param contour_labels: whether to plot dose contour labels (default: False)
    :param fs: font size (default: 12)
    :param ax: axis handle
    :return: figure handle
    '''
    # If multiple dose metrics specified, create figure and plot each in a separate axis
    if is_iterable(dose_key):
        fig, axes = plt.subplots(len(dose_key), 1, figsize=(1.5, 2 * len(dose_key)))
        fig.subplots_adjust(hspace=0.5, wspace=.7)
        for key, ax in zip(dose_key, axes.ravel()):
            plot_P_DC_map(
                P=P, DC=DC, dose_key=key, cmap=cmap, color=color, s=20, 
                ncontours=ncontours, contour_labels=contour_labels, fs=fs, ax=ax)
        return fig

    # Create or retrieve figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.get_figure()
    ax.set_xlabel(Label.DC, fontsize=fs)
    ax.set_ylabel(Label.P, fontsize=fs)
    sns.despine(ax=ax)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)

    # Determine pressure axis upper limit
    Pup = 1.25 * P.max() if P is not None else 1.    

    # Compute P and DC range vectors for plotting
    nperax = 100
    Prange = np.linspace(0, Pup, nperax)  # MPa
    DCrange = np.linspace(0, 100, nperax)  # %

    # Set axis limits
    ax.set_xlim(DCrange.min(), DCrange.max())
    ax.set_ylim(Prange.min(), Prange.max())

    # Compute 2D grid of P and DC values
    Pgrid, DCgrid = np.meshgrid(Prange, DCrange, indexing='ij')

    # Plot sampled DC - P combinations, if provided
    if P is not None and DC is not None:
        ax.scatter(DC, P, c=color, edgecolors='k', zorder=80, s=s)
    
    # If aggregate metrics is provided
    if dose_key is not None:
        
        # Compute 2D array of metric values over P - DC grid
        dose_values = get_dose_metric(Pgrid, DCgrid, dose_key)

        # Fetch colormap
        cmap = sns.color_palette(cmap, as_cmap=True)
        
        # Plot dose metric colormap over DC - DC space
        sm = ax.pcolormesh(
            DCrange, Prange, dose_values,
            shading='nearest', rasterized=True, cmap=cmap)
        
        # Create axis for colorbar on the right side of the plot
        bbox = ax.get_position()
        cax = fig.add_axes([
            bbox.x1 + bbox.width * .05, 
            bbox.y0, 
            bbox.width * 0.1, 
            bbox.height
        ])
        
        # Add colorbar
        cbar = fig.colorbar(sm, pad=0.02, aspect=40, cax=cax)
        cbar.set_label(dose_key, fontsize=fs)
        cbar.set_ticks([dose_values.min(), dose_values.max()])
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
        for item in cbar.ax.get_yticklabels():
            item.set_fontsize(fs)

        # Add dose metric contours if specified 
        if ncontours > 0:
            # Get reference dose value

            # If no contour labels specified, use half of max dose value
            if not contour_labels:
                refdose = dose_values.max() * .5

            # If contour labels specified, round reference dose to nearest 10^exp value
            if contour_labels: 
                maxdose = dose_values.max()
                maxdose_factor = 10 ** np.floor(np.log10(maxdose))
                refdose = np.floor(maxdose / maxdose_factor) * maxdose_factor

            # Determine characteristic dose levels to mark on plot
            dose_levels = np.logspace(-2, 0, ncontours) * refdose
            CS = ax.contour(
                DCrange, Prange, dose_values, levels=dose_levels, colors='k')
            
            # Add contour labels, if specified
            if contour_labels:
                ax.clabel(CS, fontsize=fs, inline=True, fmt='%.2g')

    # # Finalize figure layout
    # fig.tight_layout()

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
        pairs = [[(dataset, item) for item in pair] for dataset, pair in combs.items()]
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


def plot_comparative_metrics_across_conditions(data, ykey, condkey, order=None, kind='box', fs=12,  
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
    logger.info(f'plotting {ykey} across {condkey}')
    # Aggregate output metrics across conditions and datasets
    yagg = data.groupby([Label.DATASET, condkey])[ykey].mean()

    # Reindex to enforce condition order display
    if order is not None:
        yagg = yagg.reindex(order, level=condkey)

    # Determine if condition is categorical
    is_cond_categorical = yagg.index.get_level_values(condkey).dtype == 'O'

    # Establish pairs of conditions to compare
    pairs = list(combinations(yagg.unstack().columns, 2)) 
    
    # Define plot arguments
    pltkwargs = dict(
        data=yagg.reset_index(condkey),
        x=condkey,
        y=ykey,
    )
    
    # Render categorical plot 
    fg = sns.catplot(
        kind=kind, 
        whis=False, 
        showfliers=False, 
        height=3,
        **pltkwargs
    )

    # Extract figure and axis
    fig = fg.figure
    ax = fig.axes[0]

    # Restrict y ticks if normalized y unit
    if 'normalized' in ykey:
        ax.set_yticks([0, 0.5, 1])
    
    # Show underlying data points
    sns.scatterplot(
        ax=ax, 
        hue=condkey, 
        s=50,
        edgecolor='k',
        linewidth=1.5,
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
            ax.plot(tmp.index, tmp[s], c='darkgray', zorder=10)

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
        if fs is not None:
            with sns.plotting_context(rc={'font.size': fs}):
                annotator.apply_and_annotate()
        else:
            annotator.apply_and_annotate()
    
    if is_cond_categorical:
        ax.set_xlabel(None)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # Adapt labels font size
    if fs is not None:
        ax.set_xlabel(ax.get_xlabel(), fontsize=fs)
        ax.set_ylabel(ax.get_ylabel(), fontsize=fs)
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(fs)
    
    return fig


def plot_parameter_dependency_across_lines(data, xkey, ykey, yref=0., axes=None, norm=False, 
                                           err_style='band', connect='line', fill_between=True):
    '''
    Plot comparative parameter dependency curves (with error bars) across
    mouse lines, for each responder type

    :param data: responder-type-averaged multi-line statistics dataframe
    :param xkey: input parameter name(s)
    :param ykey: output parameter name
    :param yref: reference vertical level to indicate with dashed line (optional)
    :param axes: axes list (optional)
    :param norm: whether to normalize dependency profiles
    :return: figure
    '''
    # If specific, normalize metrics for each line
    if norm:
        yrefs = (
            data[f'{ykey} - mean']
            .abs()
            .groupby(Label.LINE)
            .max()
        )
        for k in ['mean', 'sem']:
            data[f'{ykey} - {k}'] /= yrefs

    xkeys = as_iterable(xkey)
    if axes is None:
        # Create figure backbone
        fig, axes = plt.subplots(1, len(xkeys), figsize=(3 * len(xkeys), 3))
    else:
        if len(axes) != len(xkeys):
            raise ValueError(
                f'number of axes ({len(axes)}) does not match number of inputs ({len(xkeys)})')
        fig = axes[0].get_figure()

    # Determine plot function
    if connect is None:
        pltfunc = sns.scatterplot
    elif connect == 'line':
        pltfunc = sns.lineplot
                       
    # For each input parameter x
    for i, (xkey, ax) in enumerate(zip(xkeys, axes)):
        sns.despine(ax=ax)
        pltkwargs = dict(
            ax=ax,
            x=xkey, y=f'{ykey} - mean',
            hue=Label.LINE,
            palette=Palette.LINE,
        )
        
        # Get x-dependent data
        if xkey != Label.ISPTA:
            depdata = get_xdep_data(data, xkey)
            # Plot mean profile, per mouse line
            pltfunc(
                data=depdata, legend=i == len(xkeys) - 1, **pltkwargs)                
        else:
            for xk, ls in {Label.P: '--', Label.DC: '-'}.items(): 
                # Plot mean profile, per mouse line
                pltfunc(
                    data=get_xdep_data(data, xk), ls=ls,
                    legend=i == len(xkeys) - 1 and xk == Label.DC,
                    **pltkwargs
                )
            depdata = data.copy()

        # Create fill vectors, if requested 
        if fill_between:
            xfill = []
            yfill = []
        
        # Plot +/-sem error, for each mouse line
        for iline, (line, ldata) in enumerate(depdata.groupby(Label.LINE)):
            ldata = ldata.sort_values(xkey)
            ymu, ysem = ldata[f'{ykey} - mean'], ldata[f'{ykey} - sem']
            if err_style == 'band':
                ax.fill_between(
                    ldata[xkey], ymu - ysem, ymu + ysem,
                    fc=Palette.LINE[line], alpha=0.3)
            elif err_style == 'bars':
                ax.errorbar(
                    ldata[xkey], ymu, yerr=ysem,
                    c=Palette.LINE[line], fmt='.')
        
            # Append to fill vectors, if requested
            if fill_between:
                if iline % 2 == 0:
                    xfill.append(ldata[xkey])
                    yfill.append(ymu)
                else:
                    xfill.append(ldata[xkey][::-1])
                    yfill.append(ymu[::-1])

            iline += 1

        # Apply fill, if requested
        if fill_between:
            ax.fill(
                np.hstack(xfill),
                np.hstack(yfill),
                fc='silver',
            )

        # Adjust y-labels
        ax.set_ylabel(ykey)
        # Add reference vertical level
        if yref is not None:
            ax.axhline(yref, ls='--', c='k')
    
    # Clean up layout, and move legend outside of axes
    fig.tight_layout()
    sns.move_legend(axes[-1], 'upper left', bbox_to_anchor=(1, 1), frameon=False)
    
    # Return figure
    return fig


def plot_intensity_dependencies(data, ykey, ax=None, hue=Label.ROI_RESP_TYPE, yref=0., 
                                avg_color='k', add_sweep_markers=True, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_title(f'ISPTA dependency')
    else:
        fig = ax.get_figure()
    # Determine output metrics key
    logger.info(f'plotting {ykey} ISPTA dependency across responders...')
    # Plot ISPTA dependency profiles
    plot_parameter_dependency(
        data, xkey=Label.ISPTA, ykey=ykey, yref=yref, hue=hue, ax=ax, 
        # marker=None, 
        as_ispta=True, 
        avglw=0 if add_sweep_markers else 3, 
        avg_color=avg_color, **kwargs)
    # Indicate the "parameter sweep" origin of each data point
    # Plot dependencies on each parameter on same ISPTA axis
    if add_sweep_markers:
        kwargs['err_style'] = None
        kwargs['legend'] = False
        # Adapt average color to mouse line, if requested
        if avg_color == 'line':
            avg_color = Palette.LINE[data[Label.LINE].unique()[0]]
        for xkey in [Label.P, Label.DC]:
            plot_parameter_dependency(
                data, xkey=xkey, ykey=ykey, yref=None, ax=ax, hue=None, 
                weighted=True, as_ispta=True,
                avg_color=avg_color, 
                add_leg_numbers=False, 
                # lw=3,
                ls={Label.P: '--', Label.DC: '-'}[xkey],
                **kwargs
            )
        #     plot_parameter_dependency(
        #         data, xkey=xkey, ykey=ykey, yref=None, ax=ax, hue=None, weighted=True, as_ispta=True, 
        #         marker=sweep_markers[xkey], markersize=7, avg_color=avg_color, 
        #         add_leg_numbers=False, err_style=None, lw=0., **kwargs)
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
        x=Label.RUN, y=ykey, errorbar='se', 
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


def plot_responder_fraction(data, xkey, hue=Label.DATASET, xref=None, kind='line', ax=None, 
                            avg_overlay=True, avg_color='k', hue_width=False, palette=None, outliers=None,
                            legend='full', fit=None, avglw=3, err_style='band', xscale='linear',
                            avgmarker=None, **kwargs):
    ''' 
    Plot fraction of responder cells as a function of an input parameter

    :param data: statistics dataframe
    :param xkey: input parameter name
    :return: figure handle
    '''
    # Get fraction of responder per input and hue
    resp_props = get_responders_counts(data, xkey, units=hue, normalize=True)
    # Extract weights and compute weighted fraction per input level
    weights = resp_props.pop('weight')
    weighted_resp_props = resp_props.multiply(weights, axis=0).groupby(xkey).sum()
    resp_props_sem = resp_props.groupby(xkey).sem()
    # if specified, compute counts per hue level to specify line widths
    if hue is not None and hue_width:
        countsperhue = resp_props['count'].groupby(hue).max()
        if hue in resp_props.index.names:
            hvals = resp_props.index.get_level_values(hue)
        else:
            hvals = resp_props[hue]
        resp_props['count'] = hvals.map(countsperhue)
    
    # Create uniform palette, if requested
    if palette == 'uniform':
        palette = get_uniform_palette(data)
        legend = False
    if outliers is not None:
        palette = get_binary_palette(data, outliers)
     
    # Adapt average color to mouse line, if requested
    if avg_color == 'line':
        avg_color = Palette.LINE[data[Label.LINE].unique()[0]]
    
    # Define default plotting options dict
    pltkwargs = dict(
        data=resp_props.reset_index(), 
        x=xkey, 
        y='positive', 
        hue_order=None if hue is None else list(data.groupby(hue).indices.keys()),
        legend=legend,
        palette=palette,
        markeredgecolor='k',
        markersize=5,
        **kwargs
    )

    # Line plot: update plotting options 
    if kind in ('line', 'scatter'):
        pltfunc = sns.relplot
        if hue is None:
            pltkwargs.update(dict(
                marker='o',
                color='b',
                markersize=8,
                lw=3,
                errorbar='se',
            ))
        else:
            if hue_width:
                pltkwargs['size'] = 'count'

    # Categorical plot: update plotting options
    elif kind in ['bar', 'box', 'boxen', 'violin']:
        if hue == Label.DATASET:
            raise ValueError('cannot plot distributions if split by dataset')
        pltfunc = sns.catplot
        pltkwargs['color'] = 'C0'
    
    # Unknow plot type: raise error
    else:
        raise ValueError(f'unknown plot type: "{kind}"')

    # No axis provided: use figure-level function   
    if ax is None:
        fg = pltfunc(
            kind=kind,
            hue=hue,
            height=4,
            **pltkwargs
        )
        # Extract figure and axis
        fig = fg.figure
        ax = fig.axes[0]
    
    # Axis provided: use-axis level function
    else:
        pltfunc = getattr(sns, f'{kind}plot')
        pltfunc(hue=hue, ax=ax, **pltkwargs)
        fig = ax.get_figure()
        
    # Post-process figure
    sns.despine(ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel('fraction of responders')
    ax.axhline(PTHR_DETECTION, c='k', ls='--')
    
    # Add average trace if specified and compatible
    if kind in ('line', 'scatter') and avg_overlay:

        # Compute weighted mean ans SE
        wmean = weighted_resp_props['positive']
        wsem = resp_props_sem['positive']
        
        if err_style == 'band':
            ax.plot(wmean.index, wmean.values, lw=avglw, c=avg_color)
            ax.fill_between(
                wmean.index, wmean - wsem, wmean + wsem,
                fc=avg_color, ec=None, alpha=0.3)
        elif err_style == 'bars':
            ax.errorbar(
                wmean.index, wmean.values, yerr=wsem.values, 
                lw=0 if fit else avglw, elinewidth=2 if fit else avglw,
                marker='o', markersize=5, color=avg_color)

        # If fit objects provided, compute and add to axis
        if fit:
            objfunc, initfunc = fit
            pltkwargs = dict(ls='--', lw=avglw, color=avg_color)
            compute_and_add_fit(
                ax, wmean.index.values, wmean.values, objfunc, initfunc=initfunc, ls='--', lw=avglw, color=avg_color)

        # if xkey == Label.ISPTA:
        #     pbyrun = get_params_by_run(data)
        #     for xk, ls in {Label.P: '--', Label.DC: '-'}.items():
        #         idx = get_xdep_data(pbyrun, xk)[Label.ISPTA]
        #         ax.plot(
        #             wmean.loc[idx].index, wmean.loc[idx].values, 
        #             lw=3, ls=ls, color=avg_color, label=f'{xk} sweep')
        #     ax.legend()
        # else:
        #     ax.plot(wmean.index, wmean.values, lw=avglw, color=avg_color)
    

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
    
    # Adjust x-scale
    adjust_xscale(ax, xscale=xscale)

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
    # Count fraction of conditions per response type for each ROI
    cond_fracs = get_rtype_fractions_per_ROI(data)

    # set plptting arguments
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


def plot_popagg_timecourse(data, ykeys, fps, hue=Label.RUN, normalize_gby=None, ax=None, 
                           legend='full', title=None, offset=True, col_wrap=2):
    '''
    Plot population-aggregated timecourse across categories
    
    :param data: population-aggregated timeseries data
    :param ykey: name of variable from which to compute spectrum profiles
    :param fps: sampling rate
    :param normalize_gby (optional): grouping variable used to normalize profiles
    :param ax (optional): plotting axis
    :return: figure handle
    '''
    if Label.DATASET in data.index.names and ax is not None:
        raise ValueError('cannot work on unique axis for muli-dataset input')

    # Work on copy to avoid modifying original data 
    data = data.copy()
    
    # Add time along run to population-average dataframe average
    data[Label.TIME] = get_index_along_experiment(
        data.index, reset_every=Label.RUN) / fps
    
    ykeys = as_iterable(ykeys)

    # Normalize profiles across categories, if any
    if normalize_gby is not None:
        logger.info(f'normalizing {ykeys} profiles across {" & ".join(as_iterable(normalize_gby))}...')
        data[ykeys] = (data[ykeys]
            .groupby(normalize_gby)
            .apply(lambda s: s / s.max())
        )

    # Copy data for plotting purposes
    plt_data = data.copy()

    # Offset spectra by hue, if any
    if offset and hue is not None:
        yoffsets = get_offsets_by(
            data, hue, y=ykeys[0], match_idx=True, ascending=False)
        yoffsets = yoffsets.iloc[:, 0]
        for k in ykeys:
            plt_data[k] += yoffsets
    
    # Create common plot arguments
    pltkwargs = dict(
        data=plt_data,
        x=Label.TIME,
        y=ykeys[0],
        hue=hue,
        errorbar=None,
        legend=legend
    )
    
    # If data is indexed by dataset, add arguments to create 1 axis per dataset
    if Label.DATASET in data.index.names:
        pltkwargs.update(dict(
            col=Label.DATASET,
            col_wrap=col_wrap,
        ))
    
    # If no axis provided, use figure-level plot function, extract figure & axes
    if ax is None:
        fg = sns.relplot(
            kind='line',
            height=4,
            aspect=2,
            **pltkwargs
        )
        fig = fg.figure
        fig.subplots_adjust(hspace=0.2)
        axes = fig.axes
    
    # If axis provided, use it, and assemble 1-axis list
    else:
        fg = None
        fig = ax.get_figure()
        sns.lineplot(ax=ax, **pltkwargs)
        axes = [ax]    
    # Despine and remove axes y ticks
    for ax in axes:
        sns.despine(ax=ax)
        ax.set_yticks([])
    # Add yscale bar
    dy = np.diff(ax.get_ylim())[0]
    yscale = round_to_base(dy * .2, precision=1, base=.1)
    xscale_norm = .97
    yscale_norm = yscale / dy
    ystart_norm = .1
    scale_bar = Line2D(
        [xscale_norm] * 2, [ystart_norm, ystart_norm + yscale_norm],
        transform=ax.transAxes, color='k',
    )
    ax.add_artist(scale_bar)
    ax.text(
        xscale_norm + .01, ystart_norm + yscale_norm / 2, yscale,
        transform=ax.transAxes, rotation='vertical', va='center')

    # Plot additional variables
    pltkwargs['legend'] = False
    for ykey, ls in zip(ykeys[1:], ['--', '-.', ':']):
        pltkwargs['y'] = ykey
        pltkwargs['ls'] = ls
        for k in ['col', 'col_wrap']:
            if k in pltkwargs:
                del pltkwargs[k]
        if fg is not None:
            fg.map_dataframe(sns.lineplot, **pltkwargs)
        else:
            sns.lineplot(ax=ax, **pltkwargs)

    # Add trial delimiters
    ntrials_per_run = len(data.index.unique(Label.TRIAL))
    trial_delimiters = np.arange(0, ntrials_per_run * NFRAMES_PER_TRIAL, NFRAMES_PER_TRIAL) / fps
    for ax in axes:
        for t in trial_delimiters:
            ax.axvline(t, ls='--', c='k', lw=1.)

    # Adapt axes titles, if needed
    if Label.DATASET in data.index.names:
        for ax in axes:
            ax.set_title(
                (
                    ax.get_title()
                    .lstrip(f'{Label.DATASET} = ')
                    .replace('_m', '\nm')
                    .replace('_', ' ')
                ),
                fontsize=10
            )
    
    # Add global title, if any
    pltgby = [hue]
    if Label.DATASET in data.index.names:
        pltgby = [Label.DATASET] + pltgby
        pltstr = f'{ykeys[0]} time course across {" & ".join(pltgby)}'
    if title is not None:
        title = f'{title} - {pltstr}'
        if len(axes) == 1:
            ax.set_title(title)
        else:
            fig.suptitle(title, y=1.05)

    # Return figure
    return fig


def plot_popagg_frequency_spectrum(data, ykeys, fps, normalize_gby=None, fmax=None, ax=None, 
                                   fband=None, fband_color='silver', legend='full', title=None,
                                   hue=Label.RUN, **kwargs):
    '''
    Plot frequency spectrum profiles across categories
    
    :param data: population-aggregated timeseries data
    :param ykey: name of variable(s) from which to compute spectrum profiles
    :param fps: sampling rate
    :param normalize_gby (optional): grouping variable used to normalize spectrum profiles
    :param fmax (optional): upper frequency limit above which the spectrum profiles are cut
    :param ax (optional): plotting axis
    :param fband (optional): frequency band to materialize with vertical span
    :return: figure handle
    '''
    # Compute frequency spectra for each continous recording (i.e. run, per dataset)
    if Label.DATASET in data.index.names:
        gby = [Label.DATASET, Label.RUN]
        if ax is not None:
            raise ValueError('cannot work on unique axis for muli-dataset input')
    else:
        gby = [Label.RUN]
    logger.info(f'computing frequency spectra across {" & ".join(gby)}...')
    popagg_spectrums = (data[ykeys]
        .groupby(gby)
        .apply(lambda s: get_power_spectrum(s, fps, add_db=False, **kwargs))
    )

    # Extract spectrum key(s)
    ykeys_spectrum = popagg_spectrums.columns.values.tolist()[1:]
    
    # If hue not in spectrum index, add it as column
    if hue is not None and hue not in popagg_spectrums.index.names:
        hue_by_gby = data[hue].groupby(gby).first()
        expand_and_add(hue_by_gby, popagg_spectrums)

    # Normalize spectra across categories, if any
    if normalize_gby is not None:
        logger.info(f'normalizing {ykeys} spectra across {" & ".join(as_iterable(normalize_gby))}...')
        popagg_spectrums[ykeys_spectrum] = (popagg_spectrums
            .groupby(normalize_gby)
            [ykeys_spectrum]
            .apply(lambda s: s / s.max())
        )

    # Restrict to low frequencies, if specified
    if fmax is not None:
        logger.info(f'restricting output to frequencies below {fmax:.2f} Hz')
        popagg_spectrums = popagg_spectrums[popagg_spectrums[Label.FREQ] < fmax]

    # Copy data into sperate dataframe for plotting
    plt_data = popagg_spectrums.copy()
    
    # Offset spectra by hue, if any
    if hue is not None:
        yoffsets = get_offsets_by(
            popagg_spectrums, hue, y=ykeys_spectrum[0], match_idx=True, ascending=False)        
        yoffsets = yoffsets.iloc[:, 0]
        for k in ykeys_spectrum:
            plt_data[k] += yoffsets

    # Create common plot arguments with first output variable
    pltkwargs = dict(
        data=plt_data,
        x=Label.FREQ,
        y=ykeys_spectrum[0],
        hue=hue,
        errorbar=None,
        legend=legend
    )
    # If data is indexed by dataset, add arguments to create 1 axis per dataset
    if Label.DATASET in data.index.names:
        pltkwargs.update(dict(
            col=Label.DATASET,
            col_wrap=5,
        ))
    # If no axis provided, use figure-level plot function, extract figure & axes
    if ax is None:
        fg = sns.relplot(
            kind='line',
            height=4,
            aspect=.5,
            **pltkwargs
        )
        fig = fg.figure
        fig.subplots_adjust(hspace=0.2)
        axes = fig.axes
    # If axis provided, use it, and assemble 1-axis list
    else:
        fg = None
        fig = ax.get_figure()
        sns.lineplot(ax=ax, **pltkwargs)
        axes = [ax]
    
    # Despine and remove axes y ticks
    for ax in axes:
        sns.despine(ax=ax)
        ax.set_yticks([])
    
    # Plot additional variables
    pltkwargs['legend'] = False
    for ykey, ls in zip(ykeys_spectrum[1:], ['--', '-.', ':']):
        pltkwargs['y'] = ykey
        pltkwargs['ls'] = ls
        if fg is not None:
            fg.map_dataframe(kind='line', **pltkwargs)
        else:
            sns.lineplot(ax=ax, **pltkwargs)

    # Mark trial repetition frequency on graphs
    ISI = (NFRAMES_PER_TRIAL - 1) / fps  # inter-sonication interval
    ftrial = 1 / ISI
    logger.info(f'adding reference lines at trial-repetition frequency ({ftrial:.2f} Hz)') 
    for ax in axes:
        ax.axvline(ftrial, c='k', ls='--', lw=1)
    
    # Materialize frequency band of interest, if specified
    if fband is not None:
        fband_str = ' - '.join([f'{x:.2f} Hz' for x in fband])
        logger.info(f'marking {fband_str} frequency band') 
        for ax in axes:
            ax.axvspan(*fband, fc=fband_color, alpha=0.3)

    # Adapt axes titles, if needed
    if Label.DATASET in data.index.names:
        for ax in axes:
            ax.set_title(
                (
                    ax.get_title()
                    .lstrip(f'{Label.DATASET} = ')
                    .replace('_m', '\nm')
                    .replace('_', ' ')
                ), 
                fontsize=10
            )
    
    # Add global title, if any
    pltgby = [hue]
    if Label.DATASET in data.index.names:
        pltgby = [Label.DATASET] + pltgby
        pltstr = f'{ykey} frequency spectrum across {" & ".join(pltgby)}'
    if title is not None:
        title = f'{title} - {pltstr}'
        if len(axes) == 1:
            ax.set_title(title)
        else:
            fig.suptitle(title, y=1.05)

    # Return figure
    return fig


def plot_spectrogram(data, ykey, fps, mode='psd', nsegpertrial=10, gby=None, trialavg=False, 
                     colwrap=4, fmax=None, cmap='viridis', add_cbar=True, fs=12):
    ''' 
    Plot spectrogram of specific experimental variable, with optional grouping. 
    
    :param data: experiment dataframe
    :param ykey: variable from which to extract spectrogram
    :param fps: frames per second
    :param mode (optional): mode of spectrogram computation (default: 'psd')
    :param nsegpertrial (optional): number of spetrogram segments per trial (default: 10)
    :param gby (optional): optional grouping variable
    :param trialavg (optional): whether to average spectrogram across trials (default: False)
    :param colwrap (optional): number of axes per row for multi-axes figure
    :param fmax (optional): maximum frequency to plot
    :param cmap (optional): colormap
    :param add_cbar (optional): whether to add colorbar
    :param fs (optional): font size
    :return: figure handle
    '''
    # Define number of frames per segment based on requested number of segments per trial
    nperseg_dict = {
        1: 100,
        2: 50,
        5: 22,
        10: 11,
        20: 5,
        50: 2,
    }
    try:
        nperseg = nperseg_dict[nsegpertrial]
    except KeyError:
        raise ValueError(f'nsegpertrial must be one of {list(nperseg_dict.keys())}')

    # Compute trial repetition frequency
    ISI = (NFRAMES_PER_TRIAL - 1) / fps  # inter-sonication interval

    # Define generic title
    title = f'{ykey} {mode} spectrogram'

    # If grouping variable provided,
    if gby is not None:
        # Create groups
        groups = data.groupby(gby)[ykey]
        
        # Create figure
        naxes = groups.ngroups
        nrows, ncols = int(np.ceil(naxes / colwrap)), min(colwrap, naxes)
        fig, axes = plt.subplots(nrows, ncols, figsize=(nrows * 3, ncols * 3))
        fig.suptitle(f'{title} across {gby}', y=.92, fontsize=fs)
        axes = np.ravel(axes)
    
    # If no grouping variable provided, create single axis figure
    else:
        fig, ax = plt.subplots()
        groups = [(None, data[ykey])]
        nrows, ncols, naxes = 1, 1, 1
        axes = [ax]
        ax.set_title(title, fontsize=fs)

    # For each axis-group pair
    for iax, ((idx, gdata), ax) in enumerate(zip(groups, axes)):
        # Compute spectrogram
        f, t, Sxx = spectrogram(gdata.values, fps, mode=mode, nperseg=nperseg)

        # Extract segment indexes
        isegs = (t * fps).astype(int)

        # If trial-averaging requested, average spectrogram across trials
        if trialavg:
            # Assert that trial interval is a multiple of inter-segment interval
            delta_iframes = isegs[1] - isegs[0]
            if NFRAMES_PER_TRIAL % delta_iframes != 0:
                raise ValueError(
                    f'''inter-segment interval ({delta_iframes} frames) is not a divider of
                    trial interval ({NFRAMES_PER_TRIAL} frames)''')

            # Generate dataframe from spectrogram output
            _, F = np.meshgrid(t, f, indexing='ij')
            dfout = pd.DataFrame({
                Label.FREQ: F.ravel(),
                mode: Sxx.T.ravel(),
            })

            # Assign index from input data
            mux = gdata.index[isegs]
            dfout.index = mux.repeat(f.size)
            dfout.set_index(Label.FREQ, append=True, inplace=True)

            # Average across trials
            df_trialavg = (dfout
                .groupby([Label.FRAME, Label.FREQ])
                [mode]
                .mean()
                .unstack()
            )

            t = df_trialavg.index.values / fps
            Sxx = df_trialavg.values.T

        # Plot spectrogram
        dt, df = t[1] - t[0], f[1] - f[0]
        tedges = np.linspace(t[0] - dt / 2, t[-1] + dt / 2, t.size + 1)
        fedges = np.linspace(f[0] - df / 2, f[-1] + df / 2, f.size + 1)
        sm = ax.pcolormesh(tedges, fedges, Sxx, cmap=cmap)

        # Add group title, if any
        if gby is not None:
            ax.set_title(f'{gby} {idx}', fontsize=fs)

        # Set axis limits and add reference frequency
        ax.set_ylim(0, fmax if fmax is not None else fedges[-1])
        ax.axhline(1 / ISI, c='w' if cmap == 'viridis' else 'k', ls='--', lw=1.)

        # Manage conditional axis labels
        icol, irow = iax % colwrap, iax // colwrap
        if irow == nrows - 1:
            ax.set_xlabel(Label.TIME, fontsize=fs)
        else:
            ax.set_xticks([])
        if icol == 0:
            ax.set_ylabel(Label.FREQ, fontsize=fs)
        else:
            ax.set_yticks([])
    
    # Hide unused axes
    for ax in axes[naxes:]:
        ax.set_visible(False)

    # Add colorbar, if requested
    if add_cbar:
        top = axes[0].get_position().y1
        bottom = axes[-1].get_position().y0
        fig.subplots_adjust(right=0.93)
        cbar_ax = fig.add_axes([0.95, bottom, 0.02, top - bottom])
        cbar_ax.set_title(mode, fontsize=fs)
        fig.colorbar(sm, cax=cbar_ax)
        cbar_ax.tick_params(labelsize=fs-2)

    # Return figure handle
    return fig


def annotate_facets(fg, data=None, test='t-test_ind', text_format='star', loc='outside', alpha=None, **pltkwargs):
    '''
    Apply statistical test between hue pairs on each facet of a facet grid, 
    and annotate statistical results associated axes
    
    :param fg: FacetGrid object
    :param data: dataframe used to generate FacetGrid
    :param test: test type
    :param text_format: text format on annotations
    :param loc: location of annotations
    :param pltkwargs: plot keyword arguments
    '''
    # Make sure "data" field is provided
    if data is None:
        raise ValueError('"data" input is required')
    naxes = len(fg.axes)

    # Set up axis dictionary and data groups  
    if naxes == 1:
        axdict = {'all': fg.ax}
        groups = [('all', data)]
        verbose = True
    else:
        col = pltkwargs.pop('col')
        axdict = fg.axes_dict
        groups = data.groupby(col)
        logger.info(f'annotating {pltkwargs["hue"]} pairs on {groups.ngroups} {col} groups')
        verbose = False

    # Set up progress bar
    with tqdm(total=naxes - 1, position=0, leave=True) as pbar:

        # For each axis/group
        for (axkey, ax), (gkey, gdata) in zip(axdict.items(), groups):

            # Check that axis and group match
            if axkey != gkey:
                raise ValueError(f'{col} group "{gkey}" does not match axis key ("{axkey}")')

            # Generate hue pairs
            pairs = get_hue_pairs(gdata, pltkwargs['x'], pltkwargs['hue'])

            # Perform statistical tests on these pairs, and annotate results
            annotator = Annotator(ax=ax, pairs=pairs, data=gdata, **pltkwargs)
            if not verbose:
                annotator.verbose = False
            config_kwargs = dict(
                test=test, 
                loc=loc, 
            )
            if alpha is None:
                config_kwargs['text_format'] = text_format
            else:
                config_kwargs['alpha'] = alpha
                config_kwargs['pvalue_format'] = dict(pvalue_thresholds=[
                    [1e-4, '****'],
                    [1e-3, '***'],
                    [1e-2, '**'],
                    [alpha, '*'],
                    [1, 'ns']
                ])
            annotator.configure(**config_kwargs)
            annotator.apply_and_annotate()

            # Update progress bar
            pbar.update()


def plot_occurence_counts(cond_seqs, runid=None, cond=None, ax=None):
    ''' Plot distribution of presented conditions for a given run ID '''
    if (runid is None and cond is None) or (runid is not None and cond is not None):
        raise ValueError('exactly one of "runid" or "cond" parameters must be provided')
    # Count conditions occurences for given run ID
    if runid is not None:
        xlabel = f'run ID = {runid}'
        occurence_dist = cond_seqs.loc[runid, :].value_counts()
    # Count run IDs occurences for given condition
    else:
        xlabel = f'cond = {cond}'
        occurence_dist = (cond_seqs == cond).sum(axis=1)
    # Plot
    if ax is None:
        figwidth = 4
        if occurence_dist.index.dtype == 'object':
            figwidth = 2 * len(occurence_dist)
        fig, ax = plt.subplots(figsize=(figwidth, 4))
    else:
        fig = ax.get_figure()
    sns.despine(ax=ax)
    ax.set_ylabel('number of occurences')
    ax.set_xlabel(xlabel)
    ax.bar(occurence_dist.index, occurence_dist.values)
    return fig


def get_uniform_palette(data, color='silver'):
    ''' Get a uniform palette across datasets '''
    dataset_ids = data.index.unique(Label.DATASET)
    return dict(zip(dataset_ids, [color] * len(dataset_ids)))


def get_binary_palette(data, outliers, default='g', member='r'):
    ''' Get a binary palette that highlights specific datasets '''
    dataset_ids = data.index.unique(Label.DATASET)
    palette = dict(zip(dataset_ids, [default] * len(dataset_ids)))
    for outlier in outliers:
        if outlier in palette:
            palette[outlier] = member
    return palette


def plot_all_deps(data, xkeys, ykeys, height=3, **kwargs):

    # Create figure backbone
    ncols, nrows = len(xkeys), len(ykeys) + 1
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(height * ncols, height * nrows))
    axes = np.atleast_2d(axes)

    # Response strength metrics
    for i, (ykey, axrow) in enumerate(zip(ykeys, axes[:-1])):
        # Determine output metrics key
        ykey_diff = get_change_key(ykey)
        # Plot dependencies on each parameter on separate axes
        logger.info(f'plotting {ykey} stimulation parameters dependencies...')
        plot_stimparams_dependency(
            data.copy(),
            ykey=ykey_diff, 
            hue=Label.DATASET,
            avgprop='whue', 
            axes=axrow,
            errorbar=None,
            xkeys=xkeys,
            legend=False,
            **kwargs
        )

    # Proportion of responders
    for i, (xkey, ax) in enumerate(zip(xkeys, axes[-1])):
        plot_responder_fraction(
            data, xkey,
            hue=Label.DATASET,
            ax=ax,
            legend='full' if i == len(xkeys) - 1 else False,
            xscale='sqrt' if xkey==Label.ISPTA else 'linear',
            **kwargs
        )
    
    # Process legend
    has_legend = ax.get_legend() is not None
    if has_legend:
        sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1), frameon=False)
        fig.set_size_inches(height * ncols, height * nrows)
        fig.tight_layout()
    
    for ax in axes.ravel():
        ax.set_box_aspect(1)

    for axrow in axes[1:]:
        for ax in axrow:
            ax.set_title(None)
    for axrow in axes[:-1]:
        for ax in axrow:
            ax.set_xlabel(None)
    if not has_legend:
        fig.tight_layout()
    
    return fig


def get_runid_palette(param_seqs, runid):
    ''' Get palette that seperates conditions for a particular run ID '''
    # Construct palette based on first presented parameters
    runid_params = param_seqs.loc[runid, :]
    mapper = dict(zip(
        runid_params.unique(), 
        [f'C{i}' for i in range(runid_params.nunique())]
    ))
    return runid_params.map(mapper).to_dict()


def plot_response_fit(data, ykey, fit_candidates, xkey=Label.ISPTA, ax=None, xscale=None, 
                      fs=12, title=None, height=4, innercall=False, fitsweepkey=None, 
                      error_aggfunc='mean', norm=False, plot_respmetrics=False,
                      rel_xmax=None, fit_store_dict=None):
    '''
    Compute fits between input parameter and response output metrics, and plot results
    
    :param data: experiment dataframe
    :param xkey: input variable
    :param ykey: output variable to fit to input
    :param fit_candidates: dictionary of (objective function, initial parameters generator function) pairs
    :return: figure handle, and best fit function applied with its optimal parameters
    '''
    if fitsweepkey is not None and len(fit_candidates) > 1:
        raise ValueError('cannot fit multiple functions with sweep key specified')

    # Generate title if not provided and possible
    if title is None and not innercall:
        for k in data.index.names:
            if len(data.index.unique(level=k)) == 1:
                title = data.index.unique(level=k)[0]
    
    # If multiple xkeys, generate multiple axes and call function recursively
    if is_iterable(xkey):
        fig, axes = plt.subplots(1, len(xkey), figsize=(height * len(xkey), height))
        axes = np.atleast_1d(axes)
        fitfuncs = {}
        for xk, ax in zip(xkey, axes):
            _, fitfuncs[xk] = plot_response_fit(
                data, ykey, fit_candidates, xkey=xk, ax=ax, xscale=xscale, 
                fs=fs, fitsweepkey=fitsweepkey, error_aggfunc=error_aggfunc,
                innercall=True, norm=norm, plot_respmetrics=plot_respmetrics,
                rel_xmax=rel_xmax)
        harmonize_axes_limits(axes)
        if title is not None:
            fig.suptitle(title, fontsize=fs + 3)
        return fig, fitfuncs

    # Restrict data range if P or DC is given as input
    if xkey == Label.DC:
        data = get_xdep_data(data, Label.DC, add_DC0=True)    
    elif xkey == Label.P:
        data = get_xdep_data(data, Label.P)

    # Compute dose metrics if not already present  
    if xkey not in data.columns:
        data[xkey] = get_dose_metric(data[Label.P], data[Label.DC], xkey)

    # Sort by increasing input value
    data = data.sort_values(xkey)

    # Get mean and sem keys
    mu_ykey = f'{ykey} - mean'
    se_ykey = f'{ykey} - sem'

    # If normalization by fit specified
    p0, popt, yfit, r2, xdense, yfitdense = None, None, None, None, None, None
    if isinstance(norm, str) and norm == 'fit':
        # Check that only one fit candidate is provided
        if len(fit_candidates) > 1:
            raise ValueError('cannot normalize by fit with multiple fit candidates')
        # Fit candidate function to data
        norm_objfunc, norm_p0func = next(iter((fit_candidates.items())))
        # Extract x and mean y data
        xdata, ydata = data[xkey], data[mu_ykey]
        # Generate initial parameters
        p0 = norm_p0func(xdata, ydata)        
        # Perform fit between input and output variables with candidate function
        logger.info(f'fitting {ykey} to {xkey} using {norm_objfunc.__name__} function: p0 = {p0}')
        popt, _ = curve_fit(norm_objfunc, xdata, ydata, p0, maxfev=10000)
        yfit = norm_objfunc(xdata, *popt)
        # Compute error between fit prediction and data
        r2 = rsquared(ydata, yfit)
        logger.debug(f'fitting results: popt = {popt}, R2 = {r2:.2f}')
        # Compute min and max values of fit predictor across determined input range
        xbounds = np.array([xdata.min(), xdata.max()])
        if rel_xmax is not None:
            xbounds[1] *= rel_xmax
        xdense = np.linspace(*xbounds, 1000)
        yfitdense = norm_objfunc(xdense, *popt)
        ymin, ymax = yfitdense.min(), yfitdense.max()

    # If standard normalization is specified
    elif norm:
        # Extract minimum and maximum values of ykey across input data range
        ymin, ymax = data[mu_ykey].min(), data[mu_ykey].max()
    
    # If no normalization is specified
    else:
        # Set min and max values to None
        ymin, ymax = None, None
    
    # Normalize mean and sem columns, if normalization range is specified
    if ymin is not None and ymax is not None:
        # ymin = 0
        logger.info(f'normalizing {ykey} to range [{ymin:.3g}, {ymax:.3g}]')
        yptp = ymax - ymin
        data[mu_ykey] = (data[mu_ykey] - ymin) / yptp
        data[se_ykey] /= yptp
        if yfit is not None:
            yfit = (yfit - ymin) / yptp
            yfitdense = (yfitdense - ymin) / yptp

    # Fetch axis, or create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(height, height))
    else:
        fig = ax.get_figure()
    
    # Set up axis layout 
    sns.despine(ax=ax)
    ax.set_xlabel(xkey, fontsize=fs)
    ax.set_ylabel(ykey, fontsize=fs)
    
    # Plot data points, with sweep color code
    line_color = Palette.LINE[data.index.unique(level=Label.LINE).values[0]]
    subxkeys = [Label.P, Label.DC]
    for subxkey in subxkeys:
        subdata = get_xdep_data(data, subxkey, add_DC0=xkey != Label.P)
        # if subxkey != fitsweepkey:
        ax.errorbar(
            subdata[xkey], subdata[mu_ykey], 
            # yerr=subdata[se_ykey] if subxkey != fitsweepkey else None, 
            fmt='o' if subxkey != fitsweepkey else 'x', 
            c=line_color, markersize=5, 
            # alpha=alpha
        )
    
    # Initialize best fit metrics
    best_r2 = 0

    # Extract reference y profile for fit 
    if fitsweepkey is None:
        fit_data = data.copy()
    else:
        fit_data = get_xdep_data(data, fitsweepkey, add_DC0=True)
    
    # Extract x and mean y data
    xdata, ydata = fit_data[xkey], fit_data[mu_ykey]

    # For each candidate objective function
    for ifit, (objfunc, p0_func) in enumerate(fit_candidates.items()):

        if len(fit_candidates) > 1 or yfitdense is None:

            # Call function to estimate initial parameters
            p0 = p0_func(xdata, ydata)
            
            # Perform fit between input and output variables with candidate function
            logger.debug(f'fitting {ykey} to {xkey} using {objfunc.__name__} function: p0 = {p0}')
            popt, _ = curve_fit(objfunc, xdata, ydata, p0, maxfev=10000)
            yfit = objfunc(xdata, *popt)

            # Compute error between fit prediction and data
            r2 = rsquared(ydata, yfit)
            logger.debug(f'fitting results: popt = {popt}, R2 = {r2:.2f}')

        # Replace best fit information, if better fit score 
        if r2 > best_r2:
            best_r2 = r2
            best_func = objfunc
            best_popt = popt

        # Determine fit color
        c = line_color if len(fit_candidates) == 1 else f'C{ifit}'

        # Plot dense fitted profile
        if len(fit_candidates) > 1 or yfitdense is None:
            xbounds = np.array([xdata.min(), xdata.max()])
            if rel_xmax is not None:
                xbounds[1] *= rel_xmax
            xdense = np.linspace(*xbounds, 1000)
            yfitdense = objfunc(xdense, *popt)
        ax.plot(
            xdense, yfitdense, ls='--', label=f'{objfunc.__name__}: R2 = {r2:.2f}', c=c)

        if len(fit_candidates) == 1 and fit_store_dict is not None:
            if xkey not in fit_store_dict:
                fit_store_dict[xkey] = xdense
            fit_store_dict[data.index[0][0]] = yfitdense

        # If response metrics requested
        if plot_respmetrics:
            # If fit function does not have a sigmoid shape, raise warning
            if objfunc.__name__ != 'sigmoid':
                logger.warning(f'response threshold and saturation extraction only enabled with sigmoid functions')
            # Else, extract and plot response metrics
            else:
                ymaxfit = popt[-1]
                rel_yresps = {'threshold': .1, 'saturation': .9}
                yresps = {k: yrel * ymaxfit for k, yrel in rel_yresps.items()}
                xresps = {k: np.interp(yresp, yfitdense, xdense) for k, yresp in yresps.items()}
                for k, xresp in xresps.items():
                    ax.axhline(yresps[k], ls='--', c=c)
                    ax.axvline(xresp, ls='--', c=c)
                    logger.info(f'{k} {xkey} = {xresp:.2f}')
                xratio = xresps['saturation'] / xresps['threshold']
                ax.text(.5, .1, f'ratio = {xratio:.2f}', transform=ax.transAxes, fontsize=fs)

                # Extract transition data (data points between threshold and saturation)
                transition_data = data[is_within(data[xkey], (xresps['threshold'], xresps['saturation']))]

                # Perform linear regression on transition data
                transition_data = transition_data.sort_values(xkey)
                xtrans, ytrans = transition_data[xkey], transition_data[mu_ykey]
                _, _, r_value, *_ = linregress(xtrans, ytrans)
                ax.text(.5, .02, f'linear fit: R2 = {r_value**2:.2f}', transform=ax.transAxes, fontsize=fs)

        # If fit was performed on particular sweep
        if fitsweepkey is not None:
            # Get data from other sweep
            otherkey = list(set(subxkeys) - set([fitsweepkey]))[0]
            other_data = get_xdep_data(data, otherkey, add_DC0=True)
            xother, yother, yothersem = other_data[xkey], other_data[mu_ykey], other_data[se_ykey]

            # Apply fit on input range from other sweep
            yotherfit = objfunc(xother, *popt)

            # Compute prediction accuracy
            # ζ = np.mean(np.abs(yotherfit - yother) / yothersem)
            # ζ = relative_error(yother, yotherfit)
            ζ = symmetric_accuracy(yother, yotherfit, aggfunc=error_aggfunc)

            # Plot divergence between fit and data points from other sweep
            iswithin = np.logical_and(xdense >= xother.min(), xdense <= xother.max())
            ax.fill(
                np.hstack((xdense[iswithin], xother[::-1])), 
                np.hstack((yfitdense[iswithin], yother[::-1])),
                fc='silver', label=f'ζ = {ζ:.2f}'
            )
        
        # Adjust x-scale if specified
        adjust_xscale(ax, xscale=xscale)

        # Adjust tick labels fontsize
        ax.tick_params(axis='both', labelsize=fs)

    # Add legend with fit metrics
    pstr = ', '.join([f'{p:.2f}' for p in best_popt])
    logger.info(f'best {ykey} fit = {best_func.__name__}({xkey}, {pstr}): R2 = {best_r2:.2f}')    
    ax.legend(frameon=False, fontsize=fs, loc='upper left')

    # Add title, if provided
    if title is not None:
        ax.set_title(title, fontsize=fs + 3)

    # Generate best fit function applied with its optimal parameters 
    def fitfunc(x):
        return best_func(x, *best_popt)

    # Return figure and best fit function
    return fig, fitfunc


def plot_filter_frequency_response(sos, as_gain=False, fs=None, fc=None):
    '''
    Plot filter frequency response

    :param sos: second-order sections representation of the IIR filter
    :param yunit: y-axis unit ("amp" or "gain")
    :param fs: sampling frequency (optional)
    :param fc: cutoff frequencies (optional)
    :return: figure handle
    '''
    w, h = sosfreqz(sos)
    f = w / (2 * np.pi)  # normalized frequencies (1 = fs)
    y = np.abs(h)
    if as_gain:
        y = 20 * np.log10(np.maximum(y, 1e-5))
        yunit = 'gain [dB]'
    else:
        yunit = 'amplitude' 
    if fs is not None:
        f *= fs
        prefix = ''
        suffix = '(Hz)'
    else:
        prefix = 'normalized '
        suffix = '(1 = Fs)'
    fig, ax = plt.subplots()
    sns.despine(ax=ax)
    ax.set_title('Frequency Response')
    ax.set_xlabel(f'{prefix}frequency {suffix}')
    ax.set_ylabel(yunit)
    ax.grid(True)
    ax.plot(f, y)
    if fc is not None:
        for fc in as_iterable(fc):
            ax.axvline(fc, c='k', ls='--')
    return fig


def plot_fluorescence_ratios(data, mouseline):
    '''
    Plot median Fneu/F ratio for each dataset

    :param data: population average fluorescence dataframe with
    ROI and neuropil fluorescence vectors of each dataset over time
    :param mouseline: mouse line
    :return: figure handle
    '''
    # Compute median Fneu/Froi ratio for each dataset
    ratios = (data[Label.F_NEU] / data[Label.F_ROI]).rename('Fneu/F')
    median_ratios = ratios.groupby(Label.DATASET).median()

    # Create figure
    fig, ax = plt.subplots(figsize=(1.5, 5))
    sns.despine(ax=ax)

    # Create boxplot of median F/Fneu ratios
    sns.boxplot(
        data=median_ratios.reset_index(),
        ax=ax,
        y='Fneu/F',
        showfliers=False,
        color=Palette.LINE[mouseline],
    )

    # Show individual data points
    sns.stripplot(
        data=median_ratios.reset_index(),
        ax=ax,
        y='Fneu/F',
        color='k',
    )

    # Add horizontal line at 1, and adjust y-axis limits
    ax.axhline(1, c='k', ls='--')
    ax.set_ylim(.0, 1.05)

    # Adjust figure layout
    ax.tick_params(axis='both', labelsize=15)
    ax.set_ylabel('median F/Fneu', fontsize=15)
    ax.set_xticklabels([mouseline], fontsize=15)

    # Return figure 
    return fig


def plot_enriched_parameter_dependency(df, xkey=Label.ISPTA, ykey=None, yref=0., ax=None, title=None,
                                       xscale='linear', run_anova=True, run_linreg=True, textfs=10):
    ''' 
    Plot dependency of specific output variable on specific run parameter, 
    with optional ANOVA testing and linear regression 

    :param s: pandas.Series containing output variable
    :param xkey: parameter to plot against
    :param yref: reference value for output variable
    :param ax (optional): axis handle
    :param title (optional): figure title
    :param run_anova: whether to run ANOVA test (default: True)
    :param run_linreg: whether to run linear regression (default: True)
    :return: figure handle
    '''
    # Convert input series to dataframe if needed, and reset index
    df = df.copy()
    if isinstance(df, pd.Series):
        if ykey is None:
            ykey = df.name
        df = df.to_frame()
    df = df.reset_index()

    if ykey is None:
        raise ValueError('ykey must be provided')

    # If no axis handle provided, create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.get_figure()
    
    # Plot output variable vs run parameters
    sns.despine(ax=ax)
    logger.info(f'plotting {ykey} vs {xkey}')
    for hue in [Label.DATASET, None]:
        sns.lineplot(
            ax=ax,
            data=df,
            x=xkey, 
            y=ykey,
            hue=hue,
            errorbar='se' if hue is None else None,
            color='k' if hue is None else None,
            lw=0,
            marker='o',
            markersize=8 if hue is None else 6,
            err_style='bars',
        )
    
    # Adjust x-scale
    adjust_xscale(ax, xscale=xscale)

    # Add reference line and move legend
    ax.axhline(yref, ls='--', c='k')
    sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1), frameon=False)

    # Add title, if provided
    if title is not None:
        ax.set_title(title)
    
    # Set initial text vertical position
    ytext = .95
    
    # If ANOVA requested
    if run_anova:
        # Assess dependency of output variable on parameter
        logger.info(f'assessing dependence of {ykey} on {xkey} with ANOVA...')
        anova_pval = anova1d(df, xkey, ykey)
        logger.info(f'pvalue = {anova_pval:.3g}')

        # Add significance level
        ax.text(
            .05, ytext, f'ANOVA: p = {anova_pval:.3g}', transform=ax.transAxes, 
            ha='left', va='center', fontsize=textfs)
        ytext -= .07

    # If linear regression requested
    if run_linreg:
        # Regress variable on param and assess wether trend it is significant
        logger.info(f'computing linear regression of {ykey} on {xkey}')
        regout = apply_linregress(df, xkey=xkey, ykey=ykey)
        logger.info(f'pvalue = {regout.pval:.3g}')

        # Add linear regression line
        sns.regplot(
            ax=ax,
            data=df,
            x=xkey,
            y=ykey,
            color='k',
            line_kws=dict(lw=3),
        )
        # Add significance level
        ax.text(
            .05, ytext, f'linreg: p = {regout.pval:.3g}', transform=ax.transAxes, 
            ha='left', va='center', fontsize=textfs)
        ytext -= .07

    # Return figure handle
    return fig