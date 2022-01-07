# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-13 11:41:52
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-01-07 12:45:15

''' Collection of plotting utilities. '''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.patches import Rectangle
import seaborn as sns
from colorsys import hsv_to_rgb, rgb_to_hsv
from tqdm import tqdm

from logger import logger
from constants import *
from utils import get_singleton, is_iterable, plural
from postpro import *
from viewers import get_stack_viewer


def harmonize_axes_limits(axes, axkey='y'):
    ''' Harmonize axes limits'''
    if axes.ndim > 1:
        axes = axes.ravel()
    ymin = min([ax.get_ylim()[0] for ax in axes])
    ymax = max([ax.get_ylim()[1] for ax in axes])
    for ax in axes.ravel():
        ax.set_ylim(ymin, ymax)


def add_unit_diag(ax, c='k', ls='--'):
    ''' Add a diagonal line representing the Y = X relationship on the axis '''
    xlims, ylims = ax.get_xlim(), ax.get_ylim()
    lims = (min(xlims[0], ylims[0]), max(xlims[1], ylims[1]))
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.plot(lims, lims, c='k', ls='--')


def plot_table(inputs):
    '''Plot a table as a figure '''
    cellText = [[k, v] for k, v in inputs.items()]
    fig, ax = plt.subplots()
    ax.set_title('Dataset', fontsize=20)
    ax.axis('off')
    table = ax.table(cellText, loc='center')
    table.set_fontsize(14)
    table.scale(1, 3)
    fig.tight_layout()
    return fig
   

def plot_stack_histogram(stacks, title=None, yscale='log'):
    '''
    Plot summary histogram from TIF stacks.
    
    :param stacks: dictionary of TIF stacks
    :param cmap (optional): colormap
    :return: figure handle
    '''
    logger.info('plotting stack(s) histogram...')
    fig, ax = plt.subplots()
    if title is None:
        title = ''
    else:
        title = f'{title} - '
    ax.set_title(f'{title}summary histogram')
    sns.despine(ax=ax)
    for k, v in stacks.items():
        ax.hist(v.ravel(), bins=50, label=k, ec='k', alpha=0.5)
    ax.legend()
    ax.set_xlabel('pixel intensity')
    ax.set_yscale(yscale)
    ax.set_ylabel('Count')
    return fig


def plot_stack_frequency_spectrum(stacks, fs, title=None, yscale='log'):
    '''
    Plot frequency spectrum of TIF stacks.
    
    :param stacks: dictionary of TIF stacks
    :param cmap (optional): colormap
    :return: figure handle
    '''
    logger.info('computing stack(s) fft...')
    nframes = stacks[list(stacks.keys())[0]].shape[0]
    # Compute frequencies
    freqs = np.fft.rfftfreq(nframes, 1 / fs)
    # Compute FFTs along time axis
    ffts = {k: np.abs(np.fft.rfft(v, axis=0)) for k, v in stacks.items()}
    # Square FFts and average across pixels to get power spectrum for each frequency
    ps_avg = {k: np.array([(x**2).mean() for x in v]) for k, v in ffts.items()}

    logger.info('plotting stack(s) frequency spectrum...')
    fig, ax = plt.subplots()
    if title is None:
        title = ''
    else:
        title = f'{title} - '
    ax.set_title(f'{title}frequency spectrum')
    sns.despine(ax=ax)
    for k, v in ps_avg.items():
        ax.plot(freqs, v, label=k)
    ax.legend()
    ax.set_xlabel('frequency (Hz)')
    ax.set_yscale(yscale)
    ax.set_ylabel('power spectrum')
    return fig


def plot_stack_summary_frames(stack, cmap='viridis', title=None, um_per_px=None):
    '''
    Plot summary images from a TIF stack.
    
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
    for ax, (title, func) in zip(axes, plotfuncs.items()):
        ax.set_title(title)
        ax.imshow(func(stack, axis=0), cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
        if um_per_px is not None:
            npx = stack.shape[-1]
            add_scale_bar(ax, npx, um_per_px, color='w')
    return fig


def add_scale_bar(ax, npx, um_per_px, color='k'):
    '''
    Add a scale bar to a micrograph axis
    
    :param ax: axis object
    :param npx: number of pixels on each dimension of the axis image
    :param um_per_pixel: number of microns per pixel
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
    # Define and
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


def plot_suite2p_registration_images(output_ops, title=None, cmap='viridis', um_per_px=None, full_mode=False):
    ''' Plot summary registration images from suite2p processing output.

        :param output_ops: suite2p output
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
    imgs_dict = {label: output_ops.get(key, None) for label, key in imkeys_dict.items()}
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


def plot_suite2p_phase_corr_peak(output_ops):
    ''' 
    Plot peak of phase correlation with reference image over time
    from suite2p processing output.

    :param output_ops: suite2p output
    :return: figure handle    
    '''    
    if 'corrXY' not in output_ops:
        logger.warning('looks like the data was not registered -> ignoring')
        return None
    logger.info('plotting suite2p registration phase correlation peaks...')
    fig, ax = plt.subplots(figsize=(12, 3))
    sns.despine(ax=ax)
    ax.set_title('peak of phase correlation with ref. image over time')
    ax.set_xlabel('frames')
    ax.set_ylabel('phase correlation peak')
    ax.plot(output_ops['corrXY'], c='k', label='whole frame', zorder=5)
    if output_ops['nonrigid']:
        block_corrs = output_ops[f'corrXY1']
        for i, bc in enumerate(block_corrs.T):
            ax.plot(bc, label=f'block {i + 1}')
        ax.legend(bbox_to_anchor=(1, 0), loc='center left')
    return fig


def plot_suite2p_registration_offsets(output_ops, fbounds=None, title=None):
    ''' Plot registration offsets over time from suite2p processing output.

        :param output_ops: suite2p output options dictionary
        :return: figure handle    
    '''    
    if 'yoff' not in output_ops:
        logger.warning('looks like the data was not registered -> ignoring')
        return None
    logger.info('plotting suite2p registration offsets...')
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharey=True)
    if title is not None:
        fig.suptitle(title)
    for ax in axes[:-1]:
        ax.set_xticks([])
        sns.despine(ax=ax, bottom=True)
    axes[-1].set_xlabel('frames')
    sns.despine(ax=axes[-1], offset={'bottom': 10})
    if fbounds is None:
        fbounds = [0, output_ops['nframes'] - 1]
    for ax, key in zip(axes, ['y', 'x']):
        offsets = output_ops[f'{key}off'][fbounds[0]:fbounds[1] + 1]
        ax.plot(offsets, c='k', label='whole frame', zorder=5)
        if output_ops['nonrigid']:
            block_offsets = output_ops[f'{key}off1'][fbounds[0]:fbounds[1] + 1]
            for i, bo in enumerate(block_offsets.T):
                ax.plot(bo, label=f'block {i + 1}')
        ax.axhline(0, c='silver', ls='--')
        ax.set_ylabel(key)
    if output_ops['nonrigid']:
        axes[0].legend(bbox_to_anchor=(1, 0), loc='center left')
    return fig


def plot_suite2p_PCs(output_ops, nPCs=3, um_per_px=None):
    '''
    Plot average of top and bottom 500 frames for each PC across the movie
    
    :param output_ops: dictionary of outputed suite2p options
    :param um_per_pixel (optional): number of microns per pixel (for scale bar)
    :return: figure handle
    '''
    if 'regPC' not in output_ops:
        logger.warning('looks like the data was not registered -> ignoring')
        return None
    logger.info('plotting suite2p PCs average frames...')
    PCs = output_ops['regPC']
    if nPCs is not None:
        PCs = PCs[:, :nPCs]
    nPCs = PCs.shape[1]
    fig, axes = plt.subplots(nPCs, 3, figsize=(9, nPCs * 3))
    fig.suptitle(f'top {nPCs} PCs average images across movie')
    maxPCs, minPCs = PCs
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
    return fig


def plot_suite2p_PC_drifts(output_ops):
    '''
    Plot drifts of PCs w.r.t reference image
    
    :param output_ops: suite2p output options dictionary
    :return: figure handle
    '''
    if 'regDX' not in output_ops:
        logger.warning('looks like the data was not registered -> ignoring')
        return None
    PCdrifts = output_ops['regDX'].T
    PCdrifts_dict = {
        'rigid': PCdrifts[0],
        'nonrigid avg': PCdrifts[1],
        'nonrigid max': PCdrifts[2]
    }
    fig, ax = plt.subplots()
    sns.despine(ax=ax)
    ax.set_title('PC drifts w.r.t. reference image')
    ax.set_xlabel('# PC')
    ax.set_ylabel('absolute registration offset')
    for k, v in PCdrifts_dict.items():
        ax.plot(v, label=k)
    ylims = ax.get_ylim()
    ax.set_ylim(min(ylims[0], -0.1), max(ylims[1], 1.0))
    ax.legend(frameon=False)
    return fig
    

def plot_suite2p_sparse_maps(output_ops, um_per_px=None):
    ''' 
    Plot the maps of detected peaks generated at various downsampling factors of
    the sparse detection mode

    :param output_ops: dictionary of outputed suite2p options
    :param um_per_pixel (optional): number of microns per pixel (for scale bar)
    '''
    if not output_ops['sparse_mode']:
        logger.warning('looks like sparse mode was not turned on -> ignoring')
        return None
    
    logger.info('plotting suite2p sparse projection maps...')
    
    # Extract maps
    Vcorr, Vmaps = output_ops['Vcorr'], output_ops['Vmap']
    
    # Compute ratios
    refnpx = max(Vcorr.shape)
    npxs = np.array([max(x.shape) for x in Vmaps])  # get map dimensions
    ratios = npxs / refnpx  # get ratios to reference map
    ratios = np.power(2, np.round(np.log(ratios) / np.log(2)))  # round ratios to nearest power of 2
    
    # Find index of map with optimal scale for ROI detection 
    best_scale_px = output_ops['spatscale_pix'][0]
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
    for i, (ax, title, img) in enumerate(zip(axes.ravel(), titles, Vmaps)):
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


def get_image_and_cmap(output_ops, key, cmap, pad=True):
    ''' Extract a reference image from the suite2p options dictionary, pad it if needed '''
    # Get colormap
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad(color='silver')
    # Extract reference image
    refimg = output_ops[key]
    # If required, apply NaN padding to match original frame dimensions
    Lyc, Lxc = refimg.shape
    Ly, Lx = output_ops['Ly'], output_ops['Lx']
    if (Lxc < Lx) or (Lyc < Ly) and pad:
        dy, dx = (Ly - Lyc) // 2, (Lx - Lxc) // 2
        refimg = np.pad(refimg, ((dy, dy), (dx, dx)), constant_values=np.nan)
    # Return reference image and colormap as a tuple
    return refimg, cmap


def plot_suite2p_ROIs(data, output_ops, title=None, um_per_px=None, norm_mask=True,
                      superimpose=True, mode='contour', refkey='Vcorr', alpha_ROIs=1.,
                      cmap='viridis'):
    ''' Plot regions of interest identified by suite2p.

        :param data: data dictionary containing contents outputed by suite2p
        :param output_ops: dictionary of outputed suite2p options
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
    Ly, Lx = output_ops['Ly'], output_ops['Lx']

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
    refimg, cmap = get_image_and_cmap(output_ops, refkey, cmap, pad=superimpose)
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
    ''' Plot the histogram distribution of posterior probabilities of each ROIdistribution '''
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
    :return: figure handle
    '''
    data = pd.DataFrame({
        'npix': [x['npix'] for x in stats],
        'npix_soma': [x['npix_soma'] for x in stats]
    })
    data['npix_ratio'] = data['npix'] / data['npix_soma']
    fig, ax = plt.subplots()
    ax.set_title('Ratios of (# pixels ROI) / (# pixels soma)')
    sns.despine(ax=ax)
    hue = None
    if thr is not None:
        ax.axvline(thr, ls='--', c='silver')
        data['is_outlier'] = data['npix_ratio'] > thr
        hue = 'is_outlier'
    sns.histplot(data, x='npix_ratio', bins=30, ax=ax, hue=hue)
    return fig, data


def plot_all_ROIs(data, key=Label.F, delimiters=None):
    '''
    Plot all ROI traces for a particular variable

    :param data: fluorescence dataframe
    :return: figure handle
    '''
    logger.info(f'plotting {key} traces of all ROIs...')
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.despine(ax=ax)
    ax.set_xlabel('frames')
    ax.set_ylabel(key)
    nROIs = len(data.index.unique(level=Label.ROI))
    ax.set_title(f'{key} traces for all {nROIs} ROIs')
    # Plot traces of all ROIs
    for _, y in data[key].groupby(Label.ROI):
        ax.plot(y.values)
    # Plot delimiters, if any
    if delimiters is not None:
        logger.info(f'adding {len(delimiters)} delimiters')
        for iframe in delimiters:
            ax.axvline(iframe, color='k', linestyle='--')
    # Return figure
    return fig    


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
    iframes = filtered_data.index.unique(level=Label.FRAME).values
    ionset = iframes[0]
    # If trials are marked in data -> adjust onset according to 1st trial index
    if Label.TRIAL in filtered_data.index.names:
        itrials = filtered_data.index.unique(level=Label.TRIAL).values
        ionset += itrials[0] * len(iframes)

    # Create figure
    if not is_iterable(iROI):
        iROI = [iROI]
    else:
        iROI = sorted(iROI)  # sort ROIs to ensure consistent looping order 
    nROIs = len(iROI)
    npersignal /= nROIs
    fig, axes = plt.subplots(nROIs, 1, figsize=(12, nROIs * 4))
    if not is_iterable(axes):
        axes = [axes]
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
    fig, axes = plt.subplots(1, iROI.size, figsize=(iROI.size * 3.5, 4))
    fig.suptitle('Linear regressions F_ROI = A * F_NEU + B')
    for ax, ir in zip(axes, iROI):
        logger.info(f'plotting F_ROI = f(F_NEU) with linear regression for ROI {ir}...')
        subdata = data.loc[pd.IndexSlice[ir, :, :, :]]
        sns.despine(ax=ax)
        ax.set_title(f'ROI {ir}')
        ax.set_aspect(1.)
        sns.regplot(
            data=subdata, x=x, y=y, ax=ax, label='data',
            robust=True, ci=None)
        add_unit_diag(ax)
        bopt, aopt = linreg(subdata)
        logger.info(f'optimal fit: F_ROI = {aopt:.2f} * F_NEU + {bopt:.2f}')
        xvec = np.array([subdata[x].min(), subdata[x].max()])
        ax.plot(xvec, aopt * xvec + bopt, label='linear fit')
        ax.legend()
    fig.tight_layout()
    return fig


def mark_trials(ax, yconds, iROI, irun, color='C1'):
    ''' Mark trials on whole-run trace that meet specific stats condition. '''
    yconds = yconds.loc[pd.IndexSlice[iROI, irun, :]]
    for (_, _, itrial), ycond in yconds.iteritems():
        if ycond:
            istart = NFRAMES_PER_TRIAL * itrial + FrameIndex.STIM
            iend = istart + NFRAMES_PER_TRIAL
            ax.axvspan(istart, iend, fc=color, ec=None, alpha=.3)
        

def plot_cell_map(ROI_masks, Fstats, output_ops, title=None, um_per_px=None, refkey='Vcorr',
                  mode='contour', cmap='viridis', alpha_ROIs=0.7):
    ''' Plot spatial distribution of cells (per response type) on the recording plane.

        :param ROI_masks: ROI-indexed dataframe of (x, y) coordinates and weights
        :param Fstats: statistics dataframe
        :param output_ops: suite2p output options dictionary
        :param title (optional): figure title
        :return: figure handle
    '''
    logger.info('plotting cells map color-coded by response type...')

    # Fetch parameters from data
    Ly, Lx = output_ops['Ly'], output_ops['Lx']
    rtypes_per_ROI = get_response_types_per_ROI(Fstats)
    rtypes = get_default_rtypes()
    count_by_type = {k: (rtypes_per_ROI == k).sum() for k in rtypes}

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
        colors = sns.color_palette(Palette.RTYPE)
        for i, (c, mask) in enumerate(zip(colors, masks)):
            rgbs[i][mask == 1] = [*c, alpha_ROIs]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))
    if title is not None:
        ax.set_title(title)
    
    # Plot reference image 
    refimg, cmap = get_image_and_cmap(output_ops, refkey, cmap, pad=True)
    ax.imshow(refimg, cmap=cmap)
    
    # Plot cell and non-cell ROIs
    colors = sns.color_palette(Palette.RTYPE)
    for c, (rtype, idx) in zip(colors, idx_by_type.items()):
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
    labels = [f'{k} ({v})' for k, v in count_by_type.items()]
    if mode == 'contour':
        legfunc = lambda color: dict(c='none', marker='o', mfc='none', mec=color, mew=2)
    else:
        legfunc = lambda color: dict(c='none', marker='o', mfc=color, mec='none')
    leg_items = [
        Line2D([0], [0], label=label, ms=10, **legfunc(c))
        for c, label in zip(colors, labels)]
    ax.legend(handles=leg_items, bbox_to_anchor=(1, 1), loc='upper left', frameon=False)
    
    return fig


def plot_experiment_heatmap(data, key=Label.DFF, title=None, show_ylabel=True):
    '''
    Plot experiment heatmap (average response over time of each cell, culstered by similarity).
    
    :param data: experiment dataframe.
    :return: figure handle
    '''
    # Determine rows color labels from response types per cell
    rtypes = get_response_types_per_ROI(data).values
    row_cmap = dict(zip(np.unique(rtypes), sns.color_palette(Palette.RTYPE)))
    row_colors = [row_cmap[rtype] for rtype in rtypes]
    # Generate 2D table of average dF/F0 response per cell (using roi as index),
    # across runs and trials
    logger.info(f'generating (ROI x time) {key} pivot table...')
    avg_resp_per_cell = data.pivot_table(
        index=Label.ROI, columns=Label.TIME, values=key, aggfunc=np.mean)
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
    cg.ax_row_colors.set_ylabel(Label.ROI_RESP_TYPE)
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



def plot_from_data(data, xkey, ykey, xbounds=None, ybounds=None, aggfunc='mean', weightby=None, ci=CI,
                   err_style='band', ax=None, alltraces=False, hue=None, hue_order=None, col=None,
                   label=None, title=None, dy_title=0.6, markerfunc=None, max_colwrap=5, aspect=1.5, alpha=None,
                   **filter_kwargs):
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

    # Get number of ROIs in filtered data
    nROIs_filtered = len(filtered_data.index.unique(level=Label.ROI))

    ###################### Process log ######################
    s = []
    # Determine figure aspect based on col parameters

    # If col set to ROI and only ROI -> remove column assignment 
    if col == Label.ROI and nROIs_filtered == 1:
        col = None
    # If col set to ROI -> remove ROI filter info
    if col == Label.ROI and Label.ROI in filters:
        del filters[Label.ROI]

    if col is not None:
        s.append(f'grouping by {col}')
        col_wrap = min(len(filtered_data.groupby(col)), max_colwrap)
        height = 5.
        if ax is not None:
            raise ValueError(f'cannot sweep over {col} with only 1 axis')
    else:
        col_wrap = None
        height = 4.      
    if hue is not None:
        s.append(f'grouping by {hue}')
        if hue == Label.ROI and Label.ROI in filters:
            del filters[Label.ROI]
    if weightby is not None:
        if weightby not in data:
            raise ValueError(f'weighting variable ({weightby}) not found in data')
        if aggfunc != 'mean':
            raise ValueError(f'cannot use {weightby}-weighting with {aggfunc} aggregation')
        s.append(f'weighting by {weightby}')
        filters['weight'] = f'weighted by {weightby}'
    if aggfunc is not None:
        s.append('averaging')
    if ci is not None:
        s.append('estimating confidence intervals')
    s = f'{", ".join(s)} and ' if len(s) > 0 else ''
    logger.info(f'{s}plotting {aggfunc} {ykey} vs. {xkey} ...')
    # Determine color palette depending on hue parameter
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
        hue       = hue,           # hue grouping variable
        hue_order = hue_order,     # hue plotting order 
        estimator = aggfunc,       # aggregating function
        ci        = ci,            # confidence interval estimator
        err_style = err_style,     # error visualization style 
        lw        = 2.0,           # line width
        palette   = palette,       # color palette
        legend    = 'full'         # use all hue entries in the legend
    )
    if alpha is not None:
        plot_kwargs['alpha'] = alpha
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
            aspect   = aspect,   # aspect ratio of the figure
            col_wrap = col_wrap, # how many axes per row
            col      = col,      # column (i.e. axis) grouping variable
        ))
        fg = sns.relplot(**plot_kwargs)
        axlist = fg.axes.ravel()
        fig = fg.figure
    # Remove right and top spines
    sns.despine()

    ###################### Individual traces ######################
    
    # Aggregation keys = all index keys that are not "frame" 
    aggkeys = list(filter(lambda x: x is not None and x != Label.FRAME, filtered_data.index.names))
    if alltraces:
        logger.info(f'plotting individual {ykey} vs. {xkey} traces...')
        # Getting number of conditions to plot
        nconds = len(axlist) * len(axlist[0].get_lines())
        alpha_trace = 0.2  # opacity index of each trace
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
                    group_color = l.get_color()
                    # Generate pivot table for the value
                    table = gr.pivot_table(
                        index=xkey,  # index = xkey
                        columns=aggkeys,  # each column = 1 line to plot 
                        values=ykey)  # values
                    # Get response classification of each trace
                    is_resps = gr[Label.IS_RESP].groupby(aggkeys).first()                                        
                    # Plot a line for each entry in the pivot table
                    for i, (x, is_resp) in enumerate(zip(table, is_resps)):
                        if use_color_code:
                            color = {True: 'g', False: 'r'}[is_resp]
                        else:
                            color = group_color
                        ax.plot(table[x].index, table[x].values, c=color, alpha=alpha_trace, zorder=-10)
                        # Add individual trace markers if specified
                        if markerfunc is not None:
                            markerfunc(ax, table[x], color=color, alpha=alpha_trace)
                    pbar.update()

    ###################### Markers ######################

    # If indicator provided, check number of values per axis
    if label is not None:
        try:
            label_values = filtered_data[label]
        except KeyError:
            label_values = get_trial_averaged(filtered_data)[label]
        if col is not None:
            label_values_per_ax = label_values.groupby(col).unique().values
        else:
            label_values_per_ax = [label_values.unique()]
        if label == Label.ROI_RESP_TYPE:
            label_cmap = dict(zip([x[0] for x in label_values_per_ax], sns.color_palette(Palette.RTYPE)))
        else:
            label_cmap = None
        
    # For each axis
    for iax, ax in enumerate(axlist):
        # Adjust x-axis if specified
        if xbounds is not None:
            ax.set_xlim(*xbounds)
        # Adjust y-axis if specified
        if ybounds is not None:
            ylims = ax.get_ylim()
            ybounds_ax = [yb if yb is not None else yl for yb, yl in zip(ybounds, ylims)]
            ax.set_ylim(*ybounds_ax)
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
        height = fig.get_size_inches()[1]
        fig.subplots_adjust(top=1 - dy_title / height)
        fig.suptitle(title)

    # Return figure
    return fig


def marks_response_peak(ax, trace, tbounds=None, color='k', alpha=1.):
    ''' Function to mark peaks of a response trace '''
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
    ''' Plot trial responses of specific sub-datasets.
    
    :param data: experiment dataframe
    :param tbounds (optional): time limits for plot
    :param ykey (optional): key indicating the specific signals to plot on the y-axis
    :param mark_stim (optional): whether to add a stimulus mark on the plot
    :param mark_analysis_window (optional): whether to add mark indicating the response analysis interval on the plot
    :param mark_peaks: whether to mark the peaks of each identified response
    :param label (optional): add a label indicating a specific field value on the plot (when possible)
    :param title (optional): figure title (deduced if not provided)
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
            markerfunc = lambda *args, **kwargs: marks_response_peak(
                *args, tbounds=tresponse, **kwargs)

    # Add tbounds to filtering criteria
    kwargs['tbounds'] = tbounds

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


def add_numbers_on_legend_labels(leg, data, xkey, ykey, hue):
    ''' Add sample size of each hue category on the plot '''
    counts_by_hue = data.groupby([hue, xkey]).count().loc[:, ykey].unstack()
    std_by_hue = counts_by_hue.std(axis=1)
    counts_by_hue = counts_by_hue.mean(axis=1)
    for t in leg.texts:
        s = t.get_text()
        if s in counts_by_hue:
            c = counts_by_hue.loc[s]
            if std_by_hue.loc[s] == 0.:
                cs = f'{c:.0f}'
            else:
                cs = f'{c:.1f} +/- {std_by_hue.loc[s]:.1f}'
        else:
            cs = '0'
        t.set_text(f'{s} (n = {cs})')


def plot_parameter_dependency(data, xkey=Label.P, ykey=Label.SUCCESS_RATE, baseline=None,
                              add_leg_numbers=True, **kwargs):
    ''' Plot parameter dependency of responses for specific sub-datasets.
    
    :param data: trial-averaged experiment dataframe
    :param xkey (optional): key indicating the independent variable of the x-axis
    :param ykey (optional): key indicating the dependent variable of the y-axis
    :param kwargs: keyword parameters that are passed to the generic plot_from_data function
    :return: figure handle
    '''    
    # Restrict filtering criteria based on xkey
    if xkey == Label.P:
        kwargs['DC'] = DC_REF
        data = data[data[Label.DC] == DC_REF]
    elif xkey == Label.DC:
        kwargs['P'] = P_REF
        data = data[data[Label.P] == P_REF]
    else:
        raise ValueError(f'xkey must be one of ({Label.P}, {Label.DC}')
    
    # Plot
    fig = plot_from_data(data, xkey, ykey, **kwargs)

    # Add numbers on legend if needed
    hue = kwargs.get('hue', None)
    if hue is not None and add_leg_numbers:
        try:
            leg = kwargs.get('ax', fig.axes[0]).get_legend()
            add_numbers_on_legend_labels(leg, data, xkey, ykey, hue)
        except AttributeError as err:
            leg = fig.legend()
            add_numbers_on_legend_labels(leg, data, xkey, ykey, hue)

    # Add baseline if specified
    if baseline is not None:
        for ax in fig.axes:
            ax.axhline(baseline, c='k', ls='--')
    
    # Return figure
    return fig


def plot_stack_timecourse(*args, **kwargs):
    '''
    Plot the evolution of the average frame intensity over time, with shaded areas
    showing its standard deviation

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
    return fig


def get_adapted_bounds(x, nstd=None):
    '''
    Return plotting boundaries for a dataset, composed of:

    :param x: data distribution
    :param nstd: number of standard deviations away from the median allowed
    :return: (lowerbound, upperbound) tuple
    '''
    bounds = (x.min(), x.max())
    if nstd is not None:
        med, sigma = x.median(), x.std()
        bounds = (max(bounds[0], med - nstd * sigma), min(bounds[1], med + nstd * sigma))
    return bounds


def plot_stat_heatmap(data, key, expand=False, title=None, groupby=None, nstd=None, cluster=False, sort=None,
                      **kwargs):
    '''
    Plot ROI x run heatmap for some statistics
    
    :param data: multi-indexed (ROI x run x trial) statistics dataframe
    :param key: stat column key
    :param expand (optional): expand figure to clearly see each combination
    :param groupby: 
    :return figure handle
    '''
    # Determine colormap center based on stat range
    center = 0 if data[key].min() < 0 else None
    # Compute trial-averaged data for entire dataset and each groupby subgroup
    trialavg_data = groupby_and_all(
        data,
        lambda df: get_trial_averaged(df[key], full_output=True),
        groupby=groupby)

    if key in Label.RENAME_ON_AVERAGING.keys():
        key = Label.RENAME_ON_AVERAGING[key]
    # Determine figure size and title
    s = f'{key} per ROI & run'
    stitle = s
    naxes = 1
    if groupby is not None:
        stitle = f'{stitle} - by {groupby}'
        naxes += data[groupby].nunique()
    # Seperate relevant data from is_repeat
    is_repeat = trialavg_data['all'][1]
    trialavg_data = {k: v[0] for k, v in trialavg_data.items()}
    # Compute size of input data
    nROIs, nruns = [len(trialavg_data['all'].index.unique(level=k)) for k in [Label.ROI, Label.RUN]]
    # Determine whether stats is a repeated value or a real distribution
    if not is_repeat:
        s = f'trial-averaged {s}'
    # Determine figure size and create figure
    axsize = (nruns / 2, nROIs / 5) if expand else (6, 5)
    figsize = (axsize[0] * naxes, axsize[1])
    fig, axes = plt.subplots(1, naxes, figsize=figsize)
    if naxes == 1:
        axes = [axes]
    # Determine colormap range
    if key in [Label.SUCCESS_RATE, Label.IS_RESP]:
        bounds = {k: (0, 1) for k in trialavg_data.keys()}
    elif nstd is not None:
        bounds = {k: get_adapted_bounds(v, nstd=nstd) for k, v in trialavg_data.items()}
    else:
        bounds = None
    
    if sort is not None:
        sorted_iROIs = sort_ROIs(trialavg_data, sort)

    # Plot trial-averaged stat heatmap for each condition
    for ax, (k, v) in zip(axes, trialavg_data.items()):
        logger.info(f'plotting {s} - {k} trials...')
        if bounds is not None:
            kwargs.update({'vmin': bounds[k][0], 'vmax': bounds[k][1]})
        vtable = v.unstack()
        if cluster:
            vtable = clusterize_data(vtable)
        elif sort is not None:
            vtable = vtable.reindex(sorted_iROIs)
        sns.heatmap(vtable, center=center, ax=ax, **kwargs)
        if naxes > 1:
            ax.set_title(k)
    # Add title
    if title is not None:
        stitle = f'{stitle} ({title})'
    if naxes == 1:
        ax.set_title(stitle)
    else:
        fig.suptitle(stitle)
    # Return
    return fig


def plot_metrics_along_trial(data, wlen, fps, full_output=True):
    '''
    Plot a specific output metrics as a function of the sliding window position along the trial
    
    :param data: multi-inxexed (ROI, run, trial, frame, istart) series of output metrics
    :param wlen: window length (in frames)
    :param fps: frame rate (in frames per second)
    :return: figure handle (with optional derived baseline and stimulus-evoked values)
    '''
    # Compute mean and std for each start index
    y_along_trial = data.groupby(Label.ISTART).mean()
    yvar_along_trial = data.groupby(Label.ISTART).std()
    # Extract starting indexes
    istarts = y_along_trial.index
    # Interpolate value at stim frame index
    y_evoked = np.interp(FrameIndex.STIM, istarts, y_along_trial.values)
    # Extract baseline value as median
    y_baseline = y_along_trial.median()
    # Plot on figure
    fig, ax = plt.subplots()
    s = data.name
    if data.dtype == bool:
        s = f'{s} rate'
        ax.set_ylim(0, 1)
    ax.set_title(f'{s} along the trial interval')
    ax.set_xlabel(f'start time of the {wlen / fps:.1f} s long detection window (s)')
    ax.set_ylabel(s)
    sns.despine(ax=ax)
    t = (istarts - FrameIndex.STIM) / fps
    ax.plot(t, y_along_trial, label='average')
    ax.fill_between(t, y_along_trial - yvar_along_trial, y_along_trial + yvar_along_trial,
        alpha=0.2, label='+/-SD interval')
    ax.axhline(y_baseline, ls=':', c='k', label=f'baseline ({y_baseline:.2f})')
    ax.axhline(y_evoked, ls=':', c='k', label=f'stim-evoked ({y_evoked:.2f})')
    ax.axvline(0, ls='--', c='k', label='stimulus onset')
    ax.legend()
    if full_output:
        return fig, y_baseline, y_evoked
    else:
        return fig


def plot_stat_histogram(data, key, trialavg=False, title=None, groupby=None, nstd=None):
    '''
    Plot the histogram distribution of a stat
    
    :param data: multi-indexed (ROI x run x trial) statistics dataframe
    :return: figure handle
    '''
    # Compute trial-averaged data for entire dataset and each groupby subgroup
    data = groupby_and_all(
        data,
        lambda df: get_trial_averaged(df[key]) if trialavg else df[key],
        groupby=groupby)
    
    if trialavg and key in Label.RENAME_ON_AVERAGING.keys():
        key = Label.RENAME_ON_AVERAGING[key]

    # Restrict data distribution to given range around median, if specified
    if nstd is not None:
        bounds = {k: get_adapted_bounds(v, nstd=nstd) for k, v in data.items()}
        data = {k: v[(v > bounds[k][0]) & (v < bounds[k][1])] for k, v in data.items()}
    
    # Re-arrange dataset to enable hist plot
    for k in data.keys():
        data[k] = data[k].to_frame()
        if groupby is not None:
            data[k][groupby] = k
    data = pd.concat(data.values(), axis=0)
    for k in data.index.names:
        data.reset_index(level=k, inplace=True)

    # Create figure
    fig, ax = plt.subplots()
    sns.despine(ax=ax)
    parsed_title = key
    if groupby is not None:
        parsed_title = f'{parsed_title} - by {groupby}'
    if title is not None:
        parsed_title = f'{parsed_title} ({title})'
    ax.set_title(parsed_title)

    # Plot Kernel density estimation for each condition
    sns.histplot(ax=ax, data=data, x=key, hue=groupby, stat='density', element='poly')

    return fig


def plot_stat_per_ROI(data, key, title=None, groupby=None, sort=None, baseline=None):
    '''
    Plot the distribution of a stat per ROI over all experimental conditions
    
    :param data: multi-indexed (ROI x run x trial) statistics dataframe
    :return: figure handle
    '''
    # Compute trial-averaged data for entire dataset and each groupby subgroup
    trialavg_data = groupby_and_all(
        data,
        lambda df: get_trial_averaged(df[key], full_output=True),
        groupby=groupby)
    if key in Label.RENAME_ON_AVERAGING.keys():
        key = Label.RENAME_ON_AVERAGING[key]
    # Seperate relevant data from is_repeat
    is_repeat = trialavg_data['all'][1]
    trialavg_data = {k: v[0] for k, v in trialavg_data.items()}

    # If ROI sorting pattern is specified
    if sort is not None:
        sorted_iROIs = sort_ROIs(trialavg_data, sort)

    # Determine whether stats is a repeated value or a real distribution
    s = key
    s2 = s
    if not is_repeat:
        s = f'trial-averaged {s}'
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlabel('# ROI')
    ax.set_ylabel(s2)
    parsed_title = f'{s2} per ROI'
    if groupby is not None:
        parsed_title = f'{parsed_title} - by {groupby}'
    if title is not None:
        parsed_title = f'{parsed_title} ({title})'
    ax.set_title(parsed_title)
    if s2 == Label.SUCCESS_RATE:
        ax.set_ylim(0, 1)
    sns.despine(ax=ax)

    # Plot mean trace with +/-std shaded area for each condition
    for k, v in trialavg_data.items():
        logger.info(f'plotting {s} - {k} trials...')
        # Group by ROI, get mean and std
        groups = v.groupby(Label.ROI) 
        mu, sigma = groups.mean(), groups.std()
        # Re-order metrics by spsecific ROI sorting pattern if specified
        if sort is not None:
            mu, sigma = mu[sorted_iROIs], sigma[sorted_iROIs]
        # Plot metrics
        x = np.arange(mu.size)
        ax.plot(x, mu, label=k)
        ax.fill_between(x, mu - sigma, mu + sigma, alpha=0.2)
    
    # Add baseline if specified
    if baseline is not None:
        for ax in fig.axes:
            ax.axhline(baseline, c='k', ls='--')
    
    # Legend
    if groupby is not None:
        ax.legend()

    return fig


def plot_stat_per_run(data, key, title=None, groupby=None, baseline=None):
    '''
    Plot the distribution of a stat per run over all ROIs
    
    :param data: multi-indexed (ROI x run x trial) statistics dataframe
    :return: figure handle
    '''    
    # Compute trial-averaged data for entire dataset and each groupby subgroup
    trialavg_data = groupby_and_all(
        data,
        lambda df: get_trial_averaged(df[key], full_output=True),
        groupby=groupby)
    if key in Label.RENAME_ON_AVERAGING.keys():
        key = Label.RENAME_ON_AVERAGING[key]
    # Seperate relevant data from is_repeat
    is_repeat = trialavg_data['all'][1]
    trialavg_data = {k: v[0] for k, v in trialavg_data.items()}
    # Determine whether stats is a repeated value or a real distribution
    s = key
    s2 = s
    if not is_repeat:
        s = f'trial-averaged {s}'    
    # Create figure
    fig, ax = plt.subplots(figsize=(4 * len(trialavg_data), 4))
    parsed_title = f'{s2} per run'
    if title is not None:
        parsed_title = f'{parsed_title} ({title})'
    ax.set_title(parsed_title)

    # Re-arrange dataset to enable bar plot
    for k in trialavg_data.keys():
        trialavg_data[k] = trialavg_data[k].to_frame()
        if groupby is not None:
            trialavg_data[k][groupby] = k
    trialavg_data = pd.concat(trialavg_data.values(), axis=0)
    trialavg_data.reset_index(level=Label.RUN, inplace=True)
    
    # Plot bar plot with std error bars for each condition
    sns.barplot(ax=ax, data=trialavg_data, x=Label.RUN, y=key, hue=groupby, ci='sd')
    ax.set_xlabel('# run')
    ax.set_ylabel(s2)
    
    if s2 == Label.SUCCESS_RATE:
        ax.set_ylim(0, 1)
    sns.despine(ax=ax)

    # Add baseline if specified
    if baseline is not None:
        for ax in fig.axes:
            ax.axhline(baseline, c='k', ls='--')

    return fig


def plot_pct_ROIs_map(data, key=Label.IS_RESP, label='stimulus-evoked activity'):
    '''
    Plot the percentage of ROIs satisfying a given condition for each run & trial
    of the experiment
    
    :param data: multi-indexed (ROI x run x trial) statistics dataframe
    :param key: name of the column containing the statistics of interest
    :param label: descriptive label corresponding to the column of interest
    :return: figure handle
    '''
    # Fetch total number of ROIs
    nROIs = len(data.index.unique(level=Label.ROI))
    # Compute number of ROIs satisfying condition for each run & trial
    nROIs_per_run_trial = data.loc[:, key].groupby([Label.RUN, Label.TRIAL]).sum()
    # Set totals from invalid trials to NaN
    isvalid = data[Label.VALID].groupby([Label.RUN, Label.TRIAL]).first()
    iinvalids = isvalid[~isvalid].index
    nROIs_per_run_trial.loc[iinvalids] = np.nan
    # Transform into percentage
    pctROIs_per_run_trial = nROIs_per_run_trial / nROIs * 1e2
    # Create figure
    fig, ax = plt.subplots()
    # Plot % ROIs heatmap
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('silver')
    sns.heatmap(pctROIs_per_run_trial.unstack(), ax=ax, cmap=cmap, vmin=0, vmax=100)
    # Add title with label 
    ax.set_title(f'% ROIs with {label} per run & trial')
    # Return figure handle
    return fig


def plot_positive_runs_hist(n_positive_runs, resp_types, nruns, title=None):
    ''' Plot the histogram of the number of positive conditions for each ROI,
        per response type.
    '''
    # Plot histogram per response type
    fig, ax = plt.subplots()
    ax.set_xlabel(Label.NPOS_RUNS)
    ax.set_ylabel('Count')
    bins = np.arange(nruns + 2) - 0.5
    df = pd.DataFrame([resp_types, n_positive_runs]).T
    colors = sns.color_palette(Palette.RTYPE)
    for c, (label, group) in zip(colors, df.groupby(Label.ROI_RESP_TYPE)):
        ax.hist(
            group[Label.NPOS_RUNS], bins=bins, label=f'{label} (n = {len(group)})',
            fc=c, ec='k')
    ax.set_xlim(bins[0], bins[-1])
    sns.despine(ax=ax)
    # Add legend
    ax.legend(title=Label.ROI_RESP_TYPE)
    # Add separator line
    # ax.axvline(NPOS_CONDS_THR - .5, ls='--', c='k')
    # Add title
    s = 'classification by # positive conditions'
    if title is not None:
        s = f'{s} ({title})'
    ax.set_title(s)
    return fig


def plot_gaussian_histogram_fit(data, fitparams, iROI, irun, ykey=Label.DFF, nbins=100):
    ''' 
    Plot histogram distribution and fitted gaussian fit for a particular dataset,
    for a subset of ROIs of interest.
    
    :param data: timeseries data
    :param fitparams: fitted gaussian parameters data
    :param iROI: indexes of ROIs of interest
    :param irun: index of run of interest
    :param ykey (optional): column of interest in timeseries data
    :param nbins (optional): number of bins in histogram distributions
    :return: figure handle
    '''
    # Create figure
    fig, axes = plt.subplots(iROI.size, 1, figsize=(5, 3 * iROI.size), sharex=True)
    axes[-1].set_xlabel(ykey)
    # For each ROI of interest
    for ax, ir in zip(axes, iROI):
        sns.despine(ax=ax)
        # Plot histogram distribution
        ax.set_title(f'ROI {ir}')
        ax.set_ylabel('Count')
        _, xedges, _ = ax.hist(
            data.loc[pd.IndexSlice[ir, irun, :, :], ykey],
            bins=nbins, alpha=0.5)
        # Plot fitted gaussian
        xmids = (xedges[1:] + xedges[:-1]) / 2
        params = fitparams.loc[pd.IndexSlice[ir, irun]]
        ax.plot(xmids, gauss(xmids, *params.values), c='k')
        # Plot markers for mean and width of gaussian fit
        ax.vlines(
            params['x0'], 
            ymin=params['H'], 
            ymax=params['H'] + params['A'], 
            ls='--', colors='k')
        ax.hlines(
            params['H'] + params['A'] / 2,
            xmin=params['x0'] - params['sigma'],
            xmax=params['x0'] + params['sigma'],
            ls='--', colors='k')
        # Add text labels for mean and width of gaussian fit
        ax.text(0.5, 0.6, f'\u03BC({ykey}) = {params["x0"]:.3f}', transform=ax.transAxes)
        ax.text(0.5, 0.5, f'\u03C3({ykey}) = {params["sigma"]:.3f}', transform=ax.transAxes)
    # Return figure
    return fig


def plot_params_correlations(data, ykey=Label.SUCCESS_RATE, pthr=None, directional=True):
    '''
    Plot the distribution of correlation coefficients of a specific response metrics
    with input stimulation parameters (pressure & duty cycle) for each ROI.

    :param data: trial-averaged statistics dataframe per ROI & run
    :param ykey: name of the column containing the metrics of interest
    :param pthr (optional): significance threshold probability used for ROI classification
    :param directional (default: True): whether to assume a directional effect (i.e. 1-tailed test) or not (i.e. 2-tailed test)
    :return: figure handle
    '''
    xkeys = [Label.P, Label.DC]
    
    # Compute correlation coeficients with stimulation parameters
    corr_coeffs = pd.concat([
        compute_correlation_coeffs(data, xkey, ykey) for xkey in xkeys], axis=1)

    # If significance threshold probability is provided
    if pthr is not None:
        # Compute threshold correlation coefficients for statistical significance in both dimensions
        # using appropriate t-value depending on effect directionality constraint  
        rthrs = {}
        srthrs = []
        for xkey in xkeys:
            n = data[xkey].nunique()
            rthrs[xkey] = tscore_to_corrcoeff(pvalue_to_tscore(pthr, n, directional=directional), n)
            srthrs.append(f'r({xkey}, p = {pthr:.3f}, n = {n}, {"1" if directional else "2"}-tailed) = {rthrs[xkey]:.2f}')
        srthrs = '\n'.join([f'-   {x}' for x in srthrs])
        logger.info(f'setting thresholds correlation coefficients for significant dependency along each dimension:\n{srthrs}')

        # Identify significantly correlated samples across both dimensions
        corrtypes = pd.DataFrame()
        for (xkey, rthr), col in zip(rthrs.items(), corr_coeffs):
            # By default, consider only positive correlation
            corrtypes[xkey] = (corr_coeffs[col] > rthr).astype(int)
            # If specified, consider also negative correlation
            if not directional:
                corrtypes[xkey] -= (corr_coeffs[col] < -rthr).astype(int)
        # Convert both informations into string code for graphical representation
        corr_coeffs[Label.ROI_RESP_TYPE] = correlations_to_rcode(corrtypes, j=', ')
        hue = Label.ROI_RESP_TYPE
        palette = Palette.RTYPE
        legend = 'full'
    else:
        # Otherwise, use mean value of the metrics as a color code
        corr_coeffs['avg'] = data[ykey].groupby(Label.ROI).mean()
        hue = 'avg'
        palette = Palette.DEFAULT
        legend = None

    # Plot joint distributions of correlation coefficients with P & DC for each ROI
    # using correlation type as color code
    jg = sns.jointplot(
        data=corr_coeffs, **dict(zip(['x', 'y'], corr_coeffs.columns.values)),
        xlim=[-1, 1], ylim=[-1, 1], hue=hue, hue_order=get_default_rtypes(), palette=palette, legend=legend)
    
    # If significance-based classification was performed
    if pthr is not None:
        # Add sample size for each category in legend
        counts = corr_coeffs[Label.ROI_RESP_TYPE].value_counts()
        labels = {t: f'{t} ({n})' for t, n in zip(counts.index, counts.values)}
        leg = jg.ax_joint.get_legend()
        for t in leg.texts:
            txt = t.get_text()
            t.set_text(labels.get(txt, f'{txt} (0)'))

        # Plot statistical significance threshold lines
        for ax_marg, xkey, k in zip([jg.ax_marg_x, jg.ax_marg_y], xkeys, ['v', 'h']):
            rthr = rthrs[xkey]
            for ax in [ax_marg, jg.ax_joint]:
                linefunc = getattr(ax, f'ax{k}line')
                linefunc(rthr, c='k', ls='--')
                if not directional:
                    linefunc(-rthr, c='k', ls='--')
        
    # Add title
    # jg.fig.suptitle(f'{ykey} - correlation with stimulus parameters across ROIs')
    jg.fig.tight_layout()
    jg.fig.subplots_adjust(top=0.92)

    # Return figure
    if pthr is not None:
        return jg.fig, corr_coeffs[Label.ROI_RESP_TYPE]
    else:
        return jg.fig


def plot_cellcounts_by_type(data, hue=Label.ROI_RESP_TYPE, add_count_labels=True):
    '''
    Plot a summary chart of the number of cells per response type and dataset
    
    :param data: multi-indexed stats dataframe with mouse-region as an extra index dimension
    :return: figure handle
    '''
    # Restrict dataset to 1 element per ROI for each dataset
    celltypes = data.groupby([Label.MOUSEREG, Label.ROI]).first()
    # Figure out bar variable and plot orientation
    groups = [Label.MOUSEREG, Label.ROI_RESP_TYPE]
    bar = list(set(groups) - set([hue]))[0]
    axdim = {Label.ROI_RESP_TYPE: 'x', Label.MOUSEREG: 'y'}[bar]
    hue_order = {Label.ROI_RESP_TYPE: None, Label.MOUSEREG: get_default_rtypes()}[bar]
    # Plot stacked count bars
    fg = sns.displot(
        data=celltypes, multiple='stack', hue=hue, hue_order=hue_order, **{axdim: bar})
    sns.despine()
    fig = fg.figure
    if add_count_labels:
        # Count number of cells of each bar and hue
        cellcounts = celltypes.groupby([Label.ROI_RESP_TYPE, Label.MOUSEREG]).count().iloc[:, 0].rename('counts')
        nperhue = cellcounts.groupby(hue).sum().astype(int)
        nperbar = cellcounts.groupby(bar).sum().astype(int)
        # Get number of responding cells
        ntot = nperhue.sum()
        neutral_type = get_default_rtypes()[0]
        if neutral_type in nperhue:
            nneutral = nperhue.loc[neutral_type]
        else:
            nneutral = nperbar.loc[neutral_type]
        nresp = ntot - nneutral
        ax = fig.axes[0]
        ax.set_title(f'{nresp} / {ntot} ({nresp / ntot * 1e2:.0f}%) responsive cells')
        # If resp type is hue, add labels to legend
        if hue == Label.ROI_RESP_TYPE:
            leg = fg._legend
            for t in leg.texts:
                s = t.get_text()
                n = nperhue.loc[s]
                t.set_text(f'{s} (n={n}, {n / ntot * 100:.0f}%)')
            leg.set_bbox_to_anchor([1.2, 0.5])
        # If resp type is bar, add labels on top of bars
        else:
            labels = [l.get_text() for l in ax.get_xticklabels()]
            offset = nperbar.max() * 0.02
            for i, label in enumerate(labels):
                n = nperbar.loc[label]
                ax.text(i, n + offset, f'{n} ({n / ntot * 100:.0f}%)', ha='center')
            
    return fig


def plot_parameter_dependency_across_datasets(data, xkey, ykey, qbounds, ybounds=(0, None)):
    '''
    Plot the parameter dependency of a specific variable across mouse-region datasets
    
    :param data: multi-indexed stats dataframe with mouse-region as an extra index dimension
    :param xkey: name of the stimulation parameter of interest
    :param ykey: name of the output variable of interest
    :param qbounds: specific quantile interval to restrict the dataset
    :return: figure handle
    '''
    # Restrict data to included trials
    data = included(data)
    norig = len(data)
    # Select quantile interval for each category (mouse-region, response type and run)
    categories = [Label.MOUSEREG, Label.ROI_RESP_TYPE, Label.RUN]
    xkeys = [Label.P, Label.DC]
    subset_idx = get_quantile_indexes(data, qbounds, ykey, groupby=categories)
    data = data.loc[subset_idx, :]
    nkept = len(data)
    logger.info(f'selected {nkept} / {norig} samples ({nkept / norig * 100:.1f}% of dataset)')
    # Average data by mouse-region, response type and run 
    avg_data = data.groupby(categories).mean()
    # Make sure input columns are correclty resolved post-averaging
    # (avoid "almost-identical" duplicates) 
    avg_data = resolve_columns(avg_data, xkeys)
    # Plot parameter dependency of ykey, grouped by mouse-region and response type
    fig = plot_parameter_dependency(
        data, xkey=xkey, ykey=ykey, ybounds=ybounds,
        hue=Label.MOUSEREG, col=Label.ROI_RESP_TYPE,
        add_leg_numbers=False, max_colwrap=2, aspect=1., ci=None, alpha=0.5,
        title=f'q = {qbounds}')
    # Add average parameter dependency trace across mouse-regions, for each response type
    xdep_avg_data = get_xdep_data(avg_data, xkey)
    for ax, (_, group) in zip(fig.axes, xdep_avg_data.groupby(Label.ROI_RESP_TYPE)):
        sns.lineplot(data=group, x=xkey, y=ykey, ax=ax, color='BLACK', lw=3, legend=False)
    return fig


def plot_protocol(table, xkey=Label.RUNID):
    '''
    Plot the evolution of stimulus parameters over time
    
    :param info_table: summary table of the parameters pertaining to each run
    :param xkey: reference variable for time evolution (run or runID)
    :return: figure handle
    '''
    try:
        x = table[xkey]
    except KeyError:
        x = table.index.get_level_values(level=xkey)
    ykeys = [Label.P, Label.DC]
    fig, axes = plt.subplots(len(ykeys), 1)
    axes[0].set_title('evolution of stimulation parameters over runs')
    for ax, ykey in zip(axes, ykeys):
        ax.scatter(x, table[ykey])
        ax.set_ylabel(ykey)
    for ax in axes[:-1]:
        sns.despine(ax=ax, bottom=True)
        ax.set_xticks([])
    axes[-1].set_xlabel(xkey)
    sns.despine(ax=axes[-1])
    return fig