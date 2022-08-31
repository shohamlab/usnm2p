
import logging
import numpy as np
import glob
import time
from random import sample
import os
from argparse import ArgumentParser
from tifffile import TiffFile
import matplotlib.pyplot as plt
from multiprocessing import Pool
from dialog import open_folder_dialog
from logger import logger
from fileops import loadtif


def get_frameavg(fpath):
    '''
    Load TIF file and compute frame-average profile per channel
    
    :param fpath: full path to input TIF file
    :return: frame-average array
    '''
    # Load TIF stack
    logger.info(f'loading data from {os.path.basename(fpath)}')
    stack = loadtif(fpath)
    logger.info(f'loaded {stack.shape} stack')

    # Compute frame average for each channel
    frameavg = stack.mean(axis=(-2, -1))
    if frameavg.ndim == 1:
        frameavg = np.array([frameavg]).T
    logger.info(f'computed {frameavg.shape[0]} frame average vector on {frameavg.shape[1]} channels')

    # Return
    return frameavg


def plot_frameavg_profiles(tif_fpaths, save=False, mpi=False):
    '''
    Plot frame-average profiles of a given acquisition
    
    :param tif_fpaths: list of full paths to tif_files
    :param save: whether to save the output figure to disk
    :return: frame average profiles figure
    '''
    nfiles = len(tif_fpaths)
    logger.info(f'plotting frame-average profiles from {nfiles} files...')
    # Find common prefix
    out_fpath = os.path.commonprefix(tif_fpaths)
    acqname = os.path.basename(out_fpath)
    # Get frame-average profiles
    if mpi:
        with Pool() as pool:
            frameavgs = pool.map(get_frameavg, tif_fpaths)
    else:
        frameavgs = list(map(get_frameavg, tif_fpaths))
    # Transform frame averages list to array, and swap axes
    frameavgs = np.stack(frameavgs)
    frameavgs = np.swapaxes(frameavgs, 1, 2)
    frameavgs = np.swapaxes(frameavgs, 0, 1)
    nchannels, nacqs, nsamples = frameavgs.shape
    isamples = np.arange(nsamples)

    # Plot
    logger.info('plotting frame-average profiles...')
    fig, axes = plt.subplots(nchannels, figsize=(10, 3 * nchannels))
    if nchannels == 1:
        axes = [axes]
    axes[0].set_title(acqname)
    for i, ax in enumerate(axes):
        ax.set_xlabel('frames')
        ax.set_ylabel(f'channel {i + 1}')
    for ax, ychannel in zip(axes, frameavgs):
        for i, yacq in enumerate(ychannel):
            ax.plot(isamples, yacq, label=f'acq {i + 1}', alpha=0.5, lw=1)
        yavg = ychannel.mean(axis=0)
        ystd = ychannel.std(axis=0)
        ax.plot(isamples, yavg, label='avg', c='k')
        ax.fill_between(isamples, yavg - ystd, yavg + ystd, fc='k', alpha=0.3)
    if nfiles <= 10:
        for ax in axes:
            ax.legend(loc='center right')    
    if save:
        outfpath = f'{out_fpath}.png'
        logger.info(f'saving output figure as {outfpath}')
        fig.savefig(outfpath)
    
    return fig


if __name__ == '__main__':

    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument('--mpi', default=False, action='store_true', help='run with multiprocessing')
    parser.add_argument('-s', '--save', default=False, action='store_true', help='save output figure as PNG')
    parser.add_argument('-n', '--nfilesmax', type=int, default=-1, help='max number of files')
    args = parser.parse_args()
    logger.setLevel(logging.INFO)

    # Select root data directory
    datadir = open_folder_dialog()
    if datadir is None:
        logger.error('no input data directory chosen')
        quit()

    # Look for a batch of files in that directory
    logger.info(f'looking for TIF files in {datadir}...')
    tif_files = glob.glob(os.path.join(datadir, '*.tif'))
    nfiles = len(tif_files)
    
    # Make sure all file have the same (or very close) size
    sizes = list(set([os.path.getsize(f) for f in tif_files]))
    if len(sizes) > 1:
        atol = 10  # absolute tolerance (in bytes)
        diffs = [abs(x - sizes[0]) for x in sizes]
        assert all(x <= atol for x in diffs), f'differing stack sizes: {sizes}'
    
    # Find common root
    root = os.path.commonprefix(tif_files)
    logger.info(f'found {nfiles} files under the common root {root}*.tif')
    
    # Select random subset of files if number of files exceeds limit 
    if nfiles > args.nfilesmax:
        logger.info(f'restricting input to subset of {args.nfilesmax} files...')
        tif_files = sample(tif_files, args.nfilesmax)
    
    # Plot stacks frame-average profiles
    t0 = time.perf_counter()
    fig = plot_frameavg_profiles(tif_files, save=args.save, mpi=args.mpi)
    tcomp = time.perf_counter() - t0
    logger.info(f'completed in {tcomp:.2f}s')
    plt.show()