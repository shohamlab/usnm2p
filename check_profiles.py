# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-08-15 16:34:13
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-08-19 13:31:00

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from config import dataroot
from logger import logger
from multiprocessing import Pool
from fileops import get_output_equivalent, get_data_folders, get_sorted_filelist, get_stack_frameavg
from parsers import P_TIFFILE

logger.setLevel(logging.INFO)

''' Command line script to check Bergamo USNM2P data '''


def plot_frameavg_profiles(ytrials, nchannels=2):
    ''' Plot frame-average profiles '''
    fig, axes = plt.subplots(nchannels, 1, figsize=(6, 2 * nchannels))
    axes[-1].set_xlabel('frames')
    for i, (ax, ychannel) in enumerate(zip(axes, ytrials.T)):
        ax.set_ylabel(f'channel {i + 1}')
        nframes, ntrials = ychannel.shape
        iframes = np.arange(nframes)
        ym, ystd = ychannel.mean(axis=-1), ychannel.std(axis=-1)
        ax.plot(iframes, ym, c='k')
        if ntrials > 1: 
            ax.fill_between(iframes, ym - ystd, ym + ystd, fc='k', alpha=0.3)
    return fig


if __name__ == '__main__':

    # Create command line parser
    parser = ArgumentParser()

    # Add dataset arguments
    parser.add_argument('-l', '--mouseline', help='mouse line', default='cre_sst')
    parser.add_argument('-d', '--expdate', help='experiment date')
    parser.add_argument('-m', '--mouseid', help='mouse number')
    parser.add_argument('-r', '--region', help='brain region')
    parser.add_argument('--layer', help='Cortical layer')

    parser.add_argument(
        '--mpi', default=False, action='store_true', help='enable multiprocessing')

    # Parse command line arguments
    args = parser.parse_args()

    # Input directory for raw data
    datadir = os.path.join(dataroot, args.mouseline)
    figsdir = get_output_equivalent(dataroot, 'raw', 'figs')  # Directory for output figures

    # List subfolders containing TIF files
    filters = [x for x in [args.mouseid, args.expdate, args.region] if x is not None]
    tif_folders = get_data_folders(datadir, include_patterns=filters)

    # Loop through each subfolder containing TIF files
    for tif_folder in tif_folders:
        logger.info(f'processing files in "{tif_folder}"')
        # Get TIF stack files inside that folder
        fnames = get_sorted_filelist(tif_folder, pattern=P_TIFFILE)
        raw_fpaths = [os.path.join(tif_folder, fname) for fname in fnames]
        # Get frame-average profiles per channel for every stack file
        if args.mpi:
            with Pool() as pool:
                ytrials = pool.map(get_stack_frameavg, raw_fpaths)
        else:
            ytrials = list(map(get_stack_frameavg, raw_fpaths))
        fig = plot_frameavg_profiles(np.stack(ytrials))
        dmr = os.path.basename(tif_folder)
        fig.suptitle(dmr)
        save_fpath = os.path.join(figsdir, 'frameavg', f'{dmr}.png')
        logger.info(f'saving {dmr} frame-average profiles figure...')
        fig.savefig(save_fpath)
