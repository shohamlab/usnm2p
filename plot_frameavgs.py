# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-08-15 16:34:13
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-08-19 16:28:46

''' Command line script to plot USNM2P frame-average data '''

import logging
import numpy as np
from random import sample
import os
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from multiprocessing import Pool

from dialog import open_folder_dialog
from logger import logger
from fileops import get_output_equivalent, get_data_folders, get_sorted_filelist, get_stack_frame_aggregate
from plotters import plot_frameavg_profiles
from config import dataroot
from parsers import P_TIFFILE, group_by_run

logger.setLevel(logging.INFO)


if __name__ == '__main__':

    # Create command line parser
    parser = ArgumentParser()
    
    # Add dataset arguments
    parser.add_argument('-l', '--mouseline', help='mouse line', default='cre_sst')
    parser.add_argument('-d', '--expdate', help='experiment date')
    parser.add_argument('-m', '--mouseid', help='mouse number')
    parser.add_argument('-r', '--region', help='brain region')
    parser.add_argument('--layer', help='Cortical layer')
    
    # Add runtime arguments
    parser.add_argument(
        '--mpi', default=False, action='store_true', help='enable multiprocessing')
    parser.add_argument(
        '-s', '--save', default=False, action='store_true', help='save output figure as PNG')
    parser.add_argument(
        '--details', default=False, action='store_true', help='plot individual traces')
    parser.add_argument(
        '--dialog', default=False, action='store_true', help='get folder selection dialog')
    parser.add_argument(
        '-n', '--nfilesmax', type=int, default=-1, help='max number of files')
    parser.add_argument(
        '--nofirst', default=False, action='store_true', help='discard first trials')
    
    # Parse command line arguments
    args = parser.parse_args()
    
    # Select root data directory
    if args.dialog:
        datadir = open_folder_dialog(initialdir=dataroot)
        if datadir is None:
            logger.error('no input data directory chosen')
            quit()
        else:
            tif_folders = get_data_folders(datadir)
            if len(tif_folders) == 0:
                tif_folders = [datadir]
    # Otherwise, populate from dataroot, and filter     
    else:
        datadir = os.path.join(dataroot, args.mouseline)
        filters = [x for x in [args.mouseid, args.expdate, args.region] if x is not None]
        tif_folders = get_data_folders(datadir, include_patterns=filters)
    
    # Get directory for figure output
    if 'raw' in dataroot:
        figsdir = get_output_equivalent(dataroot, 'raw', 'figs')
    else:
        figsdir = datadir
    
    # Loop through each folder containing TIF files
    for tif_folder in tif_folders:
        
        # Get TIF stack files inside that folder
        logger.info(f'looking for TIF files in {tif_folder}...')
        fnames = get_sorted_filelist(tif_folder, pattern=P_TIFFILE)
        nfiles = len(fnames)
        logger.info(
            f'found {nfiles} files under the common root {os.path.commonprefix(fnames)}*.tif')
        
        # Remove first trials, if specified
        if args.nofirst:
            logger.info('removing first trials')
            fbr = group_by_run(fnames)
            filt_fnames = []
            for irun, (code, flist) in fbr.items():
                filt_fnames = filt_fnames + flist[1:]
            fnames = filt_fnames 
            nfiles = len(fnames)
            logger.info(f'input list restricted to {nfiles} files')
        
        # Select random subset of files if number of files exceeds limit 
        if args.nfilesmax > 0 and nfiles > args.nfilesmax:
            logger.info(f'restricting input to subset of {args.nfilesmax} files...')
            fnames = sample(fnames, args.nfilesmax)
            nfiles = len(fnames)
        
        # Get filepaths
        tif_fpaths = [os.path.join(tif_folder, fname) for fname in fnames]
        nfiles = len(tif_fpaths)
        
        # Make sure all file have the same (or very close) size
        sizes = list(set([os.path.getsize(f) for f in tif_fpaths]))
        if len(sizes) > 1:
            atol = 10  # absolute tolerance (in bytes)
            diffs = [abs(x - sizes[0]) for x in sizes]
            assert all(x <= atol for x in diffs), f'differing stack sizes: {sizes}'

        # Get stacks frame-average profiles
        if args.mpi:
            with Pool() as pool:
                frameavgs = np.array(pool.map(get_stack_frame_aggregate, tif_fpaths))
        else:
            frameavgs = np.array(list(map(get_stack_frame_aggregate, tif_fpaths)))

        # Plot frame-average profiles
        title = os.path.basename(tif_folder)
        fig = plot_frameavg_profiles(frameavgs, details=args.details, title=title)
        if args.save:
            save_fpath = os.path.join(figsdir, 'frameprofiles', f'{title}.png')
            logger.info(f'saving output figure as {save_fpath}')
            fig.savefig(save_fpath)
    
    plt.show()