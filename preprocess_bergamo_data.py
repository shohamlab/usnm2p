# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-08-15 16:34:13
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-08-19 16:16:22

import os
import logging
from argparse import ArgumentParser

from config import dataroot
from constants import *
from utils import get_singleton
from logger import logger
from fileops import get_data_folders, get_sorted_filelist
from parsers import P_TIFFILE, get_info_table
from correctors import correct_tifs
from resamplers import resample_tifs
from stackers import stack_trial_tifs, split_multichannel_tifs
from substitutors import StackSubstitutor
from fileops import process_and_save

logger.setLevel(logging.INFO)

''' Command line script to preprocess Bergamo USNM2P data '''

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

    # Add resampling arguments
    parser.add_argument('--refsr', help='reference sampling rate (Hz)', default=BERGAMO_SR)
    parser.add_argument('--targetsr', help='target sampling rate (Hz)', default=BRUKER_SR)

    # Parse command line arguments
    args = parser.parse_args()

    # Resampling parameters
    ref_sr = args.refsr  # Hz
    target_sr = args.targetsr   # Hz

    # Input directory for raw data
    datadir = os.path.join(dataroot, args.mouseline)

    # List subfolders containing TIF files
    filters = [x for x in [args.mouseid, args.expdate, args.region] if x is not None]
    tif_folders = get_data_folders(datadir, include_patterns=filters)

    # Loop through each subfolder containing TIF files
    for tif_folder in tif_folders:
        logger.info(f'processing files in "{tif_folder}"')
        # Get TIF stack files inside that folder
        fnames = get_sorted_filelist(tif_folder, pattern=P_TIFFILE)
        raw_fpaths = [os.path.join(tif_folder, fname) for fname in fnames]
        # Detrend TIF stacks using linear regression
        corrected_fpaths = correct_tifs(raw_fpaths, input_root='raw', overwrite=True, mpi=args.mpi)
        # Resample TIF stacks
        resampled_fpaths = resample_tifs(
            corrected_fpaths, ref_sr, target_sr, input_root='corrected', mpi=args.mpi)
        # Stack trial TIFs of every run in the stack list
        stacked_fpaths = stack_trial_tifs(resampled_fpaths, overwrite=False)
        # Extract number of frames per trial 
        raw_info_table = get_info_table(stacked_fpaths)
        nframes_per_trial = get_singleton(raw_info_table, Label.NPERTRIAL)
        logger.info(f'number of frames per trial: {nframes_per_trial}')
        # Split channels from run stacks
        split_fpaths = split_multichannel_tifs(stacked_fpaths, overwrite=False)
        # Substitute problematic frames in every TIF stack and save outputs in specific directory 
        submap = [(FrameIndex.STIM - 1, FrameIndex.STIM)]
        ss = StackSubstitutor(submap, repeat_every=nframes_per_trial)
        input_root = 'split'
        substituted_fpaths = process_and_save(
            ss, split_fpaths, input_root, overwrite=False, mpi=args.mpi)
