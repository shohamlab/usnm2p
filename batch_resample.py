# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-08-15 16:34:13
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-08-15 16:46:57

import os
import logging
import glob
import matplotlib.pyplot as plt

from config import dataroot
from logger import logger
from fileops import process_and_save, get_data_folders
from resamplers import StackResampler

logger.setLevel(logging.INFO)

# Input directory for raw data
input_root = 'raw'
line = 'cre_sst'
datadir = os.path.join(dataroot, line)

# Resampling parameters
ref_sr = 30.00  # Hz
target_sr = 3.56   # Hz
sr = StackResampler(ref_sr=ref_sr, target_sr=target_sr)

# List subfolders containing TIF files
tif_folders = get_data_folders(datadir)

# Loop through each subfolder
for tif_folder in tif_folders:
    logger.info(f'resampling TIF files in "{tif_folder}"')
    # List and sort TIF files
    raw_stack_fpaths = sorted(glob.glob(os.path.join(tif_folder, '*.tif')))
    # Resample each file
    resampled_stack_fpaths = process_and_save(
        sr, raw_stack_fpaths, input_root, overwrite=False)