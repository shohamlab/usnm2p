# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-08-15 16:34:13
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-08-16 15:11:09

import os
import logging

from config import dataroot
from logger import logger
from fileops import get_data_folders
from resamplers import resample_tifs

logger.setLevel(logging.INFO)

# Input directory for raw data
line = 'cre_sst'
datadir = os.path.join(dataroot, line)

# Resampling parameters
ref_sr = 30.00  # Hz
target_sr = 3.56   # Hz

# List subfolders containing TIF files
tif_folders = get_data_folders(datadir)

# Loop through each subfolder
for tif_folder in tif_folders:
    # Resample TIF stacks found inside that folder
    resample_tifs(tif_folder, ref_sr, target_sr)
