# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-08-15 16:34:13
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-08-16 16:49:16

import os
import logging

from config import dataroot
from logger import logger
from fileops import get_data_folders, get_output_equivalent
from resamplers import StackResampler
from stackers import split_multichannel_tifs

logger.setLevel(logging.INFO)

# Input directory for raw data
line = 'cre_sst'
rawdir = os.path.join(dataroot, line)

# Input directory for stacked data
ref_sr = 30.00  # Hz
target_sr = 3.56   # Hz
sr = StackResampler(ref_sr=ref_sr, target_sr=target_sr)
datadir = get_output_equivalent(rawdir, 'raw', f'stacked/{sr.code}')
print(datadir)

# List subfolders containing TIF files
tif_folders = get_data_folders(datadir)

# Loop through each subfolder
for tif_folder in tif_folders:
    # Stack trial TIFs of every run in the folder
    output_fpaths = split_multichannel_tifs(tif_folder, overwrite=False)
