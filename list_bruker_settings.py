# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-12-11 17:29:03
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-12-11 18:05:09

import os
from argparse import ArgumentParser
import logging

from constants import DEFAULT_LINE
from fileops import get_data_folders
from parsers import parse_acquisition_settings
from logger import logger

logger.setLevel(logging.INFO)

# Parse command line arguments
parser = ArgumentParser()
parser.add_argument('-l', '--mouseline', help='mouse line', default=DEFAULT_LINE)
args = parser.parse_args()
mouseline  = args.mouseline
datadir = os.path.abspath('.')

# Get raw list of subolders containing tifs
tif_folders = get_data_folders(
    datadir, 
    exclude_patterns=['MIP', 'References', 'incomplete', 'duplicated'], 
    include_patterns=[mouseline])
tif_folders_str = '\n'.join([f'  - {os.path.basename(x)}' for x in tif_folders])
logger.info(f'Identified folders containing TIF files:\n{tif_folders_str}')

# Extract acquisition settings from each run
logger.info('extracting acquisition settings...')
daq_settings = parse_acquisition_settings(tif_folders)
print(daq_settings)