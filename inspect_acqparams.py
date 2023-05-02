# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2023-05-02 15:47:25
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-05-02 15:58:27

''' Utility script to inspect acquisition parameters across runs & datasets '''

import os
import logging
from constants import *

from parsers import resolve_mouseline, parse_acquisition_settings
from fileops import get_data_root, get_dataset_params, get_data_folders
from logger import logger
from nbutils import get_notebook_parser

logger.setLevel(logging.INFO)


if __name__ == '__main__':

    # Create command line parser
    parser = get_notebook_parser(
        'dataset_analysis.ipynb',
        line=True, date=True, mouse=True, region=True, layer=True)

    # Extract command line arguments
    args = vars(parser.parse_args())

    # Extract data root
    dataroot = get_data_root()

    # Extract candidate datasets combinations from folder structure
    datasets = get_dataset_params(root=dataroot, analysis_type=args['analysis_type'])

    # For each candidate dataset
    for d in datasets:
        # Construct dataset ID
        dataset_id = f'{d["expdate"]}_{d["mouseid"]}_{d["region"]}'
        if d['layer'] != DEFAULT_LAYER:
            dataset_id = f'{dataset_id}_{d["layer"]}'
        
        # Extract data raw data directory
        datadir = os.path.join(dataroot, d['analysis_type'], d['mouseline'], dataset_id)

        # Get raw list of subolders containing tifs, sorted by run ID
        tif_folders = get_data_folders(
            datadir, 
            exclude_patterns=['MIP', 'References', 'incomplete', 'duplicated'], 
            include_patterns=[resolve_mouseline(d['mouseline'])], 
            sortby=Label.RUNID
        )

        # Extract acquisition settings from each run, and outlier runs
        ref_daq_settings, _ = parse_acquisition_settings(tif_folders)
        print(ref_daq_settings)