# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-14 19:25:20
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-10-25 10:01:23

''' 
Collection of utilities to run suite2p batches, retrieve suite2p outputs and filter said
outputs according to specific criteria.     
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from suite2p import run_s2p

from constants import *
from logger import logger
from fileops import parse_overwrite


def run_suite2p(*args, overwrite=True, **kwargs):
    '''
    Wrapper around run_s2p function that first checks for existence of suite2p output files
    and runs only if files are absent or if user allowed overwrite.
    '''
    suite2p_keys = ['iscell', 'stat', 'F', 'Fneu', 'spks']
    # For each input directory
    for inputdir in kwargs['db']['data_path']:
        # Check for existence of suite2p subdirectory
        suite2pdir = os.path.join(inputdir, 'suite2p', 'plane0')
        if os.path.isdir(suite2pdir):
            # Check for existence of any suite2p output file
            if all(os.path.isfile(os.path.join(suite2pdir, f'{k}.npy')) for k in suite2p_keys):
                # Warn user if any exists, and act according to defined overwrite behavior
                logger.warning(f'suite2p output files already exist in "{suite2pdir}"')
                if not parse_overwrite(overwrite):
                    return
    # If execution was not canceled, run the standard function
    return run_s2p(*args, **kwargs)


def get_suite2p_data(dirpath, cells_only=False, withops=False):
    '''
    Locate suite2p output files given a specific output directory, and load them into a dictionary.
    
    :param dirpath: full path to the output directory containing the suite2p data files.
    :param cells_only: boolean stating whether to filter out non-cell entities from dataset.
    :param withops: whether include a dictionary of options and intermediate outputs.
    :return: suite2p output dictionary.
    '''
    if not os.path.isdir(dirpath):
        raise ValueError(f'"{dirpath}" directory does not exist')
    keys = ['iscell', 'stat', 'F', 'Fneu', 'spks']
    data = {k : np.load(os.path.join(dirpath, f'{k}.npy'), allow_pickle=True) for k in keys} 
    # If specified, restrict dataset to cells only
    if cells_only:
        iscell = data.pop('iscell')[:, 0]  # the full "iscell" has a second column with the probability of being a cell
        cell_idx = np.array(iscell.nonzero()).reshape(-1)
        data = {k : v[cell_idx] for k, v in data.items()}
    if withops:
        data['ops'] = np.load(os.path.join(dirpath, f'ops.npy'), allow_pickle=True).item()
    nROIs = len(data['stat'])
    logger.info(f'extracted data contains {nROIs} ROIs')
    if ROI_KEY not in data:
        data[ROI_KEY] = np.arange(nROIs)
    return data


def filter_suite2p_data(data, ROI_idx, criterion_key):
    '''
    Small utility function to filter suite2p data.
    
    :param data: suite2p outuput dictionary
    :param ROI_idx: list of indexes of the ROIs to be conserved
    :param criterion_key: key indicating the filter criterion
    :return: tuple with filtered suite2p output dictionary and new ROI indexes
    '''
    filterkey = 'is_filtered'
    if filterkey not in data:
        data[filterkey] = {}
    if criterion_key not in data[filterkey]:
        data[filterkey][criterion_key] = False
    if data[filterkey][criterion_key]:
        logger.warning(f'suite2p data already filtered according to "{criterion_key}" criterion -> ignoring')
    else:
        logger.info(f'filtering suite2p data to "{criterion_key}" criterion...')
        data = {k: v[ROI_idx] if isinstance(v, np.ndarray) else v for k, v in data.items()}
        data[filterkey][criterion_key] = True
    logger.info(f'filtered data contains {len(data[ROI_KEY])} ROIs')
    return data