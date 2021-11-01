# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-14 19:25:20
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-10-29 15:09:28

''' 
Collection of utilities to run suite2p batches, retrieve suite2p outputs and filter said
outputs according to specific criteria.     
'''

import pprint
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from suite2p import run_s2p

from constants import *
from logger import logger
from fileops import parse_overwrite


def run_suite2p(*args, overwrite=True, **kwargs):
    '''
    Wrapper around run_s2p function that performs several additional features:
    1. checks for existence of suite2p output files in the targeted directory
    2. if these files exist, the output options dictionary is loaded and compared to the one passed as input
    3. if these run options do not match exaclty, ask the user whether the suite2p data should be overwritten. 
    
    The suite2p function is then run only if (1) no output files are present or (2) the run options 
    differ from the previous ones and the user allowed overwrite.
    '''
    suite2p_keys = ['iscell', 'stat', 'F', 'Fneu', 'spks', 'ops']
    # For each input directory
    for inputdir in kwargs['db']['data_path']:
        # Check for existence of suite2p subdirectory
        suite2pdir = os.path.join(inputdir, 'suite2p', 'plane0')
        if os.path.isdir(suite2pdir):
            # Check for existence of any suite2p output file
            if all(os.path.isfile(os.path.join(suite2pdir, f'{k}.npy')) for k in suite2p_keys):
                # Warn user if any exists, and act according to defined overwrite behavior
                logger.info(f'found suite2p output files in "{suite2pdir}"')
                # Extract input and output options dictionaries
                opsin = kwargs['ops']
                opsout = np.load(os.path.join(suite2pdir, 'ops.npy'), allow_pickle=True).item()
                # Check that all input keys are in the output options dict
                diffkeys = opsin.keys() - opsout.keys()
                if len(diffkeys) > 0:
                    raise ValueError(f'the following input keys are not found in the output options: {pprint.pformat(diffkeys)}')
                # Compare input and output values for matching keys
                comparekeys = opsin.keys() - REWRITTEN_S2P_KEYS
                difftuples = [(k, opsin[k], opsout[k]) for k in comparekeys if opsout[k] != opsin[k]]
                # If differing values are found, that means suite2p is intended to be run
                # with different options -> overwrite warning
                if len(difftuples) > 0:
                    diffkeys, invals, outvals = zip(*difftuples)
                    diffdata = pd.DataFrame(
                        {'current value': invals, 'value on disk': outvals}, index=diffkeys)
                    logger.warning(f'the following suite2p run options differ from those found in suite2p output directory":\n{diffdata}')
                    if not parse_overwrite(overwrite):
                        return
                # otherwise return
                else:
                    logger.info('run options match 100% -> ignoring')
                    return
    # If execution was not canceled, run the standard function
    return run_s2p(*args, **kwargs)


def get_suite2p_data(dirpath, cells_only=False, withops=False):
    '''
    Locate suite2p output files given a specific output directory, and load them into a dictionary.
    
    :param dirpath: full path to the output directory containing the suite2p data files.
    :param cells_only: boolean stating whether to filter out non-cell entities from dataset.
    :param withops: whether include a dictionary of options and intermediate outputs.
    :return: suite2p output dictionary with extra "roi" and "is_valid" column
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
    data[ROI_KEY] = np.arange(nROIs)
    data[IS_VALID_KEY] = np.ones(nROIs).astype(bool)
    return data


def update_suite2p_data_validity(data, is_valid):
    '''
    Update the validity of each cell in the suite2p dataset according to a new vector of
    validity status for each cell.
    
    :param data: suite2p output dictionary
    :param is_valid: boolean array of validity status for each ROI
    :return: suite2p output with filtered "is_valid" list
    '''
    logger.info(f'filtering suite2p data...')
    is_valid = is_valid.astype(bool)
    data[IS_VALID_KEY] = data[IS_VALID_KEY] & is_valid
    logger.info(f'filtered data contains {data[IS_VALID_KEY].sum()} valid ROIs')


def get_filtered_suite2p_data(data, is_valid=None):
    '''
    Get a filtered version of the suite2p dataset containing only data for valid ROIs
    
    :param data: suite2p output dictionary
    :param is_valid: boolean array of validity status for each ROI
    :return: filtered suite2p output dictionary containing only valid ROIs
    '''
    fdata = data.copy()  # create copy so as to not modify the original
    if is_valid is None:
        is_valid = fdata.pop(IS_VALID_KEY)  # extract validity status array
    ivalids = is_valid.nonzero()[0]  # extract indexes of valid ROIs
    return {k: v[ivalids] if isinstance(v, np.ndarray) else v for k, v in fdata.items()}