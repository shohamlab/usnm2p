# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-14 19:25:20
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-11-12 08:58:41

''' 
Collection of utilities to run suite2p batches, retrieve suite2p outputs and filter said
outputs according to specific criteria.     
'''

import pprint
import os
import numpy as np
import pandas as pd
from suite2p import run_s2p, default_ops, version

from constants import *
from logger import logger
from fileops import parse_overwrite

suite2p_output_files = [f'{k}.npy' for k in ['iscell', 'stat', 'F', 'Fneu', 'spks', 'ops']]


def compare_options(input_ops, ref_ops):
    '''
    Compare an input suite2p options dictionary to a reference options dictionary, identify
    and report differing values. 
    
    :param input_ops: dictionary of input suite2p run options
    :param stored_ops: dictionary of reference suite2p run options (either ref options or options loaded from a previous run)
    :return: dataframe of identified differing values (or None if none found)
    '''
    # Check that all keys of input options are in the ref options dict
    diffkeys = input_ops.keys() - ref_ops.keys()
    if len(diffkeys) > 0:
        raise ValueError(
            f'the following input keys are not found in the reference options: {pprint.pformat(diffkeys)}')
    # Limit comparison to fields that are not usuallty rewritten upon suite2p running
    comparekeys = input_ops.keys() - REWRITTEN_S2P_KEYS
    # Compare input and ref values for matching keys
    difftuples = [(k, input_ops[k], ref_ops[k]) for k in comparekeys if ref_ops[k] != input_ops[k]]
    # If differing values are found, return a summary table
    if len(difftuples) > 0:
        diffkeys, input_vals, ref_vals = zip(*difftuples)
        return pd.DataFrame(
            {'input': input_vals, 'reference': ref_vals}, index=diffkeys)
    # Otherwise return None
    else:
        return None

    
def get_option_code(key, val):
    ''' Get a string code for an option key-value pair '''
    if isinstance(val, bool):
        if val:
            return key
        else:
            return f'not_{key}'
    else:
        return f'{key}_{val}'


def get_options_code(ops):
    ''' Get a string code for a dictionary of options '''
    return '_'.join(sorted([get_option_code(k, v) for k, v in ops.items()]))


def run_s2p_and_rename(ops=None, db=None, overwrite=True):
    '''
    Wrapper around run_s2p function that performs several additional features:
    1. checks for options that differ from the default options, and generates a suite2p sub-directory string code
    2. checks for existence of suite2p output files in the input directories
    3. if these files exist, compare options dictionary and ask the user what to do if they do not match exactly
    4. exclude folders that have been tagged throughout the options comparison process 
    5. run suite2p on remaining folders
    6. rename the output suite2p sub-directory in all folder, according to the code initially defined
    
    :param ops: suite2p options dictionary
    :param db: suite2p database dictionary
    :param overwrite (optional): what to do in case of potential overwrite
    :return: suite2p sub-dir code established from run options
    '''
    logger.info(f'running suite2p {version} with the following options:\n{pprint.pformat(ops)}')
    if db is None:
        raise ValueError('"db" keyword argument must be provided')
    # Get dictionary of options that differ from default options
    diff_from_default_ops = compare_options(ops, default_ops())['input'].to_dict()
    # Derive options code and, from it, final suite2p sub-directory
    s2p_basedir = 'suite2p'
    if diff_from_default_ops is not None:
        s2p_basedir = f'suite2p_{get_options_code(diff_from_default_ops)}'
    logger.info(f'data will be saved in suite2p base directory "{s2p_basedir}"')

    # For each input directory
    excluded_dirs = []
    for inputdir in db['data_path']:
        # Check for existence of suite2p subdirectory
        plane0dir = os.path.join(inputdir, s2p_basedir, 'plane0')
        if os.path.isdir(plane0dir):
            # Check for existence of all suite2p output files
            if all(os.path.isfile(os.path.join(plane0dir, k)) for k in suite2p_output_files):
                # Warn user if any exists, and act according to defined overwrite behavior
                logger.info(f'found suite2p output files in "{plane0dir}"')
                # Extract and compare input and stored options dictionaries
                stored_ops = np.load(os.path.join(plane0dir, 'ops.npy'), allow_pickle=True).item()
                diff_values = compare_options(ops, stored_ops)
                if diff_values is not None:
                    # If differing values are found, that means suite2p is intended to be run
                    # with different options -> overwrite warning
                    logger.warning(f'the following suite2p run options differ from those found in suite2p output directory":\n{diff_values}')
                    if not parse_overwrite(overwrite):
                        # If overwrite is not allowed -> exclude folder from run
                        excluded_dirs.append(inputdir)
                else:
                    # If no difference with previous run -> exclude folder from run
                    logger.info('run options match 100% -> ignoring')
                    excluded_dirs.append(inputdir)
                    
    # Apply folders exclusion
    db['data_path'] = list(set(db['data_path']) - set(excluded_dirs))
    if len(db['data_path']) > 0:
        # If folders remaining -> run suite2p on them
        opsout = run_s2p(ops=ops, db=db)
        if s2p_basedir != 'suite2p':
            # If options differed from defaults, rename all suite2p output directories
            for inputdir in db['data_path']:
                dirin = os.path.join(inputdir, 'suite2p')
                dirout = os.path.join(inputdir, s2p_basedir)
                logger.info(f'renaming "{dirin}" to "{dirout}"')
                os.rename(dirin, dirout)
    else:
        logger.info('empty data path -> no run')
    
    return s2p_basedir


def get_suite2p_data(dirpath, cells_only=False, withops=False, s2p_basedir=None):
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
        if s2p_basedir is not None:
            for k in ['save_path0', 'save_folder', 'save_path', 'ops_path', 'reg_file']:
                data['ops'][k] = data['ops'][k].replace('suite2p', s2p_basedir)
    nROIs = len(data['stat'])
    logger.info(f'extracted data contains {nROIs} ROIs')
    return data


def filter_s2p_data(data, ivalids):
    ''' Filter suite2p output dictionary according to list of valid indexes '''
    return {k: v[ivalids] if isinstance(v, np.ndarray) else v for k, v in data.items()}
