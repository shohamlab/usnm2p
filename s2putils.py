# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-14 19:25:20
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-02-25 10:17:26

''' 
Collection of utilities to run suite2p batches, retrieve suite2p outputs and filter said
outputs according to specific criteria.     
'''

import pprint
import os
import numpy as np
import pandas as pd
from suite2p import run_s2p, default_ops, version
from suite2p.io import BinaryFile

from constants import *
from logger import logger
from fileops import parse_overwrite

default_output_files = ['ops.npy', 'data.bin']
roidetect_output_files = [f'{k}.npy' for k in ['iscell', 'stat', 'F', 'Fneu', 'spks']]


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
    # Fetch default options
    defops = default_ops()
    # Check registration status and adapt nonrigid status if needed
    is_reg = ops.get('do_registration', defops['do_registration'])
    is_nonrigid = ops.get('nonrigid', defops['nonrigid'])
    if not is_reg and is_nonrigid:
        logger.warning('opted out of registration -> setting nonrigid to False')
        ops['nonrigid'] = False
        is_nonrigid = False
    # Check nonrigid status and adapt block_size if needed
    if not is_nonrigid:
        if 'block_size' in ops:
            del ops['block_size']
    else:
        block_size = ops.get('block_size', defops['block_size'])
        ops['block_size'] = tuple(block_size)
        if tuple(ops['block_size']) == tuple(defops['block_size']):
            del ops['block_size']

    # Check diameter value and raise error if needed
    if not ops.get('sparse_mode', True) and ops.get('diameter', 0) == 0:
        raise ValueError('Cannot run non-sparse ROI detection: diameter not specified')
            
    logger.info(f'running suite2p {version} with the following options:\n{pprint.pformat(ops)}')
    if db is None:
        raise ValueError('"db" keyword argument must be provided')
    # Get dictionary of options that differ from default options
    diff_from_default_ops = compare_options(ops, defops)['input'].to_dict()
    # Derive options code and, from it, final suite2p sub-directory
    s2p_basedir = 'suite2p'
    if diff_from_default_ops is not None:
        s2p_basedir = f'suite2p_{get_options_code(diff_from_default_ops)}'
    logger.info(f'data will be saved in suite2p base directory "{s2p_basedir}"')

    # List expected output files from run options
    suite2p_output_files = default_output_files.copy()
    if ops.get('roidetect', True):
        suite2p_output_files = suite2p_output_files + roidetect_output_files.copy()

    # For each input directory
    excluded_dirs = []
    for inputdir in db['data_path']:
        # Check for existence of suite2p subdirectory
        plane0dir = os.path.join(inputdir, s2p_basedir, 'plane0')
        if os.path.isdir(plane0dir):
            # Check for existence of all suite2p output files
            missing_files = [k for k in suite2p_output_files if not os.path.isfile(os.path.join(plane0dir, k))]
            if len(missing_files) > 0:
                missing_files_str = '\n'.join([f'- {x}' for x in missing_files])
                logger.warning(f'the following output files are missing in "{plane0dir}": {missing_files_str}')
            else:
                # Warn user if any exists, and act according to defined overwrite behavior
                logger.info(f'found all suite2p output files in "{plane0dir}"')
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


def get_suite2p_options(dirpath, s2p_basedir=None):
    '''
    Locate and load suite2p output options file given a specific output directory.
    
    :param dirpath: full path to the output directory containing the suite2p options file.
    :param s2p_basedir: whether to change the suite2p base directory in the relevant options fields.
    :return: suite2p options dictionary
    '''
    ops = np.load(os.path.join(dirpath, f'ops.npy'), allow_pickle=True).item()
    if s2p_basedir is not None:
        for k in ['save_path0', 'save_folder', 'save_path', 'ops_path', 'reg_file']:
            ops[k] = ops[k].replace('suite2p', s2p_basedir)
    return ops


def get_suite2p_data(dirpath, cells_only=False, withops=False, **kwargs):
    '''
    Locate suite2p output files given a specific output directory, and load them into a dictionary.
    
    :param dirpath: full path to the output directory containing the suite2p data files.
    :param cells_only: boolean stating whether to filter out non-cell entities from dataset.
    :param withops: whether include a dictionary of options and intermediate outputs.
    :return: suite2p output dictionary with extra "roi" and "is_valid" column
    '''
    if not os.path.isdir(dirpath):
        raise ValueError(f'"{dirpath}" directory does not exist')
    data = {os.path.splitext(k)[0]: np.load(os.path.join(dirpath, k), allow_pickle=True)
            for k in roidetect_output_files} 
    # If specified, restrict dataset to cells only
    if cells_only:
        iscell = data.pop('iscell')[:, 0]  # the full "iscell" has a second column with the probability of being a cell
        cell_idx = np.array(iscell.nonzero()).reshape(-1)
        data = {k : v[cell_idx] for k, v in data.items()}
    if withops:
        data['ops'] = get_suite2p_options(dirpath, **kwargs)
    nROIs = len(data['stat'])
    logger.info(f'extracted data contains {nROIs} ROIs')
    return data


def filter_s2p_data(data, ivalids):
    ''' Filter suite2p output dictionary according to list of valid indexes '''
    return {k: v[ivalids] if isinstance(v, np.ndarray) else v for k, v in data.items()}


def open_binary_file(ops):
    ''' Open binary file linked to options dictionary '''
    return BinaryFile(Ly=ops['Ly'], Lx=ops['Lx'], read_filename=ops['reg_file'])


def get_s2p_stack(ops, bounds=None):
    '''
    Get the stack resulting from suite2p processing
    
    :param ops: suite2p output options dictionary
    :param bounds (optional): frame range boundaries
    :return: (nframes, ny, nx) data array
    '''
    with open_binary_file(ops) as fobj:
        logger.info('loading suite2p binary stack...')
        data = fobj.data
    if bounds is not None:
        logger.info(f'extracting {bounds} stack slice...')
        data = data[bounds[0]:bounds[1] + 1]
    # Multiply stack by factor 2 to compensate for suite2p input normalization
    return data * S2P_UINT16_NORM_FACTOR


def get_s2p_stack_label(ops):
    ''' Construct an appropriate label for the suite2p output stack '''
    l = []
    if ops['do_registration'] == 1:
        prefix = 'non-rigid' if ops['nonrigid'] else 'rigid'
        l.append(f'{prefix} registered')
    if ops['denoise']:
        l.append('PCA denoised')
    label = 's2p'
    if len(l) > 0:
        label = f'{label} ({" + ".join(l)})'
    return label