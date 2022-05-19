# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-12-29 12:43:46
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-05-19 18:43:44

''' Utility script to run single region analysis notebook '''

import os
import logging
from argparse import ArgumentParser
from constants import *

from fileops import get_data_root, get_dataset_params
from logger import logger
from nbutils import DirectorySwicther, execute_notebooks

logger.setLevel(logging.INFO)

if __name__ == '__main__':

    # Create command line parser
    parser = ArgumentParser()

    # Add input / output / mpi / check arguments
    parser.add_argument(
        '-i', '--input', default='single_region_analysis.ipynb', help='path to input notebook')
    parser.add_argument(
        '-o', '--outdir', default='outputs', 
        help='relative path to output directory w.r.t. this script')    
    parser.add_argument(
        '--mpi', default=False, action='store_true', help='enable multiprocessing')
    parser.add_argument(
        '--nocheck', default=False, action='store_true', help='no check before running')

    # Add dataset arguments 
    parser.add_argument('-l', '--mouseline', help='mouse line')
    parser.add_argument('-d', '--expdate', help='experiment date')
    parser.add_argument('-m', '--mouseid', help='mouse number')
    parser.add_argument('-r', '--region', help='brain region')
    
    # Add arguments about other execution parameters
    parser.add_argument(
        '--no_slack_notify', default=False, action='store_true', help='Do not notify on slack')
    parser.add_argument(
        '-k', '--kalman_gain', type=str, default=str(KALMAN_GAIN), help='Kalman filter gain')
    parser.add_argument(
        '-w', '--baseline_wlen', type=float, default=BASELINE_WLEN, help='Baseline rolling window length (s)')
    parser.add_argument(
        '-q', '--baseline_quantile', type=float, default=BASELINE_QUANTILE, help='Baseline evaluation quantile')
    parser.add_argument(
        '-y', '--ykey_postpro', type=str, default='z', choices=['z', 'dff'], help='Post-processing variable')

    # Extract command line arguments
    args = vars(parser.parse_args())
    input_nbpath = args.pop('input')
    outdir = args.pop('outdir')
    mpi = args.pop('mpi')
    nocheck = args.pop('nocheck')
    exec_args = [
        'no_slack_notify',
        'kalman_gain',
        'baseline_wlen',
        'baseline_quantile',
        'ykey_postpro'
    ]
    exec_args = {k: args.pop(k) for k in exec_args}
    if exec_args['kalman_gain'].lower() == 'none':
        exec_args['kalman_gain'] = None
    else:
        exec_args['kalman_gain'] = float(exec_args['kalman_gain'])
    exec_args['ykey_postpro'] = {'z': Label.ZSCORE, 'dff': Label.DFF}[exec_args['ykey_postpro']]
    
    # Extract candidate datasets combinations from folder structure
    datasets = get_dataset_params(root=get_data_root())

    # Filter datasets to match related input parameters
    for k, v in args.items():
        if v is not None:
            logger.info(f'restricting datasets to {k} = {v}')
            datasets = list(filter(lambda x: x[k] == v, datasets))

    # Compute number of jobs to run 
    njobs = len(datasets)
    
    # Log warning message and quit if no job was found
    if njobs == 0:
        logger.warning('found no job to run')
        quit()
    # Set multiprocessing to False in case of single job
    elif njobs == 1:
        mpi = False

    # Merge datasets and execution parameters information
    params = datasets.copy()
    for k, v in exec_args.items():
        for i in range(njobs):
            params[i][k] = v

    # Get absolute path to directory of current file (where code must be executed)
    script_fpath = os.path.realpath(__file__)
    exec_dir = os.path.split(script_fpath)[0]

    # Execute notebooks within execution directory (to ensure correct function)
    with DirectorySwicther(exec_dir) as ds:
        # Execute notebooks as a batch with / without multiprocessing
        output_nbpaths = execute_notebooks(
            params, input_nbpath, outdir, mpi=mpi, ask_confirm=not nocheck)
