# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-12-29 12:43:46
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-05-26 14:31:32

''' Utility script to run single region analysis notebook '''

import os
from itertools import product
import logging
from argparse import ArgumentParser
from constants import *

from fileops import get_data_root, get_dataset_params
from logger import logger
from nbutils import DirectorySwicther, execute_notebooks
from utils import is_iterable
from batches import create_queue

logger.setLevel(logging.INFO)

if __name__ == '__main__':

    # Create command line parser
    parser = ArgumentParser()

    # Add input / output / mpi / check arguments
    parser.add_argument(
        '-i', '--input', default='single_region_analysis.ipynb',
        help='path to input notebook')
    parser.add_argument(
        '-o', '--outdir', default='outputs', 
        help='relative path to output directory w.r.t. this script')    
    parser.add_argument(
        '--mpi', default=False, action='store_true', help='enable multiprocessing')
    parser.add_argument(
        '--nocheck', default=False, action='store_true', help='no check before running')
    parser.add_argument(
        '--batch_input', default='batch_analysis.ipynb',
        help='path to input batch notebook')
    parser.add_argument(
        '-b', '--runbatch', default=False, action='store_true',
        help='run batch analysis notebook upon completion')

    # Add dataset arguments 
    parser.add_argument('-l', '--mouseline', help='mouse line')
    parser.add_argument('-d', '--expdate', help='experiment date')
    parser.add_argument('-m', '--mouseid', help='mouse number')
    parser.add_argument('-r', '--region', help='brain region')
    
    # Add arguments about other execution parameters
    parser.add_argument(
        '--no_slack_notify', default=False, action='store_true', help='Do not notify on slack')
    parser.add_argument(
        '-k', '--kalman_gain', type=float, default=KALMAN_GAIN, nargs='+',
        help='Kalman filter gain (s)')
    parser.add_argument(
        '-w', '--baseline_wlen', type=float, default=BASELINE_WLEN, nargs='+',
        help='Baseline rolling window length (s)')
    parser.add_argument(
        '-q', '--baseline_quantile', type=float, default=BASELINE_QUANTILE, nargs='+',
        help='Baseline evaluation quantile')
    parser.add_argument(
        '-s', '--baseline_smoothing', action='store_true', help='Smooth baseline')
    parser.add_argument(
        '-j', '--no-baseline_smoothing', dest='baseline_smoothing', action='store_false')
    parser.set_defaults(baseline_smoothing=BASELINE_SMOOTHING)
    parser.add_argument(
        '-y', '--ykey_postpro', type=str, default='z', choices=['z', 'dff'], nargs='+',
        help='Post-processing variable')

    # Extract command line arguments
    args = vars(parser.parse_args())

    # Process execution arguments
    input_nbpath = args.pop('input')
    outdir = args.pop('outdir')
    mpi = args.pop('mpi')
    batch_mpi = mpi
    nocheck = args.pop('nocheck')
    runbatch = args.pop('runbatch')
    batch_input_nbpath = args.pop('batch_input')
    exec_args = [
        'no_slack_notify',
        'kalman_gain',
        'baseline_wlen',
        'baseline_quantile',
        'baseline_smoothing',
        'ykey_postpro'
    ]
    exec_args = {k: args.pop(k) for k in exec_args}
    exec_args = {k: v if is_iterable(v) else [v] for k, v in exec_args.items()}
    exec_args['ykey_postpro'] = [
        {'z': Label.ZSCORE, 'dff': Label.DFF}[y] for y in exec_args['ykey_postpro']]
    exec_queue = create_queue(exec_args)

    # Extract candidate datasets combinations from folder structure
    datasets = get_dataset_params(root=get_data_root())

    # Filter datasets to match related input parameters
    for k, v in args.items():
        if v is not None:
            # For date, apply loose "startswith" matching (e.g. to enable year-month filtering)
            if k == 'expdate':
                filtfunc = lambda x: x[k].startswith(v)
            # For all other arguments, apply strict matching
            else:
                filtfunc = lambda x: x[k] == v
            logger.info(f'restricting datasets to {k} = {v}')
            datasets = list(filter(filtfunc, datasets))            
    
    # Compute number of jobs to run
    njobs = len(datasets) * len(exec_queue)
    # Log warning message if no job was found
    if njobs == 0:
        logger.warning('found no individual job to run')
    # Otherwise, create execution parameters queue
    else:
        params = list(product(datasets, exec_queue))
        params = [{**dataset, **exec_args} for (dataset, exec_args) in params]
        # Set multiprocessing to False in case of single job
        if njobs == 1:
            mpi = False

    # If batch notebooks must also be run
    nbatchjobs = 0
    if runbatch:
        # Identify mouselines
        mouselines = list(set([d['mouseline'] for d in datasets]))
        mouselines = [{'mouseline': ml for ml in mouselines}]
        # Compute number of batch jobs to run
        nbatchjobs = len(mouselines) * len(exec_queue)
        # Log warning message if no batch job was found
        if nbatchjobs == 0:
            logger.warning('found no batch job to run')
        # Otherwise, create batch execution parameters queue
        else:
            batch_params = list(product(mouselines, exec_queue))
            batch_params = [{**ml, **exec_args} for (ml, exec_args) in batch_params]
            # Set batch multiprocessing to False in case of single batch job
            if nbatchjobs == 1:
                batch_mpi = False

    # Get absolute path to directory of current file (where code must be executed)
    script_fpath = os.path.realpath(__file__)
    exec_dir = os.path.split(script_fpath)[0]

    # Execute notebooks within execution directory (to ensure correct function)
    with DirectorySwicther(exec_dir) as ds:
        # Execute notebooks as a batch with / without multiprocessing
        if njobs > 0:
            output_nbpaths = execute_notebooks(
                params, input_nbpath, outdir, mpi=mpi, ask_confirm=not nocheck)

        # If specified, execute batch analysis notebooks once individual ones are completed
        if nbatchjobs > 0:
            batch_output_nbpaths = execute_notebooks(
                batch_params, batch_input_nbpath, outdir, mpi=batch_mpi,
                ask_confirm=False)

logger.info('all analyses completed')