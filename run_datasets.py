# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-12-29 12:43:46
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-10-28 14:23:22

''' Utility script to run low-level (single dataset) analysis notebook(s) '''

import os
from itertools import product
import logging
from argparse import ArgumentParser
from constants import *

from fileops import get_data_root, get_dataset_params
from logger import logger
from nbutils import DirectorySwicther, execute_notebooks
from utils import as_iterable
from batches import create_queue

logger.setLevel(logging.INFO)

if __name__ == '__main__':

    # Create command line parser
    parser = ArgumentParser()

    # Add input / output / mpi / check arguments
    parser.add_argument(
        '-i', '--input', default='dataset_analysis.ipynb',
        help='path to input notebook')
    parser.add_argument(
        '-o', '--outdir', default='outputs', 
        help='relative path to output directory w.r.t. this script')
    parser.add_argument(
        '--mpi', default=False, action='store_true', help='enable multiprocessing')
    parser.add_argument(
        '--nocheck', default=False, action='store_true', help='no check before running')

    # Add dataset arguments
    parser.add_argument('-a', '--analysis_type', default=DEFAULT_ANALYSIS, help='analysis type')
    parser.add_argument('-l', '--mouseline', help='mouse line')
    parser.add_argument('-d', '--expdate', help='experiment date')
    parser.add_argument('-m', '--mouseid', help='mouse number')
    parser.add_argument('-r', '--region', help='brain region')
    parser.add_argument('--layer', help='Cortical layer')

    # Add arguments about other execution parameters
    parser.add_argument(
        '--inspect', default=False, action='store_true',
        help='Inspect data from random run along processing')
    parser.add_argument(
        '--slack_notify', action='store_true', help='Notify on slack')
    parser.add_argument(
        '--no-slack_notify', dest='slack_notify', action='store_false')
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
    parser.add_argument(
        '-y', '--ykey_classification', type=str, default='dff', choices=['dff', 'evrate'], nargs='+',
        help='Classification variable')
    parser.set_defaults(
        slack_notify=True,
        baseline_smoothing=BASELINE_SMOOTHING)

    # Extract command line arguments
    args = vars(parser.parse_args())

    # Process execution arguments
    input_nbpath = args.pop('input')
    outdir = args.pop('outdir')
    mpi = args.pop('mpi')
    nocheck = args.pop('nocheck')
    exec_args = [
        'inspect',
        'slack_notify',
        'kalman_gain',
        'baseline_wlen',
        'baseline_quantile',
        'baseline_smoothing',
        'ykey_classification'
    ]
    exec_args = {k: args.pop(k) for k in exec_args}
    exec_args = {k: as_iterable(v) for k, v in exec_args.items()}
    exec_args['ykey_classification'] = [
        {'evrate': Label.EVENT_RATE, 'dff': Label.DFF}[y]
        for y in exec_args['ykey_classification']]
    exec_queue = create_queue(exec_args)

    # Extract candidate datasets combinations from folder structure
    datasets = get_dataset_params(root=get_data_root(), analysis_type=args['analysis_type'])

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

    # Get absolute path to directory of current file (where code must be executed)
    script_fpath = os.path.realpath(__file__)
    exec_dir = os.path.split(script_fpath)[0]

    # Execute notebooks within execution directory (to ensure correct function)
    with DirectorySwicther(exec_dir) as ds:
        # Execute notebooks as a batch with / without multiprocessing
        if njobs > 0:
            output_nbpaths = execute_notebooks(
                params, input_nbpath, outdir, mpi=mpi, ask_confirm=not nocheck)

logger.info('all analyses completed')