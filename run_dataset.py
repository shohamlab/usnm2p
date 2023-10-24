# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-12-29 12:43:46
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-02-14 16:15:57

''' Utility script to run low-level (single dataset) analysis notebook(s) '''

import os
from itertools import product
import logging
from constants import *

from fileops import get_data_root, get_dataset_params
from logger import logger
from nbutils import DirectorySwicther, execute_notebooks, get_notebook_parser
from utils import as_iterable
from parsers import none_or_float, none_or_str
from batches import create_queue

logger.setLevel(logging.INFO)


if __name__ == '__main__':

    # Create command line parser
    parser = get_notebook_parser(
        'dataset_analysis.ipynb',
        line=True, date=True, mouse=True, region=True, layer=True)

    # Add arguments about other execution parameters
    parser.add_argument(
        '--inspect', default=False, action='store_true',
        help='Inspect data from random run along processing')
    parser.add_argument(
        '-c', '--correction', type=none_or_str, default=GLOBAL_CORRECTION, nargs='+',
        help='Global correction method')
    parser.add_argument(
        '-k', '--kalman_gain', type=none_or_float, default=KALMAN_GAIN, nargs='+',
        help='Kalman filter gain (s)')
    parser.add_argument(
        '--alpha', type=float, default=NEUROPIL_SCALING_COEFF, nargs='+',
        help='scaling coefficient for neuropil subtraction')    
    parser.add_argument(
        '-q', '--baseline_quantile', type=none_or_float, default=BASELINE_QUANTILE, nargs='+',
        help='Baseline evaluation quantile')
    parser.add_argument(
        '--wq', type=float, default=BASELINE_WQUANTILE, nargs='+',
        help='Baseline quantile filter window size (s)')
    parser.add_argument(
        '--ws', type=none_or_float, default=BASELINE_WSMOOTHING, nargs='+',
        help='Baseline gaussian filter window size (s)')
    parser.add_argument(
        '-y', '--ykey_classification', type=str, default='zscore', choices=['dff', 'zscore', 'evrate'], nargs='+',
        help='Classification variable')
    parser.add_argument(
        '--directional', action='store_true', help='Directional classification')
    parser.add_argument(
        '--non-directional', dest='directional', action='store_false')
    parser.set_defaults(directional=True)

    # Extract command line arguments
    args = vars(parser.parse_args())

    # Process execution arguments
    input_nbpath = args.pop('input')
    outdir = args.pop('outdir')
    mpi = args.pop('mpi')
    ask_confirm = not args.pop('go')
    exec_args = [
        'inspect',
        'slack_notify',
        'correction',
        'kalman_gain',
        'alpha',
        'baseline_quantile',
        'wq',
        'ws',
        'ykey_classification',
        'directional'
    ]
    exec_args = {k: args.pop(k) for k in exec_args}
    exec_args = {k: as_iterable(v) for k, v in exec_args.items()}
    exec_args['neuropil_scaling_coeff'] = exec_args.pop('alpha')
    exec_args['baseline_wquantile'] = exec_args.pop('wq')
    exec_args['baseline_wsmoothing'] = exec_args.pop('ws')
    exec_args['ykey_classification'] = [
        {
            'evrate': Label.EVENT_RATE, 
            'dff': Label.DFF,
            'zscore': Label.ZSCORE
        }[y]
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
    # Log warning message and quit if no job was found
    if njobs == 0:
        logger.warning('found no individual job to run')
        quit()
    # Set multiprocessing to False in case of single job
    elif njobs == 1:
        mpi = False
    
    # Create execution parameters queue
    params = list(product(datasets, exec_queue))
    params = [{**dataset, **exec_args} for (dataset, exec_args) in params]

    # Get absolute path to directory of current file (where code must be executed)
    script_fpath = os.path.realpath(__file__)
    exec_dir = os.path.split(script_fpath)[0]

    # Execute notebooks within execution directory (to ensure correct function)
    with DirectorySwicther(exec_dir) as ds:
        # Execute notebooks as a batch with / without multiprocessing
        output_nbpaths = execute_notebooks(
            params, input_nbpath, outdir, mpi=mpi, ask_confirm=ask_confirm)

logger.info('done.')