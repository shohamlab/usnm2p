# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-10-27 18:16:01
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-10-27 18:58:27

''' Utility script to run high-level analysis notebook(s) '''

import os
from itertools import product
import logging
from argparse import ArgumentParser
from constants import *

from fileops import get_data_root, get_dataset_params
from logger import logger
from nbutils import DirectorySwicther, execute_notebooks
from utils import as_iterable
from batchutils import get_batch_settings
from batches import create_queue

logger.setLevel(logging.INFO)

if __name__ == '__main__':

    # Create command line parser
    parser = ArgumentParser()
    
    # Add input / output / mpi / check arguments
    parser.add_argument(
        '-i', '--input', default='mouseline_analysis.ipynb',
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

    # Add arguments about other execution parameters
    parser.add_argument(
        '--slack_notify', action='store_true', help='Notify on slack')
    parser.add_argument(
        '--no-slack_notify', dest='slack_notify', action='store_false')
    parser.set_defaults(slack_notify=True)

    # Extract command line arguments
    args = vars(parser.parse_args())

    # Process execution arguments
    input_nbpath = args.pop('input')
    outdir = args.pop('outdir')
    mpi = args.pop('mpi')
    batch_mpi = mpi
    nocheck = args.pop('nocheck')
    exec_args = [
        'slack_notify',
    ]
    exec_args = {k: args.pop(k) for k in exec_args}
    exec_args = {k: as_iterable(v) for k, v in exec_args.items()}
    exec_queue = create_queue(exec_args)

    # Get mouselines input directories
    mouselines = ['line3', 'sst', 'pv']
    if args['mouseline'] is not None:
        mouselines = [mouselines[mouselines.index(args['mouseline'])]]
    trialavg_dirs = {}
    for mouseline in mouselines:
        trialavg_dirs[mouseline] = get_batch_settings(
            args['analysis_type'], mouseline, None, KALMAN_GAIN,
            BASELINE_WLEN, BASELINE_QUANTILE, True, Label.DFF)[1]
    trialavg_dirs = {k: v for k, v in trialavg_dirs.items() if os.path.isdir(v)}
    
    # Compute number of jobs to run
    njobs = len(trialavg_dirs) * len(exec_queue)
    # Log warning message if no job was found
    if njobs == 0:
        logger.warning('found no individual job to run')
    # Otherwise, create execution parameters queue
    else:
        params = list(product(trialavg_dirs.keys(), exec_queue))
        params = [{'mouseline': mouseline, **exec_args} for (mouseline, exec_args) in params]
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