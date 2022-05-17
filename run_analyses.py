# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-12-29 12:43:46
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-05-16 11:15:37

''' Utility script to run single region analysis notebook '''

import os
import logging
from argparse import ArgumentParser
from constants import KALMAN_GAIN, PEAK_CORRECTION_QUANTILE

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
        '-k', '--kalman_gain', type=float, default=KALMAN_GAIN, help='Kalman filter gain')
    parser.add_argument(
        '-q', '--peak_correction_quantile', default=PEAK_CORRECTION_QUANTILE, 
        help='Peak correction quantile')

    # Extract command line arguments
    args = vars(parser.parse_args())
    input_nbpath = args.pop('input')
    outdir = args.pop('outdir')
    mpi = args.pop('mpi')
    nocheck = args.pop('nocheck')
    exec_args = ['kalman_gain', 'peak_correction_quantile']
    exec_args = {k: args.pop(k) for k in exec_args}
    q = exec_args['peak_correction_quantile']
    exec_args['peak_correction_quantile'] = None if q.lower() == 'none' else float(q)

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
