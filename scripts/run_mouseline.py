# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-10-27 18:16:01
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2024-08-09 12:37:07

''' Utility script to run high-level analysis notebook(s) '''

import os
from itertools import product

from usnm2p.constants import *
from usnm2p.logger import logger
from usnm2p.nbutils import DirectorySwicther, execute_notebooks, get_notebook_parser, parse_notebook_exec_args
from usnm2p.batchutils import get_batch_settings


if __name__ == '__main__':

    # Create command line parser
    parser = get_notebook_parser(
        '../notebooks/mouseline_analysis.ipynb',
        line=True)
    
    # Parse command line arguments
    args = vars(parser.parse_args())

    # Extract general execution parameters
    input_nbpath, outdir, mpi, ask_confirm, proc_queue = parse_notebook_exec_args(args)

    # Get mouselines input directories
    mouselines = ['line3', 'sst', 'pv', 'sarah_line3']
    if args['mouseline'] is not None:
        mouselines = [mouselines[mouselines.index(args['mouseline'])]]
    trialavg_dirs = {}
    for mouseline in mouselines:
        trialavg_dirs[mouseline] = get_batch_settings(
            args['analysis'], mouseline, None, GLOBAL_CORRECTION[mouseline], KALMAN_GAIN, 
            NEUROPIL_SCALING_COEFF, BASELINE_QUANTILE, BASELINE_WQUANTILE, BASELINE_WSMOOTHING, 
            TRIAL_AGGFUNC, YKEY_CLASSIFICATION, DIRECTIONAL_DETECTION
        )[1]
    trialavg_dirs = {k: v for k, v in trialavg_dirs.items() if os.path.isdir(v)}
    
    # Compute number of jobs to run
    njobs = len(trialavg_dirs) * len(proc_queue)
    # Log warning message if no job was found
    if njobs == 0:
        logger.warning('found no individual job to run')
    
    # Otherwise, create execution parameters queue
    else:
        params = list(product(trialavg_dirs.keys(), proc_queue))
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
                params, input_nbpath, outdir, mpi=mpi, ask_confirm=ask_confirm)

logger.info('done.')