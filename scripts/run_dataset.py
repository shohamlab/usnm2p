# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-12-29 12:43:46
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-07-04 16:08:09

''' Utility script to run low-level (single dataset) analysis notebook(s) '''

import os
from itertools import product

from usnm2p.constants import *
from usnm2p.fileops import get_data_root, get_dataset_params, restrict_datasets
from usnm2p.logger import logger
from usnm2p.nbutils import DirectorySwicther, execute_notebooks, get_notebook_parser, parse_notebook_exec_args


if __name__ == '__main__':

    # Create command line parser
    parser = get_notebook_parser(
        '../notebooks/dataset_analysis.ipynb',
        line=True, date=True, mouse=True, region=True, layer=True)

    # Parse command line arguments
    args = vars(parser.parse_args())

    # Extract general execution parameters
    input_nbpath, outdir, mpi, ask_confirm, proc_queue = parse_notebook_exec_args(args)
    
    # Extract candidate datasets combinations from folder structure
    dataroot = get_data_root(kind=DataRoot.STACKED)
    datasets = get_dataset_params(root=dataroot, analysis=args['analysis'])

    # Filter datasets to match related input parameters
    datasets = restrict_datasets(datasets, **args)

    # Rename "analysis" to "analysis_type" for consistency with notebook execution
    for dataset in datasets:
        dataset['analysis_type'] = dataset.pop('analysis', args['analysis'])
    
    # Compute number of jobs to run
    njobs = len(datasets) * len(proc_queue)
    # Log warning message and quit if no job was found
    if njobs == 0:
        logger.warning('found no individual job to run')
        quit()
    # Set multiprocessing to False in case of single job
    elif njobs == 1:
        mpi = False
    
    # Create execution parameters queue
    params = list(product(datasets, proc_queue))
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