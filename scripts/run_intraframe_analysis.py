# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2025-07-09 17:20:08
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-07-09 17:49:15

''' Utility script to run intraframe analysis notebook '''

import os
from itertools import product

from usnm2p.constants import *
from usnm2p.fileops import get_data_root, get_dataset_params, restrict_datasets
from usnm2p.logger import logger
from usnm2p.nbutils import DirectorySwicther, execute_notebooks, get_notebook_parser, parse_notebook_exec_args


if __name__ == '__main__':

    # Create command line parser
    parser = get_notebook_parser(
        '../notebooks/intraframe_analysis.ipynb', line=True)

    # Parse command line arguments
    args = vars(parser.parse_args())
    print(args)

    # Extract general execution parameters
    input_nbpath, outdir, mpi, ask_confirm, proc_queue = parse_notebook_exec_args(args)
    proc_queue = {}

    dataroot = os.path.join(get_data_root(kind=DataRoot.ROWAVG), args['analysis'])
    if args['mouseline'] is None:
        mouselines = os.listdir(dataroot)
    else:
        if not os.path.isdir(os.path.join(dataroot, args['mouseline'])):
            logger.error(f'mouseline "{args["mouseline"]}" not found in {dataroot}')
            quit()
        mouselines = [args['mouseline']]

    # Create execution parameters queue
    params = [dict(mouseline=mouseline, analysis_type=args['analysis']) for mouseline in mouselines]
 
    # Compute number of jobs to run
    njobs = len(params)
    # Log warning message and quit if no job was found
    if njobs == 0:
        logger.warning('found no individual job to run')
        quit()
    # Set multiprocessing to False in case of single job
    elif njobs == 1:
        mpi = False

    # Get absolute path to directory of current file (where code must be executed)
    script_fpath = os.path.realpath(__file__)
    exec_dir = os.path.split(script_fpath)[0]

    # Execute notebooks within execution directory (to ensure correct function)
    with DirectorySwicther(exec_dir) as ds:
        # Execute notebooks as a batch with / without multiprocessing
        output_nbpaths = execute_notebooks(
            params, input_nbpath, outdir, mpi=mpi, ask_confirm=ask_confirm)

logger.info('done.')