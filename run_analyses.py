# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-12-29 12:43:46
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-05-10 13:21:40

''' Utility script to run single region analysis notebook '''

import os
import logging
from argparse import ArgumentParser

from fileops import get_data_root, get_dataset_params
from logger import logger
from nbutils import DirectorySwicther, execute_notebooks

logger.setLevel(logging.INFO)

if __name__ == '__main__':

    # Create command line parser
    parser = ArgumentParser()
    parser.add_argument(
        '-i', '--input', default='single_region_analysis.ipynb', help='path to input notebook')
    parser.add_argument(
        '-o', '--outdir', default='outputs', 
        help='relative path to output directory w.r.t. this script')    
    parser.add_argument(
        '--mpi', default=False, action='store_true', help='enable multiprocessing')
    parser.add_argument('-l', '--line', help='mouse line')
    parser.add_argument('-d', '--date', help='experiment date')
    parser.add_argument('-m', '--mouse', help='mouse number')
    parser.add_argument('-r', '--region', help='brain region')

    # Extract input notebook, output directory and execution parameters from command line arguments
    args = vars(parser.parse_args())
    input_nbpath = args.pop('input')
    outdir = args.pop('outdir')
    mpi = args.pop('mpi')
    
    # Rename parameters to avoid mix-up with other variables during notebook execution 
    args['mouseline'] = args.pop('line')
    args['expdate'] = args.pop('date')
    args['mouseid'] = args.pop('mouse')

    # Extract candidate parameter combinations from folder structure
    pdicts = get_dataset_params(root=get_data_root())

    # Filter to match input parameters
    for k, v in args.items():
        if v is not None:
            logger.info(f'restricting datasets to {k} = {v}')
            pdicts = list(filter(lambda x: x[k] == v, pdicts))
    
    njobs = len(pdicts)
    # Log warning message and quit if no job was found
    if njobs == 0:
        logger.warning('found no job to run')
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
        output_nbpaths = execute_notebooks(pdicts, input_nbpath, outdir, mpi=mpi, ask_confirm=True)
