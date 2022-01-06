# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-12-29 12:43:46
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-01-06 11:48:01

import os
import logging
from argparse import ArgumentParser

from fileops import get_data_root, get_date_mouse_region_combinations
from logger import logger
from nbutils import DirectorySwicther, execute_notebooks

logger.setLevel(logging.INFO)

''' Utility script to run single region analysis notebook '''

# Input & output
input_nbpath = 'single_region_analysis.ipynb'  # path to input notebook 
outdir = 'outputs'  # relative path to output directory (w.r.t. this script)

# Default parameters
defaults = {
    'line': 'line3',  # mouse line
    'mouse': 'mouse12',  # mouse number
    'region': 'region1',  # brain region
    'date': '11122019'  # experiment date
}

if __name__ == '__main__':

    # Create command line parser
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', default=input_nbpath, help='input notebook path')
    parser.add_argument('-o', '--outdir', default=outdir, help='output directory path')
    for k, v in defaults.items():
        parser.add_argument(f'-{k[0]}', f'--{k}', default=v)
    parser.add_argument('--locate', default=False, action='store_true')
    parser.add_argument('--mpi', default=False, action='store_true')

    # Extract input notebook, output diectory and execution parameters from command line arguments
    args = vars(parser.parse_args())
    input_nbpath = args.pop('input')
    outdir = args.pop('outdir')
    locate = args.pop('locate')
    mpi = args.pop('mpi')
    if locate:
        # If locate was turned on, extract parameter combinations from folder structure
        pdicts = get_date_mouse_region_combinations(root=get_data_root())
    else:
        # Otherwise, construct unique combination from inputs
        pdicts = [args]

    # Set multiprocessing to False in case of single execution
    if len(pdicts) == 1:
        mpi = False

    # Get absolute path to directory of current file (where code must be executed)
    script_fpath = os.path.realpath(__file__)
    exec_dir = os.path.split(script_fpath)[0]

    # Execute notebooks within execution directory (to ensure correct function)
    with DirectorySwicther(exec_dir) as ds:
        # Execute notebooks as a batch with / without multiprocessing
        output_nbpaths = execute_notebooks(pdicts, input_nbpath, outdir, mpi=mpi, ask_confirm=True)
