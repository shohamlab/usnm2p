# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2024-01-21 16:20:11
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2024-08-07 17:35:37

from argparse import ArgumentParser

from usnm2p.nbutils import execute_notebook
from usnm2p.logger import logger

if __name__ == '__main__':

    # Parse projection function from command line
    parser = ArgumentParser()
    parser.add_argument(
        '-p', '--projfunc', type=str, default='median', 
        choices=['median', 'mean'], help='Project function')
    args = parser.parse_args()

    # Construct parameter dictionary
    pdict = {'projfunc': args.projfunc}

    # Execute notebook
    input_nbpath = '../notebooks/fov_analysis.ipynb'
    outdir = '../outputs'
    execute_notebook(pdict, input_nbpath, outdir)

    # Log termination
    logger.info('done.')