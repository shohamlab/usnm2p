# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-08-15 16:34:13
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2024-08-09 14:34:19

''' Pre-process input dataset(s) from Bergamo system '''

# External packages
from argparse import ArgumentParser

# Internal modules
from usnm2p.constants import *
from usnm2p.logger import logger
from usnm2p.fileops import get_data_root
from usnm2p.parsers import add_dataset_arguments
from usnm2p.bergamo_utils import preprocess_bergamo_datasets


if __name__ == '__main__':

    # Create command line parser
    parser = ArgumentParser()

    # Add dataset arguments
    add_dataset_arguments(parser)

    # Processing arguments
    parser.add_argument(
        '--mpi', default=False, action='store_true', help='enable multiprocessing')
    parser.add_argument(
        '-o', '--overwrite', default=False, action='store_true', help='overwrite existing files')
    parser.add_argument(
        '-c', '--corrupted', default=False, action='store_true', help='corrupted dataset to be detrended')
    parser.add_argument(
        '--refsr', help='reference sampling rate (Hz)', default=BERGAMO_SR)
    parser.add_argument(
        '--targetsr', help='target sampling rate (Hz)', default=BRUKER_SR)

    # Parse command line arguments
    args = parser.parse_args()
    kwargs = vars(parser.parse_args())

    # Resampling parameters
    ref_sr = kwargs.pop('refsr')  # Hz
    target_sr = kwargs.pop('targetsr')  # Hz

    # Pre-process Bergamo datasets
    rawdataroot = get_data_root(kind=DataRoot.RAW_BERGAMO)
    preprocess_bergamo_datasets(
        rawdataroot,
        ref_sr, target_sr, 
        **kwargs,
    )

    logger.info('done')
