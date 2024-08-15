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
        '--fps', type=float, help='target sampling rate (Hz)')
    parser.add_argument(
        '--smooth', default=False, action='store_true', help='smooth upon resampling')
    parser.add_argument(
        '-c-', '--correct', type=str, help='global correction method')

    # Parse command line arguments
    args = parser.parse_args()
    kwargs = vars(parser.parse_args())

    # Pre-process Bergamo datasets
    rawdataroot = get_data_root(kind=DataRoot.RAW_BERGAMO)
    preprocess_bergamo_datasets(
        rawdataroot,
        **kwargs,
    )

    logger.info('done')
