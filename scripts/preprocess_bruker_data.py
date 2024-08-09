# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2024-08-08 18:02:55
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2024-08-09 14:12:36

''' Pre-process (parse and stack) input dataset(s) from Bruker system '''

# External packages
from argparse import ArgumentParser

# Internal modules
from usnm2p.bruker_utils import preprocess_bruker_datasets
from usnm2p.constants import DataRoot
from usnm2p.logger import logger
from usnm2p.fileops import get_data_root
from usnm2p.parsers import add_dataset_arguments


if __name__ == '__main__':

    # Parse command line arguments
    parser = ArgumentParser(description='Process Bruker input data')
    add_dataset_arguments(parser)
    kwargs = vars(parser.parse_args())
    analysis = kwargs.pop('analysis')

    # Pre-process Bruker datasets
    rawdataroot = get_data_root(kind=DataRoot.RAW_BRUKER)
    preprocess_bruker_datasets(rawdataroot, analysis=analysis, **kwargs)
    logger.info('done')