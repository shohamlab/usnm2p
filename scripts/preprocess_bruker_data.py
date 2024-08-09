# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2024-08-08 18:02:55
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2024-08-09 10:28:37

''' Pre-process (parse and stack) input dataset(s) from Bruker system '''

# External packages
import os
from argparse import ArgumentParser
from IPython.utils import io

# Internal modules
from usnm2p.constants import DEFAULT_LAYER, FOLDER_EXCLUDE_PATTERNS, DataRoot, Label
from usnm2p.logger import logger
from usnm2p.fileops import get_data_root, get_subfolder_names, get_data_folders, get_dataset_params, save_acquisition_settings
from usnm2p.parsers import resolve_mouseline, parse_bruker_acquisition_settings
from usnm2p.stackers import stack_tifs


def preprocess_bruker_dataset(dataroot, analysis_type, mouseline, expdate, mouseid, region, layer):
    '''
    Pre-process input dataset from Bruker system, i.e.:
        - identify "acquisition" subfolders (i.e., those containing sequences of single-frame TIFs)
        - generate and save multi-frame TIF stacks for all subfolders
        - extract acquisition settings from Bruker XML file for each subfolder
        - save acquisition settings as JSON file in the output stacks directory
    
    :param dataroot: root directory for Bruker input data
    :param analysis_type: analysis type (e.g., 'main', 'offset', 'buzzer')
    :param mouseline: mouse line (e.g., 'line3', 'sst', 'pv')
    :param expdate: experiment date, formatted as 'yyyymmdd' (e.g., '20210804')
    :param mouseid: mouse ID (e.g., 'mouse1')
    :param region: imaged region (e.g., 'region1', 'region2')
    :param layer: imaged cortical layer (e.g., 'layer2-3', 'layer5')
    '''
    # Construct dataset ID
    dataset_id = f'{expdate}_{mouseid}_{region}'
    if layer != DEFAULT_LAYER:
        dataset_id = f'{dataset_id}_{layer}'

    # Extract input data directory
    datadir = os.path.join(dataroot, analysis_type, mouseline, dataset_id)
    if not os.path.exists:
        raise ValueError(f'input data directory "{datadir}" does not exist')
    logger.info(f'processing Bruker input from "{datadir}"')

    # Get raw list of subolders containing tifs, sorted by run ID
    tif_folders = get_data_folders(
        datadir, 
        exclude_patterns=FOLDER_EXCLUDE_PATTERNS, 
        include_patterns=[resolve_mouseline(mouseline)], 
        sortby=Label.RUNID
    )

    # Turn off TIF reading warning
    inkey, outkey = DataRoot.RAW_BRUKER, DataRoot.PREPROCESSED
    with io.capture_output() as captured:  
        # Generate stacks for all TIF folders in the input data directory
        stack_fpaths = [stack_tifs(tf, inkey, outkey, overwrite=False, verbose=False) for tf in tif_folders]

    # Extract acquisition settings from Bruker XML files
    daq_settings = parse_bruker_acquisition_settings(tif_folders)
    
    # Save acquisition settings as JSON file in the output stacks directory
    save_acquisition_settings(os.path.dirname(stack_fpaths[0]), daq_settings)


def preprocess_bruker_datasets(dataroot, analysis=None, mouseline=None):
    '''
    Pre-process a set of input datasets from Bruker system

    :param dataroot: root directory for Bruker input data
    :param analysis: analysis type (e.g., 'main', 'offset', 'buzzer')
    :param mouseline: mouse line (e.g., 'line3', 'sst', 'pv')
    '''
    # If analysis type not specified, extract analysis subfolders from input data directory
    # and call function recursively for each analysis type
    if analysis is None:
        for analysis in get_subfolder_names(dataroot):
            logger.info(f'processing Bruker inputs for "{analysis}" analysis')
            preprocess_bruker_datasets(dataroot=dataroot, analysis=analysis, mouseline=mouseline)
        return

    # List datasets in input data directory, filtered by analysis type
    datasets = get_dataset_params(root=dataroot, analysis_type=analysis)

    # If mouseline specified, restrict datasets to those matching the mouseline
    if mouseline is not None:
        datasets = [d for d in datasets if d['mouseline'] == mouseline]
    
    # Process each dataset
    for dataset in datasets:
        preprocess_bruker_dataset(
            dataroot,
            analysis, 
            dataset['mouseline'], 
            dataset['expdate'], 
            dataset['mouseid'], 
            dataset['region'], 
            dataset['layer']
        )


if __name__ == '__main__':

    # Parse command line arguments
    parser = ArgumentParser(description='Process Bruker input data')
    parser.add_argument(
        '-a', '--analysis', type=str, help='analysis type (e.g., "main", "offset", "buzzer")')
    parser.add_argument(
        '-m', '--mouseline', type=str, help='mouse line (e.g., "line3", "sst", "pv")')
    args = parser.parse_args()
    analysis = args.analysis
    mouseline = args.mouseline

    # Process input datasets from Bruker system
    rawdataroot = get_data_root(kind=DataRoot.RAW_BRUKER)
    preprocess_bruker_datasets(rawdataroot, analysis=analysis, mouseline=mouseline)
    logger.info('done')