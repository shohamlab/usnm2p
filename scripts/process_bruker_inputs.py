
''' Process (parse and stack) input data from Bruker system '''

# External packages
import os
import json
from argparse import ArgumentParser
from IPython.utils import io

# Internal modules
from usnm2p.constants import DEFAULT_LAYER, FOLDER_EXCLUDE_PATTERNS, Label
from usnm2p.logger import logger
from usnm2p.fileops import get_data_root, get_data_folders, get_dataset_params
from usnm2p.parsers import resolve_mouseline, parse_bruker_acquisition_settings
from usnm2p.stackers import stack_tifs


def process_bruker_input(dataroot, analysis_type, mouseline, expdate, mouseid, region, layer):
    '''
    Process input data from Bruker system, i.e.:
        - identify "acquisition" subfolders (i.e., those containing sequences of single-frame TIFs)
        - extract acquisition settings from Bruker XML file for each subfolder
        - compare settings across subfolders, and exclude folders with "outlier" settings
        - generate and save multi-frame TIF stacks for all remaining subfolders
        - save common dictionary of acquisition settings as JSON file in the output stacks directory
    
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

    # Extract acquisition settings from each run, and outlier runs
    daq_settings, todiscard = parse_bruker_acquisition_settings(tif_folders)

    # If outlier runs were found, exclude corresponding folders from analysis 
    if len(todiscard) > 0:
        discard_str = '\n'.join([f'   - {os.path.basename(f)}' for f in todiscard])
        logger.warning(f'excluding the following folders from analysis:\n{discard_str}')
        tif_folders = [f for f in tif_folders if f not in todiscard]

    # Log final folders list
    folders_str = '\n'.join([f'   - {os.path.basename(f)}' for f in tif_folders])
    logger.info(f'final folders list:\n{folders_str}')

    # Turn off TIF reading warning
    with io.capture_output() as captured:  
        # Generate stacks for all TIF folders in the input data directory
        stack_fpaths = [stack_tifs(tf, overwrite=False, verbose=False, output_key='stackednew') for tf in tif_folders]
    
    # Save acquisition settings as JSON file in the output stacks directory
    stack_dir = os.path.dirname(stack_fpaths[0])
    logger.info(f'saving acquisition settings to "{stack_dir}"')
    daq_settings_fpath = os.path.join(stack_dir, 'daq_settings.json')
    with open(daq_settings_fpath, 'w') as f:
        json.dump(daq_settings.to_dict(), f, indent=4)


def process_bruker_inputs(dataroot, analysis=None, mouseline=None):
    '''
    Process a set of input datasets from Bruker system

    :param dataroot: root directory for Bruker input data
    :param analysis: analysis type (e.g., 'main', 'offset', 'buzzer')
    :param mouseline: mouse line (e.g., 'line3', 'sst', 'pv')
    '''
    # If analysis type not specified, extract analysis subfolders from input data directory
    # and call function recursively for each analysis type
    if analysis is None:
        for analysis in os.listdir(dataroot):
            logger.info(f'processing Bruker inputs for "{analysis}" analysis')
            process_bruker_inputs(dataroot=dataroot, analysis=analysis, mouseline=mouseline)
        return

    # List datasets in input data directory, filtered by analysis type
    datasets = get_dataset_params(root=dataroot, analysis_type=analysis)

    # If mouseline specified, restrict datasets to those matching the mouseline
    if mouseline is not None:
        datasets = [d for d in datasets if d['mouseline'] == mouseline]
    
    # Process each dataset
    for dataset in datasets:
        process_bruker_input(
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
    process_bruker_inputs(get_data_root(), analysis=analysis, mouseline=mouseline)
    logger.info('done')