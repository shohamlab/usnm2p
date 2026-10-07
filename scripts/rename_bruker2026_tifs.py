# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2026-09-30 16:32:33
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2026-10-06 23:02:14
from usnm2p.bruker_utils import *
from usnm2p.constants import *
from usnm2p.logger import logger
from usnm2p.fileops import get_data_root
from usnm2p.parsers import parse_date_mouse_region

''' 
Rename single-frame TIFs from Bruker system (new 2026 experiments) to make 
them compatible with the rest of the analysis pipeline. 
'''

# Constant parameters
analysis = 'PRF'  # analysis type
mouseline = 'theo_line3'  # mouse line
fs = 3.58  # acquisition frame rate (Hz)
npertrial = 100  # number of frames per trial
dataroot = get_data_root(kind=DataRoot.RAW_BRUKER)  # root directory for raw Bruker data


# Main script
if __name__ == '__main__':
    # Ask user for analysis type
    analysis = input(f'enter analysis type (default: "{analysis}"): ') or analysis

    # Ask user for mouse line
    mouseline = input(f'enter mouse line (default: "{mouseline}"): ') or mouseline

    # Ask user for dataset directory name
    dataset_id = input('enter dataset directory (usually a "<date>_<mouse>_<region>[_<layer>]" pattern): ')
    
    # Parse mouse ID, experiment date, region and optional layer from dataset name
    try:
        expdate, mouseid, region, layer = parse_date_mouse_region(dataset_id)
    except ValueError as e:
        logger.error(f'error parsing dataset ID: {e}')
        quit()

    # Log dataset parameters
    logmsg = '\n'.join([
        f'  - analysis: {analysis}',
        f'  - mouse line: {mouseline}',
        f'  - experiment date: {expdate}',
        f'  - mouse ID: {mouseid}',
        f'  - region: {region}',
        f'  - layer: {layer}',
        f'  - acquisition frame rate: {fs} Hz',
        f'  - number of frames per trial: {npertrial}',
    ])
    logger.info(f'current dataset:\n{logmsg}')

    # Extract input data directory
    datadir = os.path.join(dataroot, analysis, mouseline, dataset_id)
    if not os.path.exists(datadir):
        raise ValueError(f'input data directory "{datadir}" does not exist')

    # Ask user for confirmation
    answer = input(f'rename TIFs in "{datadir}" directory? (y[es]/n[o]/t[est])?: ')
    answer = answer[0].lower()
    if answer not in ('t', 'y'):
        logger.info('aborted by user')
        quit()
    if answer == 't':
        test = True
        logger.info('test mode: no files will be renamed')
    else:
        test = False
        logger.info('renaming TIFs...')

    # Rename Bruker 2026 TIFs
    rename_bruker_2026_tifs(
        datadir, mouseline, npertrial, fs, test=test)