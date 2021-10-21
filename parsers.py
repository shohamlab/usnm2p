# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-14 19:29:19
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-10-20 18:39:11

import re
import os
import pandas as pd
from constants import *

''' Collection of parsing utilities. '''

# General tif file pattern
P_TIFFILE = re.compile('.*tif')

P_RAWFOLDER = re.compile(f'^{P_LINE}_{P_TRIAL_LENGTH}_{P_FREQ}_{P_DUR}_{P_FREQ}_{P_MPA}_{P_DC}-{P_RUN}$', re.IGNORECASE)
P_RAWFILE = re.compile(f'{P_RAWFOLDER.pattern[:-1]}_{P_CYCLE}_{P_CHANNEL}_{P_FRAME}.ome.tif$', re.IGNORECASE)
P_STACKFILE = re.compile(f'{P_RAWFOLDER.pattern[:-1]}.tif$', re.IGNORECASE)


def parse_experiment_parameters(name):
    '''
    Parse experiment parameters from a file/folder name.
    
    :param name: file / folder name from which parameters must be extracted
    :return: dictionary of extracted parameters
    '''
    # Try first with folder pattern
    israwfile = False
    mo = P_RAWFOLDER.match(name)
    # If no match detected, try with file pattern
    if mo is None:
        israwfile = True
        mo = P_RAWFILE.match(name)
    # If still no match, try with stack file pattern
    if mo is None:
        israwfile = False
        mo = P_STACKFILE.match(name)
    # If still no match, throw error
    if mo is None:
        raise ValueError(f'"{name}" does not match the experiment naming pattern')
    # Extract and parse folder-level parameters
    params = {
        LINE_LABEL: mo.group(1),  # line name
        NPERTRIAL_LABEL: int(mo.group(2)),
        UNKNOWN: float(mo.group(3)),
        DUR_LABEL: float(mo.group(4)) * 1e-3,  # s
        FPS_LABEL: float(mo.group(5)),  
        P_LABEL: mo.group(6),  # MPa
        DC_LABEL: float(mo.group(7)),  # %
        RUN_LABEL: int(mo.group(8))
    }
    # Fix for pressure (replacing first zero by decimal dot)
    if '.' not in params[P_LABEL]:
        params[P_LABEL] = float(f'.{params[P_LABEL][1:]}')
    # If file, add file-level parameters
    if israwfile:
        params.update({
            CYCLE_LABEL: int(mo.group(9)),
            CH_LABEL: int(mo.group(10)),
            FRAME_LABEL: int(mo.group(11))
        })
    # Return parameters dictionary
    return params


def get_info_table(folders, index_key='run', ntrials_per_run=None, discard_unknown=True):
    '''
    Parse a list of input folders and aggregate extracted parameters into an info table.
    
    :param folders: list of absolute paths to data folders
    :param index_key (optional): name to give to the dataframe index.
    :param ntrials_per_run (optional): number of trials per run (added as an extra column if not none).
    :param discard_unknown (optional): whether to discard unknown (???) keys from table
    :return: pandas dataframe with parsed parameters for each folder.
    '''
    basenames = [os.path.basename(x) for x in folders]
    pdicts = [parse_experiment_parameters(x) for x in basenames]
    info_table = pd.DataFrame(pdicts)
    info_table['code'] = basenames
    if index_key is not None:
        info_table.index.name = index_key
    if ntrials_per_run is not None:
        info_table[NTRIALS_LABEL] = ntrials_per_run
    if discard_unknown:
        del info_table[UNKNOWN]
    return info_table
