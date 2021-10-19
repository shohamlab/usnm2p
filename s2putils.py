# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-14 19:25:20
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-10-18 11:50:57

import os
import numpy as np
import matplotlib.pyplot as plt
from suite2p import run_s2p
from suite2p.io import BinaryFile
from colorsys import hsv_to_rgb

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

from constants import *
from logger import logger
from fileops import parse_overwrite

''' Collection of utilities related to Suite2P operations. '''


def run_suite2p(*args, overwrite=True, **kwargs):
    '''
    Wrapper around run_s2p function that first checks for existence of suite2p output files
    and runs only if files are absent or if user allowed overwrite.
    '''
    suite2p_keys = ['iscell', 'stat', 'F', 'Fneu', 'spks']
    # For each input directory
    for inputdir in kwargs['db']['data_path']:
        # Check for existence of suite2p subdirectory
        suite2pdir = os.path.join(inputdir, 'suite2p', 'plane0')
        if os.path.isdir(suite2pdir):
            # Check for existence of any suite2p output file
            if all(os.path.isfile(os.path.join(suite2pdir, f'{k}.npy')) for k in suite2p_keys):
                # Warn user if any exists, and act according to defined overwrite behavior
                logger.warning(f'suite2p output files already exist in "{suite2pdir}"')
                if not parse_overwrite(overwrite):
                    return
    # If execution was not canceled, run the standard function
    return run_s2p(*args, **kwargs)


def get_suite2p_data(dirpath, cells_only=False, withops=False):
    '''
    Locate suite2p output files given a specific output directory, and load them into a dictionary.
    
    :param dirpath: full path to the output directory containing the suite2p data files.
    :param cells_only: boolean stating whether to filter out non-cell entities from dataset.
    :param withops: whether include a dictionary of options and intermediate outputs.
    :return: suite2p output dictionary.
    '''
    if not os.path.isdir(dirpath):
        raise ValueError(f'"{dirpath}" directory does not exist')
    keys = ['iscell', 'stat', 'F', 'Fneu', 'spks']
    data = {k : np.load(os.path.join(dirpath, f'{k}.npy'), allow_pickle=True) for k in keys} 
    # If specified, restrict dataset to cells only
    if cells_only:
        iscell = data.pop('iscell')[:, 0]  # the full "iscell" has a second column with the probability of being a cell
        cell_idx = np.array(iscell.nonzero()).reshape(-1)
        data = {k : v[cell_idx] for k, v in data.items()}
    if withops:
        data['ops'] = np.load(os.path.join(dirpath, f'ops.npy'), allow_pickle=True).item()
    return data


def view_registered_stack(output_ops):
    widget = widgets.IntSlider(
        value=7,
        min=0,
        max=10,
        step=1,
        description='Test:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d'
    )

    def plot_frame(t):
        with BinaryFile(Ly=REF_LY, Lx=REF_LX, read_filename=output_ops['reg_file']) as f:
            plt.imshow(f[t][0])

    interact(plot_frame, t=(0, REF_NFRAMES, 1))
