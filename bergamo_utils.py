# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-08-15 16:34:13
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-08-23 18:30:59

import logging

from constants import *
from utils import get_singleton
from logger import logger
from parsers import get_info_table
from correctors import correct_tifs
from resamplers import resample_tifs
from stackers import stack_trial_tifs, split_multichannel_tifs
from substitutors import StackSubstitutor
from fileops import process_and_save

logger.setLevel(logging.INFO)

''' Utilities to preprocess Bergamo USNM2P data '''


def preprocess_bergamo_dataset(
    input_fpaths, ref_sr, target_sr, is_corrupted=True, mpi=False, overwrite=False, **kwargs):
    ''' 
    Pre-process Bergamo dataset
    
    :param input_fpaths: filepaths to input (raw) stack files
    :param ref_sr: reference sampling rate of the input array (Hz)
    :param target_sr: target sampling rate for the output array (Hz)
    :param is_corrupted: boolean indicating whether dataset is corrupted
    :param mpi: whether to use multiprocessing or not
    :param overwrite: whether to overwrite input files if they exist
    :return: 2-tuple with:
        - filepaths to pre-processed functional stacks (list)
        - number of vframes per trial (int)
    '''
    # If dataset is corrupted, detrend TIF stacks using linear regression 
    input_root = 'raw'
    if is_corrupted:
        input_fpaths = correct_tifs(
            input_fpaths, input_root=input_root, overwrite=overwrite, mpi=mpi, **kwargs)
        input_root = 'corrected'
    # Resample TIF stacks
    resampled_fpaths = resample_tifs(
            input_fpaths, ref_sr, target_sr, input_root=input_root, mpi=mpi, **kwargs)
    # Stack trial TIFs of every run in the stack list
    stacked_fpaths = stack_trial_tifs(
        resampled_fpaths, align=is_corrupted, overwrite=overwrite)
    # Extract number of frames per trial
    raw_info_table = get_info_table(stacked_fpaths)
    nframes_per_trial = get_singleton(raw_info_table, Label.NPERTRIAL)
    logger.info(f'number of frames per trial: {nframes_per_trial}')
    # Split channels from run stacks
    split_fpaths = split_multichannel_tifs(stacked_fpaths, overwrite=overwrite)
    # Substitute problematic frames in every TIF stack and save outputs in specific directory 
    submap = [(1, 0), (FrameIndex.STIM - 1, FrameIndex.STIM)]
    ss = StackSubstitutor(submap, repeat_every=nframes_per_trial)
    input_root = 'split'
    substituted_fpaths = process_and_save(
        ss, split_fpaths, input_root, overwrite=overwrite, mpi=mpi, **kwargs)
    # Keep only files from functional channel
    channel_key = f'channel{FUNC_CHANNEL}'
    func_fpaths = list(filter(lambda x: channel_key in x, substituted_fpaths))
    return func_fpaths, nframes_per_trial