# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-15 10:13:54
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-10-21 18:00:08

import numpy as np
from scipy.stats import zscore
import pandas as pd

from constants import *
from logger import logger
from utils import get_singleton, expand_along, is_in_dataframe

''' Collection of utilities to process fluorescence signals outputed by suite2p. '''


def separate_runs(x, nruns):
    '''
    Split suite2p data into separate runs.

    :param x: 2D (ncells, nframes) suite2p data array
    :param nruns: number of runs
    :return: 3D (ncells, nruns, nperrun) data array
    '''
    # If only single run, add extra "run" dimension
    if nruns == 1:
        return np.expand_dims(x, axis=1)
    # Extract dimensions and reshape
    ncells, nframes = x.shape
    return x.reshape((ncells, nruns, -1))


def separate_trials(x, ntrials):
    '''
    Split suite2p data array into separate trials.
    
    :param x: 3D (ncells, nruns, nperrun) data array
    :param ntrials: number of trials
    :return: 4D (ncells, nruns, ntrials, npertrial) data array
    '''
    # If only single 2D array provided, add extra "run" dimension
    if x.ndim == 2:
        x = np.expand_dims(x, axis=1)
    # Extract dimensions and reshape
    ncells, nruns, nperrun = x.shape
    npertrial = nperrun // ntrials
    return x.reshape(x.shape[:-1] + (ntrials, npertrial))


def add_cells_to_table(df, cell_ROI_idx):
    '''
    Add cells info to table.

    :param df: dataframe contanining all the info about the experiment.
    :param cell_ROI_idx: list of ROI indexes corresponding to each cell.
    :return: expanded pandas dataframe with cell info
    '''
    if is_in_dataframe(df, 'cell'):
        return df
    df = expand_along(df, 'roi', cell_ROI_idx, index_key='cell')
    return df.reorder_levels(['cell', 'run']).sort_index()


def add_trials_to_table(df, ntrials=None):
    '''
    Add trials info to table.

    :param df: dataframe contanining all the info about the experiment.
    :return: expanded pandas dataframe with cell info
    '''
    if is_in_dataframe(df, 'trial'):
        return df
    if ntrials is None:
        ntrials = get_singleton(df, 'ntrials')
    return expand_along(df, 'trial', np.arange(ntrials), index_key='trial')


def add_signal_to_table(df, key, y, index_key='frame'):
    '''
    Add signal to info table.

    :param df: dataframe contanining all the info about the experiment.
    :param key: name of the column that will hold the new data
    :param y: array of signals (signals timecourse must be evolving along last array axis)
    :param index_key (optional): name of new index level to add to dataframe upon expansion
    :return: modified info table
    '''
    # Extract trial length from dataframe
    if index_key is not None and index_key in df.index.names:
        npertrial = len(set(df.index.get_level_values(index_key)))
    else: 
        npertrial = get_singleton(df, NPERTRIAL_LABEL, delete=True)
    return expand_along(df, key, y, nref=npertrial, index_key=index_key)


def add_time_to_table(df, key=TIME_LABEL):
    '''
    Add time information to info table
    
    :param df: dataframe contanining all the info about the experiment.
    :param key: name of the time column in the new info table
    :param index_key (optional): name of index level to use as reference to compute the time vector 
    :return: modified info table
    '''
    if key in df:
        logger.warning(f'"{key}" column is already present in dataframe -> ignoring')
        return df
    # Extract sampling frequency
    fps = get_singleton(df, FPS_LABEL)
    # Extract frame indexes
    iframes = df.index.get_level_values('frame')
    # Add time column and remove fps column
    df[key] = (iframes - STIM_FRAME_INDEX) / fps
    del df[FPS_LABEL]
    return df


def array_to_dataframe(x, key):
    '''
    Convert a (nsignals, npersignal) 2D array into a timeseries dataframe

    :param x: input array
    :param key: variable name to give to the signal
    :return dataframe object
    '''
    ntrials, npertrial = x.shape
    df = pd.DataFrame({FPS_LABEL: [FPS], NPERTRIAL_LABEL: [npertrial]})
    df = add_trials_to_table(df, ntrials=ntrials)
    df = add_signal_to_table(df, key, x)
    df = add_time_to_table(df)
    return df


def get_relative_fluorescence_change(F, ibaseline):
    '''
    Calculate relative fluorescence signal (dF/F0) from an absolute fluorescence signal (F).
    The signal baseline is calculated on a per-trial basis as the average of a specified
    pre-stimiulus interval.

    :param x: 4D (ncells, nruns, ntrials, npertrial) array of absolute fluorescence signals
    :param ibaseline: baseline evaluation indexes
    :return: 4D (ncells, nruns, ntrials, npertrial) array of relative change in fluorescence
    '''
    # If only single 3D array provided, add extra "run" dimension
    if F.ndim == 3:
        F = np.expand_dims(F, axis=1)
    # Extract F dimensions
    ncells, nruns, ntrials, npertrial = F.shape
    # Extract baseline fluoresence signals and average across time for each cell and trial 
    F0 = F[:, :, :, ibaseline].mean(axis=-1)  # (ncells, nruns, ntrials) array
    # Add 4th axis (of dim 1) to F0 to enable broadcasting with F
    F0 = np.expand_dims(F0, axis=3)
    # Return relative change in fluorescence
    return (F - F0) / F0


def compute_z_scores(x):
    '''
    Compute z-scores (i.e. number of standard deviations from the mean) of a signal array on a per-run basis.

    :param x: 4D (ncells, nruns, ntrials, npertrial) signal array
    :return: 4D (ncells, nruns, ntrials, npertrial) array of z-scores
    '''
    # Get array dimensions
    ncells, nruns, ntrials, npertrial = x.shape
    # Stack trials that belong to the same run sequentially
    x = x.reshape((ncells, nruns, ntrials * npertrial))
    # Compute z-scores along entire runs
    z = zscore(x, axis=-1)
    # Separate trials in resulting z-score distribution
    return z.reshape((ncells, nruns, ntrials, npertrial))


def classify_by_response_type(dFF, full_output=False):
    '''
    Classify cells by response type, based on analysis of z-score distributions of
    relative fluorescence change signals.

    :param dFF: 4D (ncells, nruns, ntrials, npertrial) array of relative change in fluorescence
    :param full_output (optional): whether to return also the distribution of identified peak z-scores per cell.
    :return: list of response type (-1: negative, 0: neutral, +1: positive) for each cell
    '''
    # Compute z-scores on a per-run basis
    logger.info('computing z-score distributions')
    z = compute_z_scores(dFF)
    # Restrict analysis to z-scores within the "response interval"
    z_resp = z[:, :, :, I_RESPONSE]
    # Average across trials to obtain mean response z-score timecourse per cell and run
    logger.info('averaging')
    z_resp_run_avg = z_resp.mean(axis=2)
    # Compute min and max z-score across response AND across runs for each given cell
    z_resp_min_per_cell = z_resp_run_avg.min(axis=-1).min(axis=-1)
    z_resp_max_per_cell = z_resp_run_avg.max(axis=-1).max(axis=-1)
    # Classify cells according to their max and min z-scores
    is_positive = z_resp_max_per_cell >= ZSCORE_THR_POSITIVE
    is_negative = np.logical_and(~is_positive, z_resp_min_per_cell <= ZSCORE_THR_NEGATIVE)
    is_neutral = np.logical_and(~is_positive, ~is_negative)
    # Cast bool -> int
    is_positive, is_negative, is_neutral = [x.astype(int) for x in [is_positive, is_negative, is_neutral]] 
    # Make sure response categories have mutually exclusive populations
    assert all(is_positive + is_negative + is_neutral == 1.), 'error'
    # Compute vector of response type per cell
    resp_types = is_positive - is_negative
    # Return response types vector and optional z-distributions
    if full_output:
        return resp_types, (z_resp_min_per_cell, z_resp_max_per_cell)
    else:
        return resp_types 
