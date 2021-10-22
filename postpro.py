# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-15 10:13:54
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-10-22 18:00:59

import numpy as np
from scipy.stats import zscore
import pandas as pd

from constants import *
from logger import logger
from utils import get_singleton, expand_along, is_in_dataframe, is_iterable

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


def add_cells_to_table(data, cell_ROI_idx):
    '''
    Add cells info to table.

    :param data: dataframe contanining all the info about the experiment.
    :param cell_ROI_idx: list of ROI indexes corresponding to each cell.
    :return: expanded pandas dataframe with cell info
    '''
    if is_in_dataframe(data, 'cell'):
        return data
    data = expand_along(data, 'roi', cell_ROI_idx, index_key='cell')
    return data.reorder_levels(['cell', 'run']).sort_index()


def add_trials_to_table(data, ntrials=None):
    '''
    Add trials info to table.

    :param data: dataframe contanining all the info about the experiment.
    :return: expanded pandas dataframe with cell info
    '''
    if is_in_dataframe(data, 'trial'):
        return data
    if ntrials is None:
        ntrials = get_singleton(data, NTRIALS_LABEL)
        del data[NTRIALS_LABEL]
    return expand_along(data, 'trial', np.arange(ntrials), index_key='trial')


def add_signal_to_table(data, key, y, index_key='frame'):
    '''
    Add signal to info table.

    :param data: dataframe contanining all the info about the experiment.
    :param key: name of the column that will hold the new data
    :param y: array of signals (signals timecourse must be evolving along last array axis)
    :param index_key (optional): name of new index level to add to dataframe upon expansion
    :return: modified info table
    '''
    # Extract trial length from dataframe
    if index_key is not None and index_key in data.index.names:
        npertrial = len(set(data.index.get_level_values(index_key)))
    else: 
        npertrial = get_singleton(data, NPERTRIAL_LABEL, delete=True)
    return expand_along(data, key, y, nref=npertrial, index_key=index_key)


def add_time_to_table(data, key=TIME_LABEL):
    '''
    Add time information to info table
    
    :param data: dataframe contanining all the info about the experiment.
    :param key: name of the time column in the new info table
    :param index_key (optional): name of index level to use as reference to compute the time vector 
    :return: modified info table
    '''
    if key in data:
        logger.warning(f'"{key}" column is already present in dataframe -> ignoring')
        return data
    # Extract sampling frequency
    fps = get_singleton(data, FPS_LABEL)
    # Extract frame indexes
    iframes = data.index.get_level_values('frame')
    # Add time column and remove fps column
    data[key] = (iframes - STIM_FRAME_INDEX) / fps
    del data[FPS_LABEL]
    return data


def array_to_dataframe(x, key):
    '''
    Convert a (nsignals, npersignal) 2D array into a timeseries dataframe

    :param x: input array
    :param key: variable name to give to the signal
    :return dataframe object
    '''
    ntrials, npertrial = x.shape
    data = pd.DataFrame({FPS_LABEL: [FPS], NPERTRIAL_LABEL: [npertrial]})
    data = add_trials_to_table(data, ntrials=ntrials)
    data = add_signal_to_table(data, key, x)
    data = add_time_to_table(data)
    return data


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


def filter_data(data, icell=None, irun=None, itrial=None, rtype=None, P=None, DC=None, tbounds=None, full_output=False):
    ''' Filter data according to specific criteria.
    
    :param data: experiment dataframe
    :param icell (optional): cell index(es)
    :param irun: run index(es)
    :param itrial: trial index(es)
    :param rtype (optional): response type code(s), within (-1, 0, 1)
    :param P (optional): pressure amplitude value(s) (MPa)
    :param DC (optional): duty cycle value(s) (%)
    :param tbounds (optional): time limits
    :param full_output (optional): whether to return also a dictionary of filter labels.
    :return: filtered dataframe (and potential filters dictionary)
    '''
    # Initialize empty filters dictionary
    filters = {}    
    # Filter data based on provided cell, run and trial indexes
    logger.info('sub-indexing data...')
    subindex = [slice(None)] * 3
    plural = lambda x: 's' if is_iterable(x) else ''
    if icell is not None:
        subindex[0] = icell
        filters['cell'] = f'cell{plural(icell)} {icell}'
    if irun is not None:
        subindex[1] = irun
        filters['run'] = f'run{plural(irun)} {irun}'
    if itrial is not None:
        subindex[2] = itrial
        filters['trial'] = f'trial{plural(itrial)} {itrial}'
    data = data.loc[tuple(subindex)]

    logger.info('filtering data...')
    # Initialize global inclusion criterion
    include = np.ones(len(data)).astype(bool)
    # Refine inclusion criterion based on response type
    if rtype is not None:
        if icell is not None:  # cannot filter on response type if cell index is provided
            raise ValueError(f'only 1 of "icell" and "rtype" can be provided')
        include = include & (data[RESP_LABEL] == rtype)
        filters[RESP_LABEL] = f'{LABEL_BY_TYPE[rtype]} cells'
    # Refine inclusion criterion based on stimulation parameters
    if P is not None:
        include = include & (data[P_LABEL] == P)
        filters[P_LABEL] = (f'P = {P} MPa')
    if DC is not None:
        include = include & (data[DC_LABEL] == DC)
        filters[DC_LABEL] = f'DC = {DC} %'
    # Refine inclusion criterion based on time range (not added to filters list because obvious)
    if tbounds is not None:
        include = include & (data[TIME_LABEL] >= tbounds[0]) & (data[TIME_LABEL] <= tbounds[1])
    # Slice data according to filters
    data = data[include]

    # Complete labels based on selected data
    # Cell(s) selected -> indicate response type(s) 
    if icell is not None and rtype is None:
        parsed_rtype = data.groupby('cell').first()[RESP_LABEL][icell]
        rcode = [LABEL_BY_TYPE[x] for x in parsed_rtype] if is_iterable(parsed_rtype) else [LABEL_BY_TYPE[parsed_rtype]]
        filters['cell'] += f' ({", ".join(list(set(rcode)))})'
        # Single run selected -> indicate corresponding stimulation parameters
        if irun is not None and P is None and DC is None:
            parsed_P, parsed_DC = get_singleton(data, [P_LABEL, DC_LABEL])
            filters['run'] += f' (P = {parsed_P} MPa, DC = {parsed_DC} %)'
    # No cell selected -> indicate number of cells
    if icell is None:
        ncells = len(set(data.index.get_level_values('cell')))
        filters['ncells'] = f'({ncells} cells)'

    # Set filters to None if not filter was applied 
    if len(filters) == 0:
        filters = None
    
    # Conditional return
    if full_output:
        return data, filters
    else:
        return data
    