# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-15 10:13:54
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-11-29 15:07:50

''' Collection of utilities to process fluorescence signals outputed by suite2p. '''

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import find_peaks
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

from constants import *
from logger import logger
from utils import *


def separate_runs(data, nruns):
    '''
    Split fluorescence dataframe into separate runs.

    :param df: multi-index dataframe with 2D (ROI, frame) index
    :param nruns: number of runs
    :return: multi-index dataframe with 3D (ROI, run, frame) index
    '''
    if Label.RUN in data.index.names:
        logger.warning('data already split by run -> ignoring')
        return data
    logger.info(f'splitting fluorescence data into {nruns} separate runs...')
    nROIs, nframes_per_ROI = [len(data.index.unique(level=k)) for k in data.index.names]
    if nframes_per_ROI % nruns != 0:
        raise ValueError(f'specified number of runs {nruns} incompatible with number of frames per ROI ({nframes_per_ROI})')
    iruns = np.arange(nruns)
    nframes_per_run = nframes_per_ROI // nruns
    data[Label.RUN] = np.repeat(np.tile(iruns, (nROIs, 1)), nframes_per_run)
    iframes_per_run = np.arange(nframes_per_run)
    iframes_per_run_ext = np.tile(iframes_per_run, (nROIs * nruns, 1)).flatten()
    data = data.droplevel(Label.FRAME)
    data = data.set_index(Label.RUN, append=True)
    data[Label.FRAME] = iframes_per_run_ext
    data = data.set_index(Label.FRAME, append=True)
    return data
    

def separate_trials(data, ntrials):
    '''
    Split fluorescence dataframe into separate trials.

    :param df: multi-index dataframe with 3D (ROI, run, frame) index
    :param ntrials: number of trials
    :return: multi-index dataframe with 4D (ROI, run, trial, frame) index
    '''
    if Label.TRIAL in data.index.names:
        logger.warning(f'data already split by {Label.TRIAL} -> ignoring')
        return data
    logger.info(f'splitting fluorescence data into {ntrials} separate trials...')
    nROIs, nruns, nframes_per_run = [len(data.index.unique(level=k)) for k in data.index.names]
    if nframes_per_run % ntrials != 0:
        raise ValueError(f'specified number of trials {ntrials} incompatible with number of frames per run ({nframes_per_run})')
    itrials = np.arange(ntrials)
    nframes_per_trial = nframes_per_run // ntrials
    itrials_ext = np.repeat(np.tile(itrials, (nROIs * nruns, 1)), nframes_per_trial)
    data[Label.TRIAL] = itrials_ext
    iframes_per_trial = np.arange(nframes_per_trial)
    iframes_per_trial_ext = np.tile(iframes_per_trial, (nROIs * nruns * ntrials, 1)).flatten()
    data = data.droplevel(Label.FRAME)
    data = data.set_index(Label.TRIAL, append=True)
    data[Label.FRAME] = iframes_per_trial_ext
    data = data.set_index(Label.FRAME, append=True)
    return data


def get_window_size(wlen, fps):
    ''' Compute window size (in number of frames) from window length (in s) and fps. '''
    # Convert seconds to number of frames
    w = int(np.round(wlen * fps))
    # Adjust to odd number if needed
    if w % 2 == 0:
        w += 1
    return w


def compute_baseline(data, fps, wlen, q):
    '''
    Compute the baseline of a signal.

    :param data: multi-indexed Series object contaning the signal of interest
    :param fps: frame rate of the signal (in fps)
    :param wlen: window length (in s) to compute the fluorescence baseline
    :param q: quantile used for the computation of the fluorescence baseline 
    :return: fluorescence baseline series
    '''
    # Compute window size (in number of frames)
    w = get_window_size(wlen, fps)
    # Log process info
    wstr = (f'{wlen:.1f}s ({w} frames) sliding window')    
    qstr = f'{q * 1e2:.0f}{get_integer_suffix(q * 1e2)} percentile'
    logger.info(f'computing signal baseline as {qstr} of {wstr}')
    # Group data by ROI and run, and apply sliding window on F to compute baseline fluorescence
    groupkeys = [Label.ROI, Label.RUN]
    nconds = np.prod([len(data.index.unique(level=k)) for k in groupkeys])
    with tqdm(total=nconds - 1, position=0, leave=True) as pbar:
        def funcwrap(x):
            pbar.update()
            return apply_rolling_window(x.values, w, func=lambda x: x.quantile(q))
        return data.groupby(groupkeys).transform(funcwrap)


def find_response_peak(s, n_neighbors=N_NEIGHBORS_PEAK, return_index=False):
    '''
    Find the response peak (if any) of a signal
    
    :param s: pandas Series containing the signal
    :param n_neighbors: number of neighboring elemtns to include on each side
        to compute average value around the peak
    :param return_index: whether to also return the index of the peak
    '''
    x = s.values
    ipeaks, _ = find_peaks(x)
    if ipeaks.size == 0: # if no peak detected -> return NaN
        ipeak, ypeak = np.nan, np.nan
    else:
        # Get index of max amplitude peak within the array
        ipeak = ipeaks[np.argmax(x[ipeaks])]
        # Make sure it's not at the signal boundary
        if ipeak == 0 or ipeak == x.size - 1:
            raise ValueError(f'max peak found at signal boundary (index {ipeak})')
        # Compute average value of peak and its neighbors
        ypeak = np.mean(x[ipeak - n_neighbors:ipeak + n_neighbors + 1])
    if return_index:
        return ipeak, ypeak
    else:
        return ypeak


def add_time_to_table(data, key=Label.TIME, frame_offset=FrameIndex.STIM):
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
    logger.info('adding time info to table...')
    # Extract sampling frequency
    fps = get_singleton(data, Label.FPS)
    # Extract frame indexes
    iframes = data.index.get_level_values(Label.FRAME)
    # Add time column and remove fps column
    data[key] = (iframes - frame_offset) / fps
    del data[Label.FPS]
    # Set time as first column
    cols = data.columns
    data = data.reindex(columns=[cols[-1], *cols[:-1]])
    return data


def get_response_types_per_ROI(data):
    '''
    Extract the response type per ROI from experiment dataframe.

    :param data: experiment dataframe
    :return: pandas Series of response types per ROI
    '''
    logger.info('extracting responses types per ROI...')
    return data.groupby(Label.ROI).first()[Label.ROI_RESP_TYPE]


def get_trial_averaged(data, full_output=False):
    '''
    Compute trial-averaged statistics
    
    :param data: multi-indexed (ROI x run x trial) statistics dataframe
    :return: (trialavg_data, is_repeat) tuple
    '''
    # Group trials
    groups = data.groupby([Label.ROI, Label.RUN])
    # Compute average of stat across trials
    trialavg_data = groups.agg(mean_str)
    if isinstance(trialavg_data, pd.DataFrame):  # DataFrame case
        # Remove time column if present
        if Label.TIME in trialavg_data:
            del trialavg_data[Label.TIME]
        # Rename relevant input columns to their trial-averaged meaning
        cols = {}
        for k, v in Label.RENAME_ON_AVERAGING.items():
            if k in trialavg_data:
                cols[k] = v
        if len(cols) > 0:
            trialavg_data.rename(columns=cols, inplace=True)
    else:  # Series case
        # Rename input to its trial-average meaning if necessary
        if trialavg_data.name in Label.RENAME_ON_AVERAGING.keys():
            trialavg_data.name = Label.RENAME_ON_AVERAGING[trialavg_data.name]
    if full_output:
        # Compute average of stat across trials
        trialstd_data = groups.std()
        # Determine whether stats is a repeated value or a real distribution
        is_repeat = ~(trialstd_data.max() > 0)
        return trialavg_data, is_repeat
    else:
        return trialavg_data


def weighted_average(data, avg_name, weight_name):
    '''
    Compute a weighted-average of a particular column of a dataframe using the weights
    of another column.
    
    :param data: dataframe
    :param avg_name: name of the column containing the values to average
    :param weight_name: name of the coolumn containing the weights
    :return: weighted average
    '''
    d = data[avg_name]
    w = data[weight_name]
    return (d * w).sum() / w.sum()


def filter_data(data, iROI=None, irun=None, itrial=None, rtype=None, P=None, DC=None, tbounds=None, full_output=False):
    ''' Filter data according to specific criteria.
    
    :param data: experiment dataframe
    :param iROI (optional): ROI index(es)
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

    ###################### Sub-indexing ######################
    logger.info('sub-indexing data...')
    subindex = [slice(None)] * 3
    if iROI is not None:
        subindex[0] = iROI
        filters[Label.ROI] = f'{Label.ROI}{plural(iROI)} {iROI}'
    if irun is not None:
        subindex[1] = irun
        filters[Label.RUN] = f'{Label.RUN}{plural(irun)} {irun}'
    if itrial is not None:
        subindex[2] = itrial
        filters[Label.TRIAL] = f'{Label.TRIAL}{plural(itrial)} {itrial}'
    # Check for each sub-index level that elements are in global index
    for k, v in zip(data.index.names, subindex):
        if is_iterable(v) or v != slice(None):
            subset, globalset = set(as_iterable(v)), set(data.index.unique(level=k))
            if not subset.issubset(globalset):
                missings = list(subset - globalset)
                raise ValueError(f'{k}{plural(missings)} {missings} not found in dataset index')
    data = data.loc[tuple(subindex)]

    ###################### Filtering ######################
    logger.info('filtering data...')
    # Initialize global inclusion criterion
    include = np.ones(len(data)).astype(bool)
    # Refine inclusion criterion based on response type
    if rtype is not None:
        if iROI is not None:  # cannot filter on response type if ROI index is provided
            raise ValueError(f'only 1 of "iROI" and "rtype" can be provided')
        include = include & (data[Label.ROI_RESP_TYPE] == rtype)
        filters[Label.ROI_RESP_TYPE] = f'{rtype} ROIs'
    # Refine inclusion criterion based on stimulation parameters
    if P is not None:
        include = include & (data[Label.P] == P)
        filters[Label.P] = (f'P = {P} MPa')
    if DC is not None:
        include = include & (data[Label.DC] == DC)
        filters[Label.DC] = f'DC = {DC} %'
    # Refine inclusion criterion based on time range (not added to filters list because obvious)
    if tbounds is not None:
        include = include & (data[Label.TIME] >= tbounds[0]) & (data[Label.TIME] <= tbounds[1])
    # Slice data according to filters
    data = data[include]

    ###################### Filters completion ######################
    logger.info('cross-checking filters...')
    # Single run selected -> indicate corresponding stimulation parameters
    if irun is not None and P is None and DC is None:
        if Label.P in data and Label.DC in data:
            try:
                parsed_P, parsed_DC = get_singleton(data, [Label.P, Label.DC])
                filters[Label.RUN] += f' (P = {parsed_P} MPa, DC = {parsed_DC} %)'
            except ValueError as err:
                logger.warning(err)
    # No ROI selected -> indicate number of ROIs
    if iROI is None:
        nROIs = len(data.index.unique(level=Label.ROI).values)
        filters['nROIs'] = f'({nROIs} ROIs)'

    # Set filters to None if not filter was applied 
    if len(filters) == 0:
        filters = None
    
    # Conditional return
    if full_output:
        return data, filters
    else:
        return data
    

def groupby_and_all(data, func, groupby=None):
    '''
    Wrapper around pandas "groupby" that applies a function on the input dataset for
    each sub-group of the groupby category, but also on the entire dataset.
    
    :param data: dataframe
    :param func: function to apply to the dataset and each sub-dataset
    :param groupby (optional): variable defining the sub-groups
    :retrun: dictionary of function output per sub-group and for the entire dataset (key "all")
    '''
    out = {'all': func(data)}
    if groupby is not None:
        for cond, cond_data in data.groupby(groupby):
            out[cond] = func(cond_data)
    return out


def get_clustered_index(data, metric='euclidean', method='single'):
    '''
    Compute a clustered index list according observations across dimensions
    
    :param data: (observations x dimensions) dataframe for some specific variable.
    :return: index of observations outputed by the clustering algorithm
    '''
    logger.info('computing new index according to hierarchical clustering...')
    # Get dataset index
    index = data.index
    # Compute pairwise distance matrix
    Y = pdist(data, metric=metric, out=None)
    # If NaNs in matrix -> return original index
    if np.isnan(np.sum(Y)):
        logger.warning('cannot clusterize dataset with NaNs -> ignoring')
        return index
    # Cluster hierarchically using the pairwise distance matrix
    Z = linkage(Y, method=method, optimal_ordering=False)
    # Return index list from cluster output 
    return index[leaves_list(Z)]


def clusterize_data(data, *kwargs):
    '''
    Re-arrange dataset along the ROI dimension according observations across runs 
    
    :param data: (nROIs x nruns) dataframe for some specific variable.
    :return: dataframe re-indexed alonmg according to ROI clustering process 
    '''
    iROIs_clustered = get_clustered_index(data, **kwargs)
    logger.info('re-arranging dataset...')
    return data.reindex(iROIs_clustered)