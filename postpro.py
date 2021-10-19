# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-15 10:13:54
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-10-18 22:38:27

import numpy as np
from scipy.stats import zscore

from constants import *
from logger import logger
from plotters import plot_zscore_distributions


''' Collection of utilities to process fluorescence signals outputed by suite2p. '''


def moving_average(x, n=5):
    ''' Apply a monving average on a signal. '''
    if n % 2 == 0:
        raise ValueError('you must specify an odd MAV window length')
    return np.convolve(np.pad(x, n // 2, mode='symmetric'), np.ones(n), 'valid') / n


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


def bound_in_time(x, fs, tbounds):
    '''
    Restrict suite2p data array to a specific time interval per trial.
    
    :param x: 4D (ncells, nruns, ntrials, npertrial) data array
    :param fs: sampling frequency
    :param tbounds: time limits (in s, relative to the stimulus onset)
    :return: 4D (ncells, nruns, ntrials, npertrial) data array, restricted along the time dimension
    '''
    # If only single 3D array provided, add extra "run" dimension
    if x.ndim == 3:
        x = np.expand_dims(x, axis=1)
    # Convert time bounds to frame index bounds (adding stimulus onset)
    ibounds = np.asarray(tbounds) * fs + STIM_FRAME_INDEX
    # Compute valid indexes along time dimension of x array
    indexes = np.arange(x.shape[-1])
    isvalid = np.logical_and(indexes >= ibounds[0], indexes <= ibounds[1])
    # Restrict data array to valid indexes along time dimension, and return
    return x[:, :, :, isvalid]


def get_relative_fluorescence_change(F, ibaseline):
    '''
    Calculate relative fluorescence signal (dF/F0) from an absolute fluorescence signal (F0).
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


def classify_by_response_type(dFF):
    '''
    Classify cells by response type, based on analysis of z-score distributions of
    relative fluorescence change signals.

    :param dFF: 4D (ncells, nruns, ntrials, npertrial) array of relative change in fluorescence
    :return: list of response class (-1: negative, 0: neutral, +1: positive) for each cell
    '''
    # Compute z-scores on a per-run basis
    logger.info('computing z-score distributions')
    z = compute_z_scores(dFF)
    # Restrict analysis to z-scores within the "response interval"
    z_resp = z[:, :, :, I_RESPONSE]
    # Average across trials to obtain mean response z-score timecourse per cell and run
    logger.info('averaging')
    z_resp_run_avg = z_resp.mean(axis=2)
    print(z_resp_run_avg.shape)
    # Compute min and max z-score across response AND across runs for each given cell
    z_resp_min_per_cell = z_resp_run_avg.min(axis=-1).min(axis=-1)
    z_resp_max_per_cell = z_resp_run_avg.max(axis=-1).max(axis=-1)
    fig = plot_zscore_distributions(z_resp_min_per_cell, z_resp_max_per_cell)
    # Classify cells according to their max and min z-scores
    is_positive = z_resp_max_per_cell >= ZSCORE_THR_POSITIVE
    is_negative = np.logical_and(~is_positive, z_resp_min_per_cell <= ZSCORE_THR_NEGATIVE)
    is_neutral = np.logical_and(~is_positive, ~is_negative)
    # Cast bool -> int
    is_positive, is_negative, is_neutral = [x.astype(int) for x in [is_positive, is_negative, is_neutral]] 
    # Make sure response categories have mutually exclusive populations
    assert all(is_positive + is_negative + is_neutral == 1.), 'error'
    # Return vector of response type class per cell
    return is_positive - is_negative


def exponential_kernel(t, tau=1, pad=True):
    '''
    Generate exponential kernel.

    :param t: 1D array of time points
    :param tau: time constant for exponential decay of the indicator
    :param pad: whether to pad kernel at init with length = len(t)
    :return: a kernel or generative decaying exponenial function
    '''
    hemi_kernel = 1 / tau * np.exp(-t / tau)
    if pad:
        len_pad = len(hemi_kernel)
        hemi_kernel = np.pad(hemi_kernel, (len_pad, 0))
    return hemi_kernel
