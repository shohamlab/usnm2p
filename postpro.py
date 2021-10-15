# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-15 10:13:54
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-10-15 10:49:03

import numpy as np

''' Collection of utilities to process fluorescence signals outputed by suite2p. '''

# def moving_average(x, n=5):
#     ''' Apply a monving average on a signal. '''
#     if n % 2 == 0:
#         raise ValueError('you must specify an odd MAV window length')
#     return np.convolve(np.pad(x, n // 2, mode='symmetric'), np.ones(n), 'valid') / n


# def get_F_baseline(x):
#     '''
#     Compute the baseline of a signal. This function assumes that time
#     is the last dimension of the signal array.
#     '''
#     if x.ndim == 1:
#         return moving_average(x, n=N_MAV)
#     else:
#         return np.apply_along_axis(lambda a: moving_average(a, n=N_MAV), -1, x, )


def separate_trials(x, ntrials):
    '''
    Split suite2p data array into separate trials.
    
    :param x: 2D (ncells, nframes) suite2p data array
    :return: 3D (ncells, ntrials, npertrial) data array
    '''
    ncells, nframes = x.shape
    npertrial = nframes // ntrials
    return np.array([np.reshape(xx, (ntrials, npertrial)) for xx in x])


def get_relative_fluorescence_change(F, ibaseline):
    '''
    Calculate relative fluorescence signal (dF/F0) from an absolute fluorescence signal (F0).
    The signal baseline is calculated on a per-trial basis as the average of a specified
    pre-stimiulus interval.

    :param x: 3D (ncells, ntrials, npertrial) array of absolute fluorescence signals
    :param ibaseline: baseline evaluation indexes
    :return: 3D (ncells, ntrials, npertrial) array of relative change in fluorescence
    '''
    # Extract F dimensions
    ncells, ntrials, npertrial = F.shape
    # Extract baseline fluoresence signals and average across time for each cell and trial 
    F0 = F[:, :, ibaseline].mean(axis=-1)  # (ncells, ntrials) array
    # Add 3rd axis (of dim 1) to F0 to enable broadcasting with F
    F0 = np.atleast_3d(F0)
    # Return relative change in fluorescence
    return (F - F0) / F0


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
    