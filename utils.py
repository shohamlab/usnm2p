# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-11 15:53:03
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-10-14 19:29:27

import numpy as np
import operator

from constants import *
from logger import logger

''' Collection of generic utilities. '''


def isIterable(x):
    for t in [list, tuple, np.ndarray]:
        if isinstance(x, t):
            return True
    return False


# SI units prefixes
SI_powers = {
    'y': -24,  # yocto
    'z': -21,  # zepto
    'a': -18,  # atto
    'f': -15,  # femto
    'p': -12,  # pico
    'n': -9,   # nano
    'u': -6,   # micro
    'm': -3,   # mili
    '': 0,     # None
    'k': 3,    # kilo
    'M': 6,    # mega
    'G': 9,    # giga
    'T': 12,   # tera
    'P': 15,   # peta
    'E': 18,   # exa
    'Z': 21,   # zetta
    'Y': 24,   # yotta
}
si_prefixes = {k: np.power(10., v) for k, v in SI_powers.items()}
sorted_si_prefixes = sorted(si_prefixes.items(), key=operator.itemgetter(1))


def getSIpair(x, scale='lin', unit_dim=1):
    ''' Get the correct SI factor and prefix for a floating point number. '''
    if isIterable(x):
        # If iterable, get a representative number of the distribution
        x = np.asarray(x)
        x = x.prod()**(1.0 / x.size) if scale == 'log' else np.mean(x)
    if x == 0:
        return 1e0, ''
    else:
        vals = np.array([tmp[1] for tmp in sorted_si_prefixes])
        if unit_dim != 1:
            vals = np.power(vals, unit_dim)
        ix = np.searchsorted(vals, np.abs(x)) - 1
        if np.abs(x) == vals[ix + 1]:
            ix += 1
        return vals[ix], sorted_si_prefixes[ix][0]


def si_format(x, precision=0, space=' ', **kwargs):
    ''' Format a float according to the SI unit system, with the appropriate prefix letter. '''
    if isinstance(x, float) or isinstance(x, int) or isinstance(x, np.float) or\
       isinstance(x, np.int32) or isinstance(x, np.int64):
        factor, prefix = getSIpair(x, **kwargs)
        return f'{x / factor:.{precision}f}{space}{prefix}'
    elif isinstance(x, list) or isinstance(x, tuple):
        return [si_format(item, precision, space) for item in x]
    elif isinstance(x, np.ndarray) and x.ndim == 1:
        return [si_format(float(item), precision, space) for item in x]
    else:
        raise ValueError(f'cannot si_format {type(x)} objects')


def moving_average(x, n=5):
    ''' Apply a monving average on a signal. '''
    if n % 2 == 0:
        raise ValueError('you must specify an odd MAV window length')
    return np.convolve(np.pad(x, n // 2, mode='symmetric'), np.ones(n), 'valid') / n


def get_F_baseline(x):
    '''
    Compute the baseline of a signal. This function assumes that time
    is the last dimension of the signal array.
    '''
    if x.ndim == 1:
        return moving_average(x, n=N_MAV)
    else:
        return np.apply_along_axis(lambda a: moving_average(a, n=N_MAV), -1, x, )


def separate_trials(x, ntrials):
    '''
    Split suite2p data array into separate trials.
    
    :param x: 2D (ncells, nframes) suite2p data array
    :return: 3D (ncells, ntrials, npertrial) data array
    '''
    ncells, nframes = x.shape
    npertrial = nframes // ntrials
    return np.array([np.reshape(xx, (ntrials, npertrial)) for xx in x])


def get_df_over_f(F, ibaseline):
    '''
    Calculate relative fluorescence signal (dF/F0) from absolute fluorescence signal.

    Baseline is calculated on a per-trial basis as the average of the pre-stimiulus interval.

    :param F: 3D (ncells, ntrials, npertrial) fluorescence signal array
    :param ibaseline: baseline evaluation indexes
    '''
    # Extract F dimensions
    ncells, ntrials, npertrial = F.shape
    # Extract baseline fluoresence signals and average across time for each cell and trial 
    F0 = F[:, :, ibaseline].mean(axis=-1)
    # Tile F0 along the time dimension
    F0 = np.moveaxis(np.tile(F0, (npertrial, 1, 1)), 0, -1)
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


def reg_search_list(query, targets):
    '''
    Search a list or targets for match if regexp query, and return first match.
    
    :param query: regexp format query
    :param targets: list of strings to be tested
    :return: first match (if any), otherwise None
    '''
    p = re.compile(query)
    for target in targets:
        positive = p.search(target)
        if positive:
            positive = positive.string
            break
    return positive


def to_unicode(query):
    ''' Translate regexp query into unicode (effectively a fix for undefined queries) '''
    if query == 'undefined':
        return UNDEFINED_UNICODE
    else:
        return query
