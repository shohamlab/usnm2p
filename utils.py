# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-11 15:53:03
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-10-25 10:01:52

''' Collection of generic utilities. '''

import numpy as np
import operator

from constants import SI_POWERS
from logger import logger


def is_iterable(x):
    for t in [list, tuple, np.ndarray]:
        if isinstance(x, t):
            return True
    return False


si_prefixes = {k: np.power(10., v) for k, v in SI_POWERS.items()}
sorted_si_prefixes = sorted(si_prefixes.items(), key=operator.itemgetter(1))


def getSIpair(x, scale='lin', unit_dim=1):
    ''' Get the correct SI factor and prefix for a floating point number. '''
    if is_iterable(x):
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


def get_singleton(df, key, delete=False):
    '''
    Extract a singleton from a dataframe column, after checking that it only takes 1 value.
    
    :param df: dataframe object.
    :param key: column from which to extract the singleton.
    :return: singleton value
    '''
    if is_iterable(key):
        return [get_singleton(df, k) for k in key]
    try:
        values = df[key]
    except KeyError as e:
        values = df.index.get_level_values(key)
    uniques = list(set(values))
    if len(uniques) > 1:
        raise ValueError(f'multiple "{key}" values: {uniques}')
    if delete:
        del df[key]
    return uniques[0]


def is_in_dataframe(df, key, raise_error=False):
    '''
    Check if key already exists as a dataframe column or index.
    
    :param df: dataframe object
    :param key: key
    :param raise_error (optional): whether to raise an error if key found in dataframe
    :return: boolean stating whether the key has been found
    '''
    if key in df.columns or key in df.index.names:
        errstr = f'"{key}" is already present in dataframe'
        if raise_error:
            raise ValueError(errstr)
        else:
            logger.warning(f'{errstr} -> ignoring')
        return True
    return False


def expand_along(df, key, y, nref=None, index_key=None):
    '''
    Add 2D numpy array as a new column into a dataframe, and expand along
    
    :param df: input dataframe
    :param key: name of new column in which array is unraveled
    :param y: 2D array
    :param nref (optional) reference size of expected signal length
    :param index_key (optional): name of new index level to add to dataframe upon expansion
    :return: expanded dataframe
    '''
    # Check that dataframe does not already contain the signal key
    if is_in_dataframe(df, key):
        return df
    # If y does not have 2 dimensions -> reshape into 2D array
    if y.ndim != 2:
        y = y.reshape(-1, y.shape[-1])
    # Extract input dimensions
    nsignals, npersignal = y.shape
    nrecords = len(df)
    if nsignals == 1:
        y = np.tile(y, (nrecords, 1))
        nsignals, npersignal = y.shape
    # Compare signal size to reference, if any
    if nref is not None:
        if nref != npersignal:
            raise ValueError(f'signal length ({npersignal}) does not match reference length ({nref})')
    # Create copy so as to not modify original dataframe
    df_exp = df.copy()
    # Check compatibility between input dataframe and signal array
    if nrecords == nsignals:
        # If dataframe length matches number of signals -> add signals and expand
        logger.debug('adding and expanding')
        df_exp[key] = y.tolist()
        df_exp = df_exp.explode(key)
    elif nrecords == nsignals * npersignal:
        # If dataframe length matches number of elements in signal array -> reshape array and add
        logger.debug('reshaping and adding')
        df_exp[key] = np.reshape(y, nsignals * npersignal)
    else:
        # Otherwise throw incompatibility error
        raise ValueError(
            f'signal array dimensions ({y.shape}) incompatible with dataframe length ({nrecords})')
    # If index key provided -> add it as extra index dimension
    if index_key is not None:
        if index_key in df.index.names:
            logger.warning(f'"{index_key}" key already present in index -> ignoring')
        else:
            inds = np.arange(npersignal)  # fundamental set of signal indexes
            df_exp[index_key] = np.tile(inds, nsignals)  # repeat for each signal and add to expanded dataframe
            df_exp = df_exp.set_index(index_key, append=True)  # set signal index column as new index level
    # Return dataframe containing signals
    return df_exp
