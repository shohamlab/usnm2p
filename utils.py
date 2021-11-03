# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-11 15:53:03
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-11-01 18:45:18

''' Collection of generic utilities. '''

import numpy as np
import pandas as pd
import operator
import abc

from constants import SI_POWERS
from logger import logger


class StackProcessor(metaclass=abc.ABCMeta):
    ''' Generic intrface for processor objects '''

    @abc.abstractmethod
    def run(self, stack: np.array) -> np.ndarray:
        ''' Abstract run method. '''
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def code(self):
        ''' Abstract code attribute. '''
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def rootcode(self):
        ''' Abstract root code attribute '''
        raise NotImplementedError


class NoProcessor(StackProcessor):
    ''' Dummy class for no-processor objects '''

    def run(self, stack: np.array, iframes):
        raise NotImplementedError

    @property
    def ptype(self):
        return self.__class__.__name__[2:].lower()

    def __str__(self) -> str:
        return f'no {self.ptype}'

    @property
    def code(self):
        return f'no_{self.ptype}'
    
    @property
    def rootcode(self):
        raise NotImplementedError


def is_iterable(x):
    ''' Check if an object is iterbale (i.e. a list, tuple or numpy array '''
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


def add_array_to_dataframe(df, key, y, nref=None, index_key=None):
    '''
    Add 2D data array as a new column into a dataframe,
    potentially "expanding" the dataframe along the array's last dimension.
    
    :param df: input dataframe
    :param key: name of new column in which array is inserted
    :param y: 2D (nvecs, npervec) data array
    :param nref (optional): expected vector length (i.e. value for npervec)
    :param index_key (optional): name of new index level to add to dataframe upon expansion, if any
    :return: dataframe containing the new array (and potentially expanded)
    '''
    # Check that dataframe does not already contain the new key
    if is_in_dataframe(df, key):
        return df
    # If y does not have 2 dimensions -> reshape into 2D array
    if y.ndim != 2:
        y = y.reshape(-1, y.shape[-1])
    # Extract input dimensions from array and dataframe
    nvecs, npervec = y.shape
    nrecords = len(df)
    # If only 1 vector provided, assume it is valid for every record of the dataframe -> tile 
    if nvecs == 1:
        y = np.tile(y, (nrecords, 1))
        nvecs, npervec = y.shape
    # Compare vector length to provided reference, if any
    if nref is not None:
        if nref != npervec:
            raise ValueError(f'vector length ({npervec}) does not match reference length ({nref})')
    # Create copy so as to not modify original dataframe
    newdf = df.copy()
    # Check compatibility between input dataframe and data array
    if nrecords == nvecs:
        # Case 1: dataframe length matches number of vectors -> insert data arrray and expand upon vector dimension
        logger.debug('inserting and expanding')
        newdf[key] = y.tolist()
        newdf = newdf.explode(key)
    elif nrecords == nvecs * npervec:
        # Case 2: dataframe length matches number of elements in data array -> flatten data array and insert
        logger.debug('flattening and inserting')
        newdf[key] = np.reshape(y, nvecs * npervec)
    else:
        # Otherwise throw incompatibility error
        raise ValueError(
            f'data array dimensions ({y.shape}) incompatible with dataframe length ({nrecords})')
    # If index key provided -> add it as extra index dimension
    if index_key is not None:
        if index_key in df.index.names:
            logger.warning(f'"{index_key}" key already present in index -> ignoring')
        else:
            inds = np.arange(npervec)  # fundamental set of vector indexes
            newdf[index_key] = np.tile(inds, nvecs)  # repeat for each vector and add to expanded dataframe
            newdf = newdf.set_index(index_key, append=True)  # set vector index column as new index level
    # Return dataframe containing new data
    return newdf


def float_to_uint8(arr):
    ''' Transform a floating point (0 to 1) array to an 8-bit unsigned integer (0 to 255) array. '''
    return (arr * 255).astype(np.uint8)


def apply_rolling_window(x, w, func=None, warn_oversize=True):
    '''
    Generate a rolling window over an array an apply a specific function to the result.
    Defaults to a moving average.
    
    :param x: input array
    :param w: window size (number of array samples used to apply the function)
    :param func (optional): function to apply to the rolling window result
    :return: output array of equal size to the input array, with the rolling window and function applied.
    '''
    # If more than 1 dimension -> reshape to 2D, apply on each row, and reshape back to original shape
    if x.ndim > 1:
        dims = x.shape
        x = x.reshape(-1, dims[-1])
        x = np.array([apply_rolling_window(xx, w, func=func, warn_oversize=False) for xx in x])
        return x.reshape(*dims)
    # Check that window size is valid
    if w % 2 == 0:
        raise ValueError('window size must be an odd number')
    if w > x.size and warn_oversize:
        logger.warning(f'window size ({w}) is larger than array length ({x.size})')
    # If function not provided, apply mean by default
    if func is None:
        func = lambda x: x.mean()
    # Pad input array on both sides
    x = np.pad(x, w // 2, mode='symmetric')
    # Generate rolling window over array
    roll = pd.Series(x).rolling(w, center=True)
    # Apply function over rolling window object, drop NaNs and extract output array 
    return func(roll).dropna().values