# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-11 15:53:03
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-11-10 17:49:26

''' Collection of generic utilities. '''

import numpy as np
import pandas as pd
import operator
import abc

from constants import SI_POWERS, IND_LETTERS
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
    ''' Check if an object is iterbale (i.e. a list, tuple or numpy array) '''
    for t in [list, tuple, np.ndarray]:
        if isinstance(x, t):
            return True
    return False


def as_iterable(x):
    ''' Return an iterable of an object if it is not already iterable '''
    return x if is_iterable(x) else [x]


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


def array_to_dataframe(arr, name, dim_names=None):
    '''
    Convert a multidimensional array into a multi-index linearized dataframe.
    
    :param arr: multi-dimensional array
    :param name: name of the variable stored in the array
    :param dim_names (optional): names of the dimensions of the array
    :return: multi-index dataframe with linearized array as the only non-index column
    '''
    if dim_names is None:
        dim_names = IND_LETTERS[:arr.ndim]
    else:
        if len(dim_names) != arr.ndim:
            raise ValueError(f'number of dimensions names {len(dim_names)} do not match number of array dimensions ({arr.shape})')
    index = pd.MultiIndex.from_product([np.arange(x) for x in arr.shape], names=dim_names)
    return pd.DataFrame(data=arr.flatten(), columns=[name], index=index)


def arrays_to_dataframe(arrs_dict, **kwargs):
    '''
    Convert a dictionary of multidimensional arrays into a multi-index linearized dataframe.
    
    :param arrs_dict: dictionary of multi-dimensional arrays
    :return: multi-index dataframe with linearized arrays in different columns
    '''
    names, arrs = zip(*arrs_dict.items())
    assert all(x.shape == arrs[0].shape for x in arrs), 'inconsistent array shapes'
    df = array_to_dataframe(arrs[0], names[0], **kwargs)
    for name, arr in zip(names[1:], arrs[1:]):
        df[name] = arr.flatten()
    return df


def describe_dataframe_index(df):
    ''' Describe dataframe index '''
    d = {}
    for k in df.index.names:
        l = len(df.index.unique(level=k))
        key = k
        if l > 1:
            key = f'{key}s'
        d[key] = l
    return ' x '.join([f'{v} {k}' for k, v in d.items()])


def get_integer_suffix(i):
    ''' Get the suffix corresponding to a given integer '''
    return {1: 'st', 2: 'nd', 3: 'rd'}.get(int(np.round(i)) % 10, 'th')


def repeat_along(df, inds, name):
    '''
    Repeat dataframe values along new index level
    
    :param df: input dataframe
    :param inds: values of the new index level
    :param name: name of the new index level
    :return: dataframe expanded along the new index dimension
    '''
    newdf = df.copy()
    newdf[name] = [inds.values] * len(df)
    return newdf.explode(name).set_index(name, append=True)