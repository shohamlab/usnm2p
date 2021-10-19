# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-11 15:53:03
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-10-15 10:16:29

import numpy as np
import operator

from constants import SI_POWERS
from logger import logger

''' Collection of generic utilities. '''


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
