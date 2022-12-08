# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-11 15:53:03
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-12-08 14:22:09

''' Collection of generic utilities. '''

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import operator
from functools import wraps

from constants import SI_POWERS, IND_LETTERS, Label
from logger import logger


def is_iterable(x):
    ''' Check if an object is iterbale (i.e. a list, tuple or numpy array) '''
    for t in [list, tuple, np.ndarray, pd.Series]:
        if isinstance(x, t):
            return True
    return False


def as_iterable(x):
    ''' Return an iterable of an object if it is not already iterable '''
    return x if is_iterable(x) else [x]


def plural(x):
    return 's' if is_iterable(x) else ''


def swapdict(d):
    ''' Swap keys and values in a dictionary '''
    return {v: k for k, v in d.items()}

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


def moving_average(x, n=3):
    '''
    Apply moving average on first axis of a n-dimensional array
    
    :param x: n-dimensional array
    :param n: moving average window size (in number of frames)
    :return: smoothed array with exact same dimensions as x
    '''
    # Logging string
    s = f'smoothing {x.shape} array with {n} samples moving average'
    if x.ndim > 1:
        s = f'{s} along axis 0'
    logger.info(s)
    # Pad input array on both sides
    if n % 2 == 0:
        n -= 1
    w = n // 2
    wvec = [(0, 0)] * x.ndim
    wvec[0] = (w, w)
    xpad = np.pad(x, wvec, mode='symmetric')
    # Apply moving average along first axis
    ret = np.cumsum(xpad, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    # Return output
    return ret[n - 1:] / n


def apply_rolling_window(x, w, func=None, warn_oversize=True, pad=True):
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
    if pad:
        x = np.pad(x, w // 2, mode='symmetric')
    # Generate rolling window over array
    roll = pd.Series(x).rolling(w, center=True)
    # Apply function over rolling window object, drop NaNs and extract output array 
    # return np.array([func(r) for r in roll])
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


def describe_dataframe_index(df, join_str=' x '):
    ''' Describe dataframe index '''
    d = {}
    for k in df.index.names:
        l = len(df.index.unique(level=k))
        key = k
        if l > 1:
            key = f'{key}s'
        d[key] = l
    return join_str.join([f'{v} {k}' for k, v in d.items()])


def get_integer_suffix(i):
    ''' Get the suffix corresponding to a given integer '''
    return {1: 'st', 2: 'nd', 3: 'rd'}.get(int(np.round(i)) % 10, 'th')


def repeat_along_new_dim(df, name, inds):
    '''
    Repeat dataframe values along new index level
    
    :param df: input dataframe
    :param name: name of the new index level
    :param inds: values of the new index level
    :return: dataframe expanded along the new index dimension
    '''
    newdf = df.copy()
    newdf[name] = [inds.values] * len(df)
    return newdf.explode(name).set_index(name, append=True)

def repeat_along_new_dims(df, newdims):
    '''
    Repeat dataframe values along multiple new index levels
    
    :param df: input dataframe
    :param newdims: dictionary of names and indices for each new index level
    :return: dataframe expanded along all new index dimensions
    '''
    for name, inds in newdims.items():
        df = repeat_along_new_dim(df, name, inds)
    return df


def list_difference(l1, l2):
    ''' 
    Return list difference elements in the same order as in the original list
    
    :param l1: original list
    :param l2: list to "subtract" to original list
    :return: output list resulting from subtraction and re-ordering
    '''
    diff = set(l1) - set(l2)
    return [o for o in l1 if o in diff]


def expand_to_match(df, mux):
    '''
    Expand dataframe along new index dimensions to match reference index
    '''
    # Transform to dataframe if needed
    name = None 
    if isinstance(df, pd.Series):
        name = df.name
        df = df.to_frame()
    # Identify index levels present in reference index but not in data
    extra_levels = list_difference(mux.names, df.index.names)
    if len(extra_levels) == 0:
        raise ValueError('did not find any extra index levels')
    newdims = {k: mux.unique(level=k) for k in extra_levels}
    newdf = repeat_along_new_dims(df, newdims)
    if name is not None:
        return newdf.loc[:, name]
    else:
        return newdf

def expand_and_add(dfnew, dfref, prefix=''):
    '''
    Expand dataframe to match index of reference dataframe, and add it to the reference

    :param dfnew: dataframe to be expanded and added
    :param dfref: reference dataframe
    '''
    dfexp = expand_to_match(dfnew, dfref.index)
    if isinstance(dfexp, pd.Series):
        dfexp = dfexp.to_frame()
    for k in dfexp:
        key = f'{prefix}_{k}' if prefix else k
        dfref[key] = dfexp[k]
    return dfref


def shape_str(s):
    return '-by-'.join([f'{x:.0f}' for x in s])


def rectilinearize(s, iruns=None):
    ''' 
    Create expanded series following rectilinear multi-index 
    
    :param s: pandas multi-index Series object
    :param iruns: imposed runs index (optional)
    :return: augmented series 
    '''
    # Compute dimensions of rectilinear output
    nlevels = len(s.index.levels)
    dims = [s.index.unique(level=i) for i in range(nlevels)]
    
    # Extract number of runs, if not imposed 
    if Label.RUN in s.index.names:
        idim_run = s.index.names.index(Label.RUN)
        if iruns is None:
            iruns = dims[idim_run]
        else:
            dims[idim_run] = iruns
    
    # If series contains multiple datasets, apply rectlinearization by dataset
    if Label.DATASET in s.index.names:
        if len(s.index.unique(Label.DATASET)) > 1:
            return s.groupby(Label.DATASET).apply(
                lambda s: rectilinearize(s.droplevel(Label.DATASET), iruns=iruns))
    
    # If dimensions match input, return directly
    shape = [len(x) for x in dims]
    if np.prod(shape) == len(s):
        logger.info(
            f'{s.name}: original {len(s)}-rows series is alreay rectilinear -> ignoring')
        return s
    
    # Create "expanded" (rectilinear) index from input index levels
    mux_exp = pd.MultiIndex.from_product(dims)
    
    # Create new series filled with zeros
    s_exp = pd.Series(0., index=mux_exp)

    # Log process
    logger.info(
        f'{s.name}: expanding {len(s)}-rows series into {describe_dataframe_index(s_exp)} ({len(s_exp)}-rows) series ({len(s_exp) - len(s)} additional rows)')

    # Add expanded series to original series (new elements will show up as NaN)
    snew = (s + s_exp).rename(s.name)
    
    return snew


def shiftslice(s, i):
    ''' Shift a slice by some integer '''
    return slice(s.start + i, s.stop + i)


def get_mux_slice(mux):
    ''' Get a neutral multi-indexed slice '''
    return [slice(None)] * len(mux.names)


def slice_last_dim(mux, wslice):
    ''' Get a multi-indexed slice with last index dimenions '''
    idx = get_mux_slice(mux)
    idx[-1] = wslice
    return tuple(idx)

def is_mux_unique(mux):
    ''' Check whether a multi-index is unique '''
    return mux.duplicated().sum() == 0
    

def reindex_dataframe_level(df, level=0):
    ''' Update dataframe index level to a standard integer list '''
    old_vals = df.index.unique(level=level)
    new_vals = np.arange(old_vals.size)
    return df.reindex(new_vals, level=level)


def discard_indexes(data, ikey, idiscard=None):
    '''
    Discard specific values from a specific index level in a multi-indexed dataset
    
    :param data: multi-index dataframe with 4D (ROI, run, trial, frame) index
    :param ikey: index level key
    :param idiscard: index values to discard at this index level
    :return: filtered dataset
    '''
    # Cast indexes to discard as list
    idiscard = as_iterable(idiscard)
    # If no discard index specified -> return 
    if len(idiscard) == 0 or idiscard[0] is None:
        return data
    # Get indexes values present in dataset at that level
    idxs = data.index.unique(level=ikey)
    # Diff the two lists to get indexes to discard from dataset
    idiscard = sorted(list(set(idxs).intersection(set(idiscard))))
    # If no discard index specified -> return 
    if len(idiscard) == 0 or idiscard[0] is None:
        return data
    # Discard them
    logger.info(f'discarding {ikey} index values {idiscard} from dataset')
    data = data.query(f'{ikey} not in {idiscard}')
    # Return data
    logger.info(f'filtered data ({describe_dataframe_index(data)})')
    return data


def str_proof(aggfunc):
    ''' Make aggregating function compatible with string-typed iterables. '''

    @wraps(aggfunc)
    def wrapper(s):
        if s.nunique() == 0:
            # 0 unique -> only NaNs -> return NaN
            return np.nan
        elif s.nunique() == 1:
            out = s.unique()
            # 1 unique value
            if is_numeric_dtype(s):
                # If numeric, extract first non NaN value and cast it to float
                return out[~np.isnan(out)][0].astype(np.float)
            else:
                # Otherwise, assume no NaN are present and return first value
                return out[0] 
        else:
            # Multiple non NaN values
            if is_numeric_dtype(s):
                # For numeric types -> return mean
                return aggfunc(s)
            else:
                # For non-numeric type -> return NaN
                return np.nan

    return wrapper


def pbar_update(func, pbar):
    ''' Add par update feature to function object '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        out = func(*args, **kwargs)
        pbar.update()
        return out
    return wrapper


def itemize(l):
    ''' return an itemized string version of a list '''
    return '\n'.join([f' - {item}' for item in l])


def reduce_close_elements(x, ethr=1e-8):
    ''' Reduce array by resolving "almost-duplicate" elements '''
    # Get matrix of pairwise combinations
    X, Y = np.meshgrid(x, x)
    # Keep only upper triangle and transform into dataframe
    iup = np.triu_indices(x.size, k=1)
    df = pd.DataFrame({'X': X[iup], 'Y': Y[iup]})
    # Compute differences and identify pairs that contain very close elements
    df['eps'] = np.abs(df['X'] - df['Y'])
    df['isclose'] = df['eps'] < ethr
    dfclose = df[df['isclose']]
    # If close pairs remain, resolve first close pair
    if len(dfclose) > 0:
        row = dfclose.iloc[0]
        x1, x2 = row['X'], row['Y']
        i1, i2 = np.where(x == x1)[0], np.where(x == x2)[0]
        x[i1] = (x1 + x2) / 2  # assign mean to first index
        x = np.delete(x, i2)  # delete second index
        return reduce_close_elements(x)  # call function recursively
    return sorted(x)


def find_nearest(array, value):
    ''' Find nearest element to value in array '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def resolve_close_elements(x, decimals=8, **kwargs):
    ''' Resolve array by updating "almost-duplicate" elements to their reduced value '''
    xunique = reduce_close_elements(np.unique(x), **kwargs)
    elems = np.array([find_nearest(xunique, xx) for xx in x])
    if decimals is not None:
        elems = np.round(elems, decimals) 
    return elems


def resolve_columns(df, cols, **kwargs):
    ''' Resolve specific dataframe columns by updating "almost-duplicate" elements to their reduced value '''
    cols = as_iterable(cols)
    for col in cols:
        df[col] = resolve_close_elements(df[col], **kwargs)
    return df


def pressure_to_intensity(p, rho=1046.0, c=1546.3):
    '''
    Return the spatial peak, pulse average acoustic intensity (ISPPA)
    associated with the specified pressure amplitude.
    
    Default values of dennsity and speed of sound are taken from the
    IT'IS foundation database for brain tissue. 
    
    :param p: pressure amplitude (Pa)
    :param rho: medium density (kg/m3)
    :param c: speed of sound in medium (m/s)
    :return: spatial peak, pulse average acoustic intensity (W/m2)
    '''
    return p**2 / (2 * rho * c)


def intensity_to_pressure(I, rho=1046.0, c=1546.3):
    '''
    Return the pressure amplitude (in Pa) associated with the specified
    spatial peak, pulse average acoustic intensity (ISPPA).
    
    Default values of dennsity and speed of sound are taken from the
    IT'IS foundation database for brain tissue. 
    
    :param I: Isppa (W/m2)
    :param rho: medium density (kg/m3)
    :param c: speed of sound in medium (m/s)
    :return: pressure amplitude (Pa)
    '''
    return np.sqrt(I * 2 * rho * c)


def normalize_stack(x, bounds=(0, 1000)):
    '''
    Normalize stack to a given interval
    
    :param x: (nframe, Ly, Lx) stack array
    :param bounds (optional): bounds for the intensity interval
    :return rescaled stack array
    '''
    # Get input data type and related bounds
    dtype = x.dtype
    if str(dtype).startswith('int'):
        dinfo = np.iinfo(dtype)
    else:
        dinfo = np.finfo(dtype)
    dbounds = (dinfo.min, dinfo.max)
    # Make sure output bounds are within data type limits
    if bounds[0] < dbounds[0] or bounds[1] > dbounds[1]:
        raise ValueError(f'rescaling interval {bounds} exceeds possible {dtype} values')
    # Get input bounds (recasting as float to make ensure correct downstream computations)
    input_bounds = (x.min().astype(float), x.max().astype(float))
    # Get normalization factor
    input_ptp = input_bounds[1] - input_bounds[0]
    output_ptp = bounds[1] - bounds[0]
    norm_factor = input_ptp / output_ptp
    # Compute normalized array
    y = x / norm_factor - input_bounds[0] / norm_factor
    # Cast as input type and return
    return y.astype(dtype)


def html_wrap(s, wrapper, serialize=True):
    ''' Wrap HTML content inside an HTML element '''
    if is_iterable(s) and serialize:
        s = ''.join(s)
    return f'<{wrapper}>{s}</{wrapper}>'

def tr(s):
    return html_wrap(s, 'tr')

def th(s):
    return html_wrap(s, 'th')

def td(s):
    return html_wrap(s, 'td', serialize=False)

def table(s):
    return html_wrap(s, 'table')

def dict_to_html(d):
    ''' Convert dictionary to an HTML Table object. '''
    header = tr([th('Key'), td('Value')])
    rows = [tr([td(key), td(value)]) for key, value in d.items()]
    return table([header] + rows)

class DictTable(dict):
    ''' Overriden dict class enabling rendering as HTML Table in IPython notebooks. '''
    def _repr_html_(self):
        return dict_to_html(self)


def nan_proof(func):
    '''
    Wrapper around cost function that makes it NaN-proof
    
    :param func: function taking a input a pandas Series and outputing a pandas Series
    :return: modified, NaN-proof function object
    '''
    @wraps(func)
    def wrapper(s, *args, **kwargs):
        # Remove NaN values
        s2 = s.dropna()
        # Call function on cleaned input series
        out = func(s2, *args, **kwargs)
        # If output is of the same size as the cleaned input, add it to original input to retain same dimensions
        if (is_iterable(out) or isinstance(out, pd.Series)) and len(out) == s2.size:
            s[s2.index] = out
            return s
        # Otherwise return output as is
        else:
            return out
    return wrapper


def pandas_proof(func):
    '''
    Wrapper around function that makes it pandas-proof
    
    :param func: processing function
    :return: modified, pandas-proof function object
    '''
    @wraps(func)
    def wrapper(y, *args, **kwargs):
        yout = func(y)
        if isinstance(y, pd.Series):
            return pd.Series(data=yout, index=y.index)
        else:
            return yout
    return wrapper


def expdecay(x, H, A, tau, x0):
    '''
    Exponential decay function
    
    :param x: independent variable
    :param H: vertical offset
    :param A: scaling factor
    :param tau: decay time constant
    :param x0: horizontal offset
    '''
    return H + A * np.exp(-(x - x0) / tau)


def biexpdecay(x, H, A1, A2, tau1, tau2, x1, x2):
    ''' 
    Bi-exponential decay function

    :param x: independent variable
    :param H: vertical offset
    :param A1, A2: scaling factors
    :param tau1, tau2: decay time constants
    :param x1, x2: horizontal offsets
    '''
    return expdecay(x, H / 2, A1, tau1, x1) + expdecay(x, H / 2, A2, tau2, x2)


def gauss(x, H, A, x0, sigma):
    '''
    Gaussian function
    
    :param x: independent variable
    :param H: vertical offset
    :param A: gaussian amplitude
    :param x0: horizontal offset
    :param sigma: gaussian width
    '''
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma**2))


def round_to_base(x, precision=1, base=.5):
    '''
    Round to nearest base
    
    :param x: input number
    :param precision: rounding precision (number of digits)
    :param base: rounding base
    :return: rounded number
    '''
    return np.round(base * round(float(x) / base), precision)


def compute_mesh_edges(x, scale='lin'):
    '''
    Compute the appropriate edges of a mesh that quads a linear or logarihtmic distribution.
    
    :param x: the input vector
    :param scale: the type of distribution ('lin' for linear, 'log' for logarihtmic)
    :return: the edges vector
    '''
    if scale == 'log':
        x = np.log10(x)
        range_func = np.logspace
    else:
        range_func = np.linspace
    dx = x[1] - x[0]
    n = x.size + 1
    return range_func(x[0] - dx / 2, x[-1] + dx / 2, n)


def sigmoid(x, x0=0, sigma=1.):
    ''' 
    Apply sigmoid function with specific center and width
    
    :param x: input signal
    :param x0: sigmoid center (i.e. inflection point)
    :param sigma: sigmoid width
    :return sigmoid function output
    '''
    return 1 / (1 + np.exp(-(x - x0) / sigma))


def bounds(x):
    ''' Extract minimum and maximum of array simultaneously '''
    return np.array([min(x), max(x)])