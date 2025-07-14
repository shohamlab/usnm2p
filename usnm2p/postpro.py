# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-15 10:13:54
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-07-14 17:50:26

''' Collection of utilities to process fluorescence signals outputed by suite2p. '''

from collections import Counter
from itertools import combinations
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import maximum_filter1d, minimum_filter1d, gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import butter, sosfiltfilt, find_peaks, peak_widths, welch, hilbert, periodogram, stft, correlate, correlation_lags
from scipy.stats import skew, norm, ttest_ind, linregress, chi2
from scipy.stats import t as tstats
from scipy.stats import f as fstats
from scipy.interpolate import griddata, interp1d
import spectrum
import statsmodels.api as sm
from statsmodels.formula.api import ols
from functools import wraps
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from .constants import *
from .logger import logger
from .utils import *
from .parsers import parse_2D_offset
from .surrogates import generate_surrogate


# Register tqdm progress functionality to pandas
tqdm.pandas()


def split_first_dim(arr, npersplit, verbose=False):
    '''
    Split the first dimension of an array into two dimensions.

    :param arr: input array
    :param npersplit: number of elements per split (must be an integer divider of first dimension)
    :verbose (optional): whether to log process (defaults to False)
    :return: reshaped array with first dimension split into two dimensions
    '''
    shape = arr.shape
    if shape[0] % npersplit != 0:
        raise ValueError(f'cannot split first dimension length ({shape[0]}) into {npersplit}-long segments')
    nsplits = shape[0] // npersplit
    new_shape = (nsplits, npersplit) + shape[1:]
    if verbose:
        logger.info(f'splitting input array first dimension into {nsplits} {npersplit}-long segments')
    return arr.reshape(new_shape)


def ravel_first_two_dims(arr):
    '''
    Ravel the first two dimensions of an array into a single dimension.

    :param arr: input array (at least 2D)
    :return: reshaped array with first two dimensions raveled into a single dimension
    '''
    if arr.ndim < 2:
        raise ValueError(f'cannot ravel {arr.ndim}D array')
    shape = arr.shape
    new_shape = (shape[0] * shape[1],) + shape[2:]
    return arr.reshape(new_shape)


def extract_fluorescence_profile(F, qbase=None, avgkey=None, zscore=False, serialize=False, nsplit=None):
    '''
    Extract fluorescence profile (either absolute or relative change) from a stack,
    by averaging across a specific dimension.

    :param F: 3D (single channel) or 4D (multi-channel) fluorescence stack
    :param qbase (optional): quantile for baseline extraction, if relative change is required
    :param avgkey (optional): key specifying dimension to average across, if any. One of:
        - None: no aggregation, stack is fully expanded into 1D vector
        - 'frame': average across entire frame
        - 'row': average across column for each frame row
    :param serialize (optional): whether to serialize data if not already frame-averaged (defaults to False)
    :param nsplit (optional): number of segments to split the output profile into
    :return: 1D (if linear) or 2D (if split) frame-average or row-average fluoresence profile
    '''
    # Check valididty of averaging key
    if avgkey is not None and avgkey not in ('frame', 'row'):
        raise ValueError(f'cannot compute "{avgkey}"-average')

    # Check input dimensionality
    nd = F.ndim
    if nd > 4 or nd < 3:
        raise ValueError(f'cannot extract fluorescence profile from {nd}-dimensional input')

    # If multi-channel, run independently on each channel
    if nd == 4:
        return np.array([extract_fluorescence_profile(x, qbase, avgkey=avgkey, nsplit=nsplit) for x in F])

    # Determine average axi(e)s, if any
    if avgkey is not None:
        avgax = {'frame': (1, 2), 'row': 2}

        # Average across relevant axes to get temporal profile(s)
        logger.info(f'extracting {avgkey}-average profile')
        F = np.mean(F, axis=avgax[avgkey])

    # If baseline quantile is provided, compute relative change in fluorescence
    if qbase is not None:
        F0 = np.quantile(F, qbase, axis=0)
        F = (F - F0) / F0

    # If specified, zscore the fluorescence profile
    if zscore:
        logger.info('z-scoring fluorescence profile')
        F = (F - np.mean(F, axis=0)) / np.std(F, axis=0)

    # If requested, serialize to convert to single time-varying profile
    if serialize and F.ndim > 1:
        logger.info(f'serializing across {F.shape[1:]} {avgkey if avgkey is not None else "pixel"}s to yield single time-varying vector')
        F = F.ravel()

    # If nsplit provided, apply
    if nsplit is not None:
        F = split_first_dim(F, nsplit)
    
    # Return profile
    return F


def process_intraframe_fluorescence(y, fidx, fps, npre=3, wroll=15e-3, verbose=True):
    '''
    Process intraframe fluorescence signal.

    :param y: multi-indexed (*:, frame, row) series containing fluorescence data
    :param fidx: FrameIndexer object
    :param fps: frame sampling frequency (frame per second)
    :param npre: number of pre-stimulus frames to average from to get predictive
        scanning-induced intraframe variation profile
    :param wt: size of rolling window (in seconds) for "rolling max" computation
    :param verbose: whether to log different process steps
    :return: pandas dataframe with pulse-evoked dips
    '''
    # Check that data contains frame and row indexes
    idxdims = y.index.names
    for k in [Label.FRAME, Label.ROW]:
        if k not in idxdims:
            raise ValueError(f'"{k}" dimension not found in data index')
    
    # If data contains extra dims, call function recursively for each extra dims group
    extradims = [k for k in idxdims if k not in [Label.FRAME, Label.ROW]]
    if extradims:
        logger.info(f'processing intraframe fluorescence signal across {extradims} combinations')
        yout = (y
            .groupby(extradims)
            .apply(lambda yy: process_intraframe_fluorescence(
                yy.droplevel(extradims), fidx, fps, npre=npre, wroll=wroll, 
                verbose=False))
        )
        if isinstance(yout, pd.DataFrame):
            yout = yout.stack(level=[Label.FRAME, Label.ROW])
        return yout

    # Store series name, and temporarily replace it
    name = y.name
    y.name = 'tmp'
    
    # Subtract pre-stimulus baseline level
    y = y - y.loc[:fidx.iref].median()

    # Extract number of rows per frame
    lpf = y.index.get_level_values(Label.ROW).max() + 1

    # Interpolate values between the last row of each frame to extract "physiological response trend"
    if verbose:
        logger.info('interpolating slow physiological response trend')
    yresp = y.copy()
    yresp.loc[pd.IndexSlice[:, :lpf - 1]] = np.nan
    yresp.loc[pd.IndexSlice[0, 0]] = y.loc[pd.IndexSlice[0, 0]]
    yresp = yresp.interpolate(method='linear')

    # Subtract interpolated response trend from original profile
    if verbose:
        logger.info('subtracting interpolated response trend')
    y = y - yresp

    # Compute average intra-frame variation (scanning artifact) profile from n preceding frames
    if verbose:
        logger.info(f'computing average scanning-induced intra-frame variation profile from {npre} baseline frames')
    ypreavg = y.loc[pd.IndexSlice[fidx.iref - npre:fidx.iref - 1]].groupby(Label.ROW).mean()

    # Expand average profile across frames, and subtract
    if verbose:
        logger.info('subtracting baseline intra-frame variation profile')
    yscan = expand_to_match(ypreavg, y.index)
    yscan = yscan.swaplevel(0, 1).sort_index()
    y = y - yscan

    # Apply frame-by-frame rolling filter to extract baseline
    if verbose:
        logger.info('applying frame-by-frame rolling filter to extract remaining intra-frame LF baseline')
    w = int(np.round(wroll * fps * lpf))  # Rolling window size (in frames)
    if w % 2 == 0:  # Ensure odd window size
        w += 1
    ybaseline = (y
        .groupby(Label.FRAME)
        .transform(lambda x: gaussian_filter1d(quantile_filter(x.values, w, 1), w // 2))
    )

    # Subtract frame-by-frame baseline fit from baseline corrected profile
    if verbose:
        logger.info('subtracting frame-by-frame baseline fit')
    y = y - ybaseline

    # Reset series name and return
    return y.rename(name)


def interp_over_smooth_pulse_relative_time(y, npts=50):
    ''' 
    Harmonize pulse-relative relative vectors across pulses and other dimensions
    '''
    logger.info(f'interpolating {y.name} values over smooth {npts}-points pulse-relative time vector...')
    trel = y.index.get_level_values(Label.PULSERELTIME)
    smooth_trel = np.linspace(trel.min(), trel.max(), npts)

    def interpfunc(x):
        return pd.Series(
            np.interp(
                smooth_trel, x.index.get_level_values(Label.PULSERELTIME), x, left=np.nan, right=np.nan), 
            index=pd.Index(smooth_trel, name=Label.PULSERELTIME), name=x.name)

    return (y
        .groupby([k for k in y.index.names if k != Label.PULSERELTIME])
        .apply(interpfunc)
    )


def split_by_pulse(y, fidx, fps, dur, PRF, onset=0., verbose=True, name=None):
    '''
    Split continous fluorescence profile by pulse 

    :param y: (:, frame, row) indexed series containing fluorescence data
    :param fidx: FrameIndexer object
    :param fps: frame sampling frequency (frames per second)
    :param dur: stimulus duration (s)
    :param PRF: stimulus PRF (Hz)
    :param onset: stimulus onset (s)
    :param verbose: whether to log process (default to True, unless in recursive call)
    :param name (optional): name to give to procesed fluorescence data. If None, the original name is used
    :return: (:, pulse, row) indexed dataframe with 2 columns:
        fluorescence data and relative tim w.r.t. pulse onset
    '''
    # Check that data contains frame and row indexes
    idxdims = y.index.names
    for k in [Label.FRAME, Label.ROW]:
        if k not in idxdims:
            raise ValueError(f'"{k}" dimension not found in data index')
    
    # If data contains extra dims, call function recursively for each extra dims group
    extradims = [k for k in idxdims if k not in [Label.FRAME, Label.ROW]]
    if extradims:
        logger.info(f'splitting fluorescence signal by pulse across {extradims} combinations')
        name = y.name
        # If "auto" onset extraction requested, set up appropriate function
        if onset == 'auto':
            if Label.DATASET not in extradims:
                raise ValueError('automatic onset extraction only works on dataset-indexed inputs')
            idataset = extradims.index(Label.DATASET)
            def onset_func(gkey):
                return DATASET_STIM_TRIG_DELAY.get(gkey[idataset], DEFAULT_STIM_TRIG_DELAY)
        # Otherwise, set up generic placeholder
        else:
            def onset_func(_):
                return onset
        # Apply function recursively and return
        return (y
            .groupby(extradims)
            .apply(lambda yy: split_by_pulse(
                yy.droplevel(extradims), fidx, fps, dur, PRF, onset=onset_func(yy.name), 
                verbose=False, name=name))
        )

    # Extract number of rows per frame and row sampling frequency and time step
    lpf = y.index.get_level_values(Label.ROW).max() + 1
    fs = fps * lpf
    dt = 1 / fs

    # Extract profile during stim window
    tbounds = np.array([onset, onset + dur])
    if verbose:
        logger.info(f'extracting profile within [{tbounds[0]:.3f} - {tbounds[1]:.3f}] s stim window')
    istart = int(np.ceil(onset * fs)) # start index = index following stim onset
    n = int(np.floor(dur * fs)) # ensure consistent number of indexes 
    ystim = y.loc[pd.IndexSlice[fidx.iref, istart:istart + n - 1]].droplevel(Label.FRAME)

    # If name provided, rename series
    if name is not None:
        ystim = ystim.rename(name)
    
    # Extract series name, and cast as dataframe
    name = ystim.name
    df = ystim.to_frame()

    # Compute pulse index and relative time w.r.t pulse onset
    if verbose:
        logger.info('computing pulse index and relative time w.r.t pulse onset')
    tvec = np.arange(ystim.size) / fs + dt - onset % dt
    df[Label.PULSE] = (tvec // (1 / PRF)).astype(int)
    df[Label.PULSERELTIME] = ((tvec % (1 / PRF)) - 1 / fs) * 1e3  # ms

    # Re-index bybpulse and rel time, and extract series of interest
    y = df.reset_index().set_index([Label.PULSE, Label.PULSERELTIME])[name]

    # Max-rectify within-pulse profiles
    if verbose:
        logger.info('max-rectifying pulse profiles')
    y = y.groupby(Label.PULSE).transform(lambda x: x- x.max())

    # Return dataframe
    return y


def get_onoff_times(dur, PRF, DC, onset=0):
    '''
    Get on and off time in a pulse train

    :param dur: total duration of the pulse train (s)
    :param PRF: pulse repetition frequency (Hz)
    :param DC: duty cycle (%)
    :param onset: onset time of the pulse train (s). Defaults to 0.
    :return: 2D array of ON and OFF time on and time off vectors (s)
    '''
    npulses = int(np.round(dur * PRF))  # number of pulses in the burst
    ton = np.arange(npulses) / PRF + onset  # pulse onset times
    toff = ton + (DC * 1e-2) / PRF  # pulse offset times
    return np.array([ton, toff]).T


def get_stimon_mask(dims, fps, tpulses):
    '''
    Construct mask of "stim-on" pixels on stimulus frame 

    :param dims: 2D dimensions of the frame (in pixels)
    :param fps: frame sampling frequency (Hz)
    :param tpulses: list of onset and offset times for each pulse
    :return mask: 2D numpy array
    '''
    # Compute number of pixels in frame
    npixels = np.prod(dims)
    # Compute single pixel sampling frequency (Hz)
    fs = fps * npixels
    # Construct serialized scanning time vector
    tscan = np.arange(npixels) / fs
    # Create and populate serialized mask
    mask = np.zeros(npixels)
    for ton, toff in tpulses:
        idxs = np.logical_and(tscan >= ton, tscan <= toff)
        mask[idxs] = 1
    # Reshape to frame dimenions
    mask = mask.reshape(dims)
    # Return
    return mask


def frame_to_dataframe(y):
    ''' Format a 2D numpy frame array as a pandas dataframe '''
    df = pd.DataFrame(y)
    df.columns.name = Label.COL
    df.index.name = Label.ROW
    return df


def separate_runs(data, nruns):
    '''
    Split fluorescence dataframe into separate runs.

    :param df: multi-index dataframe with 2D (ROI, frame) index
    :param nruns: number of runs
    :return: multi-index dataframe with 3D (ROI, run, frame) index
    '''
    # Check that data was not already split by run
    if Label.RUN in data.index.names:
        logger.warning('data already split by run -> ignoring')
        return data
    logger.info(f'splitting fluorescence data into {nruns} separate runs...')
    # Get dimensions of dataframe index 
    nROIs, nframes_per_ROI = [len(data.index.unique(level=k)) for k in data.index.names]
    # Make sure that #frames / ROI is an integer multiple of #frames / run 
    if nframes_per_ROI % nruns != 0:
        raise ValueError(f'specified number of runs {nruns} incompatible with number of frames per ROI ({nframes_per_ROI})')
    # Add run indexes as an extra column
    iruns = np.arange(nruns)
    nframes_per_run = nframes_per_ROI // nruns
    data[Label.RUN] = np.repeat(np.tile(iruns, (nROIs, 1)), nframes_per_run)
    # Re-arrange frame indexes on a per-run basis
    iframes_per_run = np.arange(nframes_per_run)
    iframes_per_run_ext = np.tile(iframes_per_run, (nROIs * nruns, 1)).flatten()
    # Re-generate data index with new "run" level
    data = data.droplevel(Label.FRAME)
    data = data.set_index(Label.RUN, append=True)
    data[Label.FRAME] = iframes_per_run_ext
    data = data.set_index(Label.FRAME, append=True)
    # Return data
    return data
    

def separate_trials(data, ntrials):
    '''
    Split fluorescence dataframe into separate trials.

    :param data: multi-index dataframe with 3D (ROI, run, frame) index
    :param ntrials: number of trials
    :return: multi-index dataframe with 4D (ROI, run, trial, frame) index
    '''
    # Check that data was already split by run but not yet by trial
    if Label.RUN not in data.index.names:
        raise ValueError(f'attempting to split by {Label.TRIAL} but dataset missing {Label.RUN} information')
    if Label.TRIAL in data.index.names:
        logger.warning(f'data already split by {Label.TRIAL} -> ignoring')
        return data
    logger.info(f'splitting fluorescence data into {ntrials} separate trials...')
    # Get dimensions of dataframe index
    nROIs, nruns, nframes_per_run = [len(data.index.unique(level=k)) for k in data.index.names]
    # Make sure that #frames / run is an integer multiple of #frames / trial
    if nframes_per_run % ntrials != 0:
        raise ValueError(f'specified number of trials {ntrials} incompatible with number of frames per run ({nframes_per_run})')
    # Add trial indexes as an extra column
    itrials = np.arange(ntrials)
    nframes_per_trial = nframes_per_run // ntrials
    itrials_ext = np.repeat(np.tile(itrials, (nROIs * nruns, 1)), nframes_per_trial)
    data[Label.TRIAL] = itrials_ext
    # Re-arrange frame indexes on a per-trial basis
    iframes_per_trial = np.arange(nframes_per_trial)
    iframes_per_trial_ext = np.tile(iframes_per_trial, (nROIs * nruns * ntrials, 1)).flatten()
    # Re-generate data index with new "trial" level 
    data = data.droplevel(Label.FRAME)
    data = data.set_index(Label.TRIAL, append=True)
    data[Label.FRAME] = iframes_per_trial_ext
    data = data.set_index(Label.FRAME, append=True) 
    # Return data
    return data


def get_window_size(wlen, fps):
    ''' Compute window size (in number of frames) from window length (in s) and fps. '''
    # Convert seconds to number of frames
    w = int(np.round(wlen * fps))
    # Adjust to odd number if needed
    if w % 2 == 0:
        w += 1
    return w


def robust_linreg(y, x=None, norm='HuberT', add_cst=True):
    '''
    Perform linear regression on 2 signals
    
    :param y: output signal
    :param x (optional): input signal
    :param norm (default: HuberT): name of the norm used to compute the linear regression
    :param add_cst (default: True): whether to consider an additional constant in the
        linear regression model 
    :return: fitted regression parameter(s)
    '''
    if x is None:
        x = np.arange(y.size)
    if add_cst:
        x = sm.add_constant(x)
    norm = getattr(sm.robust.norms, norm)()
    model = sm.RLM(y, x, M=norm)
    fit = model.fit()
    params = fit.params
    if not add_cst:
        return [0, params[0]]
    else:
        return params
    

def force_positive_Fc(costfunc):
    '''
    Wrapper around cost function that forces positivity of the corrected fluorescence trace
    by heavily penalizing the presence of negative samples
    
    :param costfunc: cost functiopn object
    :return: modified cost function object with enforced non-negativity constraint
    '''
    @wraps(costfunc)
    def wrapper(F_ROI, F_NEU, alpha):
        cost = costfunc(F_ROI, F_NEU, alpha)
        Fc = F_ROI - alpha * F_NEU
        if Fc.min() < 0:
            cost += 1e10
        return cost
    return wrapper


def maximize_skewness(F_ROI, F_NEU, alpha):
    '''
    Maximize skewness of corrected fluorescence profile
    
    :param F_ROI: ROI fluorescence profile (1D array)
    :param F_NEU: associated neuropil fluorescence profile (1D array)
    :param alpha: candidate neuropil subtraction factor (scalar)
    :return: associated cost (scalar)
    '''
    # Compute corrected fluorescence signal
    Fc = F_ROI - alpha * F_NEU
    # Maximize skewness by penalizing negative skewness values
    return -skew(Fc)


def center_around(alpha_ref):
    '''
    Create a cost function that minimizes deviations of alpha from a reference value

    :param alpha_ref: reference alpha value
    :return: cost function object
    '''
    def costfunc(a, b, alpha):
        # Attract alpha towards reference value by penalizing absolute distance from default 
        return np.abs(alpha - alpha_ref)
    costfunc.__name__ = f'center_around({alpha_ref})'
    return costfunc


def optimize_alpha(data, costfunc, bounds=(0, 1)):
    '''
    Compute optimal neuropil subtraction coefficient that minimizes a specific cost function 

    :param data: fluorescence timeseries dataframe
    :param costfunc: cost function object
    :param bounds (default: (0, 1)): bondary values for neuropil subtraction coefficient
    :return: optimal neuropil subtraction coefficient value
    '''
    # Generate vector of alphas
    alphas = np.linspace(*bounds, 100)
    # Extract fluorescence profiles as 2D arrays
    F_ROI = data[Label.F_ROI].values
    F_NEU = data[Label.F_NEU].values
    # Compute cost for each alpha
    costs = np.array([costfunc(F_ROI, F_NEU, alpha) for alpha in alphas])
    # Return alpha corresponding to minimum cost
    return alphas[np.argmin(costs)]


def quantile_filter(x, w, q):
    '''
    Apply a quantile detection filter on signal
    
    :param x: input signal
    :param w: window size (number of samples)
    :param q: quantile value
    :return: filtered signal
    '''
    return apply_rolling_window(x, w, func=lambda x: x.quantile(q))


def skew_to_quantile(s, qthr=0.05, sigma=1.):
    ''' Function mapping a distribution skewness value to a baseline extraction quantile '''
    return sigmoid(-s, sigma=sigma, A=1 - 2 * qthr, y0=qthr)


def get_quantile_baseline_func(fs, wquantile, q=None, wsmooth=None):
    '''
    Construct a quantile-based baseline computation function
    
    :param fs: sampling rate of the signal (in fps)
    :param wquantile: window length of the quantile filter (in s)
    :param q (optional): quantile value used for the quantile filter 
    :param wsmooth (optional): width of smoothing Gaussian filter (in seconds)
    :return: baseline function object
    '''
    if q is not None:
        # Define quantile description string
        qstr = f'{q * 1e2:.0f}{get_integer_suffix(q * 1e2)} percentile'
    else:
        qstr = 'adaptive percentile'

    # If window length not given, return constant baseline computation function
    if wquantile is None:
        logger.info(f'defining baseline function as {qstr} of signal')
        return lambda s: s.quantile(q)

    # Otherwise, define quantile filter computation function
    else:
        wquantile_len = get_window_size(wquantile, fs)
        bstr = f'{wquantile:.1f}s ({wquantile_len} frames) long {qstr} filter'
        if wsmooth:
            wsmooth_len = get_window_size(wsmooth, fs)
            bstr = [bstr] + [f'{wsmooth:.1f}s ({wsmooth_len} frames) long gaussian filter']
            bstr = f'successive application of:\n{itemize(bstr)}'    
        logger.info(f'defining baseline function as {bstr}')

        # Define baseline extraction function per run
        def bfunc_per_run(y, q_):
            y = quantile_filter(y, wquantile_len, q_)
            if wsmooth is not None:
                y = gaussian_filter1d(y, wsmooth_len)
            return y
        
        # Define baseline extraction function per ROI
        def bfunc_per_ROI(s):
            if q is None:
                q_ = skew_to_quantile(skew(s))
            else:
                q_ = q
            return s.groupby(Label.RUN).transform(lambda ss: bfunc_per_run(ss, q_))
        
        return bfunc_per_ROI


def maximin(y, w):
    '''
    Apply maximin function to signal with specific window size
    
    :param y: 1D signal
    :param w: window size (number of samples)
    :return: maximin-filtered signal
    '''
    return maximum_filter1d(minimum_filter1d(y, w), w)


def get_maximin_baseline_func(fs, wmaximin, wsmooth=None):
    ''' 
    Construct a maximin baseline function

    :param fs: sampling rate (Hz)
    :param wmaximin: window size for maximum and minimum filters (in seconds)
    :param wsmooth (optional): width of smoothing Gaussian filter (in seconds)
    :return: baseline function object
    '''
    s = []
    if wsmooth is not None:
        wsmooth_len = get_window_size(wsmooth, fs)
        s.append(f'{wsmooth:.1f}s ({wsmooth_len} frames) long gaussian filter')
    wmaximin_len = get_window_size(wmaximin, fs)
    s += [
        f'{wmaximin:.1f}s ({wmaximin_len} frames) long mininum filter',
        f'{wmaximin:.1f}s ({wmaximin_len} frames) long maximum filter'
    ]
    logger.info(f'defining baseline extraction function as successive application of:\n{itemize(s)}')
    # Define baseline extraction function
    def bfunc(y):
        if wsmooth is not None:
            y = gaussian_filter1d(y, wsmooth_len)
        y = maximin(y, wmaximin_len)
        return y
    # Return baseline extraction function
    return pandas_proof(bfunc)


def get_butter_filter_func(fs, fc, order=2, kind='pass', verbose=True, pdproof=True):
    '''
    Construct zero-phase filter func
    
    :param fs: sampling frequency (Hz)
    :param fc: tuple of cutoff frequencies (Hz)
    :param order: filter order
    :param kind: filter type ("pass" or "stop")
    :param verbose: whether to log process
    :param pdproof: whether to make filter function pandas-proof
    :return: 
        - Second-order sections representation of the IIR filter.
        - (pandas-proof or regular) filter function
    '''
    # Check validity of inputs
    all_kinds = ('pass', 'stop')
    if kind not in all_kinds:
        raise ValueError(f'invalid filter kind: "{kind}" (options are {all_kinds})')
    fc = np.asarray(fc)
    if fc.size != 2:
        raise ValueError(f'invalid cutoff frequencies: {fc} (you must provide a pair of values)')
    if any(fc != np.sort(fc)):
        raise ValueError('cutoff frequencies must be provided in ascending order')
    if fc[0] < 0:
        raise ValueError(f'invalid cutoff frequency: {fc[0]} (must be positive)')
    if fc[0] == 0. and fc[1] >= fs:
        raise ValueError(f'invalid cutoff frequencies: {fc} (no filtering needed)')
    # Determine filter range (low, high or band)
    brange = 'band'
    if fc[0] == 0.:
        brange = 'low'
        fc = fc[1]
    elif fc[1] >= fs:
        brange = 'high'
        fc = fc[0]
    if kind == 'stop' and brange != 'band':
        raise ValueError(f'cannot design "{brange}{kind}" filter') 
    btype = f'{brange}{kind}'
    # Log process
    fc_str = ' - '.join([f'{x:.3f} Hz' for x in as_iterable(fc)])
    if verbose:
        logger.info(f'defining order {order} {btype} BW filter with fc = {fc_str}')
    # Determine Nyquist frequency
    nyq = fs / 2
    # Calculate Butterworth filter second-order sections
    sos = butter(order, np.asarray(fc) / nyq, btype=btype, output='sos')
    # Define filter function
    def myfiltfunc(y):
        return sosfiltfilt(sos, y)
    # If requested, make pandas proof
    if pdproof:
        myfiltfunc = pandas_proof(myfiltfunc)
    # Return butter output and filter function
    return sos, myfiltfunc


def compute_baseline(data, bfunc):
    '''
    Compute the baseline of a signal.

    :param data: multi-indexed Series object contaning the signal of interest
    :param bfunc: baseline function
    :return: fluorescence baseline series
    '''
    # Group data by ROI and run, and apply sliding window on F to compute baseline fluorescence
    groupkeys = [Label.ROI, Label.RUN]
    nconds = np.prod([len(data.index.unique(level=k)) for k in groupkeys])
    # Apply function to each ROI & run, and log progress
    with tqdm(total=nconds - 1, position=0, leave=True) as pbar:
        baselines = data.groupby(groupkeys).transform(pbar_update(nan_proof(bfunc), pbar))
    return baselines


def find_response_peak(s, n_neighbors=0, full_output=False, **kwargs):
    '''
    Find the response peak (if any) of a signal
    
    :param s: pandas Series (or numpy array) containing the signal
    :param n_neighbors: number of neighboring elements to include on each side
        to compute average value around the peak
    :param full_output: whether to return a full dictionary of peak properties
    '''
    # If series input, convert to array and extract index
    idx = None
    if isinstance(s, pd.Series):
        idx = s.index
        s = s.values
    
    # Find peaks
    ipeaks, props = find_peaks(s, **kwargs)

    # If no peak detected, set both peak value and index to NaN
    if ipeaks.size == 0: 
        ipeak, ypeak = np.nan, np.nan
    
    # Otherwise
    else:
        # Get index of max amplitude peak within the array
        imax = np.argmax(s[ipeaks])
        ipeak = ipeaks[imax]
        # Make sure it's not at the signal boundary
        if ipeak == 0 or ipeak == s.size - 1:
            raise ValueError(f'max peak found at signal boundary (index {ipeak})')
        # Compute average value of peak and its neighbors
        ypeak = np.mean(s[ipeak - n_neighbors:ipeak + n_neighbors + 1])
        # Select properties for that peak
        props = {k: v[imax] for k, v in props.items()}
        # If input was series, convert index to series index
        if idx is not None:
            ipeak = idx[ipeak]
            for k, v in props.items():
                if k.startswith('left') or k.startswith('right'):
                    if isinstance(v, int):
                        props[k] = idx[v]
                    else:
                        props[k] = np.interp(v, np.arange(len(idx)), idx)

    props['index'] = ipeak
    props['value'] = ypeak

    # If requested, return full output
    if full_output:
        return props
    
    # Otherwise, return only peak value
    else:
        return ypeak


def convert_peak_props(props, fs, ioffset=0):
    '''
    Convert peak properties from frames to seconds
    
    :param props: dictionary of peak properties
    :param fs: sampling frequency (Hz)
    :param ioffset (optional): index offset to apply to peak index metrics
    :return: dictionary of peak properties with time units
    '''
    props = props.copy()
    for k, v in props.items():
        if k.startswith(('left', 'right', 'index', 'widths')):
            if k != 'widths':
                v -= ioffset
            props[k] = v / fs
    return props


def find_max(s, n_neighbors=0):
    x = s.values
    if n_neighbors > 0:
        w = 2 * n_neighbors + 1
        x = np.convolve(x, np.ones(w), 'valid') / w
    return x.max()


def is_bounded(x, lb, ub):
    '''
    Determine if vector lies within an interval
    
    :param x: vector (array or series)
    :param lb: interval lower bound
    :param ub: interval upper bound
    :return: boolean stating whether vector lies within the interval (True) or not (False)
    '''
    if x.min() < lb:
        return False
    if x.max() > ub:
        return False
    return True


def compute_displacement_velocity(ops, mux, um_per_pixel, fps, isubs=None, full_output=False):
    '''
    Compute displacement velocity profiles frrom registration offsets
    
    :param ops: suite2p output options dictionary
    :param mux: (run, trial, frame) multi-index object
    :param um_per_pixel: spatial resolution of the images
    :param fps: sampling frequency (in frames/second)
    :param isubs (optional): indices of frames that have been substituted by 
        by their preceding frames
    :param full_output (default: False): whether to return the entire dataframe of intermediate
        metrics or simply the resulting displacement velocity series 
    :return: Series of displacement velocity profiles
    '''
    logger.info('computing diplacement velocity over time from registration offsets...')
    # Gather pixel offsets from reference frame
    df = pd.DataFrame({f'{k[0]} (pixels)': ops[k] for k in ['xoff', 'yoff']}, index=mux)
    # Add mean of sub-block offsets if present
    if 'xoff1' in ops:
        df[Label.X_PX] += ops['xoff1'].mean(axis=1)
        df[Label.Y_PX] += ops['yoff1'].mean(axis=1)
    # Compute Euclidean distance from reference frame (in pixels)
    df[Label.DISTANCE_PX] = np.sqrt(df[Label.X_PX]**2 + df[Label.Y_PX]**2)
    # Translate distance from pixels to microns
    df[Label.DISTANCE_UM] = df[Label.DISTANCE_PX] * um_per_pixel
    # Compute absolute displacement velocity (in um/frame) for each run independently
    df[Label.SPEED_UM_FRAME] = df[Label.DISTANCE_UM].groupby(Label.RUN).diff().abs()
    # Frames substitution (if applied) can create consecutive identical frames, resulting
    # in zero (or very low) displacement velocity artifacts at specific indexes. In this case, we
    # also substitute displacement velocity at these indexes by values at the preceding indexes.
    if isubs is not None:
        logger.info(
            f'correcting displacement velocity at indices {isubs} to compensate '
            'for frames substitution...')
        # Set substituted frames velocities to NaN
        df.loc[pd.IndexSlice[:, :, isubs], Label.SPEED_UM_FRAME] = np.nan
        # Interpolate stimulus frames velocities using forward fill method
        df = df.fillna(method='ffill')
        # Reset first velocity value of each run to NaN
        df.loc[pd.IndexSlice[:, 0, 0], Label.SPEED_UM_FRAME] = np.nan
    # Translate displacement velocity to um/s
    df[Label.SPEED_UM_S] = df[Label.SPEED_UM_FRAME] * fps
    # Return
    if full_output:
        return df
    else:
        return df[Label.SPEED_UM_S]


def apply_in_window(data, wslice, ykey=None, aggfunc='mean', weights=None, verbose=True, log_completion_rate=False):
    '''
    Apply function to a given signal within a specific observation window
    
    :param data: multi-indexed fluorescence timeseries series/dataframe
    :param ykey: name of the column containing the signal of interest
    :param wslice: slice object (or index) representing the indexes of the window
    :param aggfunc: aggregation function (name or callable)
    :param weights (optional): weights to apply to the samples prior to aggregation
    '''
    # If dataframe input, extract column containing signal of interest
    if isinstance(data, pd.DataFrame):
        if ykey is None:
            raise ValueError('ykey must be specified for dataframe input')
        data = data[ykey]

    # Extract non-frame index levels in input series
    idxlevels = [k for k in data.index.names if k != Label.FRAME]

    # If window slice is an integer, convert to slice object
    if isinstance(wslice, (int, np.int64)):
        wslice = slice(wslice, wslice + 1)
    
    # Compute slice length
    wlen = wslice.stop - wslice.start + 1
    
    # If weights vector provided
    if weights is not None:
        weights = np.asarray(weights)
        # Make sure it is of the same length as the slice
        if len(weights) != wlen:
            raise ValueError(f'weights vector length ({len(weights)}) must match slice length ({wlen})')

        # Make sure they sum up to slice length
        if np.abs(weights.sum() - wlen) > 1e-6:
            raise ValueError(f'weights must sum up to slice length ({wlen})')
    
    # If verbosity specified, log process
    if verbose:
        wstr = wslice.start
        if wslice.stop > wslice.start + 1:
            wstr = f'[{wstr}-{wslice.stop}] index window'
        else:
            wstr = f'{wstr} index'
        istr = ', '.join(idxlevels)
        funcstr = aggfunc.__name__ if callable(aggfunc) else aggfunc
        logstr = f'applying {funcstr} function on {data.name} in {wstr} across {istr}'
        if weights is not None:
            logstr = f'{logstr} with weights {weights}'
        logger.info(logstr)
    mux_slice = slice_last_dim(data.index, wslice)

    # Compute vector of slice indexes
    idx_slice = pd.Index(
        np.arange(wslice.start, wslice.stop + 1), name=Label.FRAME)
    
    # Extract slice data
    sdata = data.loc[mux_slice]

    # If weights are provided, multiply slice data
    if weights is not None:
        weights = pd.Series(weights, index=idx_slice, name='weights')
        sdata = sdata * weights

    # Group by non-frame index levels and apply aggregation function 
    out = sdata.groupby(idxlevels).agg(aggfunc)

    # If specified, log completion rate
    if log_completion_rate:
        outs_found = out.notna().sum()
        nwindows = len(data.groupby(idxlevels).first())
        out_pct = outs_found / nwindows * 100
        logger.info(
            f'identified outputs in {outs_found}/{nwindows} windows ({out_pct:.1f} %)')
    
    # Return aggregated output
    return out


def detect_across_trials(func, data, iwindow=None, key=Label.ZSCORE):
    '''
    Apply detection function in a given time window across all trials

    :param func: event detection function
    :param data: multi-indexed fluorescence timeseries dataframe
    :param iwindow: list (or slice) of indexes to consider (i.e. window of interest) in the trial interval
    :param key: name of the column containing the variable of interest
    :return: multi-indexed series of detect event values across conditions    
    '''
    # Compute number of trials to consider (i.e. those not starting by a NaN)
    ntrials = data[Label.ZSCORE].groupby([Label.ROI, Label.RUN, Label.TRIAL]).first().notna().sum()
    # Restrict dataset to given trial window if specified
    if iwindow is not None:
        data = data.loc[pd.IndexSlice[:, :, :, iwindow], key]
        s = f' in {iwindow} index window'
    else:
        s = ''
    # Apply detection function for each ROI, run and trial in the dataset
    logger.info(f'applying {func.__name__} function to detect events in {key} signals across trials{s}...')
    events = data.groupby([Label.ROI, Label.RUN, Label.TRIAL]).agg(func)
    # Compute number of detected events from function output
    nevents = events.notna().sum()
    # Log events detection statistics
    logger.info(f'identified {nevents} events over {ntrials} valid trials (detection rate = {nevents / ntrials * 1e2:.1f} %)')
    # Return events
    return events


def slide_along_trial(func, data, ref_wslice, iseeds):
    '''
    Call a specific function while sliding a detection window along the trial length.

    :param func: function called on each sliding iteration
    :param data: fluorescence timeseries data
    :param ref_wslice: reference window slice (in frames)
    :param iseeds: either the index list or the number of sliding iterations along the trial length
    :return: stacked function output series with window starting index as a new index level
    '''
    # Normalize reference window
    ref_wslice = slice(0, ref_wslice.stop - ref_wslice.start)
    # Extract available frame indexes
    iframes = data.index.unique(Label.FRAME)
    nframes_per_trial = iframes.size
    # Get window length
    wlen = ref_wslice.stop - ref_wslice.start
    # If not provided, generate vector of starting positions for the analysis window
    if isinstance(iseeds, int):
        iseeds = np.round(np.linspace(0, nframes_per_trial - wlen, iseeds)).astype(int)
        iseeds = iframes[iseeds]
    duplicates = set(iseeds) - set(np.unique(iseeds))
    if len(duplicates) > 0:
        raise ValueError(f'duplicate seeding indexes ({duplicates})')
    outs = []
    # For each starting position
    for i in tqdm(iseeds):
        # Call function and get output series
        out = func(data.copy(), shiftslice(ref_wslice, i))
        # Transform to dataframe and add column with istart info
        out = out.to_frame()
        out[Label.ISTART] = i
        # Append to list
        outs.append(out)
    # Concatenate outputs
    df = pd.concat(outs, axis=0)
    # Add istart to index
    df.set_index(Label.ISTART, append=True, inplace=True)
    # Return
    return df, iseeds
    

def add_time_to_table(data, key=Label.TIME, idxkey=Label.FRAME, fidx=None, fps=None):
    '''
    Add time information to info table
    
    :param data: dataframe contanining all the info about the experiment.
    :param key: name of the time column in the new info table
    :param idxkey: name of the index level containing reference indexes
    :param fidx (optional): frame indexer object
    :return: modified info table
    '''
    if key in data:
        logger.warning(f'"{key}" column is already present in dataframe -> ignoring')
        return data
    logger.info('adding time info to table...')
    # Extract sampling frequency
    if fps is None:
        fps = get_singleton(data, Label.FPS)
        del data[Label.FPS]
    if fidx is not None:
        frame_offset = fidx.iref
    else:
        frame_offset = 0.
    # Extract frame indexes
    try:
        idxs = data.index.get_level_values(idxkey)
    except KeyError:
        idxs = data[idxkey]
    # Add time column
    data[key] = (idxs - frame_offset) / fps
    # Set time as first column
    cols = data.columns
    data = data.reindex(columns=[cols[-1], *cols[:-1]])
    return data


def get_index_along_experiment(mux, reset_every=None, runid_map=None):
    '''
    Compute frame indexes along experiment specific temporal delimiter
    
    :param mux: multi-index of experiment dataframe 
    :param reset_every (optional): temporal delimiter at which to reset the frame index
    :param runid_map (optional): run <-> run ID mapper (useful for whole-experiment indexing) 
    :return: frame index array
    '''
    if reset_every is None and runid_map is None:
        raise ValueError('run ID mapper must be provided for full experiment indexing')
    # Extract frame indexes along trials
    frame_idxs = mux.get_level_values(Label.FRAME)
    # If temporal delimiter = trial, return index array as is
    if reset_every == Label.TRIAL:
        return frame_idxs
    # Compute index offset for each trial and add them to frame indexes
    npertrial = frame_idxs.max() + 1
    trial_idx_offsets = mux.get_level_values(Label.TRIAL) * npertrial
    frame_idxs += trial_idx_offsets
    # If temporal delimiter = run, return index array as is
    if reset_every == Label.RUN:
        return frame_idxs
    # Compute index offset for each run (based on run ID), and add them to frame indexes
    ntrials_per_run = len(mux.unique(Label.TRIAL))
    runid_map -= runid_map.min()
    run_idx_offsets = mux.get_level_values(Label.RUN).map(runid_map) * ntrials_per_run * npertrial
    frame_idxs += run_idx_offsets
    # Return frame indexes
    return frame_idxs


def add_intensity_to_table(table, precision=None):
    '''
    Add information about pulse and time average acoustic intensity to a table
    
    :param table: dataframe with pressure and duty cycle columns
    :param precision (optional): precision of the intensity values
    :return: dtaframe with extra intensity columns
    '''
    if Label.ISPTA not in table:
        logger.info('deriving acoustic intensity information...')
        table[Label.ISPPA] = pressure_to_intensity(table[Label.P] * 1e6) * 1e-4  # W/cm2
        table[Label.ISPTA] = table[Label.ISPPA] * table[Label.DC] * 1e-2   # W/cm2
        if precision is not None:
            table[Label.ISPTA] = table[Label.ISPTA].round(precision)
            table[Label.ISPPA] = table[Label.ISPPA].round(precision)
    return table


def get_response_types_per_ROI(data, verbose=True):
    '''
    Extract the response type per ROI from experiment dataframe.

    :param data: experiment dataframe
    :return: pandas Series of response types per ROI
    '''
    if verbose:
        logger.info('extracting responses types per ROI...')
    return data.groupby(Label.ROI).first()[Label.ROI_RESP_TYPE]


def aggregate_along(data, level=Label.TRIAL, aggfunc=None, verbose=True):
    '''
    Aggregate data along a given level of the index
    
    :param data: multi-indexed series (or dataframe)
    :param aggfunc: aggregation function
    :param level: index level to aggregate along
    :return: aggregated data
    '''
    if aggfunc is None:
        aggfunc = lambda s: s.mean()
    if verbose:
        logger.info(f'aggregating {describe_dataframe_index(data)} data along {level} dimension...')
    gby = [n for n in data.index.names if n != level]
    return data.groupby(gby).mean()


def get_trial_aggregated(data, aggfunc=None, full_output=False):
    '''
    Compute trial-aggregated statistics
    
    :param data: multi-indexed (ROI x run x trial) statistics dataframe
    :return: data aggregated across trials
    '''
    # Default aggregating function to mean if not provided
    if aggfunc is None:
        aggfunc = np.mean # lambda s: s.mean()
    
    # If data contains any non-numeric columns, make aggfunc string-proof
    is_numeric = [is_numeric_dtype(dt) for dt in data.dtypes.values]
    if not all(is_numeric):
        logger.info('data contains non-numeric columns -> string-proofing aggfunc')
        aggfunc = str_proof(aggfunc)

    # Extract non-trial index dimensions
    gby_dims = [k for k in data.index.names if k != Label.TRIAL]
    gby_str = ", ".join(gby_dims)

    # Log process 
    logger.info(f'computing trial-{aggfunc.__name__} of data over {gby_str}...')

    # Cast to dataframe
    is_series = isinstance(data, pd.Series)
    if is_series:
        data = data.to_frame()
    
    # Group data across grouping dimensions
    groups = data.groupby(gby_dims)

    # Compute aggregate of data across trials
    agg_data = groups.agg(aggfunc)
    
    # Remove time column if present
    if Label.TIME in agg_data:
        del agg_data[Label.TIME]

    # If channel2 ROI is present, re-cast to boolean
    if Label.CH2_ROI in agg_data:
        agg_data[Label.CH2_ROI] = agg_data[Label.CH2_ROI].astype(bool)
    
    # Rename relevant input columns to their trial-aggregated meaning
    # from constants import Label
    cols = {}
    for k, v in Label.RENAME_UPON_AGG.items():
        if k in agg_data:
            cols[k] = v
    if len(cols) > 0:
        agg_data.rename(columns=cols, inplace=True)

    # Resolve close elements for relevant input columns  
    for k in Label.RESOLVE_UPON_AGG:
        if k in agg_data:
            agg_data[k] = resolve_close_elements(agg_data[k])
    
    # Identify z-score (or z-score metrics) related columns in data
    is_zscore_col = [Label.ZSCORE in col for col in data.columns]

    # If z-score columns are present
    if any(is_zscore_col):
        # Compute number of valid trials per condition group
        logger.info(f'computing number of valid trials per {gby_str}')
        nvalid_trials_per_group = groups.count().iloc[:, 0]
        # Compute corresponding z-score scaling factors
        zfactors = np.sqrt(nvalid_trials_per_group)

        # Rescale z-score columns by scaling factors
        zscore_cols = list(data.columns[is_zscore_col])
        logger.info(f'rescaling z-score columns {zscore_cols}')
        for col in zscore_cols:
            agg_data[col] = agg_data[col] * zfactors
    
    # Cast back to series if input was a series
    if is_series:
        agg_data = agg_data.squeeze()

    # Return output
    if full_output:
        # Compute std of metrics across trials
        trialstd_data = groups.std()
        # Determine whether metrics is a repeated value or a real distribution
        is_repeat = ~(trialstd_data.max() > 0)
        return agg_data, is_repeat
    else:
        return agg_data


def weighted_average(data, avg_name, weight_name):
    '''
    Compute a weighted-average of a particular column of a dataframe using the weights
    of another column.
    
    :param data: dataframe
    :param avg_name: name of the column containing the values to average
    :param weight_name: name of the column containing the weights
    :return: weighted average
    '''
    d = data[avg_name]
    w = data[weight_name]
    return (d * w).sum() / w.sum()


def filter_data(data, iROI=None, irun=None, itrial=None, idataset=None, rtype=None, P=None, DC=None, tbounds=None, full_output=False):
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
    idxnames = data.index.names
    subindex = [slice(None)] * len(idxnames)
    if iROI is not None:
        subindex[idxnames.index(Label.ROI)] = iROI
        filters[Label.ROI] = f'{Label.ROI}{plural(iROI)} {iROI}'
    if irun is not None:
        subindex[idxnames.index(Label.RUN)] = irun
        filters[Label.RUN] = f'{Label.RUN}{plural(irun)} {irun}'
    if itrial is not None:
        subindex[idxnames.index(Label.TRIAL)] = itrial
        filters[Label.TRIAL] = f'{Label.TRIAL}{plural(itrial)} {itrial}'
    if idataset is not None:
        subindex[idxnames.index(Label.DATASET)] = idataset
        filters[Label.DATASET] = f'{Label.DATASET}{plural(idataset)} {idataset}'
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
    # No ROI selected  but ROI in data index -> indicate number of ROIs
    if iROI is None and Label.ROI in data.index.names:
        if Label.DATASET in data.index.names:
            nROIs = len(data.groupby([Label.DATASET, Label.ROI]).first())
        else:
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


def histogram_fit(data, func, bins=100, p0=None, bounds=None):
    '''
    Fit a specific function to a dataset's histogram distribution
    
    :param data: data array
    :param func: function to be fitted
    :param bins (optional): number of bins in histogram  (or bin edges values)
    :param p0 (optional): initial guess for function parameters
    :param bounds (optional): bounds for function parameters
    :return: (bin mid-points, fitted function parameters) tuple
    '''
    hist, xedges = np.histogram(data, bins=bins)
    xmid = (xedges[1:] + xedges[:-1]) / 2
    try:
        popt, _ = curve_fit(func, xmid, hist, p0=p0, bounds=bounds)
    except RuntimeError:
        logger.warning('histogram fit requires more iterations than expected...')
        popt, _ = curve_fit(func, xmid, hist, p0=p0, bounds=bounds, max_nfev=1000)
    return xmid, popt


def gauss_histogram_fit(data, bins=100, plot=False):
    '''
    Fit a gaussian function to a dataset's histogram distribution
    
    :param data: data array
    :param bins (optional): number of bins in histogram  (or bin edges values)
    :param plot (default: False): whether to plot fit results
    :return: (bin mid-points, fitted function parameters) tuple
    '''
    # Compute histogram distribution
    hist, edges = np.histogram(data, bins=bins)
    mids = (edges[1:] + edges[:-1]) / 2

    # Identify location of histogram peak and estimate its width
    ipeak = np.argmax(hist)
    pw = peak_widths(hist, [ipeak])[0][0] * (mids[1] - mids[0])

    # Initial guesses
    H = hist.min()  # vertical offset -> min histogram value
    A = np.ptp(hist)  # vertical range -> histogram amplitude range
    x0 = mids[ipeak]  # gaussian mean -> index of max histogram value
    sigma = pw / 2  # gaussian width -> 1/2 of the estimated histogram peak width
    p0 = (H, A, x0, sigma)

    # Bounds
    Hbounds = (0, H + A / 2)  # vertical offset -> between 0 and min + half-range
    Abounds = (A / 2, 2 * A)   # vertical range -> between half and twice initial guess
    x0bounds = (x0 - pw, x0 + pw)  # gaussian mean -> within 2 peak width ranges
    sigmabounds = (1e-10, 3 * pw)  # sigma -> between 0 and 3 peak width ranges (non-negativity constraint)
    pbounds = tuple(zip(*(Hbounds, Abounds, x0bounds, sigmabounds)))

    # Fit gaussian to histogram distribution
    try:
        xmid, popt = histogram_fit(data, gauss, bins=bins, p0=p0, bounds=pbounds)
    except (ValueError, RuntimeError) as err:
        logger.warning(err)
        xmid = mids
        fig, ax = plt.subplots()
        ax.hist(data, bins=50)
        popt = p0
        plot = True
        
    # Optional plot
    if plot:
        fig, ax = plt.subplots()
        ax.set_xlabel('signal values')
        ax.set_ylabel('count')
        for sk in ['top', 'right']:
            ax.spines[sk].set_visible(False)
        ax.plot(mids, hist, label='histogram')
        ax.plot(mids, gauss(mids, *p0), label='initial guess')
        ax.plot(mids, gauss(mids, *popt), label='fit')
        ax.legend(frameon=False)

    # Check that x-axis parameters (mean and std) are correct
    s = popt[3]
    if s <= 0.:
        err_msg = f'non-positive stdev found during Gaussian histogram fit'
        data_desc = 'Data:\n' + '\n'.join([f'  - {x}' for x in [
            f'range = {bounds(data)}',
            f'mean = {np.mean(data)}',
            f'stdev = {np.std(data)}'
        ]])
        popt_desc = 'Optimal parameters:\n' + '\n'.join([f'  - {x}' for x in [
            f'H = {popt[0]}',
            f'A = {popt[1]}',
            f'x0 = {popt[2]}',
            f'sigma = {popt[3]}'
        ]]) 
        logger.warning(
            f'{err_msg}\n{data_desc}\n{popt_desc}\n->falling back to data mean and stdev')
        popt = (0., 1., np.mean(data), np.std(data))

    # Return outputs
    return xmid, popt


def compute_correlation_coeffs(data, xkey, ykey):
    '''
    Compute correlation coefficients between an input (independent) variable and
    an output (dependent) variable for each ROI in a dataset.

    :param data: fluorescence dataset
    :param xkey: name of the column containing the input (independent) variable 
    :param ykey: name of the column containing the output (dependent) variable 
    :return: series of correlation coefficients per ROI
    '''
    # Trial-average if required
    if Label.TRIAL in data.index.names:
        logger.info('computing trial-averaged stats...')
        data = get_trial_aggregated(data)
    # Filter data according to independent variable
    if xkey not in data:
        raise ValueError(f'"{xkey}" variable not found in dataset')
    if ykey not in data:
        raise ValueError(f'"{ykey}" variable not found in dataset')
    if xkey == Label.P:
        data = filter_data(data, DC=DC_REF)
    elif xkey == Label.DC:
        data = filter_data(data, P=P_REF)
    else:
        logger.warning(f'x-variable "{xkey}" does not look independent...')
    # Group by ROI and compute correlation coefficients
    logger.info('computing correlation coefficients...')
    r = data.groupby(Label.ROI).apply(lambda df: df.corr().loc[xkey, ykey])
    # Rename and return
    return r.rename(f'r[{xkey}, {ykey}]')


def corrcoeff_to_tscore(r, n):
    '''
    Compute t-score associated with a given correlation coefficient
    
    :param r: correlation coefficient (scalar, array or series)
    :param n: sample size
    :return: associated t-score (same type as input)
    '''
    t = r * np.sqrt((n - 2) / (1 - r**2))
    if isinstance(r, pd.Series):
        t = t.rename(f't[{r.name}]')
    return t


def tscore_to_corrcoeff(t, n):
    '''
    Compute correlation coefficient associated with a given t-score
    
    :param t: t-test score (scalar, array or series)
    :param n: sample size
    :return: associated correlation coefficient (same type as input)
    '''
    x = t**2 / (n - 2)
    r = np.sqrt(x / (1 + x))
    if isinstance(t, pd.Series):
        r = r.rename(f'r[{r.name}]')
    return r


def tscore_to_pvalue(t, n, directional=True):
    '''
    Compute p-value associated with a given t-score
    
    :param t: t-score (scalar, array or series)
    :param n: sample size
    :param directional (default: True): whether to assume a directional effect (i.e. 1-tailed test) or not (i.e. 2-tailed test)
    :return: associated p-value (same type as input)
    '''
    p = tstats.sf(np.abs(t), df=n - 1)
    if directional:
        p *= 2
    if isinstance(t, pd.Series):
        p = pd.Series(data=p, index=t.index).rename(f'p[{t.name}]')
    return p


def pvalue_to_tscore(p, n, directional=True):
    '''
    Compute t-score associated with a given p-value
    
    :param p: p-value (scalar, array or series)
    :param n: sample size
    :param directional (default: True): whether to assume a directional effect (i.e. 1-tailed test) or not (i.e. 2-tailed test)
    :return: associated t-score (same type as input)
    '''
    if not directional:
        p /= 2
    t = tstats.ppf(1 - p, df=n - 1)
    if isinstance(t, pd.Series):
        t = pd.Series(data=t, index=p.index).rename(f't[{p.name}]')
    return t


def pvalue_to_zscore(p=.05, directional=False):
    '''
    Compute the z-score corresponding to a given chance probability level
    
    :param p: associated probability
    :param directional (default: True): whether to assume a directional effect (i.e. 1-tailed test) or not (i.e. 2-tailed test)
    :return: corresponding z-score
    '''
    if not directional:
        p /= 2
    return norm.ppf(1 - p)


def get_pvalue_per_sample(p, n):
    ''' 
    Return the p-value per sample given a p-value for the aggregate over n samples
    
    :param: p: aggregate p-value over sample distribution
    :param: n: number of samples
    :return: resolved p-value per sample
    '''
    # Get inverse probability 
    pinv = 1 - p
    # Probability that all samples are inferior = product of individual probabilities
    # Assuming all samples are independent, we therefore take the n-th root to get the
    # probability at the individual sample level
    pinv_sample = np.power(pinv, 1 / n)
    # Return inverse probability
    return 1 - pinv_sample


def get_zscore_mean(pthr, w, **kwargs):
    '''
    Get the value of the mean from a z-score distribution inside a sample window
    corresponding to a given significance criterion (p-value)
    
    :param pthr: significance threshold (p-value)
    :param w: window size (or window slice)
    :return: characteristic mean z-score
    '''
    # Vectorize function
    if isinstance(w, np.ndarray):
        return np.array([get_zscore_mean(pthr, ww) for ww in w]) 
    # If slice is given, compute size
    if isinstance(w, slice):
        w = w.stop - w.start
    # If window size smaller than 1 -> return NaN
    if w < 1:
        return np.nan
    # Compute and return z-score
    return pvalue_to_zscore(pthr, **kwargs) / np.sqrt(w)


def get_zscore_maximum(pthr, w):
    '''
    Get the value of the maximum from a z-score distribution inside a sample window
    corresponding to a given significance criterion (p-value)
    
    :param pthr: significance threshold (p-value)
    :param w: window size (or window slice)
    :return: characteristic maximum z-score
    '''
    # Vectorize function
    if isinstance(w, np.ndarray):
        return np.array([get_zscore_maximum(pthr, ww) for ww in w]) 
    # If slice is given, compute size
    if isinstance(w, slice):
        w = w.stop - w.start
    # If window size smaller than 1 -> return NaN
    if w < 1:
        return np.nan
    # Compute and return z-score
    return pvalue_to_zscore(get_pvalue_per_sample(pthr, w), directional=True)


def pre_post_ttest(s, fidx, wpre=None, wpost=None, directional=False):
    '''
    Select samples from pre- and post-stimulus windows and perform a t-test
    to test for their statistical significance
    
    :param s: input pandas Series
    :param frame indexer object
    :param wpre: pre-stimulus window slice
    :param wpost: post-stimulus window slice
    :param directional (default: False): whether to expect a directional effect
    :return tuple with t-statistics and associated p-value
    '''
    if wpre is None:
        wpre = fidx.get_window_slice('pre')
    if wpost is None:
        wpost = fidx.get_window_slice('post')
    xpre = s.loc[slice_last_dim(s.index, wpost)].values
    xpost = s.loc[slice_last_dim(s.index, wpre)].values
    tstat, pval = ttest_ind(
        xpre, xpost,
        equal_var=False, 
        nan_policy='raise',
        alternative='greater' if directional else 'two-sided')
    if np.isnan(tstat):
        # If NaN raised because all elements are equal -> return 0 and 0.5
        if np.unique(xpre) == np.unique(xpost):
            return 0., .5
        raise ValueError(f't-test between {xpre} and {xpost} returned NaN') 
    return tstat, pval


def zscore_to_resp_type(z, zthr, directional=False):
    ''' 
    Convert a z-score to a response type
    
    :param z: scalar or array of z-score(s)
    :param zthr: threshold z-score for response detection
    :param directional: whether to look for directional effect
    (i.e. positive side only) or not
    :return: response type representative string(s)
    '''
    dstr = 'directional ' if directional else ''
    logger.info(f'classifying responses using {dstr}z-score thresholding ...')
    # Generate response integer code from z-score
    if directional:
        # If directional effect, classify into 0 (weak) and 1 (positive)
        rcode = (z > zthr).astype(int)
    else:
        # Otherwise, classify into 0 (weak), -1 (negative) and 1 (positive)
        rcode = (np.abs(z) > zthr).astype(int) * np.sign(z)
    # Translate response code to response type string and return
    return rcode.map(RTYPE_MAP)


def is_valid_cond(isv):
    ''' 
    Identify valid conditions per sample as those having more than a 
    specific number of valid trials 
    
    :param isv: multi-index (sample, condition, trial) validity boolean series
    :return: multi-index (sample, condition) validity boolean series
    '''
    # Compute number of valid trials per sample
    groupby = list_difference(isv.index.names, [Label.TRIAL])
    nvalid_trials = isv.groupby(groupby).sum()
    # Identify samples with a minimum number of valid trials for averaging purposes
    logger.info(f'identifying conditions with >= {MIN_VALID_TRIALS} valid trials')
    return nvalid_trials >= MIN_VALID_TRIALS


def is_valid(data, keys=None, full_output=False):
    ''' 
    Return a series with an identical index as that of the input dataframe, indicating
    which rows are valid and must be included for response analysis.

    :param data: multi-index pandas experiment dataframe
    :param keys: list of validity keys to consider
    :param full_output (default: False): whether to return also a list of used validity keys
    :return: multi-index validity column (and optional list of used validity keys)
    '''
    # If no validity keys are provided, use default ones
    if keys is None:
        keys = TRIAL_VALIDITY_KEYS
    
    # Filter out validity keys that are not present in the dataframe
    cols = [k for k in keys if k in data.columns]
    
    # Check that validity columns are boolean
    if (data[cols].dtypes != bool).any():
        raise TypeError(f'invalidity columns must be boolean')
    
    # If no validity keys are present, simply label all samples as valid 
    if len(cols) == 0:
        isv = pd.Series(True, index=data.index).rename('valid?')
        nv, ntot = len(isv), len(isv)
    
    # Otherwise, identify samples without any invalidity criterion
    else:
        colstr = '[' + ', '.join(cols) + '] tags' if len(cols) > 1 else f'"{cols[0]}" tag'
        isv = ~data[cols].any(axis=1).rename('valid?')
        nv, ntot = isv.sum(), len(isv)
        ninv = ntot - nv
        pctinv = ninv / ntot * 1e2
        logger.info(f'identified {ninv}/{ntot} ({pctinv:.1f}%) samples with {colstr}')
    
    # Identify samples with a minimum number of valid trials for averaging purposes
    isv_cond = is_valid_cond(isv)
    isv_cond_exp = expand_to_match(isv_cond, isv.index)
    
    # Update validity index with that information
    isv = np.logical_and(isv, isv_cond_exp)
    ndiff = nv - isv.sum()
    if ndiff > 0:
        logger.info(f'filtered out additional {ndiff} ({ndiff * 1e2:.1f}) samples')
    
    # Return
    if full_output:
        return isv, cols
    else:
        return isv


def valid(df, **kwargs):
    ''' Return a copy of the dataframe with only valid rows that must be included for response analysis. '''
    isv, cols = is_valid(df, full_output=True, **kwargs)
    out = df.loc[isv, :].copy()
    for k in cols:
        del out[k]
    return out


def valid_timeseries(timeseries, stats, **kwargs):
    ''' Return a copy of a timeseries dataframe with only valid rows that must be included for response analysis. '''
    isv = is_valid(stats.copy(), **kwargs)
    logger.info('adding expanded validity index to timeseries ...')
    isv_exp = expand_to_match(isv, timeseries.index)
    logger.info('filtering timeseries ...')
    return timeseries.loc[isv_exp, :]
    

def nonzero(df):
    ''' Return a copy of the dataframe with only rows corresponding to trials with non-zero pressure '''
    return df.loc[df[Label.P] > 0., :]


def get_ROI_masks(stats, iROIs):
    '''
    Get a dataframe of ROI masks containing pixel coordinates and weight information
    
    :param stats: suite2p stats dictionary
    :param iROIs: ROIs index
    :return: ROI-indexed dataframe of (x, y) coordinates and weights
    '''
    if len(stats) != len(iROIs):
        raise ValueError(f'number of ROIs in stats ({len(stats)}) does not match index length ({len(iROIs)})')
    keys = ['ypix', 'xpix', 'lam']
    masks = []
    for iROI, stat in zip(iROIs, stats):
        mask = pd.DataFrame({k: stat[k] for k in keys})
        mask[Label.ROI] = iROI
        masks.append(mask)
    return pd.concat(masks).set_index(Label.ROI, drop=True)


def is_within_quantiles(s, qmin=0.25, qmax=0.75):
    '''
    Return boolean series stating which entries are within a given 
    quantile interval of the input
    
    :param s: pandas Series  object
    :param qmin: quantile of the lower bound
    :param qmax: quantile of the upper bound
    :return: boolean series
    '''
    xmin, xmax = s.quantile([qmin, qmax])
    return (s >= xmin) & (s <= xmax)


def get_quantile_slice(s, key=None, **kwargs):
    '''
    Get the values of a series that lie within a specific quantile interval
    
    :param s: pandas Series/Dataframe  object
    :param key: column key to apply quantile selection on (for dataframe input)
    :return: series/dataframe reduced only to its quantile slice constituents
    '''
    if isinstance(s, pd.DataFrame):
        if key is None:
            raise ValueError('key must be provided for dataframe input')
        cond = is_within_quantiles(s[key], **kwargs)
    else:
        cond = is_within_quantiles(s, **kwargs)
    return s[cond]


def get_quantile_indexes(data, qbounds, ykey, groupby=None):
    '''
    Return the row indexes of the dataframe that belong to a certain quantile interval
    
    :param data: multi-indexed experiment dataframe 
    :param qbounds: 2-tuple indicating the quantile lower and upper bounds
    :param ykey: the column of interest for the quantile slice estimation
    :param groupby (optional): whether to group the data by category before quantile selection
    :return: multi-index of the experiment dataframe specific to the quantile interval
    '''
    idxlevels_in = data.index.names
    logger.debug(f'input index: {idxlevels_in}')
    if qbounds is None:
        return data.index
    qmin, qmax = qbounds
    s = f'selecting {qmin} - {qmax} quantile slice from {ykey}'
    if groupby is not None:
        # Group the data by category
        data = data.groupby(groupby)
        # Expand groupby to iterbale if it is not
        if not is_iterable(groupby):
            groupby = [groupby[0]]
        # Fogure ou
        s = f'{s} for each {" & ".join(groupby)}'
    logger.info(f'{s}...')
    # Apply quantile selection on specific column
    yquantile = data[ykey].apply(
        lambda s: get_quantile_slice(s, qmin=qmin, qmax=qmax)).reset_index(level=1, drop=True)
    # Remove duplicate levels in output index
    logger.debug(f'output index: {yquantile.index.names}')
    counts = dict(Counter(yquantile.index.names))
    duplicates = [key for key, value in counts.items() if value > 1]
    for k in duplicates:
        i = yquantile.index.names.index(k)
        logger.debug(f'removing first instance of duplicate {k} (level {i})')
        yquantile.reset_index(level=i, inplace=True, drop=True)
    # Remove variables in output index that were not in original index from output index 
    for k in yquantile.index.names:
        if k not in idxlevels_in:
            logger.debug(f'removing index level "{k}" (not in original index)')
            yquantile = yquantile.reset_index(level=k)
    logger.debug(f'cleaned output index: {yquantile.index.names}')
    # Return multi-index
    return yquantile.index


def get_data_subset(data, subset_idx):
    '''
    Select a sbuset of traces based on a ROI, run, trial multi-index
    
    :param data: (ROI, run, trial, frame) multi-indexed traces dataframe
    :param subset_idx: (ROI, run, trial) subset index
    :return: (ROI, run, trial, frame) subset traces dataframe
    '''
    logger.info(f'creating new multi-index (with frame level) corresponding to index subset...')
    mux = repeat_along_new_dim(
        pd.DataFrame(index=subset_idx), Label.FRAME, data.index.unique(level=Label.FRAME)).index
    logger.info('selecting traces data from subset...')
    return data.loc[mux, :]


def correlations_to_rcode(corrtypes, j=', '):
    ''' 
    :param corrtypes: dataframe of integer codes {-1, 0 or 1} representing positive,
        nonexistent or negative correlations with different input parameters
    :param j: string used to join between inputs
    :return: series of corresponding response type codes         
    '''
    corrtypes = corrtypes.astype(int)
    # Define map of integers to suffixes
    suffix_map = {-1: '-', 0: 'o', 1: '+'}
    codes = []
    # For each input column
    for inputkey in corrtypes:
        # Get prefix key of the column
        prefix = inputkey.split()[0]
        # Build integer to code mapping dictionary
        code_map = {k: f'{prefix}{v}' for k, v in suffix_map.items()}
        # Apply mapping dictionary over each row entry of the column
        codes.append(corrtypes[inputkey].map(code_map))
    return pd.concat(codes, axis=1).agg(j.join, axis=1)


def get_default_rtypes(directional=False):
    ''' Get default response type codes '''
    rtypes = RTYPE.categories.values.tolist()
    if directional:
        del rtypes[rtypes.index('negative')]
    return rtypes


def get_change_key(y, full_output=False):
    '''
    Get change metrics key for a specific variable name
    
    :param y: variable name
    :param full_output: whether to return pre and post metrics key as well
    return: change metrics name
    '''
    # If input is iterable, apply function recursively to each list element
    if is_iterable(y):
        return [get_change_key(yy, full_output=full_output) for yy in y]

    # Generate keys for pre, post, and diff
    y_prestim_avg = f'pre-stim avg {y}'
    y_poststim_avg = f'post-stim avg {y}'
    y_change = f'evoked {y} change'

    # Return either all keys of only diff
    if full_output:
        return (y_prestim_avg, y_poststim_avg, y_change)
    return y_change


def compute_evoked_change(data, ykey, fidx, verbose=True, full_output=False):
    ''' 
    Compute stimulus-evoked change in specific variable
    
    :param data: timeseries dataframe
    :param ykey: evaluation variable name
    :param fidx: frame indexer object
    :param iref: reference frame index (default: stimulus frame)
    :param verbose: whether to print out results
    :param full_output: whether to return pre and post metrics as well
    :return: evoked change series, or stats dataframe if full_output is True
    '''    
    # Extract keys for pre- and post-stimulus averages and change
    y_prestim_avg, y_poststim_avg, y_change = get_change_key(ykey, full_output=True)
    
    # Define prefix:slice dictionary
    sdict = {
        y_prestim_avg: fidx.get_window_slice('pre'),
        y_poststim_avg: fidx.get_window_slice('post')
    }

    # Compute metrics average in pre- and post-stimulus windows for each ROI & run
    ystats = pd.DataFrame(
        {k: apply_in_window(data[ykey], s, verbose=verbose) for k, s in sdict.items()})

    # Compute evoked change as their difference
    if verbose:
        logger.info(f'computing {y_change}...')
    ystats[y_change] = ystats[y_poststim_avg] - ystats[y_prestim_avg]

    # Return
    if full_output:
        return ystats
    else:
        return ystats[y_change]


def get_xdep_data(data, xkey, add_DC0=False, dc_ref=DC_REF, p_ref=P_REF):
    '''
    Restrict data to relevant subset to estimate parameter dependency.
    
    :param data: multi-indexed experiment dataframe
    :param xkey: input parameter of interest (pressure or duty cycle)
    :param add_DC0: for DC sweeps, whether to add (DC = 0) data taken from P = 0
    :param dc_ref: reference duty cycle (defaults to DC_REF)
    :param p_ref: reference pressure (defaults to P_REF)
    :return: multi-indexed experiment dataframe containing only the row entries
        necessary to evaluate the dependency on the input parameter
    '''
    # If pressure sweep
    if xkey == Label.P:
        # Restrict data to pressures at reference duty cycle
        return data[data[Label.DC] == dc_ref]

    # If duty cycle sweep
    elif xkey == Label.DC:
        # Restrict data to duty cycles at reference pressure
        subdata = data[data[Label.P] == p_ref]
        # If DC = 0 values is required
        if add_DC0:
            # Extract (P = 0, DC = dc_ref) data, and re-format it as (P = Pref, DC = 0)
            data0 = data[data[Label.P] == 0.].copy()
            data0[Label.P] = p_ref
            data0[Label.DC] = 0
            # Add to sub-data and sort index
            subdata = pd.concat([data0, subdata]).sort_index()
        # Return
        return subdata
    
    # If unknown sweep, do nothing
    else:
        logger.info(f'{xkey} not part of ({Label.P}, {Label.DC}) -> no filtering')
        return data


def filter_datasets(*dfs, exclude=None):
    '''
    Filter data by excluding specific datasets from analysis
    
    :param data: list of multi-dataset dataframes
    :param exclude: exclusion criterion. This can be a list of specific datasets 
        (i.e. "date-mouse-region") to be discarded, or simply a common exclusion pattern.
    :return: filtered experiment dataframes
    '''
    # Cast exclude to list (if not None)
    exclude = as_iterable(exclude) if exclude is not None else None

    # If no exclusion -> return as is
    if exclude is None or len(exclude) == 0:
        logger.warning('no exclusion criterion -> ignoring')
        return dfs if len(dfs) > 1 else dfs[0]
    
    # Identify candidate datasets from first dataframe
    candidate_datasets = dfs[0].index.unique(level=Label.DATASET).values

    # Identify all datasets that have a partial match with any exclusion candidate  
    to_exclude = []
    matched_criteria = []
    # For each exclusion criterion
    for e in exclude:
        # Loop through datasets
        for cd in candidate_datasets:
            # If partial match detected
            if e in cd:
                # Add dataset to exclusion list, if not already there
                if cd not in to_exclude:
                    to_exclude.append(cd)
                # Add exclusion criterion to matched criteria list, if not already there
                if e not in matched_criteria:
                    matched_criteria.append(e) 
    
    # Raise warning if exclusion criteria have no match 
    notthere = list(set(exclude) - set(matched_criteria))
    if len(notthere) > 0:
        logger.warning(f'no match found for "{notthere}" exclusion criteria -> ignoring')

    # Intersect exclusion list with candidate datasets
    to_exclude = list(set(candidate_datasets).intersection(set(to_exclude)))

    # If no exclusion candidate in data, return as is
    if len(to_exclude) == 0:
        logger.warning('did not find any datasets to exclude')
        return dfs if len(dfs) > 1 else dfs[0]
    
    # Exclude and return
    logger.info(
        f'excluding the following datasets from analysis:\n{itemize(to_exclude)}')
    remaining = list(set(candidate_datasets) - set(to_exclude))
    logger.info(f'{len(remaining)} datasets remaining')
    query = f'{Label.DATASET} not in {to_exclude}'
    dfs_out = [df.query(query) for df in dfs]
    return dfs_out if len(dfs_out) > 1 else dfs_out[0]


def get_reference_stats_per_run(dfs):
    '''
    Extract reference stats-per-run dataframe from a list of multi-indexed experiment dataframes

    :param dfs: list (or dictionary) of multi-indexed experiment dataframes
    :return: reference stats-per-run dataframe
    '''
    # If dictionary, extract values list
    if isinstance(dfs, dict):
        dfs = list(dfs.values())
    
    # Filter out non-pandas objects
    dfs = [df for df in dfs if isinstance(df, (pd.Series, pd.DataFrame))]

    # Extract length and index dimensions for each dataframe, 
    # and identify those that are stats
    lengths = np.array([len(df) for df in dfs])
    dims = [df.index.names for df in dfs]
    is_stats = ~np.array([Label.FRAME in dim for dim in dims])

    # If no stats dataframe, raise error
    if not is_stats.any():
        raise ValueError('no stats dataframe found in input')

    # Extract shortest stats dataframe
    istats = np.where(is_stats)[0]
    stats_lengths = lengths[istats]
    irefstats = istats[np.argmin(lengths[istats])]
    refstats = dfs[irefstats]
    
    # Remove unecessary dimensions (i.e. anything but dataset and run)
    gby = [Label.DATASET, Label.RUN] if Label.DATASET in refstats.index.names else [Label.RUN]
    refstats_per_run = refstats.groupby(gby).first()

    # Return reference stats-per-run dataframe
    return refstats_per_run


def get_param_code(data):
    ''' Get a code string from stimulation parameters column '''
    # Parse P and DC columns to strings
    P_str = data[Label.P].map('{:01.2f}MPa'.format)
    DC_str = data[Label.DC].map('{:02.0f}%DC'.format)
    # Generate new column from concatenated (P, DC) combination 
    return pd.concat([P_str, DC_str], axis=1).agg('_'.join, axis=1)


def process_runids(s):
    ''' Process run IDs  in a dataset to uncover run sequence '''
    # If multi-dataset series, apply process for each dataset separately
    if Label.DATASET in s.index.names:
        if len(s.index.unique(Label.DATASET)) > 1:
            return s.groupby(Label.DATASET).transform(process_runids)
    # Extract run ID for each run
    org_runids = s.groupby(Label.RUN).first()
    unique_ids = s.unique()
    if len(unique_ids) != len(org_runids):
        raise ValueError(f'found more run IDs ({len(unique_ids)}) than runs ({len(org_runids)})')
    # Subtract minimum value to uncover run sequence
    new_runids = org_runids - org_runids.min()
    # Replace run IDs by run sequence
    runid_map = dict(zip(org_runids.values, new_runids.values))
    return s.map(runid_map).astype(int)
        

def get_offset_code(data):
    ''' Get an offset code string from the suffix column '''
    # Parse offset from suffix
    locs = data[Label.SUFFIX].apply(parse_2D_offset).rename(Label.OFFSET)
    # Convert to string
    return locs.apply(lambda x: f'{x[0]}x_{x[1]}y')


def get_buzzer_code(data):
    ''' Get an buzzer code string from the suffix column '''
    # Parse buzzer condition from suffix
    return data[Label.SUFFIX]


def get_offset_complex(data):
    ''' Get an offset complex number from the suffix column '''
    # Parse offset from suffix
    locs = data[Label.SUFFIX].apply(parse_2D_offset).rename(Label.OFFSET)
    # Convert to complex
    return locs.apply(lambda x: x[0] + 1j * x[1])
    

def get_duplicated_runs(data, condition='param'):
    '''
    Get potential duplicated runs in a dataset 
    '''
    # Extract condition per run
    stats_per_run = get_reference_stats_per_run(data)
    if condition == 'param':
        cond_per_run = get_param_code(stats_per_run)
    elif condition == 'offset':
        cond_per_run = get_offset_code(stats_per_run)
    elif condition == 'buzzer':
        cond_per_run = get_buzzer_code(stats_per_run)
    else:
        raise ValueError(f'unknown condition: "{condition}"')
    # Check for duplicates
    isdup = cond_per_run.duplicated(keep=False)
    # Return none if no duplicates are found
    if isdup.sum() == 0:
        return None
    else:
        # Otherwise, return duplicate table
        return cond_per_run[isdup]


def get_run_mapper(pcodes):
    '''
    Get a dictionary mapping string codes based on P-DC combination with run indexes
    (used to re-mapping datasets)  
    
    :param pcodes: parameter codes
    :return: mapping dictionary
    '''
    # Get unique values and assign ieach of them to run index to form mapping dictionary
    unique_pcodes = np.sort(pcodes.unique())
    idxs = np.arange(unique_pcodes.size)
    return dict(zip(unique_pcodes, idxs))


def update_run_index(data, runidx):
    ''' Get new run indexes column and update appropriate data index level '''
    # Extract current index as dataframe
    mux = data.index.to_frame()
    # Transform to dictionary
    muxdict = {k: mux[k].values for k in mux.columns}
    # Add entry for run index
    muxdict[Label.RUN] = runidx
    # Re-construct multi-index from dictionary entries
    new_mux = pd.MultiIndex.from_arrays(list(muxdict.values()), names=muxdict.keys())
    # Update data index
    data.index = new_mux
    # Return data
    return data


def harmonize_run_index(dfs, condition='param'):
    '''
    Generate a new harmonized run index in multi-region dataset, based on a specific condition
    (e.g. P-DC combination)
    
    :param dfs: dictionary of multi-indexed, multi-dataset dataframes
    :return: dictionary of dataframes with harmonized run indexes
    '''
    # Cast input as dictionary if necessary
    if not isinstance(dfs, dict):
        if isinstance(dfs, pd.DataFrame):
            dfs = {'_': dfs}
        else:
            raise TypeError(f'invalid input type: {type(dfs)}')

    logger.info(f'harmonizing run index by {condition} across datasets')    
    # Determine condition generating function
    try:
        condfunc = {
            'param': get_param_code,
            'offset': get_offset_code,
        }[condition]
    except KeyError:
        raise ValueError(f'unknown condition key: "{condition}"')

    # Initialize empty conditions dictionary 
    conds = {}
    logger.info(f'generating expanded conditions...')

    # Get conditions from each stat dataframe
    min_ndims, ref_minkey = np.inf, None 
    for k, df in dfs.items():
        if Label.FRAME not in df.index.names:
            conds[k] = condfunc(df).rename('condition')
            if conds[k].index.nlevels < min_ndims:
                min_ndims = conds[k].index.nlevels
                ref_minkey = k
    statkeys = list(conds.keys())
    
    # Loop through timeseries dataframes
    for k, df in dfs.items():
        if Label.FRAME in df.index.names:
            # Get condition series that contains all non-frame dimensions in
            # the timeseries with the fewest extra dimensions
            refdims = list(set(df.index.names) - {Label.FRAME})
            ref_sk, ref_extrakeys = None, None
            for sk in statkeys:
                if all(kk in conds[sk].index.names for kk in refdims):
                    extrakeys = list(set(conds[sk].index.names) - set(refdims))
                    if ref_extrakeys is None or len(extrakeys) < len(ref_extrakeys):
                        ref_sk, ref_extrakeys = sk, extrakeys

            # If no reference condition is found, raise error 
            if ref_sk is None:
                raise ValueError(f'could not find reference condition stats for {k} timeseries')

            # Otherwise, get reference condition
            refcond = conds[ref_sk]

            # If extra dimensions are present, reduce dimensionality of 
            # reference condition by sampling first value for each extra dimension 
            s = f'expanding "{ref_sk}" condition'
            if len(ref_extrakeys) > 0:
                gby = list(filter(lambda x: x not in ref_extrakeys, refcond.index.names))
                s = f'{s} w/o. {ref_extrakeys} dimensions'
                refcond = refcond.groupby(gby).first()
            
            # Create expanded condition compatible with timeseries
            s = f'{s} to match "{k}" timeseries'
            logger.info(s)
            conds[k] = expand_to_match(refcond, dfs[k].index)
    
    # Get condition: run-index mapper
    mapper = get_run_mapper(conds[ref_minkey])
    logger.debug(f'run map:\n{pd.Series(mapper)}')    
    
    # Get new run indexes column and update appropriate data index level
    dfs_out = {}
    for k, df in dfs.items():
        logger.info(f'updating {k} run index')
        dfs_out[k] = update_run_index(df, conds[k].map(mapper))
        
        # SANITY CHECK: Check that RUNID column has a unique value for each dataset and run
        if Label.RUNID in df.columns:
            for (d, r), runid in df[Label.RUNID].groupby([Label.DATASET, Label.RUN]):
                if runid.nunique() > 1:
                    raise ValueError(f'found non-unique run ID for dataset {d} and run {r}: {runid.unique()}')
    
    # If only one harmonized dataframe, return it
    if len(dfs_out) == 1 and '_' in dfs_out:
        return dfs_out['_']

    # Otherwise, return harmonized dataframes dictionary
    return dfs_out


def highlight_incomplete(x, xref=None):
    if is_iterable(x):
        return [highlight_incomplete(xx, xref=x.max()) for xx in x]
    if np.isnan(x):
        return 'color:red;'
    if xref is not None:
        if x != xref:
            return 'color:orange;'
    return ''


def get_detailed_ROI_count(data, style=False):
    ''' 
    Generate HTML table showing a detailed ROI count per dataset & run,
    along with parameter references 
    
    :param data: multi-dataset experiment stats dataframe
    :return: formatted HTML table
    '''
    # Get detailed ROI count per dataset & run
    ROI_detailed_count = data.groupby([Label.DATASET, Label.RUN]).apply(
        lambda gdata: len(gdata.index.unique(Label.ROI)))
    ROI_detailed_count = ROI_detailed_count.unstack()
    # Add parametric references
    params_per_run = data[[Label.P, Label.DC]].groupby(
        [Label.DATASET, Label.RUN]).first().groupby(Label.RUN).max()
    params_per_run[Label.P] = params_per_run[Label.P].map('{:01.2f}'.format)
    params_per_run[Label.DC] = params_per_run[Label.DC].map('{:02.0f}'.format)
    ROI_detailed_count.columns = pd.MultiIndex.from_arrays(
        [ROI_detailed_count.columns, params_per_run[Label.P], params_per_run[Label.DC]])
    # Format if specified
    if style:
        ROI_detailed_count = ROI_detailed_count.style.apply(
            highlight_incomplete, axis=1).format('{:.0f}')
    return ROI_detailed_count


def get_detailed_responder_counts(data, normalize=False):
    counts = (data[Label.ROI_RESP_TYPE]
        .groupby([Label.DATASET, Label.ROI])
        .first()
        .groupby(Label.DATASET)
        .value_counts()
        .unstack()
        .fillna(0.)
        .astype(int)
    )
    counts.loc['TOTAL'] = counts.sum()
    totals = counts.sum(axis=1)
    if normalize:
        counts = counts.div(totals, axis=0)
        counts['count'] = totals
    else:
        counts['total'] = totals
    return counts


def get_plot_data(timeseries, stats, fidx, keys=None):
    '''
    Get ready-to-plot dataframe by merging timeseries and stats dataframes
    and adding time information.

    :param timeseries: timeseries dataframe
    :param stats: stats dataframe
    :param keys: list of stats keys to add to timeseries
    :return: merged dataframe 
    '''
    # If no keys are specified, use default ones
    if keys is None:
        keys = Label.MERGE_UPON_PLT
    # Otherwise, make sure that all keys are present in stats
    else:
        keys = as_iterable(keys)
        for k in keys:
            if k not in stats.columns:
                raise ValueError(f'"{k}" not in stats dataframe')
    # Reduce stats dataframe to relevant keys
    stats = stats[keys]

    # Log process
    nstats = len(stats.columns.values)
    statstr = f'{nstats}-column stats' if nstats > 5 else stats.columns.values
    logger.info(f'adding {statstr} information to timeseries...')

    # Merge timeseries and stats dataframes
    plt_data = timeseries.copy()
    expand_and_add(stats, plt_data)

    # Add time information
    add_time_to_table(plt_data, fidx=fidx)

    # Return
    return plt_data


def eta_squared(anova_table):
    '''
    Compute eta-squared (η²) from ANOVA results table.

    :param anova_table: dataFrame containing the ANOVA results.
    :return: series with factor names as keys and eta-squared values as values.
    '''
    # Compute η² for each factor
    e2 = anova_table['sum_sq'] / anova_table['sum_sq'].sum()
    # Add Residual=NaN row for compatibility with ANOVA table 
    e2['Residual'] = np.nan    
    # Return
    return e2


def omega_squared(anova_table):
    '''
    Compute omega-squared (ω²) from ANOVA results table (from Kroes & Finley, 2023).

    :param anova_table: dataFrame containing the ANOVA results.
    :return: series with factor names as keys and omega-squared values as values.
    '''
    # Compute ω² for each factor
    MS_error = anova_table.loc['Residual', 'mean_sq']
    SS_total = anova_table['sum_sq'] + (anova_table['df'] * MS_error)
    w2 = (anova_table['sum_sq'] - (anova_table['df'] * MS_error)) / (SS_total + MS_error)
    # Add Residual=NaN row for compatibility with ANOVA table 
    w2['Residual'] = np.nan
    # Return
    return w2


def parse_factor(key, categorical):
    '''
    Parse factor name and type from key and categorical flag

    :param key: factor key
    :param categorical: whether the factor is categorical
    :return: parsed factor name
    '''
    if is_iterable(key):
        if not is_iterable(categorical):
            raise ValueError('categorical must be a a tuple of booleans')
        if len(key) != len(categorical):
            raise ValueError('key and categorical must have the same length')
        return [parse_factor(k, c) for k, c in zip(key, categorical)]
    if categorical:
        return f'C({key})'
    return key


def anova_formula(ykey, xkey, categorical=True, interaction=False):
    '''
    Generate a formula string for ANOVA analysis

    :param ykey: dependent variable key
    :param xkey: independent variable key
    :param categorical: whether the independent variable is categorical
    :param interaction: whether to include interaction term in the model
    :return: formula string
    '''
    # Make sure xkey and categorical have the same length
    if len(categorical) != len(xkey):
        raise ValueError('xkey and categorical must have the same length')
    # Apply "categorical" tag to specified xkeys
    xkeys = parse_factor(xkey, categorical)
    # Link xkeys with "+" or "*" operator depending on interaction flag
    opstr = ' * ' if interaction else ' + '
    xstr = opstr.join(xkeys)
    # Construct and return formula
    return f'{ykey} ~ {xstr}'


def anova(data, ykey, xkey, categorical=True, typ=1, interaction=False, full_output=False, gby=None, verbose=True):
    '''
    Wrapper around the statsmodels ANOVA functions to perform a 1 or 2-way
    ANOVA to assess whether a dependent variable is dependent
    on two given independent variables (i.e. group, factor).

    :param data: pandas dataframe
    :param xkey: name of column(s) (or index dimension(s)) containing the independent variable(s)
    :param ykey: name of column containing the dependent variable
    :param categorical: scalar or 2-tuple of booleans indicating whether the independent variables are categorical (default: True)
    :param typ: type of ANOVA to perform (default: 2)
    :param interaction: whether to include interaction term in the model
    :param gby: grouping variables
    :param full_output: whether to return full ANOVA table or just the p-values
    :param verbose: whether to print out results
    :return: p-values for dependency of y on factor(s) and their interaction, or full ANOVA table
    '''
    # Cast xkey and categorical to lists if not already
    xkey = as_iterable(xkey)
    # Check that number of independent variables is <= 2
    if len(xkey) > 2:
        raise ValueError('ANOVA currently supports maximum 2 independent variables')
    # If categorical is a scalar, apply it to all independent variables
    if not is_iterable(categorical):
        categorical = (categorical,) * len(xkey)
    # Otherwise, check that xkey and categorical have the same length
    elif len(xkey) != len(categorical):
        raise ValueError('xkey and categorical must have the same length')
    # Check that categorical is a tuple of booleans
    if not all(isinstance(c, bool) for c in categorical):
        raise ValueError('categorical must be a scalar or a tuple of booleans')

    # Log process, if requested
    if verbose:
        formula = anova_formula(
            ykey, xkey, categorical=categorical, interaction=interaction)
        s = f'performing "{formula}" type {typ} ANOVA'
        if gby is not None:
            s = f'{s} across {gby}'
        logger.info(s)
    
    # If gby is specified, apply 2-way ANOVA to each group
    if gby is not None:
        out = {}
        for glabel, gdata in tqdm(data.groupby(gby)):
            out[glabel] = anova(
                gdata, ykey, xkey, 
                categorical=categorical, typ=typ, interaction=interaction, 
                full_output=full_output, verbose=False)
        return pd.concat(
            out, 
            axis=0,
            names=gby,
            keys=out.keys(),
        )

    # Define placeholders for dependent and independent variables to ensure compatibility
    # with statsmodels formula API
    xplaceholders = [f'x{i}' for i in range(len(xkey))]
    yplaceholder = 'y'
    col_mapper = {xk: xp for xk, xp in zip(xkey, xplaceholders)}
    col_mapper[ykey] = yplaceholder

    # Extract and rename columns of interest
    data = data.reset_index()[[*xkey, ykey]].rename(columns=col_mapper)

    # Construct placeholder formula
    placeholder_formula = anova_formula(
        yplaceholder, xplaceholders, categorical=categorical, interaction=interaction)

    # Construct OLS model with data
    model = ols(placeholder_formula, data=data).fit()

    # Extract results table for appropriate ANOVA type
    anova_table = sm.stats.anova_lm(model, typ=typ)
    anova_table.index.name = 'factor'
    
    # Construct factors mapper 
    factor_mapper = dict(zip(
        parse_factor(xkey, categorical), parse_factor(xplaceholders, categorical)))
    if interaction:
        factor_mapper[':'.join(factor_mapper.keys())] = ':'.join(factor_mapper.values())

    # Rename factor rows
    anova_table = anova_table.rename(index={v: k for k, v in factor_mapper.items()})

    # If full output requested, add effect size columns
    if full_output:
        anova_table['η²'] = eta_squared(anova_table)
        if 'mean_sq' in anova_table.columns:
            anova_table['ω²'] = omega_squared(anova_table)
    
    # Rename some table columns for compatibility with other statistical tests
    anova_table = anova_table.rename(columns={
        'sum_sq': 'SS',
        'PR(>F)': 'p-value'
    })
    
    # Return
    if full_output:
        return anova_table
    else:
        pval = anova_table['p-value'].drop('Residual')
        if len(pval) == 1:
            pval = pval.iloc[0]
        return pval


def sum_of_square_devs(x):
    return ((x - x.mean())**2).sum()


def get_crossdataset_average(data, xkey, ykey=None, hue=None, weighted=True, errprop='intra',
                             add_global_avg=False):
    '''
    Average data output variable(s) across datasets for each value of an (or a combination of) input variable(s).
    
    :param data: multi-dataset dataframe
    :param xkey: name(s) of input variable(s) whose values determine aggregation groups
    :param ykey (optional): name of output variable(s) to average. If not provided, all output variables will be considered.
    :param hue: additional aggregation groups
    :param weighted: whether to compute a weighted-average based on the number of ROIs per dataset, or not
    :param errprop: whether to compute standard errors between datasets or to propagate it from each dataset
    :return: aggregated dataframe with mean and propagated sem columns for each considered output variable 
    '''
    if hue is not None and add_global_avg:
        out_by_hue = get_crossdataset_average(
            data, xkey, ykey=ykey, hue=hue, weighted=weighted, errprop=errprop, add_global_avg=False)
        out_avg = get_crossdataset_average(
            data, xkey, ykey=ykey, hue=None, weighted=weighted, errprop=errprop, add_global_avg=False)
        outlevels = out_avg.index.names
        out_avg[hue] = 'all'
        out_avg = out_avg.set_index(hue, append=True).reorder_levels([hue, *outlevels])
        return pd.concat([out_by_hue, out_avg], axis=0)

    # Format input
    xkey = as_iterable(xkey)

    # Log process
    desc = 'computing'
    if weighted:
        desc = f'{desc} ROI-weighted'
    ss = 'dataframe' if (ykey is None or len(as_iterable(ykey)) > 1) else f'"{ykey}" series'
    xkey_str = ' & '.join(xkey)
    logger.info(
        f'{desc} average of ({describe_dataframe_index(data)}) {ss} across {xkey_str}...')

    # Gather samples collection for counts
    countgby = xkey + [Label.DATASET]
    if weighted:
        countgby.append(Label.ROI)
    samples = data.groupby(countgby).first()
    
    # Compute number of samples per group
    nsamples = (samples
        .groupby(xkey + [Label.DATASET])
        .count().iloc[:, 0]
        .rename('counts')
    )

    # Compute associated weights per dataset, for each input value
    ntot = nsamples.groupby(xkey).sum()
    weightsperdataset = nsamples / ntot

    # Derive grouping categories (besides dataset): (hue,) and input value(s)
    gby = []
    if hue is not None:
        gby.append(hue)
    gby = gby + xkey

    # Groupby by dataset and grouping categories
    groups = data.groupby([Label.DATASET] + gby)

    # Identify columns that display intra-group variation and not
    maxnvalspercat = (
        groups.first()  # Take first row for each group
        .groupby(gby).nunique()  # Count number of unique values across datasets for each group
        .max(axis=0)  # Take group with max count, for each column
    )
    constkeys = maxnvalspercat[maxnvalspercat == 1].index.values
    varkeys = maxnvalspercat[maxnvalspercat > 1].index.values
    
    # Restrict varying columns to float types only
    float_columns = data.select_dtypes(include=['float64']).columns.values
    varkeys = list(set(varkeys).intersection(set(float_columns)))
    
    # If output key not specified, assign all varying columns
    if ykey is None:
        ykey = varkeys
    ykey = as_iterable(ykey)
    
    # Initialize weighted average dataframe with constant columns
    wdata = (groups[constkeys]  # Select constant columns
        .first()  # Get first value of each group
        .groupby(gby).first()  # Remove dataset dimension 
    )
    
    # Add counts column
    if hue is not None:
        # If hue provided, add counts per hue
        countsperhue = samples.groupby(hue).count().iloc[:, 0].rename('counts')
        wdata['count'] = wdata.index.get_level_values(hue).map(countsperhue)
    else:
        # Otherwise, add constant count
        wdata['count'] = len(samples)
    
    # For each output key
    for yk in ykey:
        # Compute means and standard errors (across ROIs) for each group
        means = groups[yk].mean()
        sems = groups[yk].sem()

        # Compute weighted mean across datasets
        mean = (
            (weightsperdataset * means)  # Muliply means by dataset weights
            .groupby(gby).sum()  # Sum across datasets
            .rename('mean')
        )

        # If error propagation is required
        if errprop is not None:
            # If global error must be propagated from SEs in each dataset
            if errprop == 'intra':     
                sem = np.sqrt(
                    (weightsperdataset * sems**2)  # Muliply squared standard errors by dataset weights
                    .groupby(gby).sum()  # Sum across datasets
                    .rename('sem')
                )  # Take square root of sum (propagated SE = weighted RMS of SEs)
            elif errprop == 'inter':
                # Compute standard error between hues for each input
                sem = (means
                    .groupby(gby).sem()  # Compute SE between means of each dataset 
                    .rename('sem')
                )
            else:
                raise ValueError(f'unknown error propagation mode: "{errprop}"')

            # Concatenate outputs, and add prefix
            ykey_wdata = pd.concat([mean, sem], axis=1)
            ykey_wdata = ykey_wdata.add_prefix(f'{yk} - ')
        
        # Otherwise, just add mean
        else:
            ykey_wdata = mean.rename(yk).to_frame()

        # Add outputs to weighted average dataframe
        wdata = pd.concat([wdata, ykey_wdata], axis=1)
    
    # Sort output dataframe by xkey value
    wdata = wdata.sort_values(xkey)

    # If xkey not in original index, move it out of output index 
    for k in xkey:
        if k not in data.index.names:
            wdata = wdata.reset_index(k)
    
    # Return weighted average data
    return wdata


def get_propagg_keys(y):
    ''' 
    Get propagated average and standard error keys for an input key
    
    :param y: input key
    :return: propagated average and standard error keys
    '''
    return f'{y} - mean', f'{y} - sem'


def interpolate_2d_map(df, xkey, ykey, outkey, dx=None, dy=None, method='linear'):
    '''
    Interpolate 2D map of sparse output metrics along X and Y dimensions
    
    :param df: input dataframe
    :param xkey: name of x-coordinates column
    :param ykey: name of y-coordinates column
    :param outkey: name of output metrics column
    :param dx: step size between interpolated coordinates in x-range (mm)
    :param dy: step size between interpolated coordinates in y-range (mm)
    :param method: type of interpolation (default = "linear")
    :return: 3-tuple with:
        - x-range vector
        - y-range vector
        - 2D XY grid of interpolated values
    '''
    # Extract XY points
    xp, yp = df[xkey].values, df[ykey].values
    points = np.vstack((xp, yp)).T
    # Generate range x and y vectors
    if dx is None:
        dx = np.diff(np.sort(np.unique(xp))).mean()
    else:
        if np.mod(np.ptp(xp) / dx, 1) != 0:
            raise ValueError(
                f'invalid x step size ({dx}): should be an integer divider of x range ({np.ptp(xp)})')
    if dy is None:
        dy = np.diff(np.sort(np.unique(yp))).mean()
    else:
        if np.mod(np.ptp(yp) / dy, 1) != 0:
            raise ValueError(
                f'invalid y step size ({dy}): should be an integer divider of y range ({np.ptp(yp)})')
    xrange = np.arange(xp.min(), xp.max() + dx, dx)
    yrange = np.arange(yp.min(), yp.max() + dy, dy)
    # Generate associated interpolation meshgrid
    X, Y = np.meshgrid(xrange, yrange, indexing='ij')
    xi = np.vstack((X.ravel(), Y.ravel())).T
    # Interpolate data & reshape into XY array
    interp_values = griddata(points, df[outkey].values, xi, method=method)
    interp_map = interp_values.reshape(xrange.size, yrange.size)
    # Return outputs
    return xrange, yrange, interp_map


def align_at(s, iframe):
    ''' 
    Align all traces of a series at a specific index (or index slice average)

    :param s: input multi-index series
    :param iframe: frame index (or index slice) at which to align traces
    :return: aligned series
    '''
    if isinstance(s, pd.DataFrame):
        return s.apply(lambda x: align_at(x, iframe))
    if isinstance(iframe, slice):
        fstr = f'average of frames {iframe.start} - {iframe.stop - 1}'
    else:
        fstr = f'frame {iframe}'
    logger.info(f'aligning {s.name} traces at {fstr}...')
    s_iframe = s.loc[slice_last_dim(s.index, iframe)]
    if Label.FRAME in s_iframe.index.names:
        levels = list(s_iframe.index.names)[:-1]
        s_iframe = s_iframe.groupby(levels).mean()
    return s - s_iframe


def get_cumulative_frame_index(mux):
    ''' Get cumulative frame index across trials for trial-frame multi-index '''
    itrial = mux.get_level_values(Label.TRIAL)
    iframe = mux.get_level_values(Label.FRAME)
    nframes_per_trial = iframe.max() + 1
    return (itrial * nframes_per_trial) + iframe


def get_quantile_index(s, **kwargs):
    return np.abs(s - s.median(**kwargs)).idxmin()


def offset_per_dataset(s, rel_offset=.5):
    ''' 
    Offset values to clearly separate datasets
    for visualization purposes.
    
    :param s: multi-index experiment series
    :param rel_offset: relative offset (in units of standard deviations) to apply to each dataset
    :return: dataframe with column offset dataset
    '''
    # Determine absolute offset of each dataset
    offset_factor = s.std() * rel_offset
    datasets = s.index.unique(Label.DATASET)
    offsets = pd.Series(
        data=-np.arange(len(datasets)) * offset_factor,
        index=datasets)
    # Apply offsets and return    
    def offset_func(ss):
        idx = ss.index.unique(Label.DATASET)[0]
        offset = offsets.loc[idx]
        return ss + offset
    return s.groupby(Label.DATASET).transform(offset_func)


def exclude_outliers(data, ykey, k=10):
    '''
    Exclude data points falling outside of median +/- k * stdev variation range

    :param stats: experiment statistics dataframe
    :param timeseries: experiment timeseries dataframe
    :param ykey: name of statistics variable of interest
    :param k: standard deviation multiplication factor determining range boundaries 
    :return: data subset of samples with ykey falling within median +/- k * stdev
    '''
    if k is None:
        return data
    y = data[ykey]
    ntot = len(data)
    mu, sig = y.median(), y.std()
    bounds = (mu - sig * k, mu + sig * k)
    iswithin = np.logical_and(y > bounds[0], y < bounds[1])
    nexc = ntot - iswithin.sum()
    bounds_str = ', '.join([f'{x:.2f}' for x in bounds])
    logger.info(
        f'excluded {nexc}/{ntot} ({nexc/ntot * 1e2:.1f} %) samples falling outside [{bounds_str}] interval')
    return data[iswithin]


def tmean(x):
    ''' 
    Estimate mean by fitting t-distribution to sample distribution
    
    :param x: input distribution
    :return: estimated mean
    '''
    return tstats.fit(x)[-2]


def get_power_spectrum(y, fs, method='welch', scaling='spectrum', remove_dc=True, 
                       normalize=False, add_db=True, add_suffix=False, **kwargs):
    '''
    Compute signal power spectrum and return it as a dataframe
    
    :param y: input signal
    :param fs: sampling frequency
    :param method: method to use for power spectrum estimation
    :param scaling: whether to return raw power spectrum ("spectrum") of power spectral density ("density")
    :param remove_dc: whether to remove DC component from signal before computing power spectrum
    :param normalize: whether to normalize power spectrum to its maximal value
    :param add_db: whether to add a column with power spectrum in dB
    :param add_suffix: whether to add variable suffix to the output power spectrum column 
    :param kwargs: additional arguments to spectrum estimation function
    :return: dataframe with frequency and power spectrum columns
    '''
    # If input is a dataframe, 
    if isinstance(y, pd.DataFrame):
        # Apply function to each column, and concatenate results
        out = pd.concat([
            get_power_spectrum(
                y[k], fs, method=method, scaling=scaling, remove_dc=remove_dc,
                normalize=normalize, add_db=add_db, add_suffix=True, **kwargs)
            for k in y], 
            axis=1)
        # Remove duplicate frequency columns and return
        return out.loc[:,~out.columns.duplicated()]

    # Determine number of samples
    nsamples = len(y)

    # Determine power spectrum output column name, and convert to numpy array if necessary
    spname = Label.PSPECTRUM
    if scaling == 'density':
        spname = f'{spname} density'
    if isinstance(y, pd.Series):
        if add_suffix and y.name is not None:
            spname = f'{y.name} {spname}'
        y = y.values

    # Mean-rectify signal to remove DC spectrum component, if requested
    if remove_dc:
        y -= y.mean()
    
    # Compute power spectrum with appropriate method
    if method == 'fft':
        # standard FFT method
        freqs = np.fft.rfftfreq(nsamples, d=1 / fs)
        yfft = np.fft.rfft(y)
        Pxx_spec = np.abs(yfft)**2
        if scaling == 'density':
            Pxx_spec /= nsamples
    elif method == 'stft':
        # short-time Fourier transform method
        freqs, _, Zxx = stft(y, fs=fs, **kwargs)
        Pxx_spec = np.mean(np.abs(Zxx)**2, axis=1)
        if scaling == 'density':
            Pxx_spec /= nsamples
    elif method == 'welch':
        # Welch's method
        freqs, Pxx_spec = welch(y, fs=fs, scaling=scaling, **kwargs)
    elif method == 'periodogram':
        # standard periodogram method
        freqs, Pxx_spec = periodogram(y, fs=fs, scaling=scaling, **kwargs)
    else:
        # Determine PSD kwargs for Spectrum package methods
        PSD_kwargs = dict(
            sampling=fs,
            scale_by_freq=False,
            **kwargs)

        if method == 'pma':
            # MA model
            pobj = spectrum.pma(y, 64, 128, **PSD_kwargs)
        elif method == 'pyule':
            # Yule Walker method
            pobj = spectrum.pyule(y, 7, **PSD_kwargs) 
        elif method == 'pburg':
            # Burg method
            pobj = spectrum.pburg(y, 7, **PSD_kwargs)
        elif method == 'pcovar':
            # covar method
            pobj = spectrum.pcovar(y, 7, **PSD_kwargs)
        elif method == 'pmodcovar':
            # mod covar method
            pobj = spectrum.pmodcovar(y, 7, **PSD_kwargs)
        elif method == 'pcorrelogram':
            # correlogram method
            pobj = spectrum.pcorrelogram(y, lag=60, **PSD_kwargs)
        elif method == 'pminvar':
            # minvar method
            pobj = spectrum.pminvar(y, 7, **PSD_kwargs)
        elif method == 'pmusic':
            # MUSIC method
            pobj = spectrum.pmusic(y, 10, NSIG=4, **PSD_kwargs)
        elif method == 'pev':
            # Eigenvalues method
            pobj = spectrum.pev(y, 10, 4, **PSD_kwargs)
        elif method == 'mtaper':
            # Multi-taper method
            pobj = spectrum.MultiTapering(y, NW=4.5, **PSD_kwargs)()
        else:
            # Unknown method
            raise ValueError(f'unknown spectrum estimation method "{method}"')

        # Extract frequency and power spectrum density from object  
        freqs, Pxx_spec = np.array(pobj.frequencies()), pobj.psd

        # Scale power spectrum if requested
        if scaling == 'spectrum':
            Pxx_spec *= nsamples
    
    # Normalize power spectrum if requested
    if normalize:
        Pxx_spec /= Pxx_spec.max()
    
    # Assemble as two-column dataframe
    df = pd.DataFrame({
        Label.FREQ: freqs,
        spname: Pxx_spec
    })

    # Convert to dB if requested
    if add_db:
        df[f'{spname} (dB)'] = 10 * np.log10(df[spname])
    
    # Set index name
    df.index.name = 'freq index'

    # Return
    return df


def get_offsets_by(s, by, y=None, rel_gap=.2, ascending=True, match_idx=False, verbose=True):
    ''' 
    Compute offsets to enable data separation between categories
    
    :param s: multi-indexed pandas Series
    :param by: name of separation variable in multi-index
    :param y: name of variable(s) to compute offsets for
    :param rel_gap: relative offset gap (w.r.t. mean variation range)
    :param ascending: whether offset should be ascending or descending
    :param match_idx: whether to expand offsets to match original index size
    :return: Series of offsets per category
    '''
    # Save original index names
    org_idx_names = s.index.names

    # Cast offset variable as iterable
    by = as_iterable(by)

    # Append grouping variable(s) to index (if not already there)
    for b in by:
        if b not in s.index.names:
            if isinstance(s, pd.DataFrame) and b in s.columns:
                s = s.set_index(b, append=True)
            else:
                raise ValueError(f'grouping variable "{b}" not found in index')
    
    # If input is dataframe, restrict to column(s) of interest and add axis to operations
    kwargs = {}
    if isinstance(s, pd.DataFrame) and y is not None:
        s = s[as_iterable(y)]
        kwargs['axis'] = 0

    # Log process
    if verbose:
        logger.info(f'computing offsets by {", ".join(by)}')

    # Compute data variation range for each category
    yranges = s.groupby(by).apply(np.ptp, **kwargs)

    # Compute associated gap from mean variation range, and add to ranges
    ygap = rel_gap * yranges.mean(**kwargs)
    yranges += ygap
    
    # Compute offset as cumulative sum of ranges + gap
    yoffsets = yranges.cumsum(**kwargs)
    
    # If descending offset, adjust sign
    if not ascending:
        yoffsets = -yoffsets
    
    # If specified, expand offsets series to match original index size
    if match_idx:
        extra_mux_levels = list(filter(lambda x: x not in as_iterable(by), s.index.names))
        if len(extra_mux_levels) > 0:
            yoffsets = free_expand(yoffsets, s, verbose=verbose)
    
    # Remove grouping variable that are not in original index from offsets index
    for b in by:
        # If variable not in index, remove it from offsets index
        if b not in org_idx_names:
            yoffsets = yoffsets.droplevel(by)

    # Return offsets
    return yoffsets


def offset_by(s, by, **kwargs):
    ''' 
    Offset series according to some index level
    
    :param s: pandas Series
    :param by: offseting categorical variable 
    :return: offseted series
    '''
    # Compute and add vertical offsets to y column
    return s + get_offsets_by(s, by, match_idx=True, **kwargs)


def get_responders_counts(data, xkey, units=None, normalize=False):
    '''
    Count the number of responder cells per condition
    
    :param data: statistics dataframe
    :param xkey: input variable
    :param units: extra grouping variables
    :param normalize: whether to normalize counts per group
    :return: dataframe of responder counts (or proportion) per group
    '''
    # Restrict data to input parameter dependency range
    data = get_xdep_data(data, xkey)
    # Add extra grouping variables, if any
    groupby = [xkey]
    if units is not None:
        if isinstance(units, list):
            groupby = groupby + units
        else:
            groupby.append(units)
    # Count number of responses of each type, for each grouping variables combination
    resp_counts = (data
        .groupby(groupby)[Label.RESP_TYPE]
        .value_counts()
        .unstack()
        .fillna(0.)
        .astype(int)
    )
    # Add total count column
    resp_counts['total'] = resp_counts.sum(axis=1)
    # If normalization specified
    if normalize:
        # Convert to proportions
        resp_props = resp_counts.div(resp_counts['total'], axis=0)
        resp_props['count'] = resp_counts['total']
        del resp_props['total']
        # Compute weights per input level
        resp_props['weight'] = (
            resp_props['count']
            .groupby(xkey)
            .transform(lambda s: s / s.sum())
        )
        # Return proportions
        return resp_props
    else:
        # Otherwise, return counts
        return resp_counts


def apply_test(data, groupby, testfunc, pthr=0.05):
    ''' Apply statistical test '''
    testres = data.groupby(groupby).agg(testfunc)
    testres = pd.DataFrame(
        testres.tolist(), columns=['stat', 'pval'], index=testres.index)
    testres['H0'] = testres['pval'] >= pthr
    return testres


def get_rtype_fractions_per_ROI(data, return_nconds=False):
    ''' 
    Get the fraction of response type over all relevant conditions, per ROI
    
    :param data: multi-index (ROI, run) stats dataframe
    :param return_nconds (default=False): whether to return the number of conditions used for classification
    :return: ROI stats dataframe 
    '''
    # Determine grouping variable(s) based on input index dimensions
    if Label.DATASET in data.index.names:
        gby = [Label.DATASET, Label.ROI]
    else:
        gby = Label.ROI
    
    # Save original ROIs list
    org_ROIs = data.groupby(gby).first().index.unique()

    # Filter data to only conditions with ISPTA values above certain threshold
    classification_data = data.loc[data[Label.ISPTA] > ISPTA_THR, :]

    # Compute number of conditions used for classification
    nconds = len(classification_data.index.unique(Label.RUN))

    # Compute response type fractions
    logger.info(f'computing fraction of response occurence per ROI over {nconds} "strong ISPTA" conditions...')
    roistats = (
        classification_data[Label.RESP_TYPE]
        .groupby(gby)
        .value_counts(normalize=True)
        .unstack()
        .fillna(0.)
    )

    # Extract ROIs list after filtering and conditioning
    filt_ROIs = roistats.index.unique()

    # Add "zero" fractions for ROIs that were filtered out by ISPTA criterion
    missing_ROIs = list(set(org_ROIs) - set(filt_ROIs))
    if len(missing_ROIs) > 0:
        logger.info(
            f'adding blank (0.) proportions for ROIs filtered out by ISPTA criterion {missing_ROIs}...')
        for ROI in missing_ROIs:
            roistats.loc[ROI, :] = 0.
    
    # Return output(s)
    if return_nconds:
        return roistats, nconds
    else:
        return roistats


def mylinregress(x, y, robust=False, intercept=True, return_model=False):
    '''
    Perform robust or standard linear regression between 2 1D arrays

    :param x: independent variable
    :param y: dependent variable
    :param robust: whether to perform robust linear regression
    :param intercept: whether to fit with or without intercept
    :param return_model: whether to return the model object 
        in addition to fit output (default = False)
    :return: fit output as a pandas Series, and optionally the model object
    '''
    # If intercept requested, add constant to input vector
    if intercept:
        x = sm.add_constant(x)
    
    # Construct OLS or RLM linear regression model, depending on "robust" flag
    if robust:
        model = sm.RLM(y, x)
    else:
        model = sm.OLS(y, x)

    # Fit model
    fit = model.fit()

    # Create fit output series
    fit_output = pd.Series(dtype=np.float64)

    # Extract fit parameters (slope and intercept)
    slopeidx = 0
    if intercept:
        fit_output['intercept'] = fit.params[0]
        slopeidx = 1
    else:
        fit_output['intercept'] = 0.
    fit_output['slope'] = fit.params[slopeidx]

    # Extract associated p-value for the slope
    fit_output['pval'] = fit.pvalues[slopeidx]

    # If OLM, extract R-squared value
    if not robust:
        fit_output['r2'] = fit.rsquared
    # Otherwise, compute R-squared value manually
    else:
        fit_output['r2'] = rsquared(y, fit.predict(x))

    # If specified, return fit output and model object
    if return_model:
        return fit_output, fit
    # Otherwise, return fit output
    else:
        return fit_output


def apply_linregress(df, xkey=Label.TRIAL, ykey=None, **kwargs):
    ''' 
    Apply linear regression between two column series

    :param df: input pandas Dataframe / Series object
    :param xkey: name of column / index dimension to use as input vector
    :param ykey: name of column to use as output vector (optional for series)
    :param robust: whether to use robust regression
    :return: pandas Series with regression output metrics 
    '''
    # Extract input vector
    if xkey in df.index.names:
        x = df.index.get_level_values(xkey)
    else:
        if isinstance(df, pd.Series):
            raise ValueError('xkey must be an index dimension for Series inputs')
        x = df[xkey].values
    
    # Extract output vector
    if isinstance(df, pd.Series):
        y = df.values
    else:
        if ykey is None:
            raise ValueError('ykey must be specified for DataFrame inputs')
        y = df[ykey].values
    
    return mylinregress(x, y, **kwargs)


def assess_significance(data, pthr, pval_key='pval', sign_key=None):
    '''
    Categorize responses by comparing p-values to a significance threshold, 
    and adding a potential "sign" measure to the output 
    '''
    sig = data[pval_key] < pthr
    if sign_key is not None:
        sig = (sig * np.sign(data[sign_key])).astype(int)
    return sig


def classify_ROIs(data, directional=False, return_nconds=False):
    ''' 
    Classify ROIs based on fraction of each response type across experiment
    
    :param data: trial-aggragated stats dataframe
    :param directional: whether to use uni-directional (i.e., weak vs. positive) classification 
        of tertiary (weak, positive, negative) classification
    :param return_nconds: whether to return the number of conditions used for classification
    :return: ROI classification stats dataframe
    '''
    # Compute fraction of response occurence in "strong" ISPTA conditions, for each ROI
    roistats, nconds = get_rtype_fractions_per_ROI(data, return_nconds=True)

    # Classify ROIs based on proportion of conditions in each response type
    logger.info('classiying ROIs as a function of their response occurence fractions...')

    # Add missing type columns filled with zeros 
    if 'weak' not in roistats.columns:
        roistats['weak'] = 0.
    if 'positive' not in roistats.columns:
        logger.warning('no positive responses found')
        roistats['positive'] = 0.
    if not directional and 'negative' not in roistats.columns:
        logger.warning('no negative responses found')
        roistats['negative'] = 0.

    # Extract non-weak response type with highest occurence frequency, for each ROI
    nonweakresps = (roistats
        .drop('weak', axis=1)
        .agg(['idxmax', 'max'], axis=1)
    )

    # Assess whether these occurence frequencies are above defined threshold
    isbignonweakfrac = nonweakresps['max'] >= PROP_CONDS_THR

    # Classify ROIs based on their response fractions
    roistats[Label.ROI_RESP_TYPE] = 'weak'
    roistats.loc[isbignonweakfrac, Label.ROI_RESP_TYPE] = nonweakresps.loc[isbignonweakfrac, 'idxmax']

    # Return output(s)
    if return_nconds:
        return roistats, nconds
    else:
        return roistats


def get_params_by_run(data, extra_dims=None):
    ''' 
    Get parameters by run
    
    :param data: multi-index stats dataframe
    :param extra_dims (optional): extra index dimensions to conserve in output
    :return: dataframe with parameters by run (and potential extra dimensions)
    '''
    # Check that run is in data index
    if Label.RUN not in data.index.names:
        raise ValueError(f'"{Label.RUN}" not found in index dimensions')
    gby = [Label.RUN]

    # Check validity of extra dimensions if any
    if extra_dims is not None:
        if isinstance(extra_dims, str):
            extra_dims = [extra_dims]
        for d in extra_dims:
            if d not in data.index.names:
                raise ValueError(f'"{d}" not found in index dimensions')
        # Add extra dimensions to groupby list, in same order as in original index
        gby = gby + extra_dims
        gby = list(filter(lambda x: x in gby, data.index.names))

    # Parameter keys to extract
    inputkeys = [Label.P, Label.DC, Label.ISPTA]
    
    # Extract first value of each parameter for each group
    first_params_by_run = data[inputkeys].groupby(gby).first()

    # Return
    return first_params_by_run


def find_in_dataframe(df, key):
    ''' Find a column in a dataframe, be it in index or as a data column '''
    if key in df.index.names:
        return df.index.get_level_values(key)
    else:
        return df[key]


def free_expand(s, ref_df, verbose=True):
    '''
    Expand series to a higher-dimensional dataframe
    
    :param s: input pandas Series object
    :param ref_df: reference dataframe to match index dimensions to
    :param verbose: whether to log process
    :return: expanded pandas Series object
    '''
    if verbose:
        # Log process
        s_desc = describe_dataframe_index(s)
        if isinstance(s, pd.Series):
            s_desc = f'({s_desc}) "{s.name}" series'
        elif isinstance(s, pd.DataFrame):
            s_desc = f'({s_desc}) input dataframe'
        ref_desc = f'({describe_dataframe_index(ref_df)}) reference dataframe'
        logger.info(f'expanding {s_desc} to match {ref_desc}')

    # If dataframe input, expand each constituent column
    if isinstance(s, pd.DataFrame):
        return pd.concat([free_expand(s[k], ref_df, verbose=False) for k in s], axis=1)

    # Extract index dimensions of series to expand
    gby = list(s.index.names)
   
    # Create function to find value in series and return an expanded vector
    def find_and_expand(df):
        idx_small = [find_in_dataframe(df, k)[0] for k in gby]
        val_small = s.loc[tuple(idx_small)]
        return pd.Series(index=df.index, data=val_small)

    # Group by small index dimensions and apply function
    out = ref_df.groupby(gby).apply(find_and_expand).rename(s.name)

    # Remove redundant index dimensions generated by apply, if any
    outdims = list(out.index.names)
    while len(outdims) > len(set(outdims)):
        out = out.droplevel(0)
        outdims = list(out.index.names)
    
    # Remove extra index dimensions generated by apply, if any
    extra_levels = list(filter(lambda x: x not in ref_df.index.names, out.index.names))
    if len(extra_levels) > 0:
        out = out.droplevel(extra_levels)

    # Sort index and return
    return out.sort_index()


def free_expand_and_add(smalldf, largedf, prefix=None):
    exp_smalldf = free_expand(smalldf, largedf)
    if prefix is not None:
        exp_smalldf = exp_smalldf.add_prefix(prefix)
    for k in exp_smalldf:
        largedf[k] = exp_smalldf[k]
    return largedf


def get_popavg_data(data, ykey=None):
    '''
    Compute population-average data
    
    :param data: multi-index stats dataframe
    :param ykey (optional): output column to average
    :return: population average stats dataframe
    '''
    # Extract grouping variables and group data accordingly
    gby = list(data.index.names)
    iroi = gby.index(Label.ROI)
    del gby[iroi]
    gby_str = '(' + ', '.join(gby) + ')'
    ykey_str = ykey + ' ' if ykey is not None else ''
    logger.info(f'computing {ykey_str}population average data across {gby_str}...')
    groups = data.groupby(gby)

    # Identify columns that vary across ROIs or not
    maxnvalspercat = groups.nunique().max(axis=0)
    constkeys = maxnvalspercat[maxnvalspercat == 1].index.values
    varkeys = maxnvalspercat[maxnvalspercat > 1].index.values

    # Remove object-typed columns from variable columns, if any
    objcols = data.dtypes[data.dtypes == object].index.values
    objvarkeys = list(set(varkeys).intersection(set(objcols)))
    if len(objvarkeys) > 0:
        logger.warning(f'ignoring object-typed columns {objvarkeys} in population average data')    
        varkeys = [k for k in varkeys if k not in objcols]

    if ykey is not None:
        varkeys = list(set(varkeys).intersection(as_iterable(ykey)))

    # Compute population-average dataframe 
    return pd.concat([
        groups[constkeys].first(),  # constant columns: first value of each group 
        groups[varkeys].mean()  # variable columns: mean of each group
    ], axis=1)


def compare_halves(df, ykey, testfunc, **testkwargs):
    ''' Compare distributions of specific variable between trials in both sequence halves '''
    # Extract trial sequences for both halves
    y1 = df[df['half'] == 1][ykey]
    y2 = df[df['half'] != 1][ykey]
    # Compare their distributions
    stat, pval = testfunc(y1, y2, nan_policy='raise', **testkwargs)
    # Average both distributions, and compute difference of means
    y1, y2 = y1.mean(), y2.mean()
    ydiff = y2 - y1
    # Return mmeans, differences, test stat, and p-value 
    return pd.Series(dict(zip(
        ['half1', 'half2', 'diff', 'stat', 'pval'], 
        [y1, y2, ydiff, stat, pval]
    )))


def extract_hilbert(s):
    '''
    Extract signal envelope mangnitude and instantaneous phase from a signal

    :param s: pandas series containing the signal
    :return: pandas dataframe containing the signal envelope and phase
    '''
    # Compute hilbert transform
    h = hilbert(s)
    
    # Return phase and envelope as dataframe
    return pd.DataFrame({
        Label.ENV: np.abs(h), 
        Label.PHASE: np.angle(h)
    }, index=s.index)


def remove_frames(df, fslice):
    '''
    Remove a range of frames from a dataframe
    
    :param df: input dataframe
    :param fslice: slice object specifying frames to remove
    :return: dataframe with frames removed
    '''
    # Identify frames falling inside defined frame slice
    is_in_slice = is_within(df.index.get_level_values(Label.FRAME), bounds(fslice))
    # Create copy of input data to avoid modifying it directly
    dfout = df.copy()
    # Remove identified frames
    dfout = dfout[~is_in_slice]
    # Return dataframe with frames removed
    return dfout


def circ_corrcl(alpha, x):
    '''
    Compute correlation between a circular and a linear variable

    Adapted from circ_corrcl function of CircStat package matlab toolbox:
    *Berens, P. (2009). CircStat: A MATLAB Toolbox for Circular Statistics. 
    J. Stat. Soft. 31. 10.18637/jss.v031.i10.*

    :param alpha: circular variable
    :param x: linear variable
    :return: circular-linear correlation coefficient, and associated p-value
    '''
    # Check that input dimensions match
    if len(alpha) != len(x):
        raise ValueError(f'Input dimensions ({len(alpha)}, {len(x)}) do not match')
    
    # Extract number of samples
    n = len(alpha)

    # If input is a pandas series, extract values and log process
    if isinstance(alpha, pd.Series) and isinstance(x, pd.Series):
        logger.info(f'computing circular-linear correlation between {alpha.name} and {x.name} (n = {n})')
        alpha, x = alpha.values, x.values

    # Compute circular-linear correlation coefficient
    rcx = np.corrcoef(np.cos(alpha), x)[0, 1]
    rsx = np.corrcoef(np.sin(alpha), x)[0, 1]
    rcs = np.corrcoef(np.cos(alpha), np.sin(alpha))[0, 1]
    rho = np.sqrt((rcx**2 + rsx**2 - 2 * rcx * rsx * rcs) / (1 - rcs**2))

    # Compute associated p-value from chi-square CDF with 2 degrees of freedom
    pval = 1 - chi2(2).cdf(n * rho**2)

    # Return correlation coefficient and p-value
    return rho, pval


def circ_mean(alpha):
    ''' Circular mean '''
    # Project circular data onto unit circle
    p = np.exp(1j * alpha)
    # Return angle of mean resultant vector
    return np.angle(p.mean())


def circ_corrcc(alpha, beta):
    '''
    Compute correlation between two circular variables

    Adapted from circ_corrcc function of CircStat package matlab toolbox:
    *Berens, P. (2009). CircStat: A MATLAB Toolbox for Circular Statistics.
    J. Stat. Soft. 31. 10.18637/jss.v031.i10.*

    :param alpha: first circular variable
    :param beta: second circular variable
    :return: circular-circular correlation coefficient, and associated p-value
    '''
    # Check that input dimensions match
    if len(alpha) != len(beta):
        raise ValueError(f'Input dimensions ({len(alpha)}, {len(beta)}) do not match')
    
    # Extract number of samples
    n = len(alpha)

    # If input is a pandas series, extract values and log process
    if isinstance(alpha, pd.Series) and isinstance(beta, pd.Series):
        logger.info(f'computing circular-circular correlation between {alpha.name} and {beta.name} (n = {n})')
        alpha, beta = alpha.values, beta.values

    # Compute angular mean of each variable
    mu_alpha = circ_mean(alpha)
    mu_beta = circ_mean(beta)

    # Compute relative angles
    alphar = alpha - mu_alpha
    betar = beta - mu_beta

    # Compute circular-circular correlation coefficient
    num = np.sum(np.sin(alphar)) * np.sum(np.sin(betar))
    den = np.sqrt(np.sum(np.sin(alphar)**2) * np.sum(np.sin(betar)**2))
    rho = num / den

    # Compute associated t-statistic
    l20 = np.mean(np.sin(alphar)**2)
    l02 = np.mean(np.sin(betar)**2)
    l22 = np.mean((np.sin(alphar)**2) * (np.sin(betar)**2))
    f = n * l20 * l02 / l22
    ts = np.sqrt(f) * rho
    
    # Compute associated p-value
    pval = 2 * (1 - norm.cdf(np.abs(ts)))

    # Return correlation coefficient and p-value
    return rho, pval


def pandas_circ_corrcl(data, ckey, lkey):
    '''
    Compute circular-linear correlation between a circular and a linear variable
    in a pandas dataframe

    :param data: input dataframe
    :param ckey: circular variable key
    :param lkey: linear variable key
    :return: 2-item series with circular-linear correlation coefficient, and associated p-value
    '''
    # Extract circular and linear variables
    c = data[ckey]
    l = data[lkey]
    
    # Compute circular-linear correlation
    rho, pval = circ_corrcl(c, l)

    # Return correlation coefficient and p-value as series
    return pd.Series(dict(zip(['rho', 'pval'], [rho, pval])))


def circ_rtest(alpha):
    '''
    Compute Rayleigh test for non-uniformity of circular data

    Adapted from circ_rtest function of CircStat package matlab toolbox:
    *Berens, P. (2009). CircStat: A MATLAB Toolbox for Circular Statistics.
    J. Stat. Soft. 31. 10.18637/jss.v031.i10.*

    :param alpha: circular variable
    :return: mean resultant vector, Rayleigh test statistic, and associated p-value
    '''
    # Extract number of samples 
    n = len(alpha)
    # Project circular data onto unit circle
    p = np.exp(1j * alpha)
    # Compute mean resultant vector
    r = np.abs(p.mean())
    # Compute Rayleigh's R
    R = n * r
    # Compute Rayleigh's z
    z = R**2 / n
    # Compute associated p-value
    pval = np.exp(np.sqrt(1 + 4 * n + 4 * (n**2 - R**2)) - (1 + 2 * n))
    # Return Rayleigh test statistic and p-value
    return r, z, pval


def get_frames_indexes(iref, irelstart=-2, irelend=12, every=1):
    '''
    Get indexes of frames relative to stimulus onset
    
    :param iref: reference index of stimulus onset
    :param irelstart: relative index of first frame w.r.t. stimulus onset (default = -2)
    :param irelend: relative index of last frame w.r.t. stimulus onset (default = 12)
    :param every: frame sampling rate (default = 1, i.e. every frame)
    :return: array of frame indexes
    '''
    return (np.arange(irelend - irelstart + 1) + irelstart + iref)[::every]


def phase_clustering(phi, aggby):
    ''' 
    Compute phase clustering of data along specified dimension. 

    :param phi: multi-indexed pandas Series with angular phase data
    :param aggby: dimension along which to aggregate data to compute phase clustering
    :return: pandas Series with phase clustering values
    '''
    # Compute phase projection onto unit circle
    logger.info(f'projecting {phi.name} onto unit circle')
    xy = complex_exponential(phi)

    # Check that aggregation dimension is valid
    dims = list(phi.index.names)
    if aggby not in dims:
        raise ValueError(f'{aggby} is not a dimension of {phi.name}')

    # Derive suffix from aggregating dimension
    k = aggby[0].upper()
    suffix = f'I{k}PC'

    # Derive groupby dimensions from aggregation dimension
    gby = [d for d in dims if d != aggby]

    # Group data and compute resultant mean vector amplitude in each group
    logger.info(f'computing inter-{aggby} {phi.name} clustering')
    return (xy
        .groupby(gby)
        .mean()
        .abs()
        .rename(f'{phi.name} {suffix}')
    )


def shuffle(y, keep_index=True, key=None, gby=None, verbose=True):
    '''
    Shuffle input
    
    :param y: input (pandas Series/DataFrame object or iterable)
    :param keep_index: whether to keep original index (only for pandas inputs)
    :param key: name of specific column for which to shuffle data (optional, only for DataFrame inputs)
    :param gby: name of grouping dimension(s) inside which to shuffle data (optional)
    :param verbose: verbosity flag
    :return: input with shuffled values
    '''
    # If column key specified
    if key is not None:
        # If not a DataFrame input, raise error
        if not isinstance(y, pd.DataFrame):
            raise ValueError('key can only be specified for DataFrame inputs')
        # If key not found in columns, raise error
        if key not in y.columns:
            raise ValueError(f'"{key}" not found in input DataFrame columns')
        # Copy dataframe, 
        yout = y.copy()
        # If gby specified, add gby columns to index to enable grouping
        extraidx = []
        if gby is not None:
            for gb in as_iterable(gby):
                if gb not in yout.index.names:
                    yout = yout.set_index(gb, append=True)
                    extraidx.append(gb)
        # Shuffle specified column and keeping overall index
        yout[key] = shuffle(yout[key], keep_index=True, gby=gby)
        # Reset added index dimensions as columns
        for gb in extraidx:
            yout = yout.reset_index(gb)
        # Return
        return yout
    
    # Assess whether input is a pandas object (Series or DataFrame)
    is_pandas_input = isinstance(y, (pd.Series, pd.DataFrame))

    # In case it is not
    if not is_pandas_input:
        # If not an iterable, raise error
        if not is_iterable(y):
            raise ValueError('input must be a pandas Series/DataFrame object or an iterable')
        # Otherwise, convert to Series
        y = pd.Series(y)

    # Log process if required 
    if verbose:
        s = f'shuffling {y.name} data'
        if gby is not None:
            s += f' within {gby} groups'
        logger.info(s)

    # If specified, shuffle data along specified dimension
    if gby is not None:
        groups = y.groupby(gby)
        with tqdm(total=len(groups)) as pbar:
            yout = groups.apply(pbar_update(lambda x: x.sample(frac=1).droplevel(gby), pbar))

    # Otherwise, shuffle data along its entire length
    else:
        yout = y.sample(frac=1)

    # If specified, keep original index
    if keep_index:
        yout.index = y.index

    # If input was not pandas object, convert back to numpy array
    if not is_pandas_input:
        yout = yout.values

    # Return
    return yout


def cyclic_shift(y, key=None, n=None, keep_index=True, gby=None, verbose=True):
    '''
    Cyclically shift a pandas Series by n positions.

    :param s: pandas Series
    :param key: name of specific column for which to shift data (optional, only for DataFrame inputs)
    :param n: number of positions to shift (default: random)
    :param keep_index: whether to keep original index (default: True)
    :param gby: dimension along which to group data before shifting
    :param verbose: verbosity flag
    :return: shifted pandas Series
    '''
    # Assess whether input is a pandas series (Series or DataFrame)
    is_pandas_input = isinstance(y, (pd.Series, pd.DataFrame))

    # Extract key if pandas input
    if is_pandas_input:
        # If dataframe input, make sure column of interest is present
        if isinstance(y, pd.DataFrame):
            if key is None:
                raise ValueError('key must be specified for DataFrame inputs')
            if key not in y.columns:
                raise ValueError(f'"{key}" not found in input DataFrame columns')
        # Otherwise, extract name of series
        else:
            key = y.name
    # Otherwise, set key to empty string
    else:
        key = ''

    # Log process if required 
    if verbose:
        s = f'performing cyclic shift of {key} data'
        if gby is not None:
            s += f' within {gby} groups'
        logger.info(s)

    # If specified, shuffle data along specified dimension
    if gby is not None:
        if not is_pandas_input:
            raise ValueError('groupby option only available for pandas Series/DataFrame inputs')
        groups = y.groupby(gby)
        with tqdm(total=len(groups)) as pbar:
            yout = groups.apply(pbar_update(
                lambda x: cyclic_shift(
                    x, key=key, n=n, keep_index=keep_index, verbose=False).droplevel(gby), 
                    pbar))
        return yout

    # In case it is not
    if not is_pandas_input:
        # If not an iterable, raise error
        if not is_iterable(y):
            raise ValueError('input must be a pandas Series/DataFrame object or an iterable')
        # Otherwise, convert to Series
        y = pd.Series(y)
    
    # If dataframe input, extract column of interest
    if isinstance(y, pd.DataFrame):
        y = y[key]

    # Randomly select shift value if not provided
    if n is None:
        n = np.random.randint(len(y))

    # Check that shift value is within bounds
    if n > len(y):
        raise ValueError(f'required shift ({n}) > input length ({len(y)})')

    # Perform cyclic shift
    sout = y.reindex(index=np.roll(y.index, n))

    # Keep original index if requested
    if keep_index:
        sout.index = y.index
    
    # If input was not pandas object, convert back to numpy array
    if not is_pandas_input:
        yout = yout.values

    # Return
    return sout


def get_correlation_matrix(s, by, sort=True, shuffle=False, remove_diag=False, remove_utri=False):
    '''
    Compute correlation matrix of data grouped by `by` dimension.

    :param s: multi-indexed pandas Series with data to correlate
    :param by: dimension along which to measure pairwise correlations
    :param sort: whether to sort correlation matrix by mean correlation
    :param shuffle: whether to shuffle data before computing correlation matrix
    :param remove_diag: whether to remove diagonal elements from correlation matrix
    :param remove_utri: whether to remove upper triangular elements from correlation matrix
    :return: pandas dataframe with correlation matrix
    '''
    # Check that correlation dimension is present in index
    if by not in s.index.names:
        raise ValueError(f'"{by}" dimension not found in index of input Series') 
    logger.info(f'computing {"shuffled " if shuffle else ""}{s.name} pairwise correlations between ROIs')
    # Unstack data along correlation dimension
    df = s.unstack(by)
    # Shuffle data, if specified
    if shuffle:
        df = df.apply(lambda x: x.sample(frac=1).to_numpy())
    # Compute pairwise correlation matrix
    corrs = df.corr()
    # Remove diagonal elements, if specified
    if remove_diag:
        np.fill_diagonal(corrs.values, np.nan)
    # Sort correlation matrix by mean correlation, if specified 
    if sort:
        isort = corrs.mean().sort_values(ascending=False).index
        corrs = corrs[isort].reindex(isort)
    # Remove upper triangular elements, if specified
    if remove_utri:
        corrs.values[np.triu_indices(len(corrs))] = np.nan
    # Return correlation matrix
    return corrs


def cross_correlate(s1, s2):
    '''
    Extract the cross-correlation between two 1-dimensional signals
    
    :param s1: first input signal (pandas Series or 1D array)
    :param s2: second input signal (pandas Series or 1D array)
    :return:
        - 1D array containing discrete linear cross-correlation of s1 with s2
        - 1D array of the corresponding index lags between s1 and s2 
        - optimal index lag yielding the highest correlation between s1 and s2
    '''
    # Compute cross-correlation
    corr = correlate(s1, s2) / s1.size
    # Compute associated lags
    lags = correlation_lags(len(s1), len(s2))
    # Compute optimal lag
    opt_lag = lags[np.argmax(corr)]
    # Return
    return corr, lags, opt_lag


def optimal_cross_correlation_lag(*args):
    return cross_correlate(*args)[2]


def compute_pairwise_metric(data, by, evalfunc, include_self=True):
    ''' 
    Compute pairwise metric between each "by" pair in multi-indexed data.

    :param data: multi-indexed pandas Series with data to correlate
    :param by: index dimension along which to measure pairwise correlations
    :param evalfunc: function to apply to each pairwise combination of data
    :param include_self: whether to include self-pairs in pairwise metric evaluation
    :return: 2D pandas dataframe with pairwise metric evaluations
    '''
    # Unstack series along evaluation dimension
    table = data.unstack(by)
    # Compute pairs of indices to evaluate
    idxs = table.columns
    pairs = list(combinations(idxs, 2))
    # Add self-pairs, if specified
    if include_self:
        pairs += [(i, i) for i in idxs]
    # Initialize square output dataframe
    out_table = pd.DataFrame(columns=idxs, index=idxs)
    # Compute metric for each pair of indices, and fill output dataframe
    logger.info(f'evaluating pairwise {evalfunc.__name__} across {len(idxs)} {by}...')
    for i1, i2 in tqdm(pairs):
        out = evalfunc(table[i1], table[i2])
        out_table.loc[i1, i2] = out
        if i1 != i2:
            out_table.loc[i2, i1] = -out
    # Return
    return out_table


def compute_fit(xdata, ydata, kind, r2_crit=0.2):
    '''
    Fit a function to an X-Y profile

    :param xdata: x data range
    :param ydata: y data range
    :param kind: string identifying the type of fit to perform
    :param r2_crit: fit R2 threshold to consider fit reliable (default = 0.5)
    :return: 4-tuple with
        - optimal fit parameters
        - covariance matrix of fit parameters
        - fit goodness (reported as r-squared coefficient)
        - fit function
    '''
    # If fit == "best", choose from every possible fit
    if kind == 'best':
        kind = list(fit_functions_dict.keys()) + ['poly1', 'poly2', 'poly3']

    # If multiple fit types specified, perform fit for each and return best one
    if is_iterable(kind):
        best_popt, best_pcov, best_r2, best_objfunc = None, None, 0, None 
        for k in kind:
            try:
                popt, pcov, r2, objfunc = compute_fit(xdata, ydata, k, r2_crit=r2_crit)
                if r2 > best_r2:
                    best_popt, best_pcov, best_r2, best_objfunc = popt, pcov, r2, objfunc
            except ValueError as e:
                logger.warning(f'cannot fit {k} function: {e}')
        if best_objfunc is None:
            raise ValueError('failed to fit any of the specified functions')
        logger.info(f'best fit type: {best_objfunc.__name__} (R2 = {best_r2:.2f})')
        return best_popt, best_pcov, best_r2, best_objfunc

    # If "poly" mode specified, perform polynomial fit
    if kind.startswith('poly'):
        # Get polynomial order
        order = int(kind[4])
        # Extract fit name 
        fitname = kind
        # Perform polynomial fit
        logger.info(f'fitting order {order} polynomial to data')
        popt, pcov = np.polyfit(xdata, ydata, order, full=False, cov=True)
        # Generate function to return polynomial value for a given x
        objfunc = lambda x, *y: np.poly1d(y)(x)

    # Otherwise, get fit functions from dictionary and perform classic fit
    else:
        fitobjs = get_fit_functions(kind)
        if len(fitobjs) == 3:
            objfunc, initfunc, boundsfunc = fitobjs
        else:
            objfunc, initfunc = fitobjs
            boundsfunc = None
        # Extract fit name
        fitname = objfunc.__name__
        # Call fit initialization function, if provided
        if initfunc is not None:
            logger.debug(f'estimating initial fit parameters for {fitname} function')
            p0 = initfunc(xdata, ydata)
            p0str = '[' + ', '.join([f'{p:.2g}' for p in p0]) + ']'
        else:
            p0, p0str = None, None
        
        # Call fit bounds function, if provided
        if boundsfunc is not None:
            logger.debug(f'estimating fit bounds for {fitname} function')
            bounds = boundsfunc(xdata, ydata)
        else:
            bounds = (-np.inf, np.inf)
        # Perform fit
        logger.info(f'computing fit with {fitname} function: p0 = {p0str}')
        try:
            popt, pcov = curve_fit(objfunc, xdata, ydata, p0, maxfev=100000, bounds=bounds)
        except RuntimeError as e:
            raise ValueError(e)
        
    # If fitting did not converge, log warning and return 
    if np.any(np.isinf(pcov)):
        raise ValueError('convergence error during fitting')
    
    # Compute output value with fit predictor
    yfit = objfunc(xdata, *popt)

    # Compute error between fit prediction and data
    r2 = rsquared(ydata, yfit)
    popt_str = '[' + ', '.join([f'{p:.2g}' for p in popt]) + ']'
    logger.info(f'fitting results: popt = {popt_str}, R2 = {r2:.2f}')

    # If fit is not good enough, log warning and return
    if r2 < r2_crit:
        raise ValueError(f'unreliable fit (R2 = {r2:.2f} < {r2_crit:.2f})')

    # Return
    return popt, pcov, r2, objfunc


def compute_fit_uncertainty(xvec, popt, pcov, objfunc, ci=.95, nsims=1000): 
    '''
    Compute uncertainty of fit parameters over an input range

    :param xvec: input range vector
    :param popt: optimal fit parameters
    :param pcov: covariance matrix of fit parameters
    :param objfunc: fit function
    :param ci: fit confidence interval (default = 0.95)
    :param nsims: number of Monte-Carlo simulations to perform to estimate
        fit confidence interval (default = 1000)
    :return: 2-tuple with lower and upper bounds of fit confidence interval
    '''
    # Compute standard error of fit parameters from covariance matrix
    perr = np.sqrt(np.diag(pcov))
    perr_str = '[' + ', '.join([f'{p:.2g}' for p in perr]) + ']'
    logger.info(f'fit parameter standard errors: {perr_str}')
    
    # Compute relative parameter errors w.r.t. optimal values
    rel_perr = np.abs(perr / popt)

    # If max relative error is too large, raise error
    max_rel_perr = np.max(rel_perr)
    if max_rel_perr > 5:
        raise ValueError(
            f'maximal fit parameter uncertainity ({max_rel_perr:.2g}) too large to estimate fit uncertainty')
        
    # Perform Monte Carlo simulation to estimate fit output distribution
    logger.info(f'computing fit uncertainty with {nsims} Monte Carlo simulations')
    yfits = np.full((nsims, xvec.size), np.nan)
    for isim in range(nsims):
        # Sample parameters from their respective confidence intervals
        sampled_params = np.random.normal(popt, perr)
        # Calculate the model output for the sampled parameters
        yfits[isim, :] = objfunc(xvec, *sampled_params)

    # If any invalid values in output samples, log warning and return 
    if np.any(np.isnan(yfits)) or np.any(np.isinf(yfits)):
        raise ValueError('invalid values in fit predictions from sampled parameters')

    # Calculate the confidence interval for the output
    alpha = 1 - ci
    yfit_lb = np.quantile(yfits, alpha / 2, axis=0)
    yfit_ub = np.quantile(yfits, 1 - alpha / 2, axis=0)
    
    # Otherwise, set confidence interval to None
    return yfit_lb, yfit_ub


def compute_predictor(*args, **kwargs):
    ''' 
    Wrapper around compute_fit to return a predictor function, i.e. the fit function
    called with optimal fit parameters
    '''
    popt, pcov, r2, objfunc = compute_fit(*args, **kwargs)
    return lambda x: objfunc(x, *popt)


def get_fit_table(Pfit='poly2', exclude=None):
    ''' 
    Generate 2D table of fit functions across cell lines and input parameters
    
    :param pfit: fit for pressure dependencies (default = 'scaled_power')
    :param exclude: lines to exclude from fit table (default = None)
    :return: pandas dataframe with fit functions
    '''
    # Determine non-pressure fit for each line
    fits_per_line = {
        'line3': 'corrected_sigmoid',
        'sarah_line3': 'corrected_sigmoid',
        'sst': 'corrected_sigmoid_decay',
        'pv':  'threshold_linear',  # 'sigmoid',
        'cre_sst': 'corrected_sigmoid',
        'cre_ndnf': 'corrected_sigmoid',
    }

    # Create empty 2D dataFframe
    fit_table = pd.DataFrame(
        columns=pd.Index(list(fits_per_line.keys()), name=Label.LINE),
        index=pd.Index([
            Label.P, 
            Label.DC, 
            Label.PSPTA, 
            Label.PSPTRMS, 
            Label.ISPTA, 
            Label.ISPTRMS
        ], name='parameter')
    )

    # Set fit for pressure dependency for all lines
    fit_table.loc[Label.P, :] = Pfit

    # Set line-specific fit for all other parameters
    for line, fit in fits_per_line.items():
        fit_table.loc[fit_table.index.drop(Label.P), line] = fit

    # Set excluded lines to None
    if exclude is not None:
        for line in as_iterable(exclude):
            fit_table.loc[:, line] = None

    # Return dataframe
    return fit_table


def classify_ternary(stats, key, rel_sigma_thr=2, plot=False):
    '''
    Classify a specific stat into "negative", "weak", and "positive" subsets 
    based on stat distribution.
    
    :param stats: stats dataframe
    :param key: key of stat column
    :param rel_sigma_thr: threshold for stat distribution relative to its central value
    :return: subsets Series with same index as in original stats dataframe 
    '''
    # Create copy of stats dataframe to avoid modifying it directly
    stats = stats.copy()

    # Fit gaussian to stat distribution
    logger.info(f'fitting Gaussian to {key} distribution...')
    *_, x0, sigma = gauss_histogram_fit(stats[key])[-1]

    # Determine classification boundaries from Gaussian parameters
    ybounds = [x0 - rel_sigma_thr * sigma, x0 + rel_sigma_thr * sigma]
    ybounds_str = ', '.join([f'{y:.2f}' for y in ybounds])
    logger.info(f'{key} bounds: ({ybounds_str}) (i.e. more than {rel_sigma_thr} sigma away from distribution center)')
    
    # Classify stat distribution into states
    logger.info(f'classifying {key} values')
    state_key = f'{key} state'
    stats[state_key] = 'weak'
    stats.loc[stats[key] < ybounds[0], state_key] = 'negative'
    stats.loc[stats[key] > ybounds[1], state_key] = 'positive'

    # If requested, plot histogram distribution of conditions, per state value
    if plot:
        fig, ax = plt.subplots()
        sns.despine(ax=ax)
        sns.histplot(
            data=stats,
            x=key, 
            ax=ax, 
            hue=state_key, 
            palette=Palette.RTYPE
        )
        for y in ybounds:
            ax.axvline(y, ls='--', c='k')
    
    # Return state series
    return stats[state_key]


def compute_vca(data, groupkey=None, **fit_kwargs):
    '''
    Perform a variance components analysis for a specific column

    :param data: multi-indexed pandas Series
    :param groupkey: key of grouping variable (optional). 
        If not specified, the first index dimension is used.
    :param fit_kwargs: additional keyword arguments to pass to fit function
    :return: statsmodels.MixedLMResults 
    '''
    # Extract index dimensions
    dimnames = list(data.index.names)
    logger.info(f'performing variance component analysis of {data.name} across {dimnames}...')
    
    # Rename series and add index dimensions to dataframe
    data = data.rename('y').reset_index()

    # Set all index dimensions as categorical
    for dim in dimnames:
        data[dim] = data[dim].astype('category')

    # Split index dimensions into grouping variable and categorical variables 
    if groupkey is not None:
        if groupkey not in dimnames:
            raise ValueError(f'"{groupkey}" not found in index dimensions')
        groupvar = groupkey
        catvars = [k for k in dimnames if k != groupkey]
    else:
        groupvar, *catvars = dimnames

    # Define main formula to fit "grand mean" model (to estimate 
    # components without any predictors)
    formula = 'y ~ 1'

    # Define variance structure of the model (i.e. random effects) with
    # random intercept at highest grouping level, to allow for variance
    # in baseline values across groups
    re_formula = '1'

    # Define variance components formula
    vc_formula = {k: f'0 + C({k})' for k in catvars}

    # Log model details
    logdetails = [
        f'main formula: {formula}',
        f'grouping variable: {groupvar}',
        f'random effects formula: {re_formula}',
        f'variance components formula: {vc_formula}',
    ]
    logdetails = [f'    - {d}' for d in logdetails]

    # Set up model
    logger.info('initializing mixed linear model with:\n' + '\n'.join(logdetails))
    model = sm.MixedLM.from_formula(
        formula, 
        re_formula=re_formula,
        vc_formula=vc_formula,
        groups=groupvar,
        data=data
    )

    # Compute VCA results
    logger.info('fitting mixed linear model...')
    res = model.fit(
        **fit_kwargs
    )

    # Return
    logger.info(f'VCA results:\n{res.summary().as_text()}')
    return res


def compute_crossROIs_correlations(data, key=None, remove_diag=True, remove_utri=True, 
                                   serialize=False, by=None):
    ''' 
    Compute pairwise correlation across ROIs for a specific statistics.
    
    :param data: multi-indexed pandas dataframe/series with data to correlate
    :param key: key of statistics to correlate (for dataframe inputs)
    :param remove_diag: whether to remove diagonal elements from correlation matrix
    :param remove_utri: whether to remove upper triangular elements from correlation matrix
    :param serialize: whether to serialize correlation matrix to 1D array
    :param by: dimension along which to measure pairwise correlations
    :return: 2D pandas dataframe with pairwise correlation matrix
    '''
    # If grouping dimension specified, compute pairwise correlations within each group
    if by is not None: 
        # groups = data.groupby(by)
        # with tqdm(total=len(groups)) as pbar:
        #     out = groups.apply(pbar_update(
        #         lambda x: compute_crossROIs_correlations(
        #             x, key=key, remove_diag=remove_diag, remove_utri=remove_utri, serialize=serialize), pbar))
        # return out.rename('R')
        return (data
            .groupby(by)
            .apply(lambda x: compute_crossROIs_correlations(
                x, key=key, remove_diag=remove_diag, remove_utri=remove_utri, serialize=True))
            .rename('R')
        )
    
    # For dataframe inputs, extract column of interest
    if isinstance(data, pd.DataFrame):
        if key is None:
            raise ValueError('key must be specified for DataFrame inputs')
        if key not in data.columns:
            raise ValueError(f'"{key}" not found in input DataFrame columns')
        data = data[key]

    # Gather non-ROI index dimensions
    extra_dims = [d for d in data.index.names if d != Label.ROI]
    
    # Unstack series into (ROI x samples) matrix
    M = data.unstack(level=extra_dims).T
    
    # Compute pairwise correlation matrix
    C = M.corr()

    # If specified, remove diagonal elements
    if remove_diag:
        np.fill_diagonal(C.values, np.nan)
    
    # If specified, remove upper triangular elements
    if remove_utri:
        C.values[np.triu_indices(len(C), k=1)] = np.nan
    
    # If specified, serialize correlation matrix to 1D array
    if serialize:
        C = C.stack().dropna().sort_index().rename('R')
        C.index.names = ['ROI1', 'ROI2']
        C.name = 'R'

    # Return 
    return C


def extract_run_index(table, P=P_REF, DC=DC_REF):
    '''
    Extract run index for a given P, DC condition

    :param table: run-indexed pandas dataframe with experiment parameters
    :param P: pressure condition (default = P_PREF) 
    :param DC: duty cycle condition (default = DC_REF)
    :return: run index for specified P, DC condition
    '''
    # Extract parameters by run from info table
    pbyrun = get_params_by_run(table)
    # Identify target condition in info table
    iscond = (pbyrun[Label.P] == P) & (pbyrun[Label.DC] == DC) 
    # If no run found for target condition, raise error
    if iscond.sum() == 0:
        raise ValueError(f'no run found for P = {P}, DC = {DC}')
    # Return run index for target condition
    return pbyrun[iscond].index[0]


def compute_covariance_matrix(y):
    '''
    Compute covariance matrix of data grouped by "by" dimension.

    :param y: 2D (features x samples) input provided as pandas series
    :return: pandas dataframe with covariance matrix
    '''
    # Squeeze index levels with only one value
    y = squeeze_multiindex(y)
    # Unstack series along sample dimension
    Y = y.unstack().T
    # Return covariance matrix
    return Y.cov()


def shuffle_columns(data):
    ''' Shuffle values within each column of a 2D array '''
    # Create empty copy of input data
    shuffled_data = np.empty_like(data)
    # Shuffle values within each column
    for col_idx in range(data.shape[1]):
        col_values = data[:, col_idx]
        np.random.shuffle(col_values)
        shuffled_data[:, col_idx] = col_values
    # Return shuffled array
    return shuffled_data


def fit_PCA(y, n_components=None, mean_correct=True, norm=True, shuffle=False, verbose=True, gby=None, **kwargs):
    '''
    Fit Principal Component tensor to a 2D pandas series.

    :param y: 2D (features x samples) input provided as pandas series
    :param n_components: number of components in the PC tensor 
    :param mean_correct: bool, whether to mean-subtract each feature before decomposition
    :param norm: bool, whether to normalize each feature before decomposition
    :param shuffle: whether to shuffle 
    :param verbose: verbosity flag
    :param gby: dimension along which to group data before fitting
    :param kwargs: additional keyword arguments to pass to PCA model
    :return: PCA model object
    '''
    # Create tensor descriptor if log requested
    if verbose:
        if n_components is not None:
            tensor_prefix = f'{n_components}-component'
            if n_components > 1:
                tensor_prefix += 's'
            tensor_prefix = f'{tensor_prefix} '
        else:
            tensor_prefix = ''
        tensor_str = f'{tensor_prefix}PC tensor'

    # If grouping dimension specified, fit PCA within each group
    if gby is not None:
        # Log if requested
        if verbose:
            fkey, skey = excluded(y, gby)
            input_str = f'({fkey} x {skey}) {y.name} input'
            if shuffle:
                input_str = f'shuffled {input_str}'
            logger.info(f'fitting {tensor_str} to {input_str} across {gby}')

        groups = y.groupby(gby)
        with tqdm(total=len(groups)) as pbar:
            out = groups.apply(pbar_update(
                lambda x: fit_PCA(
                    x, n_components=n_components, mean_correct=mean_correct, norm=norm, shuffle=shuffle, verbose=False), pbar))
        return out.rename('PCA')
    
    # Squeeze index levels with only one value
    y = squeeze_multiindex(y)

    # Check input validity
    if not isinstance(y, pd.Series) or len(y.index.names) != 2:
        raise ValueError('input must be a 2D series')
    
    # Extract feature and sample keys 
    fkey, skey = y.index.names

    # Convert series to (samples x features) array, and extract dimensions
    X = mux_series_to_array(y).T
    ns, nf = X.shape

    # Log if requested
    if verbose:
        input_str = f'({nf} {fkey}s x {ns} {skey}s) {y.name} input'
        if shuffle:
            input_str = f'shuffled {input_str}'
        logger.info(f'fitting {tensor_str} to {input_str}')
    
    # If required, shuffle columns
    if shuffle:
        X = shuffle_columns(X)

    # Standardize input
    scaler = StandardScaler(with_mean=mean_correct, with_std=norm)
    X = scaler.fit_transform(X)

    # Fit PCA model to the data
    pca = PCA(n_components=n_components, **kwargs)
    pca.fit(X)

    # Return PCA object
    return pca


def extract_PC_loadings(pca, fkey, verbose=True):
    '''
    Extract PC loadings matrix as a 2D pandas series

    :param pca: PCA model object
    :param fkey: feature key
    :param verbose: verbosity flag
    :return: 2D pandas series with (PC, features) PC loadings
    '''
    # Log if requested
    if verbose:
        logger.info(f'extracting PC loadings from PCA results')

    # If input is a series, apply function to each element
    if isinstance(pca, pd.Series):
        df = pca.apply(
            lambda x: extract_PC_loadings(x, fkey, verbose=False))
        return df.stack()
    
    # Extract (PC, features) loadings matrix from PCA model
    W = pca.components_

    # Cast as 2D series
    name = 'loading'
    s = array_to_dataframe(W, name, dim_names=['PC', fkey])[name]
    
    # Set PC index values to start from 1 instead of 0
    W = s.unstack()
    W.index = W.index + 1
    s = W.stack().rename(name)

    # Return
    return s


def extract_explained_variance(pca, cumulative=False, verbose=True):
    ''' 
    Extract fraction of variance explained by each PC from a fitted PC tensor
    
    :param pca: PCA model object
    :param cumulative: bool, whether to compute cumulative explained variance
    :return: 1D pandas series with explained variance for each PC
    '''
    # Log if requested
    if verbose:
        logger.info(f'extracting explained variance from PCA results')

    # If input is a series, apply function to each element
    if isinstance(pca, pd.Series):
        df = pca.apply(
            lambda x: extract_explained_variance(x, cumulative=cumulative, verbose=False))
        return df.stack().rename('explained variance')
        
    # Extract explained variance from PCA object
    expvar = pd.Series(
        pca.explained_variance_ratio_,
        name='explained variance'
    )
    expvar.index = expvar.index + 1
    expvar.index.name = 'PC'

    # Compute cumulative variance if requested
    if cumulative:
        expvar = expvar.cumsum()
        expvar.name = f'cumulative {expvar.name}'
        expvar.index.name = '# PCs'
    
    # Return
    return expvar


def transform_PCA(y, pca, mean_correct=True, norm=True, verbose=True):
    '''
    Transform input 2D pandas series using a fitted PCA model.

    :param y: 2D (features x samples) input provided as pandas series
    :param pca: PCA model object
    :param mean_correct: bool, whether to mean-subtract each feature before transformation
    :param norm: bool, whether to normalize each feature before transformation
    :param verbose: verbosity flag
    :return: transformed 2D pandas series
    '''
    # Squeeze index levels with only one value
    y = squeeze_multiindex(y)

    # Check input validity
    if not isinstance(y, pd.Series) or len(y.index.names) != 2:
        raise ValueError('input must be a 2D series')

    # Extract feature and sample keys 
    fkey, skey = y.index.names

    # Convert series to (samples x features) array, and extract dimensions
    X = mux_series_to_array(y).T
    ns, nf = X.shape

    # Log if requested
    if verbose:
        logger.info(f'transforming ({nf} {fkey}s x {ns} {skey}) {y.name} input')

    # Standardize input
    scaler = StandardScaler(with_mean=mean_correct, with_std=norm)
    X = scaler.fit_transform(X)

    # Transform input using PCA model
    Xout = pca.transform(X)

    # Return transformed data as 2D series
    return array_to_dataframe(Xout.T, y.name, dim_names=['PC', skey])


def bimodality_coefficient(data):
    ''' Compute bimodality coefficient of data '''
    # Compute mean and standard deviation of the data
    mu = np.mean(data)
    sigma = np.std(data)
    
    # Compute probability densities for each data point using a normal distribution
    pdf_values = norm.pdf(data, loc=mu, scale=sigma)
    
    # Compute the mean of the probability densities
    pdf_mean = np.mean(pdf_values)
    
    # Compute the variance of the probability densities
    pdf_variance = np.var(pdf_values)
    
    # Compute the Bimodality Coefficient
    bc = pdf_variance / (pdf_mean**2)
    
    return bc


def compute_trajectory_angles(trajectory):
    ''' 
    Compute angles between point-to-point vectors in a trajectory.

    :param trajectory: numpy array of shape (npoints, ndims) representing the trajectory
    :return: vector of angles (in radians) between point-to-point vectors
    '''
    npoints, ndims = trajectory.shape
    logger.info(f'computing tangent vectors along {npoints} points {ndims}D trajectory')
    # Compute tangent vectors along trajectory
    tangents = np.diff(trajectory, axis=0)  # (npoints - 1, ndims)
    # Compute dot products between tangent vectors
    dotprods = np.array([np.dot(A, B) for A, B in zip(tangents[:-1], tangents[1:])])
    # Compute norms products of tangent vectors
    norms = np.linalg.norm(tangents, axis=1)
    normprods = norms[:-1] * norms[1:]
    # Compute angles between tangent vectors
    return np.arccos(dotprods / normprods)


def autoreg_predict(y, wbounds, order=None, surrogate=False):
    '''
    Use auto-regression model to predict post-stim data from pre-stim data  

    :param y: 1D trial and frame indexed timeseries
    :param wbounds: bounding indexes of prediction window
    :param order: auto-regresion model order. If None, infered from the starting
        index of the prediction window
    :param surrogate: whether to use surrogate signal to fit AR model
    ''' 
    # Parse order if needed
    if order is None:
        order = wbounds[0]

    # If surrogate signal requested, generate surrogate signal as training signal
    if surrogate:
        ytrain = generate_surrogate(y)
    else:
        ytrain = y.copy()
    
    # Fit AR model to training signal and extract fitted model parameters
    ar_model = sm.tsa.AutoReg(ytrain.values, order)
    ar_params = ar_model.fit().params

    # Generate new model to predict values on input signal using 
    # AR parameters fitted on training signal 
    ar_model = sm.tsa.AutoReg(y.values, order)

    # Predict post-stim values from pre-stim values
    ypred = y.copy()
    extradims = [k for k in y.index.names if k not in (Label.FRAME, Label.FRAMEROW)]
    iref = 0
    for _, yseg in ypred.groupby(extradims):
        ibounds = iref + wbounds
        ypred.iloc[ibounds[0]:ibounds[1] + 1] = ar_model.predict(
            ar_params,
            start=ibounds[0], 
            end=ibounds[1],
            dynamic=True,
        )
        iref += yseg.size 
    
    # Return predicted signal
    return ypred


def get_quantile_intervals(nbins):
    '''
    Return quantile intervals for a given number of bins
    '''
    return pd.IntervalIndex.from_breaks(np.linspace(0, 1, nbins + 1).round(2), closed='right')


def bin_by_quantile_intervals(data, ykey, nbins=10, bin_unit=None, gby=None, add_aggregate=True, binagg_func='median', add_to_data=False):
    '''
    Bin specific variable into quantile intervals

    :param data: pandas dataframe with data to bin
    :param ykey: key of variable to bin
    :param nbins: number of quantile intervals to use
    :param bin_unit: labelling unit of output quantile intervals. One of:
        - "data": data units
        - "quantile": quantile units
        - "desc": qualitative descriptors (only valid for nbins <= 3)
    :param gby: grouping key inside which to perform binning
    :param add_aggregate: whether to add another series with aggregate value in each quantile interval
    :param binagg_func: function to use for aggregation in each quantile interval
    :return: pandas series with binned data
    '''
    # Check number of bins validity
    if nbins < 2:
        raise ValueError('number of bins must be at least 2')
    
    # If bin unit not specified, infer from number of bins
    if bin_unit is None:
        bin_unit = 'desc' if nbins <= 3 else 'quantile'

    # Generate binning labels based on bin unit
    if bin_unit == 'data':
        binlabels = None
        bin_key = f'{ykey} bin'
    elif bin_unit == 'quantile':
        binlabels = get_quantile_intervals(nbins)
        bin_key = f'{ykey} qbin'
    elif bin_unit == 'desc':
        bin_key = f'{ykey} qcat'
        if nbins == 2:
            binlabels = ['low', 'high']
        elif nbins == 3:
            binlabels = ['low', 'medium', 'high']
        else:
            raise ValueError('cannot use qualitative descriptors for more than 3 bins')
    else:
        raise ValueError(f'invalid bin unit: {bin_unit}')
    
    # Work on data copy if new columns must not be added
    if not add_to_data:
        data = data.copy()
    
    # Define binning function
    def fbin(s):
        y = pd.qcut(s, nbins, precision=2)
        if binlabels is not None:
            ycat = pd.Categorical(y)
            mapper = dict(zip(ycat.categories, binlabels))
            return y.map(mapper)
        else:
            return y
    
    # Bin pre-stim data into quantile intervals, per dose level
    suffix = ''
    if gby is not None:
        suffix = f', per {gby}'
    logger.info(f'binning {ykey} data into {nbins} quantile intervals{suffix}')
    if gby is not None:
        data[bin_key] = (data
            .groupby(gby)
            [ykey]
            .apply(fbin)
            .droplevel(0)
        )
    else:
        data[bin_key] = fbin(data[ykey])

    # If requested, compute average value in each quantile interval
    if add_aggregate:
        binagg_key = f'{bin_key} agg'
        gby = bin_key if gby is None else [*as_iterable(gby), bin_key]
        avgbybin = (data
            .groupby(gby)
            [ykey]
            .agg(binagg_func)
        )
        data[binagg_key] = (data
            .groupby(gby)
            [ykey]
            .transform(lambda s: avgbybin.loc[s.name])
        )
        return data[[bin_key, binagg_key]]

    else:
        return data[bin_key]
