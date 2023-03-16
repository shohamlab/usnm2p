# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-15 10:13:54
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-03-16 12:09:31

''' Collection of utilities to process fluorescence signals outputed by suite2p. '''

from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import maximum_filter1d, minimum_filter1d, gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import butter, sosfiltfilt, find_peaks, peak_widths, welch
from scipy.stats import skew, norm, ttest_ind, linregress
from scipy.stats import t as tstats
from scipy.stats import f as fstats
from scipy.interpolate import griddata, interp1d
import statsmodels.api as sm
from statsmodels.formula.api import ols
from functools import wraps

from constants import *
from logger import logger
from utils import *
from parsers import parse_2D_offset

# Register tqdm progress functionality to pandas
tqdm.pandas()


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


def get_quantile_baseline_func(fs, wquantile, q=None, wsmooth=None, smooth=True):
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


def get_butter_filter_func(fs, fc, order=2, btype='low'):
    '''
    Construct a pandas compatible zero-phase filter function
    
    :param fs: sampling frequency (Hz)
    :param fc: tuple of cutoff frequencies (Hz)
    :param order: filter order
    '''
    # Log process
    fc_str = ' - '.join([f'{x:.3f} Hz' for x in as_iterable(fc)])
    logger.info(f'defining order {order} {btype} BW filter with fc = {fc_str}')
    # Determine Nyquist frequency
    nyq = fs / 2
    # Calculate Butterworth filter second-order sections
    sos = butter(order, np.asarray(fc) / nyq, btype=btype, output='sos')
    # Define filter function
    def myfilter(y):
        return sosfiltfilt(sos, y)
    # Make pandas proof and return filter function
    return pandas_proof(myfilter)


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


def compute_trial_trend(s, type='constant'):
    ''' Compute trend on trial trace '''
    # Extract values
    y = s.values
    # Generate index vector
    x = np.arange(y.size)
    # Generate validity vector
    isvalid = np.ones(y.size, dtype=bool) 
    isvalid[FrameIndex.RESP_EXT] = False
    # Perform linear fit on valid indices only
    if type == 'linear':
        m, b, *_ = linregress(x[isvalid], y[isvalid])
        yfit = m * x + b
    elif type == 'constant':
        yfit = y[isvalid].mean()
    else:
        raise ValueError(f'unknown detrending type: {type}')
    return yfit


def detrend_trial(s, **kwargs):
    # Subtract fit to data and return
    return s - compute_trial_trend(s, **kwargs)


def find_response_peak(s, n_neighbors=N_NEIGHBORS_PEAK, return_index=False):
    '''
    Find the response peak (if any) of a signal
    
    :param s: pandas Series containing the signal
    :param n_neighbors: number of neighboring elements to include on each side
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


def find_max(s, n_neighbors=N_NEIGHBORS_PEAK):
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


def apply_in_window(data, ykey, wslice, aggfunc='mean', verbose=True, log_completion_rate=False):
    '''
    Apply function to a given signal within a specific observation window
    
    :param data: multi-indexed fluorescence timeseries dataframe
    :param ykey: name of the column containing the signal of interest
    :param wslice: slice object representing the indexes of the window
    :param aggfunc: aggregation function (name or callable)
    '''
    idxlevels = [k for k in data.index.names if k != Label.FRAME]
    if verbose:
        wstr = wslice.start
        if wslice.stop > wslice.start + 1:
            wstr = f'[{wstr}-{wslice.stop - 1}] index window'
        else:
            wstr = f'{wstr} index'
        istr = ', '.join(idxlevels)
        funcstr = aggfunc.__name__ if callable(aggfunc) else aggfunc
        logger.info(
            f'applying {funcstr} function on {ykey} in {wstr} across {istr} ...')
    idx_slice = slice_last_dim(data.index, wslice)
    out = data.loc[idx_slice, ykey].groupby(idxlevels).agg(aggfunc)
    if log_completion_rate:
        outs_found = out.notna().sum()
        nwindows = len(data.groupby(idxlevels).first())
        out_pct = outs_found / nwindows * 100
        logger.info(
            f'identified outputs in {outs_found}/{nwindows} windows ({out_pct:.1f} %)')
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
    wlen = FrameIndex.RESPONSE.stop - FrameIndex.RESPONSE.start
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
    

def add_time_to_table(data, key=Label.TIME, frame_offset=FrameIndex.STIM, fps=None):
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
    if fps is None:
        fps = get_singleton(data, Label.FPS)
        del data[Label.FPS]
    # Extract frame indexes
    iframes = data.index.get_level_values(Label.FRAME)
    # Add time column
    data[key] = (iframes - frame_offset) / fps
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


def add_intensity_to_table(table):
    '''
    Add information about pulse and time average acoustic intensity to a table
    
    :param table: dataframe with pressure and duty cycle columns
    :return: dtaframe with extra intensity columns
    '''
    if Label.ISPTA not in table:
        logger.info('deriving acoustic intensity information...')
        table[Label.ISPPA] = pressure_to_intensity(table[Label.P] * 1e6) * 1e-4  # W/cm2
        table[Label.ISPTA] = table[Label.ISPPA] * table[Label.DC] * 1e-2   # W/cm2
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


def get_trial_aggregated(data, aggfunc=None, full_output=False, inner_call=False):
    '''
    Compute trial-aggregated statistics
    
    :param data: multi-indexed (ROI x run x trial) statistics dataframe
    :return: data aggregated across trials
    '''
    # Default aggregating function to mean if not provided
    if aggfunc is None:
        aggfunc = lambda s: s.mean()
    # Group trials
    dims = data.index.names
    assert dims[-1] == Label.TRIAL, 'trial is not the last index dimension'
    gby = dims[:-1]
    if not inner_call:
        logger.info(f'aggregating trial data over {", ".join(gby)}')
    if Label.DATASET in gby:
        return data.groupby(Label.DATASET).progress_apply(
            lambda gdata: get_trial_aggregated(
                gdata.droplevel(Label.DATASET), 
                aggfunc=aggfunc, 
                full_output=full_output, 
                inner_call=True)
        )

    groups = data.groupby(gby)
    # Compute average of stat across trials
    agg_data = groups.agg(str_proof(aggfunc))
    # DataFrame case
    if isinstance(agg_data, pd.DataFrame):
        # Remove time column if present
        if Label.TIME in agg_data:
            del agg_data[Label.TIME]
        # Rename relevant input columns to their trial-averaged meaning
        cols = {}
        for k, v in Label.RENAME_ON_AVERAGING.items():
            if k in agg_data:
                cols[k] = v
        if len(cols) > 0:
            agg_data.rename(columns=cols, inplace=True)
    # Series case
    else:
        # Rename input to its trial-average meaning if necessary
        if agg_data.name in Label.RENAME_ON_AVERAGING.keys():
            agg_data.name = Label.RENAME_ON_AVERAGING[agg_data.name]
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
    except ValueError as err:
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


def pre_post_ttest(s, wpre=FrameIndex.PRESTIM, wpost=FrameIndex.RESPONSE, directional=False):
    '''
    Select samples from pre- and post-stimulus windows and perform a t-test
    to test for their statistical significance
    
    :param s: input pandas Series
    :param directional (default: False): whether to expect a directional effect
    :return tuple with t-statistics and associated p-value
    '''
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


def is_valid(data):
    ''' 
    Return a series with an identical index as that of the input dataframe, indicating
    which rows are valid and must be included for response analysis.

    :param data: multi-index pandas experiment dataframe
    :return: multi-index validity column
    '''
    # Identify samples without any invalidity criterion
    cols = [k for k in TRIAL_VALIDITY_KEYS if k in data.columns]
    if len(cols) > 0:
        logger.info(f'identifying samples without [{", ".join(cols)}] tags')
    isv = ~data[cols].any(axis=1).rename('valid?')
    # Identify samples with a minimum number of valid trials for averaging purposes
    isv_cond = is_valid_cond(isv)
    isv_cond_exp = expand_to_match(isv_cond, isv.index)
    # Update validity index with that information
    isv = np.logical_and(isv, isv_cond_exp)
    # Return
    return isv


def valid(df):
    ''' Return a copy of the dataframe with only valid rows that must be included for response analysis. '''
    out = df.loc[is_valid(df), :].copy()
    cols = [k for k in TRIAL_VALIDITY_KEYS if k in df.columns]
    for k in cols:
        del out[k]
    return out


def valid_timeseries(timeseries, stats):
    ''' Return a copy of a timeseries dataframe with only valid rows that must be included for response analysis. '''
    isv = is_valid(stats.copy())
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


def get_quantile_slice(s, qmin=0.25, qmax=0.75):
    '''
    Get the values of a series that lie within a specific quantile interval
    
    :param s: pandas Series object
    :param qmin: quantile of the lower bound
    :param qmax: quantile of the upper bound
    :return: series reduced only to its quantile slice constituents
    '''
    xmin, xmax = s.quantile(qmin), s.quantile(qmax)
    return s[(s >= xmin) & (s <= xmax)]


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


def get_default_rtypes():
    ''' Get default response type codes '''
    return RTYPE.categories.values.tolist()


def get_change_key(y, full_output=False):
    '''
    Get change metrics key for a specific variable name
    
    :param y: variable name
    :param full_output: whether to return pre and post metrics key as well
    return: change metrics name
    '''
    y_prestim_avg = f'pre-stim avg {y}'
    y_poststim_avg = f'post-stim avg {y}'
    y_change = f'evoked {y} change'
    if full_output:
        return (y_prestim_avg, y_poststim_avg, y_change)
    return y_change


def compute_evoked_change(data, ykey, wpre=FrameIndex.PRESTIM, wpost=FrameIndex.RESPONSE):
    ''' 
    Compute stimulus-evoked change in specific variable
    
    :param data: timeseries dataframe
    :param ykey: evaluation variable name
    :return: evoked change series
    '''
    # Compute metrics average in pre-stimulus and response windows for each ROI & run
    ypre = apply_in_window(data, ykey, wpre)
    ypost = apply_in_window(data, ykey, wpost)
    # Compute eovked change as their difference
    return (ypost - ypre).rename(get_change_key(ykey))


def add_change_metrics(timeseries, stats, ykey, npre=None, npost=None):
    '''
    Compute change in a given variable between pre and post-stimulation windows
    
    :param timeseries: timeseries dataframe
    :param stats: stats dataframe
    :param ykey: evaluation variable
    :param npre: number of pre-stimulus samples
    :param npost: number of post-stimulus samples
    :return: updated stats dataframe
    '''
    # Determine new keys
    ykey_prestim_avg, ykey_poststim_avg, ykey_diff = get_change_key(ykey, full_output=True)
    if npre is None and npost is None and ykey_diff in stats:
        logger.warning(f'default {ykey_diff} already present in stats -> ignoring')
        return stats
    logger.info(f'adding {ykey_diff} metrics to stats dataset...')
    
    # Define series averaging function
    def series_avg(s):
        return s.mean()

    # Define windows sizes if not provided
    if npre is None:
        npre = FrameIndex.PRESTIM.stop - FrameIndex.PRESTIM.start - 1
    if npost is None:
        npost = FrameIndex.RESPONSE.stop - FrameIndex.RESPONSE.start - 1

    # Compute pre-and post-stimulus windows
    wpre = slice(FrameIndex.STIM - npre, FrameIndex.STIM + 1)
    wpost = slice(FrameIndex.STIM + 1, FrameIndex.STIM + 2 + npost)
    
    # Compute evoked change
    stats[ykey_diff] = compute_evoked_change(timeseries, ykey, wpre=wpre, wpost=wpost)
    
    # Return
    return stats


def get_xdep_data(data, xkey):
    '''
    Restrict data to relevant subset to estimate parameter dependency.
    
    :param data: multi-indexed experiment dataframe
    :param xkey: input parameter of interest (pressure or duty cycle)
    :return: multi-indexed experiment dataframe containing only the row entries
        necessary to evaluate the dependency on the input parameter
    '''
    if xkey == Label.P:
        return data[data[Label.DC] == DC_REF]
    elif xkey == Label.DC:
        return data[data[Label.P] == P_REF]
    else:
        logger.warning(f'{xkey} not part of ({Label.P}, {Label.DC}) -> no filtering')
        return data


def exclude_datasets(*dfs, to_exclude=None):
    '''
    Exclude specific datasets from analysis
    
    :param data: list of multi-dataset dataframes
    :param to_exclude: date-mouse-region combinations to be discarded
    :return: filtered experiment dataframes
    '''
    # If no exclusion -> return as is
    if to_exclude is None or len(to_exclude) == 0:
        logger.warning('empty exclude list -> ignoring')
        return dfs if len(dfs) > 1 else dfs[0]
    # Identify candidate datasets from first dataframe
    candidate_datasets = dfs[0].index.unique(level=Label.DATASET).values
    # Raise warning if exclusion candidates not found in data 
    notthere = list(set(to_exclude) - set(candidate_datasets))
    if len(notthere) > 0:
        logger.warning(f'{notthere} datasets not found -> ignoring') 
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
    first_by_run = data.groupby([Label.RUN]).first()
    if condition == 'param':
        cond_per_run = get_param_code(first_by_run)
    elif condition == 'offset':
        cond_per_run = get_offset_code(first_by_run)
    elif condition == 'buzzer':
        cond_per_run = get_buzzer_code(first_by_run)
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


def check_run_order(data, condition='param'):
    ''' Check run order consistency across datasets '''
    logger.info('checking for run order consistency across datasets...')
    # Extract condition per run for each dataset
    first_by_dataset_and_run = data.groupby([Label.DATASET, Label.RUN]).first()
    if condition == 'param':
        cond_per_run = get_param_code(first_by_dataset_and_run).unstack()
    elif condition == 'offset':
        cond_per_run = get_offset_code(first_by_dataset_and_run).unstack()
    else:
        raise ValueError(f'unrecognized condition: "{condition}"')
    # Drop duplicates to get unique sequences of parameters
    unique_cond_sequences = cond_per_run.drop_duplicates()
    nseqs = len(unique_cond_sequences) 
    # If differing sequences
    if nseqs > 1:
        # Get matches per sequence
        matches = {}
        for i, (seqlabel, ref_seq) in enumerate(unique_cond_sequences.iterrows()):
            key = f'seq {i}'
            matches[key] = []
            for iseq, seq in cond_per_run.iterrows():
                if seq.equals(ref_seq):
                    matches[key].append(iseq)
            if len(matches[key]) == 1:
                matches[seqlabel] = matches.pop(key)
        nmatches = [k if len(v) == 1 else f'{k} ({len(v)} matches)'
                    for k, v in matches.items()]
        unique_cond_sequences = unique_cond_sequences.transpose()
        unique_cond_sequences.columns = nmatches

        # Raise error
        raise ValueError(
            f'different run orders across datasets:\n{unique_cond_sequences}')


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
    mux = data.index.to_frame()
    muxdict = {k: mux[k].values for k in mux.columns}
    muxdict[Label.RUN] = runidx
    new_mux = pd.MultiIndex.from_arrays(list(muxdict.values()), names=muxdict.keys())
    data.index = new_mux
    return data


def harmonize_run_index(trialagg_timeseries, popagg_timeseries, stats, trialagg_stats, condition='param'):
    '''
    Generate a new harmonized run index in multi-region dataset, based on a specific condition
    (e.g. P-DC combination)
    
    :param trialagg_timeseries: multi-indexed timeseries dataframe containing multiple datasets
    :param stats: multi-indexed stats dataframe containing multiple datasets
    :return: dataframes tuple with harmonized run indexes
    '''
    logger.info(f'harmonizing run index by {condition} across datasets...')    
    # Determine condition generating function
    try:
        condfunc = {
            'param': get_param_code,
            'offset': get_offset_code,
        }[condition]
    except KeyError:
        raise ValueError(f'unknown condition key: "{condition}"')
    
    # Get conditions from extended and trial-aggregated stats
    trialagg_stats_conds = condfunc(trialagg_stats).rename('condition')
    stats_conds = condfunc(stats).rename('condition')
    
    # Expand trialagg conditions on frames to get conditions compatible with trialagg timeseries
    trialagg_timeseries_conds = expand_to_match(trialagg_stats_conds, trialagg_timeseries.index)
    
    # Remove ROI dimension from trial-aggregated conditions
    nonROI_groupby = list(filter(lambda x: x != Label.ROI, trialagg_stats_conds.index.names))
    popagg_trialagg_stats_conds = trialagg_stats_conds.groupby(nonROI_groupby).first()
    # Expand on trials to get conditions compatible with popagg timeseries
    popagg_timeseries_conds = expand_to_match(popagg_trialagg_stats_conds, popagg_timeseries.index)
    
    # Get stimparams: run-index mapper
    mapper = get_run_mapper(trialagg_stats_conds)
    logger.debug(f'run map:\n{pd.Series(mapper)}')
    
    # Get new run indexes column and update appropriate data index level
    stats = update_run_index(stats, stats_conds.map(mapper))
    trialagg_stats = update_run_index(trialagg_stats, trialagg_stats_conds.map(mapper))
    trialagg_timeseries = update_run_index(
        trialagg_timeseries, trialagg_timeseries_conds.map(mapper))
    popagg_timeseries = update_run_index(
        popagg_timeseries, popagg_timeseries_conds.map(mapper))

    # Return harmonized dataframes
    return trialagg_timeseries, popagg_timeseries, stats, trialagg_stats


def highlight_incomplete(x, xref=None):
    if is_iterable(x):
        return [highlight_incomplete(xx, xref=x.max()) for xx in x]
    if np.isnan(x):
        return 'color:red;'
    if xref is not None:
        if x != xref:
            return 'color:orange;'
    return ''


def get_detailed_ROI_count(data):
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
    # Format
    return ROI_detailed_count.style.apply(
        highlight_incomplete, axis=1).format('{:.0f}')


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


def get_plot_data(timeseries, stats):
    '''
    Get ready-to-plot dataframe by merging timeseries and stats dataframes
    and adding time information.

    :param timeseries: timeseries dataframe
    :param stats: stats dataframe
    :return: merged dataframe 
    '''
    logger.info('merging timeseries and stats information...')
    plt_data = timeseries.copy()
    expand_and_add(stats, plt_data)
    add_time_to_table(plt_data)
    return plt_data


def compute_1way_anova(data, xkey, ykey):
    '''
    Perform a 1-way ANOVA to assess whether a dependent variable is dependent
    on a given independent variable (i.e. group, factor).

    :param data: pandas dataframe
    :param xkey: name of column containing the independent variable
    :param ykey: name of column containing the dependent variable
    :return: p-value for dependency
    '''
    # Rename columns of interest to ensure statsmodels compatibility
    data = data.rename(columns={xkey: 'x', ykey: 'y'})
    # Construct OLS model with data
    model = ols('y ~ x', data=data).fit()
    # Extract results table for 1-way ANOVA 
    anova_table = sm.stats.anova_lm(model, typ=2)
    # Extract relevant p-value for dependency
    F = anova_table.loc['x', 'PR(>F)']
    return F



def sum_of_square_devs(x):
    return ((x - x.mean())**2).sum()


def anova1d(data, xkey, ykey):
    '''
    Detailed 1-way ANOVA
    
    :param data: multi-indexed experiment dataframe
    :param xkey: name of column holding the values of the independent variable
    :param ykey: name of column holding the values of the dependent variable
    :return: p-value computing from F-score
    '''
    # Compute problem dimensions
    k = data[xkey].nunique()  # number of conditions
    npergroup = data.groupby(xkey)[ykey].agg(lambda s: s.notna().sum())  # number of valid (non-NaN) observations in each condition
    N = npergroup.sum()  # overall number of valid observations
    
    # Compute degrees of freedom
    df_between = k - 1
    df_within = N - k
    df_total = N - 1

    # Compute means
    Gm = data[ykey].mean()  # grand mean
    M = data.groupby(xkey)[ykey].mean()  # means per group

    # Compute sums of squares
    ss_within = data.groupby(xkey)[ykey].apply(sum_of_square_devs).sum()
    ss_between = (npergroup * (M - Gm)**2).sum()
    ss_total = ss_between + ss_within

    # Compute F-score
    ms_within = ss_within / df_within
    ms_between = ss_between / df_between
    F = ms_between / ms_within

    # Compute resulting statistics
    p = fstats.sf(F, df_between, df_within)  # p-value
    # eta_sqrd = ss_between / ss_total  # effect size
    # om_sqrd = (ss_between - (df_between * ms_within)) / (ss_total + ms_within)  # corrected effect size

    # Return p-value
    return p


def get_weighted_ss_across(data, ykey, groupby):
    ''' 
    Group data by factor value, compute squared deviations from grand mean 
    for each group, weigh them by group sizes and return sum.

    :param data: input data
    :param ykey: name of variable of interest
    :param groupby: name of grouping variable
    :return: sum of squared deviations per group, weighted by group size
    '''
    # Grand mean
    ybar = data[ykey].mean()
    # Number of occurences per factor value
    df = data.groupby(groupby)[ykey].count()
    # Group mean for each factor value
    groupmeans = data.groupby(groupby)[ykey].mean()
    # Squared deviation from grand mean for each factor value
    ss = (groupmeans - ybar)**2
    # Weigh squared deviations by number of occurences
    weighed_ss = df * ss
    # Return weighted sum
    return weighed_ss.sum()


def get_weighted_ss_within(data, ykey, groupby):
    ''' 
    Group data by factor(s) value, compute squared deviations from group mean 
    within each group, weigh them by group sizes and return sum.

    :param data: input data
    :param ykey: name of variable of interest
    :param groupby: name of grouping variable(s)
    :return: ???
    '''
    # Number of occurences per factor(s) value
    df = data.groupby(groupby)[ykey].count() - 1
    # Mean variation inside each group
    ss = data.groupby(groupby)[ykey].std()**2
    # Return weighted sum of variations within each group
    return (df * ss).sum()


def anova2d(data, ykey, factors, alpha=None, interaction=True):
    ''' 
    Perform 2d ANOVA on dataset
    
    :param data: input dataframe 
    :param ykey: name of variable of interest
    :param factors: names of the 2 factors of interest
    :param alpha: critical p-value for significance (optional)
    :param interaction: whether to consider interaction across the 2 factors
    :return: ANOVA table
    '''
    # Check that inputs are correct
    if len(factors) != 2:
        raise ValueError('exactly 2 factors must be provided')
    for k in [ykey, *factors]:
        if k not in data and k not in data.index.names:
            raise ValueError(f'{k} not found in input dataset') 
    # Count number of observations per factor combinations
    npergroup = (data
        .groupby(factors)[ykey]
        .count()
        .reset_index()
        .rename(columns={ykey: 'count'})
    )
    # # If number varies across groups, raise error
    # if npergroup['count'].nunique() > 1:
    #     raise ValueError(
    #         f'Number of observations varies across groups:\n{npergroup}')
    
    # Create ANOVA backbone table
    int_key = f'{factors[0]} x {factors[1]}'
    within_key = 'Within Groups'
    tot_key = 'Total' 
    SS, df, MS, F, pval, Fcr = 'SS', 'df', 'MS', 'F', 'P-value', 'F crit'
    colnames = [SS, df, MS, F, pval]
    if alpha is not None:
        colnames.append(Fcr)
    rownames = [*factors]
    if interaction:
        rownames.append(int_key)
    rownames = rownames + [within_key, tot_key]
    table = pd.DataFrame(index=rownames, columns=colnames)
    table.index.name = 'Source of Variation'

    # Calculate degrees of freedom
    a, b = data[factors].nunique().values  # Number of unique values of each factor
    table.loc[factors, df] = a - 1, b - 1   # Individual dfs
    if interaction:
        table.loc[int_key, df] = (a - 1) * (b - 1)  # interaction df
        table.loc[within_key, df] = data.shape[0] - a * b   # Intra group df
    else:
        table.loc[within_key, df] = data.shape[0] - a - b + 1  # Intra group df
    table.loc[tot_key, df] = data.shape[0] - 1  # Total df

    # For each factor, calculate weighted sum of squared deviations across factor values
    for factor in factors: 
        table.loc[factor, SS] = get_weighted_ss_across(data, ykey, factor)

    # Calculate weighted sum of squared deviations within group for each factors combination
    table.loc[within_key, SS] = get_weighted_ss_within(data, ykey, factors)

    # Calculate total sum of squared deviations from grand mean across all observations
    table.loc[tot_key, SS] = ((data[ykey] - data[ykey].mean())**2).sum()

    # Calculate interaction SS terms as difference between total SS and sum of all other SS terms
    if interaction:
        table.loc[int_key, SS] = table.loc[tot_key, SS] - (table.loc[factors + [within_key], SS]).sum()

    # Calculate MS terms (i.e. relative variability terms) as SS / df ratios
    MSkeys = [*factors, within_key]
    if interaction:
        MSkeys.append(int_key)
    for key in MSkeys:
        table.loc[key, MS] = table.loc[key, SS] / table.loc[key, df]

    # Calculate F scores as ratio of variability between groups / variability within groups
    Fkeys = [*factors]
    if interaction:
        Fkeys.append(int_key)
    for key in Fkeys:
        table.loc[key, F] = table.loc[key, MS] / table.loc[within_key, MS]

    # Calculate corresponding p-values
    for key in Fkeys:
        table.loc[key, pval] = fstats.sf(
            table.loc[key, F], 
            table.loc[key, df], 
            table.loc[within_key, df]
        )

    # F critical 
    if alpha is not None:
        for key in Fkeys:
            table.loc[key, Fcr] = fstats.ppf(
                1 - alpha, 
                table.loc[key, df], 
                table.loc[within_key, df]
            )

    # Return final anova table
    return table


def get_crossdataset_average(data, xkey, ykey=None, hue=None, weighted=True, errprop='intra',
                             add_global_avg=False):
    '''
    Average data output variable(s) across datasets for each value of an (or a combination of) input variable(s).
    
    :param data: multi-dataset dataframe
    :param xkey: name(s) of input variable(s) whose values determine aggregation groups
    :param ykey (optional): name of output variable(s) to average. If not provided, all output variables will be considered.
    :param hue: ???
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

        # Concatenate outputs, and add prefix if needed
        ykey_wdata = pd.concat([mean, sem], axis=1)
        if len(ykey) > 1:
            ykey_wdata = ykey_wdata.add_prefix(f'{yk} - ')

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
    return (itrial * NFRAMES_PER_TRIAL) + iframe


def get_quantile_index(s, **kwargs):
    return np.abs(s - s.median(**kwargs)).idxmin()


def get_trial_anchor_points(s):
    ''' 
    Get interpolation anchor points for a trial timeseries

    :param s: trial timeseries
    :return: anchor points indexes
    '''
    # Reduce index to frame only
    s.index = s.index.get_level_values(Label.FRAME)
    # Extract pre and post response windows
    spre = s.loc[slice_last_dim(s.index, FrameIndex.PRESTIM)]
    spost = s.loc[slice_last_dim(s.index, FrameIndex.BASELINE)]
    # Extract anchor points from given quantiles in each window
    ianchors = [get_quantile_index(spre), get_quantile_index(spost)]
    return ianchors


def spline_interp_run(s, ianchors=None):
    ''' 
    Interpolate run timeseries using a cubic spline

    :param s: input multi-index series
    :return: spline-interpolated series
    '''
    # Extract anchor points if not given
    if ianchors is None:
        ianchors = s.groupby([Label.ROI, Label.RUN, Label.TRIAL]).apply(
            get_trial_anchor_points).explode().rename(Label.FRAME)
        ianchors = ianchors.to_frame().set_index(Label.FRAME, append=True).index
    # Otherwise, propagate them through every trial
    else:
        levels = {k: s.index.unique(level=k) for k in s.index.names[:-1]}
        levels[Label.FRAME] = ianchors
        ianchors = pd.MultiIndex.from_product(
            list(levels.values()), names=list(levels.keys()))
    # Compute cumulative index
    df = s.rename('y').to_frame()
    df['x'] = get_cumulative_frame_index(df.index)
    # Select data at anchor points, plus end points if not there already
    df_anchors = df.loc[ianchors, :]
    if ianchors[0][-1] != 0:
        df_anchors = pd.concat([df.head(1), df_anchors], axis=0)
    if ianchors[-1][-1] != NFRAMES_PER_TRIAL - 1:
        df_anchors = pd.concat([df_anchors, df.tail(1)], axis=0)
    # Construct cubic slice interpolator
    finterp = interp1d(
        df_anchors['x'], df_anchors['y'], kind='cubic', fill_value='extrapolate',
        assume_sorted=True)
    # Apply interpolator to entire series and return
    return pd.Series(data=finterp(df['x']), index=s.index, name=s.name)



def spline_interp_trial(s):
    ''' 
    Interpolate trial timeseries using a cubic spline

    :param s: trial timeseries
    :param ianchor: frame index (relative to trial) of anchoring points
    :return: spline-interpolated series
    '''
    mux = s.index
    # Reduce index to frame only
    s.index = s.index.get_level_values(Label.FRAME)
    # Extract pre and post response windows
    spre = s.loc[slice_last_dim(s.index, FrameIndex.PRESTIM)]
    spost = s.loc[slice_last_dim(s.index, FrameIndex.BASELINE)]
    # Extract anchor points from given quantiles in each window
    ianchors = [get_quantile_index(spre), get_quantile_index(spost)]
    yanchors = s.loc[ianchors].values
    # Interpolate with cubic spline
    finterp = interp1d(
        ianchors, yanchors, kind='cubic', 
        fill_value='extrapolate', assume_sorted=True)
    yinterp = finterp(s.index.values)
    # Return as series with original index
    return pd.Series(data=yinterp, index=mux, name=s.name)


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


def get_cubic_fit(x, y):
    ''' Get a cubic fit '''
    p = np.poly1d(np.polyfit(x, y, 3))
    xfit = np.linspace(*bounds(x), 100)
    return xfit, p(xfit)


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


def compute_ROI_vs_trial_anova(s, interaction=True):

    # Set all index dims as columns
    data = s.reset_index()

    # Fix ROI indexes
    iROIs_org = data[Label.ROI].unique() 
    iROIs_fix = np.arange(len(iROIs_org))
    ROI_mapper = dict(zip(iROIs_org, iROIs_fix))
    data[Label.ROI] = data[Label.ROI].map(ROI_mapper)

    # # Perform 2-way ANOVA with statsmodels
    # data = data.rename(columns={s.name: 'y'})
    # formula = f'y ~ C({Label.TRIAL}) + C({Label.ROI})'
    # if interaction:
    #     formula = f'{formula} + C({Label.TRIAL}):C({Label.ROI})'
    # # logger.info(f'fitting model to "{formula}"...')
    # model = ols(formula, data=data).fit()
    # # logger.info('computing anova output')
    # aov_table = sm.stats.anova_lm(model, typ=2)
    # Ftrial, Froi = aov_table.loc[[f'C({Label.TRIAL})', f'C({Label.ROI})'], 'F'].values

    # Perform 2-way ANOVA with custom function
    aov_table = anova2d(data, s.name, [Label.TRIAL, Label.ROI], interaction=interaction)

    Ftrial, Froi = aov_table.loc[[Label.TRIAL, Label.ROI], 'F'].values
    
    # Return F-scores
    return Ftrial, Froi


def tmean(x):
    ''' 
    Estimate mean by fitting t-distribution to sample distribution
    
    :param x: input distribution
    :return: estimated mean
    '''
    return tstats.fit(x)[-2]


def get_power_spectrum(y, fs):
    ''' Compute signal power spectrum and return it as a dataframe '''
    freqs, Pxx_spec = welch(y, fs, scaling='spectrum')
    df = pd.DataFrame({
        Label.FREQ: freqs,
        Label.PSPECTRUM: Pxx_spec
    })
    df.index.name = 'freq index'
    return df


def offset_by(s, by, ykey=None, rel_ygap=.5):
    ''' Offset series according to some index level '''
    if isinstance(s, pd.DataFrame):
        if ykey is None:
            raise ValueError(f'ykey argument must be provided for dataframe inputs')
        # Work on copy to avoid offset accumulation upon multiple calls 
        scopy = s.copy()
        scopy[ykey] = offset_by(s[ykey], by, rel_ygap=rel_ygap)
        return scopy
    # Compute and add vertical offsets to y column
    yranges = s.groupby(by).apply(np.ptp)
    yranges += rel_ygap * yranges.mean() 
    yoffsets = -yranges.cumsum()
    extra_mux_levels = set(s.index.names) - set([by])
    if len(extra_mux_levels) > 0:
        yoffsets = expand_to_match(yoffsets, s.index)
    return s + yoffsets


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
            .apply(lambda s: s / s.sum())
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


def get_rtype_fractions_per_ROI(data):
    ''' 
    Get the fraction of response type over all relevant conditions, per ROI
    
    :param data: multi-index (ROI, run) stats dataframe
    :return: ROI stats dataframe 
    '''
    if Label.DATASET in data.index.names:
        gby = [Label.DATASET, Label.ROI]
    else:
        gby = Label.ROI
    # Save original ROIs list
    org_ROIs = data.groupby(gby).first().index.unique()
    # Filter data to only conditions with ISPTA values above certain threshold
    class_data = data.loc[data[Label.ISPTA] > ISPTA_THR, :]
    # Compute number of conditions used for classification
    nconds = len(class_data.index.unique(Label.RUN))
    # Compute response type fractions
    logger.info(f'computing fraction of response occurence per ROI over {nconds} "strong ISPTA" conditions...')
    roistats = (
        class_data[Label.RESP_TYPE]
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
    
    return roistats


def apply_linregress(s, xkey=Label.TRIAL):
    ''' 
    Apply linear regression on series index:values distribution

    :param s: input pandas Series object
    :param xkey: name of index dimension to use as input vector
    :return: pandas Series with regression output metrics 
    '''
    res = linregress(s.index.get_level_values(xkey), y=s.values)
    return pd.Series({
            'slope': res.slope,
            'intercept': res.intercept,
            'rval': res.rvalue,
            'pval': res.pvalue,
            'stderr': res.stderr,
            'intercept_stderr': res.intercept_stderr,
    })


def assess_significance(data, pthr, pval_key='pval', sign_key=None):
    '''
    Categorize responses by comparing p-values to a significance threshold, 
    and adding a potential "sign" measure to the output 
    '''
    sig = data[pval_key] < pthr
    if sign_key is not None:
        sig = (sig * np.sign(data[sign_key])).astype(int)
    return sig


def classify_ROIs(data):
    ''' 
    Classify ROIs based on fraction of each response type across experiment
    
    :param data: trial-aggragated stats dataframe
    :return: ROI classification stats dataframe
    '''
    # Compute fraction of response occurence in "strong" ISPTA conditions, for each ROI
    roistats = get_rtype_fractions_per_ROI(data)

    # Extract type and proportion of maximum non-weak proportion, per ROI
    nonweakresps = roistats.drop('weak', axis=1).agg(['idxmax', 'max'], axis=1)

    # Classify ROIs based on proportion of conditions in each response type
    logger.info('classiying ROIs as a function of their response occurence fractions...')
    roistats[Label.ROI_RESP_TYPE] = 'weak'
    cond = nonweakresps['max'] >= PROP_CONDS_THR
    roistats.loc[cond, Label.ROI_RESP_TYPE] = nonweakresps.loc[cond, 'idxmax']

    # Return
    return roistats


def get_params_by_run(data):
    ''' Get parameters by run '''
    inputkeys = [Label.P, Label.DC, Label.ISPTA]
    return data[inputkeys].groupby(Label.RUN).first()


def find_in_dataframe(df, key):
    ''' Find a column in a dataframe, be it in index or as a data column '''
    if key in df.index.names:
        return df.index.get_level_values(key)
    else:
        return df[key]


def free_expand(s, ref_df, verbose=True):
    ''' Expand series to a higher-dimensional dataframe '''
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
        return pd.concat([free_expand(s[ss], ref_df, verbose=False) for ss in s], axis=1)

    # Extract index dimensions of series to expand
    gby = list(s.index.names)
   
    # Create function to find value in series and return an expanded vector
    def find_and_expand(df):
        idx_small = [find_in_dataframe(df, k)[0] for k in gby]
        val_small = s.loc[tuple(idx_small)]
        return pd.Series(index=df.index, data=val_small)

    # Groupby small idnex dimensions and apply function
    out = ref_df.groupby(gby).apply(find_and_expand).rename(s.name)

    # Remove extra index dimensions generated by apply
    out = out.droplevel(list(range((len(gby)))))

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
    if ykey is not None:
        varkeys = list(set(varkeys).intersection(as_iterable(ykey)))

    # Compute population-average dataframe 
    return pd.concat([
        groups[constkeys].first(),  # constant columns: first value of each group 
        groups[varkeys].mean()  # variable columns: mean of each group
    ], axis=1)
