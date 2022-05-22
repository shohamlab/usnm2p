# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-15 10:13:54
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-05-22 18:00:31

''' Collection of utilities to process fluorescence signals outputed by suite2p. '''

from collections import Counter
from argon2 import PasswordHasher
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt, find_peaks, peak_widths
from scipy.stats import skew, norm
from scipy.stats import t as tstats
from scipy.stats import f as fstats
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
import statsmodels.api as sm
from statsmodels.formula.api import ols
from functools import wraps

from constants import *
from logger import logger
from utils import *


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


def discard_indexes(data, ikey=Label.TRIAL, idiscard=None):
    '''
    Discard specific values from a specific index level in a multi-indexed dataset
    
    :param data: multi-index dataframe with 4D (ROI, run, trial, frame) index
    :param ikey: index level key
    :param idiscard: index values to discard at this index level
    :return: filtered dataset
    '''
    # Cast indexes to discard as list
    if not is_iterable(idiscard):
        idiscard = [idiscard]
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


def get_window_size(wlen, fps):
    ''' Compute window size (in number of frames) from window length (in s) and fps. '''
    # Convert seconds to number of frames
    w = int(np.round(wlen * fps))
    # Adjust to odd number if needed
    if w % 2 == 0:
        w += 1
    return w


def linreg(data, xkey=Label.F_NEU, ykey=Label.F_ROI, norm='HuberT', add_cst=True):
    '''
    Perform linear regression on 2 columns of a dataset
    
    :param data: pandas dataframe contaning the variables of interest
    :param xkey: name of the column containing the independent variable X
    :param ykey: name of the column containing the dependent variable Y
    :param norm (default: HuberT): name of the norm used to compute the linear regression
    :param add_cst (default: True): whether to consider an additional constant in the
        linear regression model 
    :return: fitted regression parameter(s)
    '''
    Y = data[ykey].values
    X = data[xkey].values
    if add_cst:
        X = sm.add_constant(X)
    norm = getattr(sm.robust.norms, norm)()
    model = sm.RLM(Y, X, M=norm)
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
    alphas = np.arange(*bounds, .01)
    # Extract fluorescence profiles as 2D arrays
    F_ROI = data[Label.F_ROI].values
    F_NEU = data[Label.F_NEU].values
    # Compute cost for each alpha
    costs = np.array([costfunc(F_ROI, F_NEU, alpha) for alpha in alphas])
    # Return alpha corresponding to minimum cost
    return alphas[np.argmin(costs)]


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


def filter_signal(y, fs, fc, order=2):
    '''
    Apply zero-phase filter to signal
    
    :param y: signal array
    :param fs: sampling frequency (Hz)
    :param fc: tuple of cutoff frequencies (Hz)
    :param order: filter order
    '''
    if isinstance(y, pd.Series):
        yf = filter_signal(y.values, fs, fc, order=order)
        return pd.Series(data=yf, index=y.index)
    fc = np.asarray(fc)
    # Determine Butterworth type and cutoff
    btype = 'band'
    if fc[0] == 0.:
        btype = 'low'
        fc = fc[1]
    elif fc[1] == np.inf:
        btype = 'high'    
        fc = fc[0]
    logger.info(f'{btype}-pass filtering signal (cutoff = {fc} Hz)...')
    # Determine Nyquist frequency
    nyq = fs / 2
    # Calculate Butterworth filter coefficients
    b, a = butter(order, fc / nyq, btype=btype)
    # Filter signal forward and backward (to ensure zero-phase) and return
    return filtfilt(b, a, y)


def compute_baseline(data, fps, wlen, q, smooth=True):
    '''
    Compute the baseline of a signal.

    :param data: multi-indexed Series object contaning the signal of interest
    :param fps: frame rate of the signal (in fps)
    :param wlen: window length (in s) to compute the fluorescence baseline
    :param q: quantile used for the computation of the fluorescence baseline 
    :param smooth (default: False): whether to smooth the baseline by applying an additional
        moving average (with half window size) to the sliding window output
    :return: fluorescence baseline series
    '''
    qstr = f'{q * 1e2:.0f}{get_integer_suffix(q * 1e2)} percentile'
    if wlen is None:
        steps_str = [f'{qstr} of signal']
        # Define constant baseline computation function
        def bfunc(s):
            return s.quantile(q)
    else:
        # Compute window size (in number of frames)
        w = get_window_size(wlen, fps)
        if smooth:
            w2 = w // 2
            if w2 % 2 == 0:
                w2 += 1
        wstr = f'{wlen:.1f}s ({w} frames) sliding window'
        steps_str = [f'{qstr} of {wstr}']
        if smooth:
            steps_str.append(f'mean of {w2 / fps:.1f}s ({w2} frames) sliding window')
        # Define rolling baseline computation function
        def bfunc(s):
            b = apply_rolling_window(s.values, w, func=lambda x: x.quantile(q))
            if smooth:
                b = apply_rolling_window(b, w2, func=lambda x: x.mean())
            return b
    if len(steps_str) > 1:
        steps_str = '\n'.join([f'  - {s}' for s in steps_str])
        steps_str = f'successive application of:\n{steps_str}'
    else:
        steps_str = steps_str[0]
    logger.info(f'computing signal baseline as {steps_str}')
    # Group data by ROI and run, and apply sliding window on F to compute baseline fluorescence
    groupkeys = [Label.ROI, Label.RUN]
    nconds = np.prod([len(data.index.unique(level=k)) for k in groupkeys])
    # Apply function to each ROI & run, and log progress
    with tqdm(total=nconds - 1, position=0, leave=True) as pbar:
        baselines = data.groupby(groupkeys).transform(pbar_update(nan_proof(bfunc), pbar))
    return baselines


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


def compute_displacement_velocity(ops, mux, um_per_pixel, fps, substituted=True, full_output=False):
    '''
    Compute displacement velocity profiles frrom registration offsets
    
    :param ops: suite2p output options dictionary
    :param mux: (run, trial, frame) multi-index object
    :param um_per_pixel: spatial resolution of the images
    :param fps: sampling frequency (in frames/second)
    :param substituted (default: True): whether stimulus frames have been substituted
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
    # Stimulus frames substitution (if applied) creates consecutive identical frames, resulting
    # in zero (or very low) displacement velocity artifacts at stimulus index. In this case, we
    # also substitute displacement velocity at stimulus indexes by values at the preceding indexes.
    if substituted:
        logger.info('correcting displacement velocity at stimulus indexes to compensate for stimulus frames substitution...')
        # Set stimulus frames velocities to NaN
        df.loc[pd.IndexSlice[:, :, FrameIndex.STIM], Label.SPEED_UM_FRAME] = np.nan
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


def shiftslice(s, i):
    return slice(s.start + i, s.stop + i)


def apply_in_window(func, data, ykey, wslice, verbose=True, log_completion_rate=True):
    '''
    Apply function to a given signal within a specific observation window
    
    :param data: multi-indexed fluorescence timeseries dataframe
    :param ykey: name of the column containing the signal of interest
    :param wslice: slice object representing the indexes of the window
    '''
    idxlevels = [k for k in data.index.names if k != Label.FRAME]
    if verbose:
        logger.info(
            f'applying {func.__name__} function on {ykey} in {wslice.start}-{wslice.stop} index window '
            f'across {", ".join(idxlevels)} ...')
    if len(idxlevels) == 1:
        window_data = data.loc[pd.IndexSlice[:, wslice], ykey]
    elif len(idxlevels) == 2:
        window_data = data.loc[pd.IndexSlice[:, :, wslice], ykey]
    elif len(idxlevels) == 3:
        window_data = data.loc[pd.IndexSlice[:, :, :, wslice], ykey]
    else:
        raise ValueError(f'cannot apply {func} func: too many index levels ({len(idxlevels)}')
    out = window_data.groupby(idxlevels).agg(func)
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


def get_response_types_per_ROI(data):
    '''
    Extract the response type per ROI from experiment dataframe.

    :param data: experiment dataframe
    :return: pandas Series of response types per ROI
    '''
    logger.info('extracting responses types per ROI...')
    return data.groupby(Label.ROI).first()[Label.ROI_RESP_TYPE]


def get_trial_averaged(data, full_output=False):
    '''
    Compute trial-averaged statistics
    
    :param data: multi-indexed (ROI x run x trial) statistics dataframe
    :return: (trialavg_data, is_repeat) tuple
    '''
    # Group trials
    groups = data.groupby([Label.ROI, Label.RUN])
    # Compute average of stat across trials
    trialavg_data = groups.agg(mean_str)
    if isinstance(trialavg_data, pd.DataFrame):  # DataFrame case
        # Remove time column if present
        if Label.TIME in trialavg_data:
            del trialavg_data[Label.TIME]
        # Rename relevant input columns to their trial-averaged meaning
        cols = {}
        for k, v in Label.RENAME_ON_AVERAGING.items():
            if k in trialavg_data:
                cols[k] = v
        if len(cols) > 0:
            trialavg_data.rename(columns=cols, inplace=True)
    else:  # Series case
        # Rename input to its trial-average meaning if necessary
        if trialavg_data.name in Label.RENAME_ON_AVERAGING.keys():
            trialavg_data.name = Label.RENAME_ON_AVERAGING[trialavg_data.name]
    if full_output:
        # Compute std of metrics across trials
        trialstd_data = groups.std()
        # Determine whether metrics is a repeated value or a real distribution
        is_repeat = ~(trialstd_data.max() > 0)
        return trialavg_data, is_repeat
    else:
        return trialavg_data


def weighted_average(data, avg_name, weight_name):
    '''
    Compute a weighted-average of a particular column of a dataframe using the weights
    of another column.
    
    :param data: dataframe
    :param avg_name: name of the column containing the values to average
    :param weight_name: name of the coolumn containing the weights
    :return: weighted average
    '''
    d = data[avg_name]
    w = data[weight_name]
    return (d * w).sum() / w.sum()


def filter_data(data, iROI=None, irun=None, itrial=None, ireg=None, rtype=None, P=None, DC=None, tbounds=None, full_output=False):
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
    if ireg is not None:
        subindex[idxnames.index(Label.MOUSEREG)] = ireg
        filters[Label.MOUSEREG] = f'{Label.MOUSEREG}{plural(ireg)} {ireg}'
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
        if Label.MOUSEREG in data.index.names:
            nROIs = len(data.groupby([Label.MOUSEREG, Label.ROI]).first())
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
    

def groupby_and_all(data, func, groupby=None):
    '''
    Wrapper around pandas "groupby" that applies a function on the input dataset for
    each sub-group of the groupby category, but also on the entire dataset.
    
    :param data: dataframe
    :param func: function to apply to the dataset and each sub-dataset
    :param groupby (optional): variable defining the sub-groups
    :retrun: dictionary of function output per sub-group and for the entire dataset (key "all")
    '''
    out = {'all': func(data)}
    if groupby is not None:
        for cond, cond_data in data.groupby(groupby):
            out[cond] = func(cond_data)
    return out


def get_clustered_index(data, metric='euclidean', method='single'):
    '''
    Compute a clustered index list according observations across dimensions
    
    :param data: (observations x dimensions) dataframe for some specific variable.
    :return: index of observations outputed by the clustering algorithm
    '''
    logger.info('computing new index according to hierarchical clustering...')
    # Get dataset index
    index = data.index
    # Compute pairwise distance matrix
    Y = pdist(data, metric=metric, out=None)
    # If NaNs in matrix -> return original index
    if np.isnan(np.sum(Y)):
        logger.warning('cannot clusterize dataset with NaNs -> ignoring')
        return index
    # Cluster hierarchically using the pairwise distance matrix
    Z = linkage(Y, method=method, optimal_ordering=False)
    # Return index list from cluster output 
    return index[leaves_list(Z)]


def clusterize_data(data, **kwargs):
    '''
    Re-arrange dataset along the ROI dimension according observations across runs 
    
    :param data: (nROIs x nruns) dataframe for some specific variable.
    :return: dataframe re-indexed alonmg according to ROI clustering process 
    '''
    iROIs_clustered = get_clustered_index(data, **kwargs)
    logger.info('re-arranging dataset...')
    return data.reindex(iROIs_clustered)


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
        plt.figure()
        plt.hist(data, bins=50)
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
    assert s > 0, f'Error: negative standard deviation found during Gaussian fit ({s})'
    mu = popt[2]
    # if mu < 0:
    #     logger.warning(f'negative mean found during Gaussian fit ({mu})')

    # Return outputs
    return xmid, popt


def sort_ROIs(data, pattern):
    ''' Get sorted ROI index from dataset according to specific metrics '''
    # Compute average metrics per ROI on the 'all' condition
    avg_per_ROI = data['all'].groupby(Label.ROI).mean()
    # sort average metrics in specified direction
    ascending = {'ascend': True, 'descend': False}[pattern]
    avg_per_ROI = avg_per_ROI.sort_values(ascending=ascending)
    # Extract sorted ROI indexes
    return avg_per_ROI.index


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
        data = get_trial_averaged(data)
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


def pvalue_to_zscore(p=.05, directional=True):
    '''
    Compute the z-score corresponding to a given chance probability level
    
    :param p: associated probability
    :param directional (default: True): whether to assume a directional effect (i.e. 1-tailed test) or not (i.e. 2-tailed test)
    :return: corresponding z-score
    '''
    if not directional:
        p /= 2
    return norm.ppf(1 - p)


def is_valid(df):
    ''' 
    Return a series with an identical index as that of the input dataframe, indicating
    which rows are valid and must be included for response analysis.
    '''
    cols = [k for k in TRIAL_VALIDITY_KEYS if k in df.columns]
    if len(cols) > 0:
        logger.info(f'identifying samples not without [{", ".join(cols)}] tags')
    out = ~df[cols].any(axis=1)
    return out.rename('valid?')


def valid(df):
    ''' Return a copy of the dataframe with only valid rows that must be included for response analysis. '''
    out = df.loc[is_valid(df), :]
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


def get_threshold_metric(ykey, navg=1):
    ''' Get threshold metrics depending on the variable used for post-processing '''
    if ykey == Label.ZSCORE:
        return pvalue_to_zscore(PTHR_DETECTION)
    elif ykey == Label.DFF:
        return 0.1 / navg
    else:
        raise ValueError(f'unknown post-processing variable: {ykey}')


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
    return ['non-responsive', 'responsive']


def reclassify(data, ykey, ythr=None, nposthr=None):
    ''' 
    Reclassify dataset based on a new significance threshold number of responsive conditions
    
    :param data: stats dataframe
    :param ykey: variable to use for significance thresholding
    :param ythr (optional): significance threshold
    :param nposthr (optional): threshold number of positive conditions
    :return: updated stats dataframe
    '''
    # If nposthr not given, infer it from dataset
    if nposthr is None:
        isresp_vs_npos = data.groupby(Label.NPOS_CONDS).first()[Label.IS_RESP_ROI]
        nposthr = isresp_vs_npos[isresp_vs_npos == True].index[0]
    # If ythr is given, classify positive conditions according to significance threshold
    if ythr is not None:
        data[Label.POS_COND] = data[ykey] > ythr
        nposconds_per_roi = data[Label.POS_COND].groupby(
            [Label.MOUSEREG, Label.ROI]).sum().rename(Label.NPOS_CONDS)
        data[Label.NPOS_CONDS] = nposconds_per_roi
    # Classify cells according to threshold number of positive conditions
    data[Label.IS_RESP_ROI] = data[Label.NPOS_CONDS] >= nposthr
    data[Label.ROI_RESP_TYPE] = data[Label.IS_RESP_ROI].map(
        {True: 'responsive', False: 'non-responsive'})
    # # Log new classification summary
    # counts_by_type = data.groupby(
    #     [Label.MOUSEREG, Label.ROI])[Label.ROI_RESP_TYPE].first().value_counts()
    # logger.info(f'{counts_by_type.sum()} cells now organized as:\n{counts_by_type}')
    # Return new data
    return data


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
    raise ValueError(f'xkey must be one of ({Label.P}, {Label.DC}')


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


def compute_metrics_vs_nposthr(data, nposthrs, ykeys, evalfunc='max'):
    '''
    Compute metrics evolution as a function of nposthr
    
    :param data: stats dataframe
    :param nposthrs: vector of threshold number of positive conditions
    :param ykeys: columns of interest
    :param evalfunc: evaluation function applied to get metrics of interest
    '''
    metrics_vs_nposthr = []
    # For each nposthr
    for nposthr in nposthrs:
        # Re-classify data
        tmp = reclassify(data.copy(), nposthr=nposthr)
        # Group by dataset, ROI and response type and extract columns of interest
        groups = tmp.groupby([Label.MOUSEREG, Label.ROI, Label.ROI_RESP_TYPE])[ykeys]
        # Extract metrics of interest from these columns
        metrics = groups.agg(evalfunc)
        # Add information about nposthr
        metrics[Label.NPOS_CONDS] = nposthr
        # Append to list
        metrics_vs_nposthr.append(metrics)
    # Concatenate into single dataframe
    metrics_vs_nposthr = pd.concat(metrics_vs_nposthr, axis=0)
    # Add nposthr to index to avoid duplicate indexes
    metrics_vs_nposthr.set_index(Label.NPOS_CONDS, append=True, inplace=True)
    # Return metrics dataframe
    return metrics_vs_nposthr


def exclude_datasets(data, to_exclude):
    '''
    Exclude specific datasets from analysis
    
    :param data: global experiment dataframe
    :param to_exclude: mouse-region combinations to be discarded
    :return: filtered experiment dataframe 
    '''
    default_k = list(data.keys())[0]
    candidate_mouseregs = data[default_k].index.unique(level=Label.MOUSEREG).values
    notthere = list(set(to_exclude) - set(candidate_mouseregs))
    if len(notthere) > 0:
        logger.warning(f'{notthere} not found in dataset -> ignoring') 
    to_exclude = list(set(candidate_mouseregs).intersection(set(to_exclude)))
    if len(to_exclude) == 0:
        logger.warning('did not find any mouse-region to exclude from dataset')
        return data
    logger.info(f'excluding {to_exclude} datasets from analysis')
    query = f'{Label.MOUSEREG} not in {to_exclude}'
    return {k: v.query(query) for k, v in data.items()}


def add_change_metrics(timeseries, stats, ykey):
    '''
    Add a change metrics to a stats table
    
    :param timeseries: mutli-indexed timeseries dataframe
    :param stats: mutli-indexed stats dataframe
    :param ykey: name of the variable for which to compute the relative change
    '''
    if ykey not in timeseries:
        raise ValueError(f'{ykey} not in timeseries data')

    # Determine variable of interest for output metrics
    ykey_peak = f'peak {ykey}'
    ykey_peak_baseline = f'baseline {ykey_peak}'
    ykey_peak_corrected = f'corrected {ykey_peak}'

    # If change metrics is already present in stats, return
    if ykey_peak_corrected in stats:
        logger.warning(f'{ykey_peak_corrected} already in stats dataframe -> ignoring')
        return stats

    # Extract stimulus-evoked peak
    stats[ykey_peak] = apply_in_window(
        find_max, timeseries, ykey, FrameIndex.RESPONSE)
    # Detect peaks while sliding detection window along trial interval
    logger.info('identifying peaks while sliding detection window across trial interval...')
    peaks_along_trial, _ = slide_along_trial(
        lambda data, w: apply_in_window(
            find_max, data, ykey, w,
            verbose=False, log_completion_rate=False),
        timeseries, FrameIndex.RESPONSE, NSEEDS_PER_TRIAL)
    # Take the 30-th percentile of detected peak values as the baseline
    stats[ykey_peak_baseline] = peaks_along_trial.groupby(
        [Label.ROI, Label.RUN]).quantile(PEAK_CORRECTION_QUANTILE)
    
    # Subtract baseline from peak to get relative increase
    logger.info(f'computing {ykey_peak_corrected}...')
    stats[ykey_peak_corrected] = (stats[ykey_peak] - stats[ykey_peak_baseline])
    
    # Return stats dataframe
    return stats


def harmonize_run_index(data):
    '''
    Generate a new harmonized run index in multi-region dataset, based on P-DC combination
    '''
    # Remove run index
    data = data.droplevel(Label.RUN)
    # Parse P and DC columns to strings
    P_str = data[Label.P].map('{:.2f}MPa'.format)
    DC_str = data[Label.DC].map('{:.0f}%DC'.format)
    # Generate new run column from (P, DC) combination 
    data[Label.RUN] = P_str + DC_str
    # Add as index dimension
    data.set_index(Label.RUN, append=True, inplace=True)
    # Return 
    return data


def get_subdataset(data, mousereg):
    subdata = {}
    for k, df in data.items():
        tmp = df[df.index.get_level_values(level=Label.MOUSEREG) == mousereg]
        subdata[k] = tmp.droplevel(Label.MOUSEREG)
    return subdata


def get_plot_data(timeseries, stats):
    logger.info('merging timeseries and stats information...')
    plt_data = timeseries.copy()
    expand_and_add(stats, plt_data)
    add_time_to_table(plt_data)
    return plt_data