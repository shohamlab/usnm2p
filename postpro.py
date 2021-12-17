# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-15 10:13:54
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-12-17 17:47:57

''' Collection of utilities to process fluorescence signals outputed by suite2p. '''

import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import skew
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
import statsmodels.api as sm
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


def remove_trials(data, iremove):
    '''
    Remove specific trial indexes from fluoresence dataset
    
    :param data: multi-index dataframe with 4D (ROI, run, trial, frame) index
    :param iremove: indexes of trials to remove
    :return: filtered dataset
    '''
    # Cast indexes to remove as list
    if not is_iterable(iremove):
        iremove = [iremove]
    # Get trials indexes present in dataset
    itrials = data.index.unique(level=Label.TRIAL)
    # Diff the two lists to get indexes to keep and indexes absent from dataset
    iabsent = sorted(list(set(iremove) - set(itrials)))
    ikeep = sorted(list(set(itrials) - set(iremove)))
    iremove = sorted(list(set(itrials).intersection(set(iremove))))
    # Convert ikeep to slice if possible (much faster data extraction)
    ibounds = min(ikeep), max(ikeep)
    if ikeep == list(range(ibounds[0], ibounds[1] + 1)):
        ikeep = slice(ibounds[0], ibounds[1])
    # Log warning message in case of absent indexes
    if len(iabsent) > 0:
        logger.warning(f'trials {iabsent} not found in dataets (already removed?) -> ignoring') 
    # Restrict dataset to keep indexes
    logger.info(f'removing trials {iremove} for each run...')
    data = data.loc[pd.IndexSlice[:, :, ikeep, :], :]
    # Return
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
    # Compute window size (in number of frames)
    w = get_window_size(wlen, fps)
    if smooth:
        w2 = w // 2
        if w2 % 2 == 0:
            w2 += 1   
    # Log process info
    wstr = (f'{wlen:.1f}s ({w} frames) sliding window')
    qstr = f'{q * 1e2:.0f}{get_integer_suffix(q * 1e2)} percentile'
    steps_str = [f'{qstr} of {wstr}']
    if smooth:
        steps_str.append(f'mean of {w2 / fps:.1f}s ({w2} frames) sliding window')
    if len(steps_str) > 1:
        steps_str = '\n'.join([f'  - {s}' for s in steps_str])
        steps_str = f'successive application of:\n{steps_str}'
    logger.info(f'computing signal baseline as {steps_str}')
    # Group data by ROI and run, and apply sliding window on F to compute baseline fluorescence
    groupkeys = [Label.ROI, Label.RUN]
    nconds = np.prod([len(data.index.unique(level=k)) for k in groupkeys])
    def bfunc(s):
        b = apply_rolling_window(s.values, w, func=lambda x: x.quantile(q))
        if smooth:
            b = apply_rolling_window(b, w2, func=lambda x: x.mean())
        return b
    with tqdm(total=nconds - 1, position=0, leave=True) as pbar:
        return data.groupby(groupkeys).transform(pbar_update(bfunc, pbar))


def find_response_peak(s, n_neighbors=N_NEIGHBORS_PEAK, return_index=False):
    '''
    Find the response peak (if any) of a signal
    
    :param s: pandas Series containing the signal
    :param n_neighbors: number of neighboring elemtns to include on each side
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


def find_peaks_across_trials(data, iwindow, key=Label.ZSCORE):
    '''
    Find peaks in a given time window across all trials

    :param data: multi-indexed fluorescence timeseries dataframe
    :param iwindow: list (or slice) of indexes to consider (i.e. window of interest) in the trial interval
    :param key: name of the column containing the variable of interest
    :return: multi-indexed series of peak values across conditions    
    '''
    logger.info(f'identifying peak {key} in {iwindow} index window...')
    window_data = data.loc[pd.IndexSlice[:, :, :, iwindow], key]
    peaks = window_data.groupby(
        [Label.ROI, Label.RUN, Label.TRIAL]).agg(find_response_peak)
    npeaks, ntrials = peaks.notna().sum(), len(peaks)
    logger.info(f'identified {npeaks} peaks over {ntrials} trials (detection rate = {npeaks / ntrials * 1e2:.1f} %)')
    return peaks


def slide_along_trial(func, data, wlen, iseeds):
    '''
    Call a specific function while sliding a detection window along the trial length.

    :param func: function called on each sliding iteration
    :param data: fluorescence timeseries data
    :param wlen: window length (in frames)
    :param iseeds: either the index list or the number of sliding iterations along the trial length
    :return: stacked function output series with window starting index as a new index level
    '''
    # Generate vector of starting positions for the analysis window
    if isinstance(iseeds, int):
        iseeds = np.round(np.linspace(0, NFRAMES_PER_TRIAL - wlen, iseeds)).astype(int)
    logger.info(f'applying {func,__name__} function at {iseeds.size} seeds along trial length...')
    lvl = logger.getEffectiveLevel()
    logger.setLevel(logging.WARNING)
    outs = []
    # For each starting position
    for i in tqdm(iseeds):
        # Call function and get output series
        out = func(data, slice(i, i + wlen))
        # Save its name
        name = out.name
        # Rename with start index and append to list
        outs.append(out.rename(i))
    logger.setLevel(lvl)
    # Concatenate output series into dataframe
    df = pd.concat(outs, axis=1)
    # Stack them ot yield output series with new index level
    s = df.stack()
    # Specify index level name
    s.index.set_names('istart', level=-1, inplace=True)
    # Rename series with function output name, and return
    return s.rename(name)
    

def add_time_to_table(data, key=Label.TIME, frame_offset=FrameIndex.STIM):
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
    fps = get_singleton(data, Label.FPS)
    # Extract frame indexes
    iframes = data.index.get_level_values(Label.FRAME)
    # Add time column and remove fps column
    data[key] = (iframes - frame_offset) / fps
    del data[Label.FPS]
    # Set time as first column
    cols = data.columns
    data = data.reindex(columns=[cols[-1], *cols[:-1]])
    return data


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
        # Compute average of stat across trials
        trialstd_data = groups.std()
        # Determine whether stats is a repeated value or a real distribution
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


def filter_data(data, iROI=None, irun=None, itrial=None, rtype=None, P=None, DC=None, tbounds=None, full_output=False):
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
    subindex = [slice(None)] * 3
    if iROI is not None:
        subindex[0] = iROI
        filters[Label.ROI] = f'{Label.ROI}{plural(iROI)} {iROI}'
    if irun is not None:
        subindex[1] = irun
        filters[Label.RUN] = f'{Label.RUN}{plural(irun)} {irun}'
    if itrial is not None:
        subindex[2] = itrial
        filters[Label.TRIAL] = f'{Label.TRIAL}{plural(itrial)} {itrial}'
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
    # No ROI selected -> indicate number of ROIs
    if iROI is None:
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


def gauss_histogram_fit(data, bins=100):
    '''
    Fit a gaussian function to a dataset's histogram distribution
    
    :param data: data array
    :param bins (optional): number of bins in histogram  (or bin edges values)
    :return: (bin mid-points, fitted function parameters) tuple
    '''
    # Histogram distribution
    hist, edges = np.histogram(data, bins=bins)
    mids = (edges[1:] + edges[:-1]) / 2
    ptp = np.ptp(data)
    
    # Initial guesses
    H = hist.min()  # vertical offset -> min histogram value
    A = np.ptp(hist)  # vertical range -> histogram amplitude range
    x0 = mids[np.argmax(hist)]  # gaussian mean -> index of max histogram value
    sigma = ptp / 4  # gaussian width -> 1/4 of the data range 
    p0 = (H, A, x0, sigma)

    # Bounds
    Hbounds = (0., H + A / 2)  # vertical offset -> between 0 and min + half-range
    Abounds = (A / 2, 2 * A)   # vertical range -> between half and twice initial guess
    x0bounds = (x0 - ptp / 2, x0 + ptp / 2)  # gaussian mean -> within half-range of initial guess
    sigmabounds = (1e-10, ptp)  # sigma -> between 0 and full range (non-negativity constraint)
    pbounds = tuple(zip(*(Hbounds, Abounds, x0bounds, sigmabounds)))

    # Fit gaussian to histogram distribution
    xmid, popt = histogram_fit(data, gauss, bins=bins, p0=p0, bounds=pbounds)

    s = popt[3]
    assert s > 0, f'Error: negative standard deviation found during Gaussian fit ({s})'
    
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