# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2026-09-11 13:44:14
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2026-09-14 16:28:13

import glob
import os
import struct
import numpy as np
from scipy import signal, optimize
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .logger import logger
from .constants import Label

''' Utilities for the analysis of temperature measurement experiments '''

# Labels
TIME_KEY = 'time (s)'
REL_TIME_KEY = 'relative time (s)'
TEMP_KEY = 'T (°C)'
REL_TEMP_KEY = f'Δ{TEMP_KEY}'
STIMDUR_KEY = 'tstim (s)'
MAX_REL_TEMP_KEY = f'max-evoked {REL_TEMP_KEY}'

COND_KEYS = ['condition', 'location', 'x (mm)', 'y (mm)', 'z (mm)', Label.P]
NONAGG_KEYS = ['timestamp', 'elapsed time (s)']

# Conversion constants
ADC_TO_VOLTS = 50.354e-6  # V per ADC unit, from Intan RHD2000 datasheet (TO CHECK)

# Osensa fiber-optic temperature probe conversion constants, from Misi. (TO CHECK)
OSENSA_TZERO_C = 0.0
OSENSA_TSPAN_C = 100.0
OSENSA_RA_OHM = 149.3
OSENSA_I_MIN_A = 4e-3
OSENSA_I_MAX_A = 20e-3
OSENSA_GAIN_C_PER_V = OSENSA_TSPAN_C / (OSENSA_RA_OHM * (OSENSA_I_MAX_A - OSENSA_I_MIN_A))

# Analysis constants
TARGET_FS = 120  # target sampling rate when downsampling loaded data (Hz)
LOWPASS_FC = 30  # Hz, lowpass filter cutoff frequency for analog signal
OVERVIEW_FS = 4   # sampling rate for longitudinal overview (Hz)
GROUP_GAP = 1.0  # s, edges further apart than this start a new burst

# Triggered average properties
STIM_ONSET = 0.5  # s, window length before each trigger burst onset
RESPONSE_WINDOW = (0., 3.0)  # s, time window containing expected response after each trigger burst onset
BASELINE_PRE = 0.4  # s, pre-onset span averaged and subtracted from each trial

# Double-exponential fit bounds
FIT_MIN_CELSIUS_AMPLITUDE = 0.01  # degC
FIT_MAX_CELSIUS_AMPLITUDE = 3.0  # degC
FIT_MAX_TAU_RISE = 1.0  # s
FIT_MAX_TAU_DECAY = 5.0  # s


def load_experiment_log(folder):
    '''
    Load experiment log data from CSV file starting with 'thermal_experiment_log_' in folder.

    :param folder: experiment data folder
    :return: pandas DataFrame with log data, indexed by trial
    '''
    # Search for CSV file starting with 'thermal_experiment_log_' in folder
    logger.info(f'searching for experiment log in "{folder}"')
    glob_pattern = os.path.join(folder, 'thermal_experiment_log_*.csv')
    log_fpaths = glob.glob(glob_pattern)

    # Make sure there is exactly one log file found
    if len(log_fpaths) == 0:
        raise FileNotFoundError(f'No log file found in {folder} matching pattern {glob_pattern}')
    elif len(log_fpaths) > 1:
        raise FileExistsError(f'Multiple log files found in {folder} matching pattern {glob_pattern}: {log_fpaths}')
    log_fpath = log_fpaths[0]

    # Load log data from CSV file into pandas DataFrame, and set index name to 'trial'
    logger.info(f'loading experiment log from "{log_fpath}"')
    data = pd.read_csv(log_fpath)
    data.index.name = 'trial'

    # Convert timestamp column to pandas datetime format, with microsecond precision
    data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')

    # Compute and add elapsed time column (s) since first trial, in seconds
    data['elapsed time (s)'] = (data['timestamp'] - data['timestamp'][0]).dt.total_seconds()

    # Return log data
    return data


def get_intan_layout(folder):
    '''
    Extract sample rate (Hz), sample count and analog channel count for a Intan RHD recording.

    The rate comes straight out of info.rhd's fixed-offset header: magic number
    (4B) + version major/minor (2x int16) + sample_rate (float32). The channel
    count is inferred from the analogin.dat / time.dat file-size ratio (uint16
    vs int32 samples).

    :param folder: path to the Intan RHD recording folder
    :return: sample rate (Hz), sample count, analog channel count
    '''
    logger.info(f'loading info from "{folder}" recording')

    # Check for info file existence
    info_fpath = os.path.join(folder, 'info.rhd')
    if not os.path.exists(info_fpath):
        raise FileNotFoundError(f'info.rhd file not found in "{folder}"')

    # Load info.rhd header and extract sample rate
    logger.info('getting sample rate from info.rhd header')
    with open(info_fpath, 'rb') as f:
        magic_number, = struct.unpack('<I', f.read(4))
        if magic_number != 0xc6912702:
            raise ValueError(f'{folder}/info.rhd is not a valid Intan RHD header')
        f.read(4)  # version major/minor, unused here
        fs, = struct.unpack('<f', f.read(4))
    logger.info(f'sample rate: {fs} Hz')

    # Get sample count and analog channel count from file sizes
    logger.info('getting sample count and analog channel count from file sizes')
    time_bytes = os.path.getsize(os.path.join(folder, 'time.dat'))
    analog_bytes = os.path.getsize(os.path.join(folder, 'analogin.dat'))
    nsamples = time_bytes // 4
    nchannels = round(2 * analog_bytes / time_bytes)
    logger.info(f'sample count: {nsamples}, analog channel count: {nchannels}')

    # Return sample rate, sample count and analog channel count
    return fs, nsamples, nchannels


def load_analog_downsampled(folder, target_fs=TARGET_FS, channel=0, chunk_bins=100_000):
    '''
    Load and mean-downsample one analogin channel straight off disk, in chunks.
    
    time.dat is checked for continuity rather than read in full; 
    if it ever has gaps, this assumption needs revisiting.

    :param folder: path to the Intan RHD recording folder
    :param target_fs: target downsampled rate (Hz)
    :param channel: analogin channel index (0-based)
    :param chunk_bins: number of bins to process in each chunk
    :return: 4-tuple with time vector (s), voltage vector (V), raw rate (Hz), downsampled rate (Hz)
    '''
    # Get recording layout
    raw_fs, n_samples, n_channels = get_intan_layout(folder)

    # Check time.dat continuity
    time_fpath = os.path.join(folder, 'time.dat')
    t_raw = np.memmap(time_fpath, dtype=np.int32, mode='r')
    if int(t_raw[-1]) - int(t_raw[0]) + 1 != n_samples:
        raise ValueError(f'{time_fpath} is not contiguous; sample times need reading in full')

    # Virtually load analogin file using memmap
    logger.info('loading analogin.dat as memmap')
    analogin_fpath = os.path.join(folder, 'analogin.dat')
    analog = np.memmap(analogin_fpath, dtype=np.uint16, mode='r')

    # Determine downsampling factor and true downsampled rate
    ds_factor = max(int(round(raw_fs / target_fs)), 1)
    ds_fs = raw_fs / ds_factor
    n_bins = n_samples // ds_factor

    # Read and downsample the requested channel in chunks 
    # to avoid reading the whole file into memory
    logger.info(f'downsampling channel {channel} from {raw_fs} Hz to {ds_fs:.3f} Hz')
    analog_ds = np.empty(n_bins, dtype=np.float64)
    for start_bin in range(0, n_bins, chunk_bins):
        stop_bin = min(start_bin + chunk_bins, n_bins)
        block = np.asarray(
            analog[(start_bin * ds_factor) * n_channels:(stop_bin * ds_factor) * n_channels],
            dtype=np.float64)
        block = block.reshape(-1, n_channels)[:, channel]
        analog_ds[start_bin:stop_bin] = block.reshape(stop_bin - start_bin, ds_factor).mean(axis=1)

    # Convert downsampled ADC vector to volts
    logger.info('converting downsampled ADC vector to volts')
    v_ds = analog_ds * ADC_TO_VOLTS

    # Generate downsampled time vector (s)
    logger.info('generating downsampled time vector')
    t_s = (np.arange(n_bins) + 0.5) / ds_fs

    # Return outputs
    return t_s, v_ds, raw_fs, ds_fs


def get_active_digital_lines(words, chunk):
    ''' 
    Determine which digitalin bits are ever high, without unpacking the whole file.

    :param words: memmap of the digital input DAT file
    :param chunk: number of words to read in each chunk
    :return: array of active digitalin line ces (0-based)
    '''
    # Initialize accumulator for OR-reduction of all words, 
    acc = np.uint16(0)

    # Perform bitwise OR-reduction of all words in chunks, 
    # to avoid reading the whole file into memory 
    for start in range(0, words.size, chunk):
        acc |= np.bitwise_or.reduce(np.asarray(words[start:start + chunk]))

    # Check which bits are ever high
    is_high = ((int(acc) >> np.arange(16)) & 1).astype(bool)

    # Return indices of active digitalin lines (0-based)
    return np.where(is_high)[0]


def load_digital_rising_edges(folder, line=None, chunk=20_000_000):
    '''
    Sample indices of every rising edge on one digitalin line, found in
    chunks off a memory map. Only the requested bit is ever unpacked.

    :param folder: path to the Intan RHD recording folder
    :param line: digitalin line index (0-based). If not specified, 
        all lines are checked and the first one with rising edges is returned.
    :param chunk: number of words to read in each chunk
    :return: array of sample indices where the specified digitalin line rises
    '''
    # Virtually load the digitalin file as memmap
    logger.info('loading digitalin.dat as memmap')
    digitalin_fpath = os.path.join(folder, 'digitalin.dat')
    words = np.memmap(digitalin_fpath, dtype=np.uint16, mode='r')

    # If no line is specified, find the first one with rising edges
    if line is None:
        logger.info('no digitalin line specified, checking all lines for rising edges')
        active_lines = get_active_digital_lines(words, chunk)
        logger.info(f'active digitalin lines: {active_lines}')
        for l in active_lines:
            edges = load_digital_rising_edges(folder, line=l, chunk=chunk)
            if edges.size > 0:
                logger.info(f'found rising edges on line {l}, returning them')
                return edges
        raise ValueError('no rising edges found on any digitalin line')

    # Initialize list of rising edge indices and previous bit value
    edges = []
    previous = np.int8(0)

    # Read the digitalin file in chunks, unpack the requested bit, and find rising edges
    for start in range(0, words.size, chunk):
        bits = ((np.asarray(words[start:start + chunk]) >> line) & 1).astype(np.int8)
        edges.append(np.where(np.diff(np.concatenate(([previous], bits))) == 1)[0] + start)
        previous = bits[-1]

    # Concatenate all rising edge indices and return
    return np.concatenate(edges)


def volts_to_degc(v):
    '''
    Convert Osensa probe voltage (analogin channel 0, across burden resistor RA) to degC.
    Linear 4-20 mA current-loop scaling, matching the vendor's MATLAB:
        temp = Tzero + (V - Imin*RA) / (Imax*RA - Imin*RA) * Tspan
    '''
    logger.info('converting Osensa probe voltage to degC')
    v_min = OSENSA_I_MIN_A * OSENSA_RA_OHM  # V
    v_max = OSENSA_I_MAX_A * OSENSA_RA_OHM  # V
    return OSENSA_TZERO_C + (v - v_min) / (v_max - v_min) * OSENSA_TSPAN_C


def filter_temperature_recording(y, fs, fc=LOWPASS_FC):
    '''
    Lowpass filter the temperature recording.

    :param y: temperature recording vector
    :param fs: sampling rate of the recording (Hz)
    :param fc: cutoff frequency for lowpass filter (Hz)
    '''
    logger.info(f'Lowpass filtering temperature trace at {fc} Hz')    

    # Generate 2nd-order lowpass Butterworth filter coefficients
    sos = signal.butter(2, fc, btype='low', fs=fs, output='sos')

    # Apply zero-phase lowpass filter (effectively 4th order) to the whole continuous trace, and return
    return signal.sosfiltfilt(sos, y)


def plot_longitudinal_temperature_recording(recording, display_fs=OVERVIEW_FS, title=None):
    '''
    Plot the longitudinal temperature recording.

    :param recording: pandas DataFrame with temperature data indexed by sample
    :param display_fs: sampling rate for display purposes (Hz)
    :param title: optional title for the plot
    '''
    # Compute the sampling interval and sampling rate from the time vector
    dt = recording[TIME_KEY][1] - recording[TIME_KEY][0]
    fs = 1 / dt

    # Compute downsampling factor and downsample the recording for display purposes
    ds_factor = max(int(round(fs / display_fs)), 1)
    effective_display_fs = fs / ds_factor
    recording = recording.iloc[::ds_factor, :].copy()

    # Add minutes column
    recording['time (min)'] = recording[TIME_KEY] / 60.0

    # Plot the longitudinal temperature recording
    logger.info(f'plotting longitudinal temperature recording at {effective_display_fs:.3f} Hz')
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.despine(ax=ax)
    sns.lineplot(
        data=recording,
        x='time (min)',
        y=TEMP_KEY,
        ax=ax
    )

    # Add figure title
    tit = 'longitudinal temperature recording'
    if title is not None:
        tit += f' - {title}'
    ax.set_title(tit)

    # Return figure
    return fig


def load_thermal_experiment(folder, plot=False):
    '''
    Load and pre-process thermal experiment data from a given folder, 
    including analog temperature recordings, digital trigger edges, 
    and experiment log data.

    :param folder: path to the thermal experiment data folder
    :param plot: if True, plot the longitudinal temperature recording
    :return: tuple containing:
        - recording: pandas DataFrame with temperature recording data indexed by sample
        - edge_times: array of sample indices where the digital trigger rises
        - log_data: pandas DataFrame with experiment log data indexed by trial
    '''
    # Load downsampled Itan analog recording data
    t_ds, v_ds, fs, ds_fs = load_analog_downsampled(folder)

    # Convert voltage to temperature in degC using Osensa probe calibration
    temp_ds = volts_to_degc(v_ds)

    # Lowpass the whole continuous trace
    temp_filtered = filter_temperature_recording(temp_ds, ds_fs)
    
    # Assemble a sample-indexed DataFrame
    recording = pd.DataFrame({
        TIME_KEY: t_ds,
        TEMP_KEY: temp_filtered,
    })
    recording.index.name = 'sample'

    # Plot the longitudinal temperature recording if requested
    if plot:
        plot_longitudinal_temperature_recording(recording, title=os.path.basename(folder))

    # Load rising edge times from Intan digital channel (trigger signal)
    edge_times = load_digital_rising_edges(folder) / fs

    # Load experiment log data from CSV file
    log_data = load_experiment_log(folder)

    return recording, edge_times, log_data


def interpolate_trial_data(data, ykey, target_reltime):
    '''
    Interpolate trial time traces along a specific relative time vector.

    :param data: pandas DataFrame with trial time traces, indexed by trial
    :param ykeys: list of column names to interpolate
    :param target_reltime: relative time vector to interpolate onto
    :return: time-indexed pandas Series with interpolated trial time trace
    '''
    # Interpolate trace along the target relative time vector
    return pd.Series(
        np.interp(
            x=target_reltime,
            xp=data[REL_TIME_KEY].values,
            fp=data[ykey].values,
            left=np.nan,
            right=np.nan
        ),
        index=pd.Index(target_reltime, name=TIME_KEY),
        name=ykey
    )


def get_time_harmonized_data(data):
    '''
    Harmonize trial time traces along a common relative time vector.

    :param data: pandas DataFrame with trial time traces, 
        containing signal column(s) and trial index column
    :return: pandas Series with harmonized time traces,
        indexed by trial and relative time
    '''
    # Compute common relative time range across all trials
    groups = data.groupby('trial')[REL_TIME_KEY]
    rel_tbounds = (groups.min().max(), groups.max().min())
    logger.info(f'harmonizing trial time traces along common [{rel_tbounds[0]:.3f}, {rel_tbounds[1]:.3f}] s time vector')

    # Define common relative time vector for all trials
    min_trial_duration = rel_tbounds[1] - rel_tbounds[0]
    dt = data[REL_TIME_KEY].diff().median().round(6)
    common_reltime = np.arange(0, min_trial_duration + dt / 2, dt) + rel_tbounds[0]

    # Interpolate all trials along the common relative time vector
    temp_harmonized = (
        data
        .groupby('trial')
        .apply(lambda df: interpolate_trial_data(df, TEMP_KEY, common_reltime))
        .stack()
        .rename(TEMP_KEY)
    )

    # Remove data preceding first trial onset
    temp_harmonized = temp_harmonized.loc[0:]

    # Return
    return temp_harmonized


def assign_trials(recording, edge_times, log_data, min_gap=GROUP_GAP):
    ''' 
    Assign trial indices and relative time to each sample in the temperature recording,
    based on the rising edge times of the digital trigger signal and the experiment log data.

    :param recording: pandas DataFrame with temperature recording data indexed by sample
    :param edge_times: array of sample indices where the digital trigger rises
    :param log_data: pandas DataFrame with experiment log data indexed by trial
    :param min_gap: minimum gap between trials (default: GROUP_GAP)
    :return: pandas DataFrame with harmonized trial time traces, indexed by trial and relative time,
        including trial information from the log data
    '''
    # Group edges by burst IDs (1 burst = 1 trial) 
    burst_ids = np.concatenate(([0], np.cumsum(np.diff(edge_times) > min_gap)))

    # Identify the index (and time) of the first edge of each burst (trial)
    i_newburst = np.concatenate(([0], np.where(np.diff(burst_ids) == 1)[0] + 1))
    stim_onset_times = edge_times[i_newburst]
    ntrials = stim_onset_times.size

    # Compute effective inter-trial interval (ITI) from the Intan recording
    min_interval = np.diff(stim_onset_times).min()  # s

    # Check that the number of trials matches the number of rows in the log data
    if ntrials != log_data.shape[0]:
        raise ValueError(f'Number of trials ({ntrials}) does not match number of rows in log data ({log_data.shape[0]})')

    # Compute the clock offset between the Intan recording and the experiment log
    clock_offset = stim_onset_times - log_data['elapsed time (s)']

    # Compute the relative clock drift over the experiment
    clock_drift = clock_offset.max() - clock_offset.min()
    if clock_drift > 1.0:
        logger.warning(f'{clock_drift:.2f} s relative clock drift over experiment -> check the burst/row pairing')
    else:
        logger.info(f'relative clock drift throughout experiment: {clock_drift:.2f} s')

    # Assign trial index to each sample in the temperature recording
    trial_start_times = stim_onset_times - STIM_ONSET
    itrial = trial_start_times.searchsorted(recording[TIME_KEY]) - 1
    recording['trial'] = itrial 

    # Compute relative time of each sample with respect to the stim onset for each trial
    reltime = recording[TIME_KEY] - trial_start_times[itrial] - STIM_ONSET
    reltime[itrial < 0] = np.nan  # samples before first trial have no relative time
    reltime[reltime > min_interval] = np.nan  # samples after last trial have no relative time
    recording[REL_TIME_KEY] = reltime

    # Harmonize relative time across trials and add trial information
    harmonized_recording = get_time_harmonized_data(recording)

    # Add trial information from the experiment log to the harmonized recording
    logger.info('adding trial information to harmonized recording')
    harmonized_recording = (
        harmonized_recording
        .to_frame()
        .join(log_data)
    )
    
    # Return the harmonized recording with trial information
    return harmonized_recording


def load_and_process_thermal_experiment(folder, allow_recursive=True, trial_offset=0, **kwargs):
    '''
    Load and process thermal experiment data from a given folder. If the folder
    does not contain an info.rhd file, the function will recursively search for 
    subfolders containing thermal experiment data, with max depth=1.

    :param folder: path to the thermal experiment data folder
    :return: trial and time indexed experiment data with log information
    '''
    # Look for the info.rhd file in the folder
    info_fpath = os.path.join(folder, 'info.rhd')

    # If it does not exist
    if not os.path.exists(info_fpath):
        # If no recursive search is allowed, raise an error
        if not allow_recursive:
            raise FileNotFoundError(f'info.rhd file not found in "{folder}" and recursive search is disabled')

        # Try to call function recursively on all subfolders
        data = {}
        trial_offset = 0
        for item in os.listdir(folder):
            sub_path = os.path.join(folder, item)
            if os.path.isdir(sub_path):
                data[item] = load_and_process_thermal_experiment(
                    sub_path, allow_recursive=False, trial_offset=trial_offset, **kwargs)
                trial_offset = data[item].index.get_level_values('trial').max() + 1

        # If no subfolders contain thermal experiment data, raise an error
        if not data:
            raise FileNotFoundError(f'No thermal experiment data found in "{folder}" or its subfolders')

        # Concatenate all subfolder data into a single DataFrame, and return
        return pd.concat(data, names=['condition'])
    
    # Load the thermal experiment data
    recording, edge_times, log_data = load_thermal_experiment(folder, **kwargs)

    # Assign trials to the recording based on the edge times and log data
    data = assign_trials(recording, edge_times, log_data)

    # If trial offset, apply it
    if trial_offset != 0:
        idx_names = data.index.names
        df_reset = data.reset_index()
        df_reset['trial'] = df_reset['trial'] + trial_offset
        data = df_reset.set_index(idx_names)

    # Return the harmonized recording with trial information
    return data


def detrend_trial(y):
    '''
    Detrend a single trial of temperature data using linear detrending.

    :param y: pandas Series with trial temperature trace, indexed by relative time
    :return: pandas Series with detrended trial temperature trace, indexed by relative time
    '''
    # Identify baseline mask
    t = y.index.get_level_values(TIME_KEY)
    baseline_mask = ~np.logical_and(
        t > RESPONSE_WINDOW[0],
        t < RESPONSE_WINDOW[1]
    )

    # Fit a linear trend (y = mx + c) on baseline samples
    slope, intercept = np.polyfit(t[baseline_mask], y[baseline_mask], 1)

    # Calculate the trend line across the entire signal
    full_trend = slope * t + intercept

    # Subtract the trend from the original signal
    y_detrended = y - full_trend

    # Re-add signal vertical offset
    y_detrended += y[baseline_mask].mean() 

    # Return
    return pd.Series(y_detrended, index=y.index, name=y.name)


def compute_deltaT(y, baseline_pre=BASELINE_PRE):
    '''
    Compute relative temperature changes with respect to the pre-stimulus baseline for each trial.

    :param y: pandas Series with trial temperature trace, indexed by relative time
    :param baseline_pre: pre-onset span (s) averaged and subtracted from each trial
    :return: pandas series with relative temperature changes, indexed by relative time
    '''
    # Compute pre-stimulus baseline
    t = y.index.get_level_values(TIME_KEY)
    baseline_mask = np.logical_and(t > -baseline_pre, t < 0)
    y0 = y[baseline_mask].mean()

    # Return relative temperature change
    return y - y0


def condition_keys(index):
    ''' Extract condition keys present in the data '''
    return [k for k in COND_KEYS if k in index]


def compute_trial_average(data):
    '''
    Aggregate temperature data over trials for each condition 
    
    :param data: pandas DataFrame with trial time traces,
        indexed by trial and relative time
    :return: pandas DataFrame with trial-averaged data for each condition, 
        indexed by condition(s) and relative time
    '''
    # Identify over which to average trial data
    cond_keys = condition_keys(list(data.columns) + list(data.index.names))

    # Identify columns to average
    aggkeys = list(set(data.columns) - set(NONAGG_KEYS + cond_keys))

    # Aggregate the data by averaging over trials for each condition
    logger.info(f'computing trial-average data by {cond_keys}')
    return (
        data
        .groupby([*cond_keys, TIME_KEY])
        [aggkeys]
        .mean()
    )


def plot_evoked_thermal_response(ax=None, data=None, y=None, **kwargs):
    '''
    Plot the evoked thermal response for a given ykey (e.g., relative temperature change)
    as a function of relative time.

    :param ax: matplotlib Axes object to plot on. If None, a new figure and axes are created.
    :param data: pandas DataFrame with trial time traces, indexed by trial and relative time
    :param y: column name of the y-axis variable to plot (e.g., relative temperature change)
    :param kwargs: additional keyword arguments to pass to sns.lineplot
    '''
    # Set the axes and despine    
    if ax is None:
        ax = plt.gca()
    kwargs['ax'] = ax
    sns.despine(ax=ax)

    # Plot the evoked thermal response using seaborn lineplot
    sns.lineplot(
        data=data,
        x=TIME_KEY,
        y=y,
        **kwargs
    )

    # Add horizontal line at y=0 for reference
    ax.axhline(0, color='k', linestyle='--', linewidth=1)

    # If stim duration is available in the data, add a shaded region to indicate the stimulus period
    if STIMDUR_KEY in data.columns:
        tstim = data[STIMDUR_KEY].iloc[0]
        ax.axvspan(0., tstim, color='silver', alpha=0.5)


def double_exp(t, t0, amplitude, tau_rise, tau_decay, offset):
    '''
    Rise-times-decay double-exponential transient.

    :param t: time array
    :param t0: time of onset
    :param amplitude: amplitude factor of the transient
    :param tau_rise: rise time constant
    :param tau_decay: decay time constant
    :param offset: baseline offset
    :return: double-exponential transient
    '''
    # Compute the relative time since onset 
    trel = t - t0

    # Initialize the output array
    y = np.zeros_like(t)

    # Get mask of time points where the transient is active (after onset)
    m = trel >= 0

    # Compute the double-exponential transient for active time points
    y[m] = (1 - np.exp(-trel[m] / tau_rise)) * np.exp(-trel[m] / tau_decay)

    # Return the transient with amplitude and offset applied
    return offset + amplitude * y


def double_exp_peak(t, t0, peak, tau_rise, tau_decay, offset):
    '''
    double_exp renormalized so that 'peak' is the transient's actual peak
    height rather than a scale factor on the un-normalized shape.

    Setting the derivative to zero gives exp(-t/tau_rise) = tr / (tr + td) at
    the maximum, from which we can extract the closed-form peak height.
    '''
    # Compute the unit peak value based on the rise and decay time constants
    w = tau_rise / (tau_rise + tau_decay)
    unit_peak = (1 - w) * np.power(w, tau_rise / tau_decay)

    # Return the double_exp transient with the peak renormalized to the specified peak height
    return double_exp(t, t0, peak / unit_peak, tau_rise, tau_decay, offset)


def fit_double_exp(y):
    '''
    Fit double_exp_peak to one mean transient, or return None if the
    transient is at the noise floor or the fit does not converge.

    - tau_rise is not identifiable on this data: the probe rises almost linearly while the
        0.2 s stimulus is on, which is the tau_rise -> infinity limit of the model,
        so tau_rise sits on its FIT_MAX_TAU_RISE bound for most conditions. 
    - t0 (~170 ms here) absorbs the probe's onset lag. Quote the peak, the time to
        peak and tau_decay; treat a saturated tau_rise as a lower bound.

    :param y: time-indexed series of temperature trace
    :return: optimal parameters for the double_exp_peak fit, or None if the fit fails
    '''
    # Extract time vector from index
    t = y.index.get_level_values(TIME_KEY).values

    # Initialize output series
    sopt = pd.Series(
        index=pd.Index([
            't0',
            'peak',
            'tau_rise',
            'tau_decay',
            'offset'
        ], name='parameter'),
        dtype=float
    )
    
    # Extract transient peak amplitude from the trace, and return None 
    # if it is below the minimum threshold 
    ypeak = y[t >= 0].max()
    if ypeak < FIT_MIN_CELSIUS_AMPLITUDE:
        return sopt

    # Set initial guess for the fit parameters
    p0 = [
        0.1,  # onset time
        ypeak,  # peak amplitude
        0.1,  # rise time constant
        0.3,  # decay time constant
        0.0,   # baseline offset
    ]

    # Set search bounds for the fit parameters
    pbounds = [
        (0.0, .5),  # t0 bounds
        (0, min(5 * ypeak, FIT_MAX_CELSIUS_AMPLITUDE)),  # peak bounds
        (1e-3, FIT_MAX_TAU_RISE),  # tau_rise bounds
        (1e-2, FIT_MAX_TAU_DECAY),  # tau_decay bounds
        (-0.05, 0.05),  # offset
    ]
    bounds = ([b[0] for b in pbounds], [b[1] for b in pbounds])

    # Attempt to fit the double_exp_peak function to the trace using curve_fit,
    # and return None if the fit does not converge
    try:
        popt, _ = optimize.curve_fit(
            double_exp_peak,
            t,
            y.values,
            p0=p0,
            bounds=bounds
        )
    except RuntimeError as e:
        logger.warning(e)
        return sopt

    # Return the optimal parameters for the double_exp_peak fit
    return pd.Series(dict(zip(sopt.index, popt)))


def longest_common_substring(strs):
    if not strs:
        return ''

    # Start with the shortest string to minimize iterations
    shortest = min(strs, key=len)
    n = len(shortest)

    # Search from longest possible substring down to shortest
    for length in range(n, 0, -1):
        for i in range(n - length + 1):
            sub = shortest[i : i + length]
            if all(sub in s for s in strs):
                return sub
    return ''


def melt_by_condition(data, ymap, melt_key):
    '''
    Melt trial-averaged data by condition, renaming ykeys according to ymap.
    '''
    # Find common demoniator in ymap keys
    common_key = longest_common_substring(ymap.keys())

    # Melt
    return (
        data
        .rename(columns=ymap)
        .reset_index()
        .melt(
            id_vars=list(data.index.names),
            value_vars=list(ymap.values()),
            var_name=melt_key,
            value_name=common_key,
        )
        .set_index(list(data.index.names) + [melt_key])
    )


def select_subset(data, trial_stats, cond):
    '''
    Extract a subset of temperature data and associated trial statistics
    for a given combination of conditions

    :param data: pandas DataFrame with trial time traces, indexed by trial and relative time
    :param trial_stats: pandas dataframe/series of trial statistics
    :param cond: tuple of conditions combination
    :return: 2-tuple with subset data and stats
    '''
    cond_stats = trial_stats.loc[cond]
    cond_data = data.loc[cond_stats.index.get_level_values('trial')]
    return cond_data, cond_stats 


def plot_pressure_dependence(data, max_ΔT, ax=None, title=None):
    ''' 
    Plot pressure dependence of stim-evoked temperature change and its peak.

    :param data: pandas DataFrame with trial time traces, indexed by trial and relative time
    :param max_ΔT: pandas Series with max-evoked temperature change for each trial
    '''
    # Create/retrieve axis and figure
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    
    # Add title, if specified
    if title is not None:
        ax.set_title(title)

    # Plot pressure-dependent temperature traces
    logger.info(f'plotting pressure-dependent evoked temperature change traces')
    sns.despine(ax=ax)
    for units in ['trial', None]:
        plot_evoked_thermal_response(
            ax=ax,
            data=data.reset_index(),
            y=REL_TEMP_KEY,
            hue=Label.P,
            palette='flare',
            estimator=None if units is not None else 'mean',
            errorbar=None,
            units=units,
            alpha=0.2 if units is not None else 1.0,
            lw=.5 if units is not None else 2.0,
            legend='full' if units is None else False
        )
    sns.move_legend(ax, 'upper right', title=Label.P, frameon=False)

    # Perform quadratic fit to the pressure-dependence of max_ΔT
    logger.info('performing quadratic fit on max_ΔT vs pressure')
    Pvals = max_ΔT.index.get_level_values(Label.P)
    quadfit_popt = np.polyfit(Pvals, max_ΔT.values, deg=2)
    quadfit_pred = np.polyval(quadfit_popt, Pvals)
    quadfit_r2 = np.corrcoef(max_ΔT.values, quadfit_pred)[0, 1] ** 2

    ylims = ax.get_ylim()
    ax.set_ylim(ylims[0], ylims[1] + 0.2 * (ylims[1] - ylims[0]))

    # Add inset axis
    inset_ax = ax.inset_axes([0.60, 0.25, 0.3, 0.4])

    # Plot dose-dependence of max-evoked temperature change at focus
    logger.info(f'plotting dose-dependence of max-evoked temperature change at focus')
    sns.despine(ax=inset_ax)
    sns.lineplot(
        ax=inset_ax,
        data=max_ΔT.reset_index(),
        x=Label.P,
        y=MAX_REL_TEMP_KEY,
        errorbar='sd',
        ls='',
        err_style='bars',
        err_kws={'elinewidth': 1.5, 'capsize': 10},
    )
    inset_ax.set_xticks(Pvals.unique())
    inset_ax.set_xticklabels(Pvals.unique())

    # Add quadratic fit to the dose-dependence of max-evoked temperature change at focus
    Pdense = np.linspace(Pvals.min(), Pvals.max(), 100)
    quadfit_pred_dense = np.polyval(quadfit_popt, Pdense)
    inset_ax.plot(
        Pdense,
        quadfit_pred_dense,
        color='k',
        lw=2,
        ls='--',
        label=f'∝P² (R² = {quadfit_r2:.3f})'
    )
    inset_ax.legend(frameon=False)

    return fig