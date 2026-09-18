# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2026-09-11 13:44:14
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2026-09-18 15:45:19

import os
import struct
import numpy as np
from scipy import signal, optimize
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from natsort import natsorted
import re
import warnings
from tables import NaturalNameWarning

from .logger import logger
from .constants import *
from .utils import *

''' Utilities for the analysis of temperature measurement experiments '''

# Condition keys that should be conserved upon trial-averaging
COND_KEYS = [
    Label.SPECIMEN,
    Label.REC,
    Label.LOCATION,
    Label.X_MM,
    Label.Y_MM,
    Label.Z_MM,
    Label.P,
    Label.ISPPA,
    Label.ISPTA
]

# Keys that should be discarded upon trial-averaging
NONAGG_KEYS = [
    Label.TIMESTAMP,
    Label.ELAPSED_TIME
]

# Intan RHD2000 acquisition system constants
ADC_TO_VOLTS = 50.354e-6  # V per ADC unit, from Intan RHD2000 datasheet
VOLTAGE_DIVIDER_RA = 4610.  # voltage divider resistor A, between Osensa and Intan anaog input (Ohm)
VOLTAGE_DIVIDER_RB = 4610.   # voltage divider resistor B, between Osensa and ground (Ohm)

# Osensa fiber-optic temperature probe constants
OSENSA_IRANGE = (4e-3, 20e-3)  # current range (A) for Tzero to (Tzero + Tspan), from FTX-300-LUX+ manual
OSENSA_RA = 197.5  # 149.3  # Ω (from Misi, TO MEASURE)
OSENSA_TZERO = -40. # °C (probe-specific, TO CHECK WITH COMPANY)
OSENSA_TSPAN = 160. # °C (probe-specific, TO CHECK WITH COMPANY)
OSENSA_SENSOR_TIP_OFFSET = 1e-3  # delta z (m) from the tip of the fiber to the sensing zone
OSENSA_PROBE_TOFFSET = 0.  # probe-specific temprature offset (°C) to be added to the measured temperature, if known

# Analysis constants
TARGET_FS = 120  # target sampling rate when downsampling loaded data (Hz)
LOWPASS_FC = 30  # Hz, lowpass filter cutoff frequency for analog signal
OVERVIEW_FS = 4   # sampling rate for longitudinal overview (Hz)
GROUP_GAP = 1.0  # s, edges further apart than this start a new burst
STIM_ONSET = 0.5  # s, window length before each trigger burst onset
PROG_TRIGGER_DELAY = 0.15  # s, delay between trigger onset and actual stimulus onset (due to hardware limitations with programmatic trigger)
RESPONSE_WINDOW = (0., 3.0)  # s, time window containing expected response after each trigger burst onset
BASELINE_PRE = 0.4  # s, pre-onset span averaged and subtracted from each trial

# Double-exponential fit bounds
FIT_MIN_CELSIUS_AMPLITUDE = 0.01  # °C
FIT_MAX_CELSIUS_AMPLITUDE = 10.0  # °C
FIT_MAX_TAU_RISE = 1.0  # s
FIT_MAX_TAU_DECAY = 5.0  # s

# Pressure-related constants
P_RANGE = (0, 2.5)  # pressure range (MPA) used throughout experiments, for consistent color-coding
ISPTA_RANGE = (0, 130)  # ISPTA range (W/cm2) used throughout experiments, for consistent color-coding
COVERGLASS_ATTENUATION_FACTOR = 0.70  # pressure attenuation factor between measurements in free-field and 2x150um coverglass (average of 2 transducers, measured by Theo)

# Regexp patterns
LOG_FILEPATTERN = 'thermal_experiment_log_([A-Za-z0-9]+)_T(\d+)_([A-Za-z0-9]+)_(\d+\.\d+)MHz(.*).csv'
MOUSE_PATTERN = 'M([0-9]+)'

# Shared hue parameters for traces plotting  
TRACE_HUE_KWARGS = dict(
    hue=Label.ISPTA,
    palette='flare',
)


def get_thermal_data_root():
    ''' Get root directory for thermal data '''
    # dataroot = '/Volumes/shohas01lab/shohas01labspace/Theo/US_thermal_experiments'
    # if not os.path.exists(dataroot):
    #     logger.info(f'could not connect to R-drive -> using local data directory')
    dataroot = '/Users/tlemaire/Documents/data/US_thermal_experiments'
    logger.info(f'root data directory: "{dataroot}"')
    return dataroot


def parse_log_filename(fname):
    '''
    Parse the thermal experiment log filename
    
    :param fname: filename of the thermal experiment log
    :return: dictionary containing:
        - specimen
        - transducer ID
        - calibration condition
        - carrier frequency
        - recording condition, if present in the filename
    '''
    # Parse log file name
    mo = re.search(LOG_FILEPATTERN, fname)
    if mo is None:
        raise ValueError(f'Filename "{fname}" does not match expected pattern {LOG_FILEPATTERN}')
    specimen = mo.group(1)
    transducer = f'T{mo.group(2)}'
    calib_cond = mo.group(3)
    fMHz = float(mo.group(4))
    rec_cond = mo.group(5)
    d = {
        Label.SPECIMEN: specimen,
        'transducer ID': transducer,
        'calibration condition': calib_cond,
        Label.FREQ_MHZ: fMHz,
    }
    if len(rec_cond) > 0:
        d['recording condition'] = rec_cond
    return d


def find_log_file(folder):
    ''' 
    Find log file in experiment folder
    
    :param folder: experiment data folder
    :return: name of log file (if found and unique) in the folder
    '''
    # Search for CSV file starting with 'thermal_experiment_log_' in folder
    items = os.listdir(folder)
    log_fnames = [item for item in items if re.search(LOG_FILEPATTERN, item)]
    log_fpaths = [os.path.join(folder, fname) for fname in log_fnames]

    # Make sure there is exactly one log file found
    if len(log_fpaths) == 0:
        raise FileNotFoundError(f'No log file found in {folder} matching pattern {LOG_FILEPATTERN}')
    elif len(log_fpaths) > 1:
        raise FileExistsError(f'Multiple log files found in {folder} matching pattern {LOG_FILEPATTERN}: {log_fpaths}')

    # Return log file name
    return os.path.basename(log_fpaths[0])


def load_experiment_log(folder):
    '''
    Load experiment log data from CSV file starting with 'thermal_experiment_log_' in folder.

    :param folder: experiment data folder
    :return: pandas DataFrame with log data, indexed by trial
    '''
    # Find log file in folder
    log_fname = find_log_file(folder)
    log_fpath = os.path.join(folder, log_fname)

    # Load log data from CSV file into pandas DataFrame, and set index name to 'trial'
    logger.info(f'loading experiment log from "{log_fname}"')
    data = pd.read_csv(log_fpath)
    data.index.name = Label.TRIAL

    # Convert timestamp column to pandas datetime format, with microsecond precision
    data[Label.TIMESTAMP] = pd.to_datetime(data[Label.TIMESTAMP], format='%Y-%m-%d %H:%M:%S.%f')

    # Compute and add elapsed time column (s) since first trial, in seconds
    data[Label.ELAPSED_TIME] = (data[Label.TIMESTAMP] - data[Label.TIMESTAMP][0]).dt.total_seconds()

    # Parse log file name
    exp_info = parse_log_filename(os.path.basename(log_fpath))
    calib_cond = exp_info['calibration condition']

    # If specimen is a mouse (means no coverglass) and transducer calibration conditions are not free-field
    # adjust pressure values to free-field equivalent using the coverglass attenuation factor 
    is_mouse = re.match(MOUSE_PATTERN, exp_info[Label.SPECIMEN]) is not None
    if is_mouse and exp_info['calibration condition'] != 'free-field':
        logger.warning(f'"open craniotomy" mouse experiment but transducer calibrated in "{calib_cond}" conditions -> adjusting pressure values to free-field equivalent using attenuation factor {COVERGLASS_ATTENUATION_FACTOR}')
        data[Label.P] = (data[Label.P] / COVERGLASS_ATTENUATION_FACTOR).round(2)

    # Return log data
    return data


def _skip_qstring(f):
    ''' Skip a Qt-style QString in an Intan RHD header file. '''
    n_bytes, = struct.unpack('<I', f.read(4))
    if n_bytes != 0xFFFFFFFF:
        f.seek(n_bytes, 1)


def get_intan_layout(folder):
    '''
    Extract sample rate (Hz),board mode, sample count and analog channel count for a Intan RHD recording.

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

    # Load info.rhd header and extract information
    with open(info_fpath, 'rb') as f:
        magic_number, = struct.unpack('<I', f.read(4))
        if magic_number != 0xc6912702:
            raise ValueError(f'{folder}/info.rhd is not a valid Intan RHD header')

        # Get RHD version number
        major, minor = struct.unpack('<hh', f.read(4))
        rhd_version = float(f'{major}.{minor}')

        # Get sample rate (Hz) from header
        fs, = struct.unpack('<f', f.read(4))

        # Skip over the rest of the header fields that are not needed
        f.seek(2, 1)       # DSP enabled: int16
        f.seek(6 * 4, 1)   # six float32 DSP/bandwidth fields
        f.seek(2, 1)       # notch mode: int16
        f.seek(2 * 4, 1)   # two float32 impedance-test fields
        for _ in range(3):  # three QStrings
            _skip_qstring(f)

        # If RHD version >= 1.3, skip over the number of temperature sensors and read the board mode
        if rhd_version >= 1.3:
            f.seek(2, 1)  # number of temperature sensors
            board_mode, = struct.unpack('<h', f.read(2))
        else:
            board_mode = None

    # Log the extracted information
    logger.info(f'Intan RHD info: version = {rhd_version}, board mode = {board_mode}, sample rate = {fs} Hz')

    # Get sample count and analog channel count from file sizes
    time_bytes = os.path.getsize(os.path.join(folder, 'time.dat'))
    analog_bytes = os.path.getsize(os.path.join(folder, 'analogin.dat'))
    nsamples = time_bytes // 4
    nchannels = round(2 * analog_bytes / time_bytes)
    logger.info(f'sample count: {nsamples}, analog channel count: {nchannels}')

    # Return sample rate, board mode, sample count and analog channel count
    return fs, board_mode, nsamples, nchannels


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
    raw_fs, board_mode, n_samples, n_channels = get_intan_layout(folder)

    # Check time.dat continuity
    time_fpath = os.path.join(folder, 'time.dat')
    t_raw = np.memmap(time_fpath, dtype=np.int32, mode='r')
    if int(t_raw[-1]) - int(t_raw[0]) + 1 != n_samples:
        raise ValueError(f'{time_fpath} is not contiguous; sample times need reading in full')

    # Virtually load analogin file using memmap
    logger.info(f'loading analog input data, channel {channel}')
    analogin_fpath = os.path.join(folder, 'analogin.dat')
    analog = np.memmap(analogin_fpath, dtype=np.uint16, mode='r')

    # Determine downsampling factor and true downsampled rate
    ds_factor = max(int(round(raw_fs / target_fs)), 1)
    ds_fs = raw_fs / ds_factor
    n_bins = n_samples // ds_factor

    # Read and downsample the requested channel in chunks 
    # to avoid reading the whole file into memory
    logger.info(f'downsampling signal from {raw_fs} Hz to {ds_fs:.3f} Hz')
    analog_ds = np.empty(n_bins, dtype=np.float64)
    for start_bin in range(0, n_bins, chunk_bins):
        stop_bin = min(start_bin + chunk_bins, n_bins)
        block = np.asarray(
            analog[(start_bin * ds_factor) * n_channels:(stop_bin * ds_factor) * n_channels],
            dtype=np.float64)
        block = block.reshape(-1, n_channels)[:, channel]
        analog_ds[start_bin:stop_bin] = block.reshape(stop_bin - start_bin, ds_factor).mean(axis=1)

    # Convert downsampled ADC vector to volts
    logger.info('converting signal from ADC units to volts')
    if board_mode != 0:
        raise ValueError(f'board mode {board_mode} not supported for voltage conversion')
    v_intan_ds = analog_ds * ADC_TO_VOLTS
    k = VOLTAGE_DIVIDER_RA / (VOLTAGE_DIVIDER_RA + VOLTAGE_DIVIDER_RB)
    v_source_ds = v_intan_ds / k

    # Generate downsampled time vector (s)
    t_s = (np.arange(n_bins) + 0.5) / ds_fs

    # Return outputs
    return t_s, v_source_ds, raw_fs, ds_fs


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


def load_digital_rising_edges(folder, line=None, chunk=20_000_000, verbose=True):
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
    if verbose:
        logger.info('loading digital input data')
    digitalin_fpath = os.path.join(folder, 'digitalin.dat')
    words = np.memmap(digitalin_fpath, dtype=np.uint16, mode='r')

    # If no line is specified, find the first one with rising edges
    if line is None:
        active_lines = get_active_digital_lines(words, chunk)
        logger.info(f'active digital input lines: {active_lines}')
        for l in active_lines:
            edges = load_digital_rising_edges(folder, line=l, chunk=chunk, verbose=False)
            if edges.size > 0:
                logger.info(f'found rising edges on line {l}, returning them')
                return edges
        raise ValueError('no rising edges found on any digital input line')

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
    Convert Osensa probe voltage (analogin channel 0, across burden resistor RA) to °C.
    Linear 4-20 mA current-loop scaling, matching the vendor's MATLAB:
        temp = Tzero + (V - Imin*RA) / (Imax*RA - Imin*RA) * Tspan
    '''
    logger.info('converting Osensa probe voltage signal to temprature (°C)')
    osensa_vrange = np.array(OSENSA_IRANGE) * OSENSA_RA  # V
    norm_v = (v - osensa_vrange[0]) / (osensa_vrange[1] - osensa_vrange[0])
    return OSENSA_TZERO + norm_v * OSENSA_TSPAN + OSENSA_PROBE_TOFFSET


def filter_temperature_recording(y, fs, fc=LOWPASS_FC):
    '''
    Lowpass filter the temperature recording.

    :param y: temperature recording vector
    :param fs: sampling rate of the recording (Hz)
    :param fc: cutoff frequency for lowpass filter (Hz)
    '''
    logger.info(f'lowpass filtering temperature signal at {fc} Hz')    

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
    dt = recording[Label.TIME][1] - recording[Label.TIME][0]
    fs = 1 / dt

    # Compute downsampling factor and downsample the recording for display purposes
    ds_factor = max(int(round(fs / display_fs)), 1)
    effective_display_fs = fs / ds_factor
    recording = recording.iloc[::ds_factor, :].copy()

    # Add minutes column
    recording[Label.TIME_MIN] = recording[Label.TIME] / MIN_TO_S

    # Plot the longitudinal temperature recording
    s = 'longitudinal temperature recording'
    if title is not None:
        s = f'{title} {s}'
    logger.info(f'plotting {s}')
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.despine(ax=ax)
    sns.lineplot(
        data=recording,
        x=Label.TIME_MIN,
        y=Label.TEMP,
        ax=ax
    )

    # Add horizontal lines for min and max temperature values
    Tmin = recording[Label.TEMP].min()
    Tmax = recording[Label.TEMP].max()
    for T in [Tmin, Tmax]:
        ax.axhline(T, color='k', linestyle='--', alpha=0.5)

    # Add text with temperature range
    Trange = Tmax - Tmin
    ax.text(
        0.02, 0.9, f'Trange = {Trange:.2f} °C', transform=ax.transAxes, ha='left', va='top')

    # Add figure title, if specified
    if title is not None:
       ax.set_title(title)

    # Return figure
    return fig


def load_thermal_experiment(folder, plot=False, figdict=None, prefix=None):
    '''
    Load and pre-process thermal experiment data from a given folder, 
    including analog temperature recordings, digital trigger edges, 
    and experiment log data.

    :param folder: path to the thermal experiment data folder
    :param plot: if True, plot the longitudinal temperature recording
    :param figdict: dictionary to store the generated figures
    :return: tuple containing:
        - recording: pandas DataFrame with temperature recording data indexed by sample
        - edge_times: array of sample indices where the digital trigger rises
        - log_data: pandas DataFrame with experiment log data indexed by trial
    '''
    # Load downsampled Itan analog recording data
    t_ds, vsource_ds, fs, ds_fs = load_analog_downsampled(folder)

    # Convert voltage to temperature in °C using Osensa probe calibration
    temp_ds = volts_to_degc(vsource_ds)

    # Lowpass the whole continuous trace
    temp_filtered = filter_temperature_recording(temp_ds, ds_fs)
    
    # Assemble a sample-indexed DataFrame
    recording = pd.DataFrame({
        Label.TIME: t_ds,
        Label.TEMP: temp_filtered,
    })
    recording.index.name = Label.SAMPLE

    # Plot the longitudinal temperature recording if requested
    if plot:
        title = os.path.basename(folder)
        if prefix is not None:
            title = f'{prefix} - {title}'
        fig = plot_longitudinal_temperature_recording(recording, title=title)
        if figdict is not None:
            figdict[f'{os.path.basename(folder)} recording'] = fig

    # Load rising edge times from Intan digital channel (trigger signal)
    edge_times = load_digital_rising_edges(folder) / fs

    # Load experiment log data from CSV file
    log_data = load_experiment_log(folder)

    return recording, edge_times, log_data


def interpolate_trial_data(data, ykey, target_reltime):
    '''
    Interpolate trial time traces along a specific relative time vector.

    :param data: pandas DataFrame with trial time traces, indexed by trial
    :param ykey: name of the column to interpolate
    :param target_reltime: relative time vector to interpolate onto
    :return: time-indexed pandas Series with interpolated trial time trace
    '''
    # Interpolate trace along the target relative time vector
    return pd.Series(
        np.interp(
            x=target_reltime,
            xp=data[Label.REL_TIME].values,
            fp=data[ykey].values,
            left=np.nan,
            right=np.nan
        ),
        index=pd.Index(target_reltime, name=Label.TIME),
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
    if isinstance(data.index, pd.MultiIndex):
        gby = excluded(data.index, Label.TIME)
    else:
        gby = Label.TRIAL
    groups = data.groupby(gby)[Label.REL_TIME]
    rel_tbounds = (groups.min().max(), groups.max().min())
    logger.info(f'harmonizing trial time traces along common [{rel_tbounds[0]:.3f}, {rel_tbounds[1]:.3f}] s time vector')

    # Define common relative time vector for all trials
    min_trial_duration = rel_tbounds[1] - rel_tbounds[0]
    dt = data[Label.REL_TIME].diff().median().round(6)
    common_reltime = np.arange(0, min_trial_duration + dt / 2, dt) + rel_tbounds[0]

    # Defien harmonizer function
    def harmonizer(df):
        # Compute new temperature
        newtemp = (
            interpolate_trial_data(df, Label.TEMP, common_reltime)
            .rename(Label.TEMP)
        )

        # Cast as dataframe to avoid "smart-unstacking" by pandas
        newdf = newtemp.to_frame()

        # Check if there are other columns besides temperature and relative time 
        othercols = [k for k in df.columns if k not in [Label.TEMP, Label.TIME, Label.REL_TIME, *as_iterable(gby)]]

        # If so, verify that each other column contains a singleton 
        # before broadcasting it on new time index
        if len(othercols) > 0:
            assign_dict = {}
            for k in othercols:
                if df[k].nunique() != 1:
                    raise ValueError(f'column "{k}" contains multiple values for trial {df.name}')
                assign_dict[k] = df[k].iloc[0]
            newdf = newdf.assign(**assign_dict)

        # Return 
        return newdf

    # Interpolate all trials along the common relative time vector
    harmonized_data = (
        data
        .groupby(gby)
        .apply(harmonizer)
    )

    # Remove data preceding first trial onset
    if isinstance(harmonized_data.index, pd.MultiIndex):
        harmonized_data = harmonized_data[harmonized_data.index.get_level_values(Label.TRIAL) >= 0]
    else:
        harmonized_data = harmonized_data.loc[0:]

    # If only 1 output columns, return series
    if len(harmonized_data.columns) == 1:
        return harmonized_data[harmonized_data.columns[0]]
    # Otherwise, return dataframe
    else:
        return harmonized_data


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
    logger.info(f'extracting stim onset times from {burst_ids.max() + 1} identified trigger bursts')
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
    logger.info(f'assigning trial indices in the temperature recording')
    trial_start_times = stim_onset_times - STIM_ONSET
    itrial = trial_start_times.searchsorted(recording[Label.TIME]) - 1
    recording[Label.TRIAL] = itrial 

    # Compute relative time of each sample with respect to the stim onset for each trial
    logger.info('computing relative time of each sample with respect to the stim onset for each trial')
    reltime = recording[Label.TIME] - trial_start_times[itrial] - (STIM_ONSET + PROG_TRIGGER_DELAY)
    reltime[itrial < 0] = np.nan  # samples before first trial have no relative time
    reltime[reltime > min_interval] = np.nan  # samples after last trial have no relative time
    recording[Label.REL_TIME] = reltime

    # Harmonize relative time across trials and add trial information
    logger.info('harmonizing relative time across trials')
    harmonized_recording = get_time_harmonized_data(recording)

    # Add trial information from the experiment log to the harmonized recording
    logger.info('adding trial information to recording')
    harmonized_recording = (
        harmonized_recording
        .to_frame()
        .join(log_data)
    )
    
    # Return the harmonized recording with trial information
    return harmonized_recording


def load_and_process_thermal_experiment(folder, allow_recursive=True, **kwargs):
    '''
    Load and process thermal experiment data from a given folder. If the folder
    does not contain an info.rhd file, the function will recursively search for 
    subfolders containing thermal experiment data, with max depth=1.

    :param folder: path (or list of paths) to the thermal experiment data folder(s)
    :param allow_recursive: if True, recursively search for subfolders containing thermal experiment data
    :return: trial and time indexed experiment data with log information
    '''
    # If folder is a list of paths, call the function recursively on each path and concatenate the results
    if isinstance(folder, list):
        data = {}
        for f in folder:
            specimen = os.path.basename(f)
            logger.info(f'loading data for specimen {specimen}'.center(100, '-'))
            data[specimen] = load_and_process_thermal_experiment(
                f, allow_recursive=allow_recursive, prefix=specimen, **kwargs)
        data = pd.concat(data, names=[Label.SPECIMEN])
        data[Label.REL_TIME] = data.index.get_level_values(Label.TIME)
        data = get_time_harmonized_data(data)
        return data

    # Look for the info.rhd file in the folder
    info_fpath = os.path.join(folder, 'info.rhd')

    # If it does not exist
    if not os.path.exists(info_fpath):
        # If no recursive search is allowed, raise an error
        if not allow_recursive:
            raise FileNotFoundError(f'info.rhd file not found in "{folder}" and recursive search is disabled')

        # Try to call function recursively on all subfolders
        data = {}
        subitems = natsorted(os.listdir(folder))
        for item in subitems:
            sub_path = os.path.join(folder, item)
            if os.path.isdir(sub_path):
                data[item] = load_and_process_thermal_experiment(
                    sub_path, allow_recursive=False, **kwargs)

        # If no subfolders contain thermal experiment data, raise an error
        if not data:
            raise FileNotFoundError(f'No thermal experiment data found in "{folder}" or its subfolders')

        # Concatenate all subfolder data into a single DataFrame, and return
        return pd.concat(data, names=[Label.CONDITION])
    
    # Look for processed data file in the folder
    log_fcode = os.path.splitext(find_log_file(folder))[0]
    process_data_fname = log_fcode.replace('_log_', '_processed_data_') + '.h5'
    process_data_fpath = os.path.join(folder, process_data_fname)

    # If processed data file exists, load it
    if os.path.exists(process_data_fpath):
        logger.info(f'loading processed data from "{process_data_fname}"')
        data = pd.read_hdf(process_data_fpath, key='data')
    
    # Otherwise
    else:
        # Load the thermal experiment data
        recording, edge_times, log_data = load_thermal_experiment(folder, **kwargs)

        # Assign trials to the recording based on the edge times and log data
        data = assign_trials(recording, edge_times, log_data)

        # Save processed data in dedicated file
        logger.info(f'saving processed data to "{process_data_fname}"')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=NaturalNameWarning)
            data.to_hdf(process_data_fpath, key='data', mode='w')

    # If data contains "z (mm)" scanning column, add sensor tip offset to it
    if Label.Z_MM in data.columns:
        logger.info(f'adding sensor tip offset of {OSENSA_SENSOR_TIP_OFFSET*1e3:.1f} mm to "{Label.Z_MM}" scanning column')
        data[Label.Z_MM] += OSENSA_SENSOR_TIP_OFFSET * 1e3  # mm

    # Return the harmonized recording with trial information
    return data


def detrend_trial(y):
    '''
    Detrend a single trial of temperature data using linear detrending.

    :param y: pandas Series with trial temperature trace, indexed by relative time
    :return: pandas Series with detrended trial temperature trace, indexed by relative time
    '''
    # Identify baseline mask
    t = y.index.get_level_values(Label.TIME)
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
    t = y.index.get_level_values(Label.TIME)
    baseline_mask = np.logical_and(t > -baseline_pre, t < 0)
    y0 = y[baseline_mask].mean()

    # Return relative temperature change
    return y - y0


def condition_keys(data):
    ''' Extract condition keys present in the data '''
    candidate_keys = list(data.columns) + list(data.index.names)
    return [k for k in COND_KEYS if k in candidate_keys]


def compute_trial_average(data):
    '''
    Aggregate temperature data over trials for each condition 
    
    :param data: pandas DataFrame with trial time traces,
        indexed by trial and relative time
    :return: pandas DataFrame with trial-averaged data for each condition, 
        indexed by condition(s) and relative time
    '''
    # Identify over which to average trial data
    cond_keys = condition_keys(data)

    # Identify columns to average
    aggkeys = list(set(data.columns) - set(NONAGG_KEYS + cond_keys))

    # Aggregate the data by averaging over trials for each condition
    logger.info(f'computing trial-average data by {cond_keys}')
    return (
        data
        .groupby([*cond_keys, Label.TIME])
        [aggkeys]
        .mean()
    )


def plot_evoked_thermal_response(ax=None, data=None, y=None, hue=None, **kwargs):
    '''
    Plot the evoked thermal response for a given ykey (e.g., relative temperature change)
    as a function of relative time.

    :param ax: matplotlib Axes object to plot on. If None, a new figure and axes are created.
    :param data: pandas DataFrame with trial time traces, indexed by trial and relative time
    :param y: column name of the y-axis variable to plot (e.g., relative temperature change)
    :param hue: column name of the variable to use for color-coding different conditions
    :param kwargs: additional keyword arguments to pass to sns.lineplot
    '''
    # Set the axes and despine    
    if ax is None:
        ax = plt.gca()
    kwargs['ax'] = ax
    sns.despine(ax=ax)

    # Determine the hue normalization range based on the hue variable
    hue_norm = {
        Label.P: P_RANGE,
        Label.ISPTA: ISPTA_RANGE,
    }.get(hue, None)

    # Plot the evoked thermal response using seaborn lineplot
    sns.lineplot(
        data=data,
        x=Label.TIME,
        y=y,
        hue=hue,
        hue_norm=hue_norm,
        **kwargs
    )

    # Add horizontal line at y=0 for reference
    ax.axhline(0, color='k', linestyle='--', linewidth=1)

    # If stim duration is available in the data, add a shaded region to indicate the stimulus period
    if Label.DUR in data.columns:
        tstim = data[Label.DUR].iloc[0]
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
    t = y.index.get_level_values(Label.TIME).values

    # Initialize output series
    sopt = pd.Series(
        index=pd.Index([
            't_0 (s)',
            'peak (°C)',
            f'{Label.TAU}_rise (s)',
            f'{Label.TAU}_decay (s)',
            'offset (°C)'
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
        0.1,  # onset time (s)
        ypeak,  # peak amplitude (°C)
        0.1,  # rise time constant (s)
        0.3,  # decay time constant (s)
        0.0,   # baseline offset (°C)
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
    return pd.Series(data=popt, index=sopt.index)


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
    cond_data = data.loc[cond_stats.index.get_level_values(Label.TRIAL)]
    return cond_data, cond_stats 


def plot_XZ_slice(M, ax=None, title=None, vcontour=None, **kwargs):
    '''
    Plot XZ slice of specific key 

    :param M: dataframe representing (X, Z) matrix of output metric 
    :param ax: axis object. If none, a new figure is created
    :param title: optional axis title
    :param vcontour: optional value at which to draw contours 
    :return: figure object and quad mesh
    '''
    # Create/retrieve dfigure and axis
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # Set up axis
    ax.set_aspect('equal')
    ax.set_xlabel(Label.X_MM)
    ax.set_ylabel(Label.Z_MM)

    # Add title if specified
    if title is not None:
        ax.set_title(title)

    # Extract X and Z coordinates, and construct associated edges 
    x, z = M.columns.values, M.index.values
    xedges = (x[:-1] + x[1:]) / 2
    xedges = np.array([2 * x[0] - xedges[0], *xedges, 2 * x[-1] - xedges[-1]])
    zedges = (z[:-1] + z[1:]) / 2
    zedges = np.array([2 * z[0] - zedges[0], *zedges, 2 * z[-1] - zedges[-1]])

    # Plot heatmap
    sm = ax.pcolormesh(xedges, zedges, M, **kwargs)

    # If specified, add focus contours
    if vcontour is not None:
        ax.contour(x, z, M, levels=[vcontour], colors='w', linewidths=2)

    # Return
    return fig, sm


def plot_ispta_dependence(data, max_ΔT, ax=None, title=None, Itarget=None):
    ''' 
    Plot ispta dependence of stim-evoked temperature change and its peak

    :param data: pandas DataFrame with trial time traces, indexed by trial and relative time
    :param max_ΔT: pandas Series with max-evoked temperature change for each trial
    :param Itarget: target ISPTA for which to compute the predicted max-evoked temperature change from the linear fit
    :return: matplotlib Figure object containing the plot, and optionally the predicted max-evoked temperature change at target
    '''
    # Create/retrieve axis and figure
    add_dt_text = False
    if ax is None:
        fig, ax = plt.subplots()
        add_dt_text = True
    else:
        fig = ax.get_figure()
    
    # Add title, if specified
    if title is not None:
        ax.set_title(title)

    # Plot ISPTA-dependent temperature traces
    logger.info(f'plotting ISPTA-dependent evoked temperature change traces')
    sns.despine(ax=ax)
    for units in [Label.TRIAL, None]:
        plot_evoked_thermal_response(
            ax=ax,
            data=data.reset_index(),
            y=Label.REL_TEMP,
            hue=Label.ISPTA,
            palette='flare',
            estimator=None if units is not None else 'mean',
            errorbar=None,
            units=units,
            alpha=0.2 if units is not None else 1.0,
            lw=.5 if units is not None else 2.0,
            legend='full' if units is None else False
        )
    sns.move_legend(ax, 'upper right', title=Label.ISPTA, frameon=False)

    # Perform least-square linear fit to the ISPTA-dependence of max_ΔT
    logger.info('performing least-square linear fit on max_ΔT vs ISPTA')
    ISPTA_vals = np.array(max_ΔT.index.get_level_values(Label.ISPTA))

    # Solve y = m * x without an intercept
    m, residuals, *_ = np.linalg.lstsq(
        ISPTA_vals[:, np.newaxis], max_ΔT.values, rcond=None)
    slope, ss_res = m[0], residuals[0]

    # Compute R2
    ss_tot = np.sum((max_ΔT - np.mean(max_ΔT)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    logger.info(f'least-square linear fit: slope = {slope:.3f} °C/(W/cm²), R² = {r2:.2f}')

    # Add inset axis
    inset_ax = ax.inset_axes([0.60, 0.25, 0.25, 0.3])

    # Plot ISPTA-dependence of max-evoked temperature change at focus
    logger.info(f'plotting ISPTA-dependence of max-evoked temperature change at focus')
    sns.despine(ax=inset_ax)
    sns.lineplot(
        ax=inset_ax,
        data=max_ΔT.reset_index(),
        x=Label.ISPTA,
        y=Label.MAX_REL_TEMP,
        errorbar='sd',
        ls='',
        err_style='bars',
        err_kws={'elinewidth': 1.5, 'capsize': 10},
    )

    # Add linear fit to the ISPTA-dependence of max-evoked temperature change at focus
    inset_ax.axline(slope=slope, xy1=(0, 0), color='k', lw=1)
    inset_ax.set_title(f'∝ISPTA (R² = {r2:.3f})')

    # If Itarget is specified, compute and plot the predicted max-evoked temperature change at that value
    if Itarget is not None:
        max_ΔT_pred = slope * Itarget
        logger.info(f'predicted max-evoked temperature change at {Itarget} W/cm²: {max_ΔT_pred:.3f} °C')
        inset_ax.plot([Itarget] * 2, [0, max_ΔT_pred], color='k', ls=':', lw=1.5)
        inset_ax.plot([0, Itarget], [max_ΔT_pred] * 2, color='k', ls=':', lw=1.5)
        if add_dt_text:
            inset_ax.text(
                0., .95 * max_ΔT_pred, f'ΔT = {max_ΔT_pred:.2f} °C',
                ha='left', va='top', fontsize=8
            )

    # Return output(s)
    if Itarget is not None:
        return fig, max_ΔT_pred
    else:
        return fig


def plot_max_deltaT_per_location(max_ΔT, title):
    '''
    Plot bar graph of max ΔT across multiple dimensions

    :param max_ΔT: 2D dataframe of maximal temprature elevation
    :return: figure object
    '''
    # Extract x and hue keys
    xkey = max_ΔT.index.name
    hue = max_ΔT.columns.name
    
    # Plot bar graph of max ΔT
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_title(title)
    sns.despine(ax=ax)
    sns.barplot(
        ax=ax,
        data=max_ΔT.stack().rename(Label.MAX_REL_TEMP).reset_index(),
        y=Label.MAX_REL_TEMP,
        x=Label.LOCATION,
        hue=Label.SPECIMEN,
    )

    # Add mean +/- std text for each x value
    for iloc, xval in enumerate(max_ΔT.index):
        xdata = max_ΔT.loc[xval]
        text = f'{xdata.mean():.2f} ± {xdata.std():.2f}'
        ax.text(
            x=iloc,
            y=xdata.max() + 0.02,
            s=text,
            ha='center',
            va='bottom',
            fontsize=10,
            color='k'
        )
        ylims = ax.get_ylim()
        ax.set_ylim(0, 1.05 * ylims[1])

    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Move legend
    sns.move_legend(ax, 'center left', bbox_to_anchor=(1, 0.5), frameon=False)

    # Return figure
    return fig
