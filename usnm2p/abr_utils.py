# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2024-08-22 12:16:31
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2024-08-22 15:53:15

# External packages
import pyabf
import numpy as np
import pandas as pd
from scipy.signal import hilbert

# Internal modules
from .logger import logger
from .constants import Label


def load_abf_file(fpath):
    '''
    Load an ABF file.

    :param fpath: path to ABF file
    :return: multi-indexed pandas DataFrame with time and voltage vectors for each sweep
    '''
    # Load file
    if not fpath.endswith('.abf'):
        raise ValueError(f'{fpath} is not an ABF file')
    logger.info(f'loading {fpath} ABF file')
    abf = pyabf.ABF(fpath)

    # Extract metadata
    nchannels = abf.channelCount
    nsweeps = abf.sweepCount
    npersweep = abf.sweepPointCount
    rec_size = nsweeps * npersweep
    sr = abf.sampleRate  # Hz
    tunit = abf.sweepUnitsX.replace('sec', 's')
    yunit = abf.sweepUnitsY

    # Get elapsed time vector
    tfull = abf.getAllXs()
    assert tfull.shape[0] == rec_size, f'time vector shape {tfull.shape} does not match expected number of sweeps x samples ({rec_size})'
    dt = tfull[1] - tfull[0]
    assert int(np.round(1 / dt)) == sr, f'expected sample rate {sr} does not match actual sample rate {1 / (dt)}'

    # Get data matrix
    y = abf.data
    assert y.shape[0] == nchannels, f'data shape {y.shape} does not match expected number of channels ({nchannels})'
    assert y.shape[1] == rec_size, f'data shape {y.shape} does not match expected number of sweeps x samples ({rec_size})'

    # Generate multi-index from number of sweeps and points per sweep
    mux = pd.MultiIndex.from_product([range(nsweeps), range(npersweep)], names=['sweep', 'sample'])

    # Generate time vector
    t = mux.get_level_values(level='sample') / sr  # s

    # Generate DataFrame filled with time and channel data
    df = pd.DataFrame(data={
        Label.TIME: t,
        f'elapsed time ({tunit})': tfull
    }, index=mux)
    for i, yy in enumerate(y):
        df[f'y{i} ({yunit})'] = yy

    # Return dataframe
    return df


def load_abr_data(fpath, yunit='mV'):
    '''
    Load ABR data from an ABF file
    
    :param fpath: path to ABF file
    param yunit: output unit for ABR data
    :return: pandas DataFrame with time, stimulus and ABR data
    '''
    # Load ABF data as pandas DataFrame
    data = load_abf_file(fpath)

    # Remove 'elapsed time' column
    del data['elapsed time (s)']

    # Rename data columns appropriately
    mapper = {
        'y0 (V)': f'stim ({yunit})',
        'y1 (V)': f'ABR ({yunit})'
    }
    data.rename(columns=mapper, inplace=True)

    # Rescale ABR data if necessary
    if yunit == 'mV':
        for k in mapper.values():
            data[k] *= 1e3

    # Return DataFrame
    return data


def extract_stim_frequency(y, dt, roundto=10):
    '''
    Extract stimulus frequency from stimulus waveform
    
    :param y: stimulus waveform vector
    :param dt: time step (s)
    :param roundto: rounding frequency unit (Hz)
    :return: stimulus frequency (Hz)
    '''
    # Compute FFT
    f = np.fft.rfftfreq(y.size, d=dt)
    Y = np.fft.rfft(y)
    
    # Extract frequency with highest power amplitude
    fcarrier = f[np.argmax(np.abs(Y))]

    # If rounding unit specified, round to nearest unit
    if roundto is not None:
        fcarrier = np.round(fcarrier / roundto) * roundto

    # Return result
    return fcarrier


def extract_stim_window(y, rel_thr=0.05):
    '''
    Extract stimulus window from stimulus waveform
    
    :param y: stimulus waveform vector
    :param rel_thr: relative amplitude threshold for stimulus window detection
    :return: tuple of stimulus window boundaries indexes
    '''
    # Extract waveform envelope
    yenv = np.abs(hilbert(y))

    # Compute time at which envelope rises above and drops below 
    # defined fraction of its maximum
    ythr = rel_thr * yenv.max()
    idx = np.where(yenv > ythr)[0]
    ibounds = (idx[0], idx[-1])

    # Return
    return ibounds