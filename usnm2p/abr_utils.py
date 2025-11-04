# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2024-08-22 12:16:31
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-10-30 13:12:59

''' Utilities for the anlysis of ABR data and the prediciton of ABR responses to US '''

# External packages
import re
import pyabf
import numpy as np
import pandas as pd
from scipy.signal import hilbert
import seaborn as sns

# Internal modules
from .logger import logger
from .constants import Label, Pattern, HZ_TO_KHZ, S_TO_MS
from .utils import is_within, dB_to_amplitude, resample
from .wfutils import get_waveform, get_spectrogram


# Regexp pattern for parsing filenames
ABR_pattern = re.compile(f'^{Pattern.DATE}_(pre|post)deafening_(left|right|x)ear_(\d+)kHz_(\d+|x)dB_(\d+)$')  


def parse_ABR_parameters(fcode):
    ''' 
    Parse ABR parameters from file code (file name without extension)
    
    :param fcode: file code
    :return: dictionary with parsed parameters
    '''
    # If input is a list, tuple, numpy array or pandas index, parse each element
    # and return a DataFrame with the results
    if isinstance(fcode, (list, tuple, np.ndarray, pd.Index)):
        return pd.concat({k: pd.Series(parse_ABR_parameters(k)) for k in fcode}, names=['dataset']).unstack()

    # Parse file code
    match = ABR_pattern.match(fcode)

    # Check if pattern matched
    if match is None:
        raise ValueError(f'file code {fcode} does not match ABR pattern')
    
    # Extract parameters and return
    year, month, day, condition, side, freq, level, acq = match.groups()
    return {
        'date': f'{year}-{month}-{day}',
        'condition': f'{condition}-deafening',
        'ear': side,
        'freq (kHz)': int(freq),
        'level (dB)': int(level) if level != 'x' else np.nan,
        'acq': int(acq)
    }


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


def compute_predicted_ABR(ABR_thresholds, impulse_ABR, *args, fbounds=(2e3, 90e3), norm_factor=None, nreps=1, **kwargs):
    '''
    Compute predicted ABR signal evoked by specific US waveform

    :param ABR_thresholds: frequency-indexed ABR threshold (dB SPL) as pandas Series
    :param impulse_ABR: time-indexed impulse ABR signal (uV) as pandas Series
    :param args: arguments for get_waveform function
    :param fbounds: frequency bounds for ABR signal (default = mouse hreading range [2 - 90 kHz])
    :param norm_factor: normalization factor for final ABR response signal (default = None) 
    :param nreps: number of repetitions (useful if randomness injected into input signal)
    :param kwargs: arguments for get_spectrogram function
    :return: time-indexed evoked ABR signal (uV) as pandas Series
    '''
    # If more than 1 rep, call function recursively and stack outputs in multi-indexed series 
    if nreps > 1:
        outputs = {}
        for irep in range(nreps):
            outputs[irep] = compute_predicted_ABR(
                ABR_thresholds, impulse_ABR, *args, fbounds=fbounds,
                norm_factor=norm_factor, nreps=1, **kwargs)
        return pd.concat(outputs, names=['repetition'], axis=0)

    # Construct US waveform
    t, P_t, _ = get_waveform(*args, **kwargs)

    # Compute waveform spectrogram
    freqs, times, I_ft = get_spectrogram(t, P_t)

    # Derive spectrogram over time and restrict to frequency range of interest
    dI_ft = np.abs(np.diff(I_ft) / np.diff(times))
    fmask = is_within(freqs, fbounds)
    freqs = freqs[fmask]  # Hz
    dI_ft = dI_ft[fmask]  # yunit * Hz

    # Extract hearing threshold over frequency range of interest
    h_dB = np.interp(
        freqs,  # Hz
        ABR_thresholds.index / HZ_TO_KHZ,  # Hz 
        ABR_thresholds.values   # dB SPL
    )
    h = dB_to_amplitude(h_dB)  # convert dB SPL to amplitude scale
    
    # Scale by ABR sensitivity (1 / threshold) over frequency range of interest
    weighted_dI_ft = (dI_ft.T / h).T

    # Sum over frequencies
    s_t = np.sum(weighted_dI_ft, axis=0)

    # Resample s_t to match higher impulse_ABR sampling rate
    dt_impulse_ms = impulse_ABR.index.values[1] - impulse_ABR.index.values[0]  # ms
    dt_impulse = dt_impulse_ms / S_TO_MS  # s
    tdense, s_t_dense = resample(times[:-1], s_t, dt_impulse)

    # Convolve with impulse ABR to get evoked ABR signal
    a_t = np.convolve(s_t_dense, impulse_ABR.values, mode='full')[:s_t_dense.size]

    # Normalize ABR signal if requested
    if norm_factor is not None:
        a_t /= norm_factor
        out_unit = 'a.u.'
    else:
        out_unit = 'uV'
    
    # Return as time-indexed pandas Series
    return pd.Series(
        a_t,  # uV
        index=pd.Index(tdense * S_TO_MS, name='time (ms)'),
        name=f'ABR ({out_unit})'
    )


def plot_ABR_vs_parameter(s, pname, punit, stim_tbounds=None, height=1, hue=None):
    '''
    Plot ABR response signal vs. input parameter
    
    :param s: (parameter, time)-indexed series of ABR response signal (a.u.)
    :param pname: name of the input parameter
    :param punit: unit of the input parameter
    :param stim_tbounds: optional tuple of (start, stop) of stimulus (ms)
    :return: figure handle
    '''
    # Extract parameter index name and values
    pkey = s.index.names[0]
    pvals = s.index.get_level_values(pkey).drop_duplicates().values

    # Plot evoked ABR for different input parameter values 
    logger.info(f'plotting evoked ABR vs. {pname}...')
    g = sns.relplot(
        data=s.reset_index(),
        kind='line',
        x='time (ms)',
        y='ABR (a.u.)',
        hue=hue,
        errorbar='se',
        row=pkey,
        row_order=pvals,
        height=height, aspect=5, 
    )
    fig = g.fig
    axes = g.axes.flatten()
    for ax, pval in zip(axes, pvals):
        ax.set_title('')
        ax.set_ylabel(pval, rotation=0, ha='right', va='center')
        ax.tick_params(left=False, labelleft=False)
        sns.despine(ax=ax, left=True)
    for ax in axes[:-1]:
        ax.tick_params(bottom=False)
        sns.despine(ax=ax, left=True, bottom=True)

    # Add unit vertical scale bar on right of top axis
    xlims = axes[0].get_xlim()
    xpos = xlims[1] - 0.05 * (xlims[1] - xlims[0])  # 5% from right edge
    ylims = axes[0].get_ylim()
    ycenter = (ylims[1] + ylims[0]) / 2
    axes[0].plot([xpos, xpos], [ycenter - .5, ycenter + .5], color='k', lw=2)
    axes[0].text(
        xpos + 0.01 * (xlims[1] - xlims[0]), ycenter, '1 a.u.', 
        rotation=90, va='center', ha='left', fontsize=10, color='k')

    # Materialize stimulus span, if time bounds are specified
    if stim_tbounds is not None:
        tstart, tstop = stim_tbounds
        for ax in axes:
            ax.axvspan(tstart, tstop, color='silver', alpha=0.5, lw=0)

    # Adjust layout and title
    fig.supylabel(f'{pname} ({punit})', x=0)
    fig.tight_layout()

    # Return figure handle
    return fig
