# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-08-15 09:29:37
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-06-03 10:48:12

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import spectrogram
from scipy.signal.windows import tukey

from .constants import S_TO_MS, HZ_TO_KHZ
from .utils import is_within
from .logger import logger
from .plotters import data_to_axis


def get_pulse_envelope(n, xramp=0, xprepad=0, xpostpad=0, rms_norm=True, nreps=1):
    '''
    Get a pulse envelope of specified size with a specific ramp-up/down fraction 
    and optional pre- and post-pulse zero padding.
    
    :param n: number of points in pulse window (whtout padding)
    :param xramp: fraction of window used for ramp-up. Defaults to zero (rectangular pulse)
    :param xprepad: relative size of pre-pulse zero padding, w.r.t pulse window. Defaults to zero.
    :param xpostpad: relative size of post-pulse zero padding, w.r.t pulse window. Defaults to zero.
    :param rms_norm: flag indicating whether to normalize the envelope to unit RMS amplitude (default = True)
    :param nreps: number of repetitions of the pulse envelope (default = 1)
    :return: pulse(s) envelope vector (potentially normalized to unit RMS amplitude)
    '''
    # Check that ramp fraction is in [0, 1]
    if xramp < 0 or xramp > 1.:
        raise ValueError(f'xramp must be in [0, 1], got {xramp}')
    
    # Generate pulse envelope with Tukey window with appropriate taper
    w = tukey(n, alpha=xramp)

    # If specified, compensate amplitude to achieve equal RMS amplitude regardless of ramp time
    if rms_norm:
        wrms = np.sqrt(np.mean(w**2))  # RMS amplitude of the pulse envelope
        w = w / wrms  # normalize to unit RMS amplitude

    # Add pre- and post-pulse padding, if any
    if xprepad > 0:
        npre = int(np.round(n * xprepad))
        w = np.hstack([np.zeros(npre), w])
    if xpostpad > 0:
        npost = int(np.round(n * xpostpad))
        w = np.hstack([w, np.zeros(npost)])

    # Repeat pulse envelope if needed
    if nreps > 1:
        w = np.tile(w, nreps)

    # Return envelope vector
    return w


def get_pulse_train_envelope(dt, dur, PRF=None, DC=100., tramp=0, tprepad=None, tpostpad=None, **kwargs):
    '''
    Construct pulse train envelope from stimulation parameters

    :param dt: time step (s) for envelope resolution
    :param dur: total duration of the pulse train (s)
    :param PRF (optional): pulse repetition frequency (Hz). Must be specified if DC < 100%.
    :param DC: duty cycle (%). Defaults to 100% (continuous wave).
    :param tramp: ramp-up time (s) for each pulse envelope
    :param tprepad: pre-train zero padding (s). If None, set to 5% of pulse train duration or 20 ms, whichever is greater
    :param tpostpad: post-train zero padding (s). If None, set to 5% of pulse train duration or 20 ms, whichever is greater
    :param kwargs: additional arguments for get_pulse_envelope function
    :return: time and envelope vectors
    '''
    # print(dt, dur, PRF, DC, tramp, tprepad, tpostpad, kwargs)
    # Determine overall number of points
    nenv = int(np.round(dur / dt)) + 1

    # If DC = 0, construct zero envelope
    if DC == 0:
        y = np.zeros(nenv)
    
    # If DC is non-zero
    else:
        # If DC is 100% (CW mode), set nperpulse = n
        if DC == 100:
            npulses = 1
            nperpulse = nenv
            ton = dur  # nominal pulse duration (s)
        
        # If DC < 100% (PW mode), set npulse and nperpulse
        else:
            if PRF is None:
                raise ValueError('PRF must be specified for pulsed waveforms (DC < 100%)')
            PRI = 1 / PRF  # pulse repetition interval (s)
            if PRI > dur:
                raise ValueError(f'waveform duration ({dur * S_TO_MS:.2f} ms) is shorter than pulse repetition interval ({PRI * S_TO_MS:.2f} ms)')
            npulses = int(np.round(dur * PRF))  # number of pulses in the burst
            nperpulse = int(np.round(nenv / npulses * DC * 1e-2))  # number of points per pulse (with DC)
            ton = (DC * 1e-2) / PRF  # nominal pulse duration (s)

        # Compute relative ramp time and post-padding fraction
        xramp = 2 * tramp / ton  # ramp fraction
        if xramp > 1:
            raise ValueError(f'ramp time ({tramp * S_TO_MS:.2f} ms) is > 50% of pulse duration ({ton * S_TO_MS:.2f} ms)')
        xpostpad = 1 / (DC * 1e-2) - 1  # offset fraction

        # Get pulse envelope
        y = get_pulse_envelope(nperpulse, xramp=xramp, xpostpad=xpostpad, nreps=npulses, **kwargs)

    # Compute effective time step post-construction, and check against requested time step
    dteff = dur / (y.size - 1)
    if not np.isclose(dteff, dt):
        logger.warning(f'effective time step ({dteff:.2e} s) does not match requested time step ({dt:.2e} s).')

    # Add pre- and post-train padding, if any
    if tprepad is None:
        tprepad = max(0.05 * dur, 0.02)  # s
    if tpostpad is None:
        tpostpad = max(0.05 * dur, 0.02)  # s
    npre = max(int(np.round(tprepad / dt)) - 1, 0)
    npost = max(int(np.round(tpostpad / dt)) - 1, 0)
    y = np.pad(y, (npre, npost), mode='constant', constant_values=0)

    # Construct time vector
    t = np.arange(y.size) * dteff  # s
    if tprepad > 0:
        t -= tprepad
    
    # Return envelope vector
    return t, y


def get_waveform(Fdrive, A, *args, npc=25, **kwargs):
    '''
    Get full waveform from waveform prameters

    :param Fdrive: carrier frequency (Hz)
    :param A: waveform amplitude
    :param args: additional arguments for get_pulse_train_envelope function
    :param npc: number of points per cycle (default = 25)
    :param kwargs: additional arguments for get_pulse_envelope function
    :return: time, waveform and waveform envelope vectors
    '''
    # Set target time step
    dt_target = 1 / (Fdrive * npc)  # s

    # Construct pulse train envelope and time vectors
    t, yenv = get_pulse_train_envelope(dt_target, *args, **kwargs)
    yenv *= A  # scale envelope by amplitude
    
    # Construct carrier vector
    ycarrier = np.sin(2 * np.pi * Fdrive * t)  # sine carrier

    # Multiply envelope by carrier to get waveform
    y = yenv * ycarrier  # waveform vector

    # Return time, waveform and envelope vectors
    return t, y, yenv


def plot_waveform(*args, ax=None, color=None, alpha=0.7, label=None,
                  ylabel=None, title=None, legend=True, zoom_tbounds=None,
                  inset_ax=None):
    '''
    Plot waveform on axis

    :param args: either a 3-tuple of (t, y, yenv) or a dictionary of waveforms
    :param ax: axis handle (default = None, creates new axis)
    :param color: color for the waveform and envelope (default = cyclic)
    :param alpha: transparency level for the waveform (default = 0.7)
    :param label: common label for the waveform and envelope (optional). If None, 
        generic "waveform" and "envelope" labels will be used.
    :param ylabel: y-axis label (optional, default = None)
    :param title: title for the plot (optional, default = None)
    :param legend: flag indicating whether to show legend (default = True)
    :param zoom_tbounds: time bounds for zoom-in on waveform (optional)
    :param inset_ax: axis handle for zoom-in axis (optional, default = None)
    :return: figure handle
    '''
    # Create/retrieve figure and axis
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
    else:
        fig = ax.get_figure()
    
    # Despine axis
    sns.despine(ax=ax)

    # Add x and y labels, and title if specified
    ax.set_xlabel('time (ms)')
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    
    # If input is a dictionary, call function recursively for each waveform
    if len(args) == 1 and isinstance(args[0], dict):
        for key, val in args[0].items():
            fig, ax, inset_ax = plot_waveform(
                *val, ax=ax, color=color, alpha=alpha, label=key, 
                ylabel=ylabel, title=title, zoom_tbounds=zoom_tbounds, 
                inset_ax=inset_ax
            )
        return fig, ax, inset_ax

    # Unpack arguments
    if len(args) != 3:
        raise ValueError(f'Expected 3 arguments (t, y, yenv), got {len(args)}')
    t, y, yenv = args

    # Plot waveform with specific color and transparency
    lh = ax.plot(t * S_TO_MS, y, label='waveform' if label is None else None, c=color, alpha=alpha)[0]
    color = lh.get_color()

    # If zoom time bounds are specified, create inset axis
    if inset_ax is None and zoom_tbounds is not None:
        inset_ax = add_zoomin_axis(ax, zoom_tbounds)

    # Plot waveform envelope (lower and upper bounds) with same color
    ax.plot(t * S_TO_MS, yenv, label='envelope' if label is None else label, c=color)
    ax.plot(t * S_TO_MS, -yenv, c=color)

    # If inset time bounds specified, plot restricted time and waveform vectors
    # on inset axis
    if zoom_tbounds is not None:
        inset_mask = is_within(t, zoom_tbounds)
        t, y, yenv = t[inset_mask], y[inset_mask], yenv[inset_mask]    
        inset_ax.plot(t * S_TO_MS, y, c=color, alpha=alpha)

    # Set y-axis limits to the envelope bounds
    if legend:
        ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))

    # Return figure handle
    return fig, ax, inset_ax


def add_zoomin_axis(ax, tbounds, xspan=0.12):
    '''
    Add zoom-in axis to the given axis with specified time bounds
    :param ax: axis handle
    :param tbounds: time bounds for the zoom-in axis (tuple of start and stop times in seconds)
    :param xspan: relative width of the zoom-in axis w.r.t. parent axis (default = 0.12)
    :return: inset axis
    '''
    # Draw figure canvas to ensure axis transformation is up-to-date
    ax.get_figure().canvas.draw()
    # Get the xmid position in data coordinates
    inset_xmid = data_to_axis(ax, (np.mean(tbounds) * S_TO_MS, 0))[0]
    # Create inset axis with specified xspan
    inset_ax =  ax.inset_axes([inset_xmid - xspan / 2, -0.05, xspan, 1.1])
    # Set inset axis limits to the specified time bounds
    inset_ax.set_xlim(tbounds[0] * S_TO_MS, tbounds[1] * S_TO_MS)
    # Remove x and y ticks from the inset axis 
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    # Return the inset axis
    return inset_ax


def get_spectrogram(t, y, Δt=1e-3, overlap=0.5, window='tukey', taper=.25, Δf=None):
    '''
    Compute waveform spectrogram with specific temporal and frequency granularities
    
    :param t: time vector of the signal
    :param y: waveform vector of the signal
    :param Δt: size of Fourier window in time domain (s)
    :param overlap: relative amount of overlap between consecutive time segments
    :param window: type of window to use for Fourier transform ('tukey', 'hann', 'hamming')
    :param taper: taper fraction for Tukey windows (default = 0.25, 0 = rectangular window)
    :param Δf (optional): frequency bin interval (Hz). If None, set to 1/Δt
    :return: frequency vector (Hz), time vector (s) and spectrogram matrix
    '''
    # If Δf is not specified, set it to 1 / Δt
    if Δf is None:
        Δf = 1 / Δt

    # Check that (Δf*Δt <= 1) condition is satisfied
    if Δf * Δt > 1:
        raise ValueError(f'Δf * Δt ({Δf * Δt:.2e}) must be <= 1 for spectrogram computation')
    
    # Extract relevant parameters
    dt = t[1] - t[0]  # time step (s)
    fs = 1 / dt  # sampling frequency (Hz)
    nfft = int(np.round(fs / Δf))  # number of FFT points per segment
    nperseg = int(np.round(Δt / dt))  # number of points per Fourier segment
    noverlap = int(np.round(overlap * nperseg))  # number of overlap points

    # Construct window input
    if window not in ['tukey', 'hann', 'hamming']:
        raise ValueError(f'Unsupported window type: {window}. Supported types are: tukey, hann, hamming.')
    if window == 'tukey':
        window = (window, taper)

    # Compute spectrogram
    freqs, times, S = spectrogram(
        y, 
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        scaling='spectrum',
        mode='magnitude'
    )

    # Offset spectrogram time vector to match original time vector
    times += t[0]  # s
    
    # Return
    return freqs, times, S


def get_power_spectrum(t, y, norm=False, dB=False):
    '''
    Get the power spectrum of a waveform.

    :param t: time vector (s)
    :param y: waveform vector
    :param norm: flag indicating whether to normalize the power spectrum (default = False)
    :param dB: flag indicating whether to convert power to dB (default = False)
    :return: frequency (Hz) and power vectors
    '''
    dt = np.diff(t)[0]  # s
    freqs = np.fft.rfftfreq(y.size, dt)  # Hz
    ps = np.abs(np.fft.rfft(y))**2  # power
    if norm:
        ps /= ps.max()
    if dB:
        ps = 10 * np.log10(ps)
    return freqs, ps


def plot_spectrogram(t, f, S, ax=None, title=None, zunit=None, cmap='viridis'):
    '''
    Plot spectrogram output as a surface plot
    
    :param t: spectrogram time vector (s)
    :param f: spectrogram feequency vector (Hz)
    :param S: spectrogram values matrix
    :param ax: axis handle (default = None)
    :param title: title for the plot (default = None)
    :param zunit: unit of spectrogram values (default = None)
    :param cmap: colormap for the surface plot (default = 'viridis')
    :return: figure handle
    '''
    if ax is None:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()
    sns.despine(ax=ax)
    ax.view_init(azim=-45, elev=30)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('frequency (kHz)')
    zlabel = 'magnitude'
    if zunit is not None:
        zlabel = f'{zlabel} ({zunit})'
        ax.set_zlabel(zlabel)
    T, F = np.meshgrid(t, f)
    ax.plot_surface(T * S_TO_MS, F * HZ_TO_KHZ, S, cmap=cmap)
    return fig
