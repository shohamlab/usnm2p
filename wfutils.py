# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-08-15 09:29:37
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-08-07 14:50:28

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import welch
from utils import si_prefixes


def get_pulse_envelope(n, PD, tonset=0, toffset=0, nreps=1):
    '''
    Get a smoothed rectangular pulse envelope with a specific ramp-up (and down) time,
    plateau duration and base duration.
    
    :param n: number of points in envelope vector
    :param PD: pulse duration (s)
    :param tonset: pulse onset time (s)
    :param toffset: OFF time following pulse (s)
    :return: time (s) and envelope vectors
    '''
    # Compute characteristic times and total duration 
    ttot = tonset + PD + toffset  # s

    # Generate time and envelope vectors
    t = np.linspace(0, ttot, n)  # s
    y = np.zeros(n)
    y[np.logical_and(t > tonset, t <= tonset + PD)] = 1

    # Repeat vectors if needed
    if nreps > 1:
        texp = t.copy()
        yexp = y.copy()
        for _ in range(2, nreps + 1):
            texp = np.hstack([texp, texp[-1] + t[1:]])
            yexp = np.hstack([yexp, y[1:]])
        t = texp
        y = yexp

    # Return time and envelope vectors
    return t, y


def get_full_waveform(t, yenv, Fdrive, npc=25):
    ''' 
    Get a full waveform from a time vector, an envelope vector and a carrier frequency.

    :param t: time vector (s)
    :param yenv: envelope vector
    :param Fdrive: carrier frequency (Hz)
    :param npc: number of points per cycle (default = 25)
    :return: dense time, waveform and envelope vectors
    '''
    # Determine sampling frequency and number of points in full waveform
    fs = Fdrive * npc  # Hz
    npts = int(np.round(t[-1] * fs))
    # Generate dense time vector 
    tdense = np.linspace(0, t[-1], npts)  # s
    # Interpolate envelope vector
    yenvdense = np.interp(tdense, t, yenv)
    # Generate carrier vector
    ycarrier = np.sin(2 * np.pi * Fdrive * tdense)
    # Generate waveform vector
    ydense = yenvdense * ycarrier
    # Return dense time, waveform and envelope vectors
    return tdense, ydense, yenvdense


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


def plot_waveform_trace(tenv, yenv, Fdrive, unit='ms', ax=None, label=None, **kwargs):
    ''' 
    Plot waveform trace from a time vector, an envelope vector and a carrier frequency.

    :param t: time vector (s)
    :param yenv: envelope vector
    :param Fdrive: carrier frequency (Hz)
    :param unit: time unit for plotting (default = 'ms')
    :param ax: axis handle (default = None)
    :param title: title for the plot (default = None)
    :param label: label for the plotted waveform (default = None)
    :param mark_regs: flag indicating whether to mark identified regions (default = True)
    :return: figure handle
    '''
    # Get figure and axis handles
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    sns.despine(ax=ax)

    # Get dense time, waveform and envelope vectors
    tdense, ydense, yenvdense = get_full_waveform(tenv, yenv, Fdrive, **kwargs)
    tfactor = si_prefixes[unit[:-1]]
    tdense /= tfactor

    # Plot waveform and its envelope
    lh, *_ = ax.plot(tdense, yenvdense, label=label)
    ax.plot(tdense, ydense, c=lh.get_color(), alpha=0.5, label='waveform' if label is None else None)
    ax.set_xlabel(f'time ({unit})')
    ax.set_ylabel('amplitude')
    ax.axhline(0, c='k', ls='--')
    if label is not None:
        ax.legend()

    # Return figure handle
    return fig


def plot_waveform_spectrum(t, y, norm=False, dB=True, ax=None, label=None, title=None, color=None):
    ''' 
    Plot a smoothed waveform from a time vector, an envelope vector and a carrier frequency.

    :param t: time vector (s)
    :param yenv: envelope vector
    :param ax: axis handle (default = None)
    :param label: label for the plotted spectrum (default = None)
    :param title: title for the plot (default = None)
    :return: figure handle
    '''
    # Get figure and axis handles
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    sns.despine(ax=ax)
    
    # Add title
    ax.set_title('waveform spectrum' if title is None else title)

    # Extract and waveform frequency spectrum
    freqs, ps = get_power_spectrum(t, y, norm=norm, dB=dB)
    ax.plot(freqs, ps, label=label, c=color)
    ax.set_xlabel('frequency (Hz)')
    ax.set_xscale('log')
    ylabel = 'power'
    if dB:
        ylabel = f'{ylabel} (dB)'
    if norm:
        ylabel = f'normalized {ylabel}'
    ax.set_ylabel(ylabel)

    if label is not None:
        ax.legend()

    # Return figure handle
    return fig
