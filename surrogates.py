# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2024-06-26 13:47:43
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2024-07-02 19:36:31


''' Collection of utilities to generate surrogate data, adapted from `pyunicorn` package '''

import numpy as np

from logger import logger
from utils import pandas_proof


def FT(y):
    '''
    Return Fourier Transform surrogates.

    Generate surrogates by Fourier transforming the `y`
    time series (assumed to be real valued), randomizing the phases and
    then applying an inverse Fourier transform. Correlated noise surrogates
    share their power spectrum and autocorrelation function with the
    original_data time series.

    .. note::
    The amplitudes are not adjusted here, i.e., the
    individual amplitude distributions are not conserved!

    :param y: 1D array representing the original time series.
    :return: the surrogate time series.
    '''
    # Calculate FFT of original_data time series
    ysur = np.fft.rfft(y)

    # Generate random phases uniformly distributed in the interval [0, 2*Pi]
    phases = np.random.uniform(low=0, high=2 * np.pi, size=ysur.size)

    # Add random phases uniformly distributed in the interval [0, 2*Pi]
    ysur *= np.exp(1j * phases)

    # Calculate IFFT and take the real part, the remaining imaginary part
    # is due to numerical errors.
    return np.real(np.fft.irfft(ysur, n=y.size))
    

def align_ranks(y, yref):
    '''
    Align the ranks of a 1D array with a reference array.

    :param y: 1D array to be aligned.
    :param yref: 1D array to be aligned with.
    :return: the aligned 1D array.
    '''
    # Copy input signal
    ycopy = y.copy()   

    # Sort copy  
    ycopy.sort()

    # Compute ranks of reference signal
    ranks = yref.argsort().argsort()

    # Return copy sorted to have the same rank order as the original signal
    return ycopy[ranks]


def AAFT(y):
    '''
    Return surrogates using the amplitude adjusted Fourier transform
    method.

    Reference: [Schreiber2000]_

    :param y: 1D array representing the original time series.
    :return: the surrogate time series.
    '''
    # Create Gaussian reference series 
    gaussian = np.random.randn(y.size)
    
    # Sort gaussian to be rank ordered like the original data
    rescaled_gaussian = align_ranks(gaussian, y)

    # Phase randomize sorted Gaussian signal
    phase_randomized_gaussian = FT(rescaled_gaussian)

    # Rescale back to amplitude distribution of original data
    return align_ranks(y, phase_randomized_gaussian)


def iAAFT(y, nit=100, output='true_amplitudes'):
    '''
    Return surrogates using the iteratively refined amplitude adjusted
    Fourier transform method.

    A set of AAFT surrogates is iteratively refined to produce a closer match 
    of both amplitude distribution and power spectrum of surrogate and original data.

    Reference: [Schreiber2000]_

    :param y: 1D array representing the original time series.
    :param nit: number of iterations / refinement steps
    :param output: type of surrogate to return:
        - "true_amplitudes": surrogates with correct amplitude distribution, 
        "true_spectrum": surrogates with correct power spectrum, 
        "both": return both outputs of the algorithm.
    :return: 1D array of the surrogate time series.
    '''
    # Get Fourier transform of original data
    fourier_transform = np.fft.rfft(y)

    # Get Fourier amplitudes
    original_fourier_amps = np.abs(fourier_transform)

    # Get sorted copy of original data
    sorted_original = y.copy()
    sorted_original.sort()

    # Get initial surrogate using AAFT method
    r = AAFT(y)

    # For each iteration
    for i in range(nit):
        # Get Fourier phases of surrogate
        r_fft = np.fft.rfft(r)
        r_phases = r_fft / np.abs(r_fft)

        # Transform back, replacing the actual amplitudes by the desired
        # ones, but keeping the phases exp(iψ)
        s = np.fft.irfft(original_fourier_amps * r_phases, n=y.size)

        # Rescale to desired amplitude distribution
        ranks = s.argsort().argsort()
        r = sorted_original[ranks]

    # Return requested output
    if output == 'true_amplitudes':
        return r
    elif output == 'true_spectrum':
        return s
    elif output == 'both':
        return (r, s)
    else:
        raise ValueError(f'Invalid output type: {output}')


@pandas_proof
def generate_surrogate(y, method='iAAFT'):
    '''
    Wrapper around surrogate data generation methods 

    :param y: original 1D timeseries
    :param method: surrogate data generation method
    :return: surrogate 1D timeseries
    '''
    if method == 'FT':
        return FT(y)
    elif method == 'AAFT':
        return AAFT(y)
    elif method == 'iAAFT':
        return iAAFT(y)
    else:
        raise ValueError(f'invalid surrogate generation method: {method}')
