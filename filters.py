# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-04 17:44:51
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-02-08 16:04:54

''' Collection of filtering utilities. '''

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from logger import logger
from fileops import StackProcessor, NoProcessor
from constants import DataRoot


class StackFilter(StackProcessor):

    @staticmethod
    def get_baseline(stack: np.array, n: int, noisefactor: float=1.) -> np.array:
        '''
        Construct baseline stack from stack.
        
        :param stack: 3D image stack
        :param n: number of frames to put in the baseline
        :param noisefactor: uniform multiplying factor applied to the noise matrix to construct the baseline
        :return: generated baseline stack
        '''
        # Extract frame dimensions from stack
        nframes, *framedims = stack.shape
        logger.info(f'constructing {n}-frames baseline from {nframes}-frames stack...')
        # Construct noise matrix spanning [-noisefactor, noisefactor] and matching the required baseline stack dimensions
        noise = noisefactor * (2 * np.random.rand(n, *framedims) - 1)
        # Estimate median and standard deviation of each pixel across the images of the stack
        pixel_med, pixel_std = np.median(stack, axis=0), np.std(stack, axis=0)
        # Add axis to both matrices to allow broadcasting with noise matrix
        pixel_med, pixel_std = [np.expand_dims(x, axis=0) for x in [pixel_med, pixel_std]]
        # Construct baseline stack by summing up median and std-scaled noise matrices
        baseline = pixel_med + noise * pixel_std
        # Bound to [0, MAX] interval and return
        return np.clip(baseline, 0, np.amax(stack))

    @property
    def rootcode(self):
        return DataRoot.FILTERED


class NoFilter(NoProcessor):
    ''' No filter instance to substitute in the code in case no filtering is wanted '''
    pass


class KalmanDenoiser(StackFilter):
    '''
    Main interface to Kalman filtering denoiser.

    Based on ImageJ plugin written by Christopher Philip Mauer (https://imagej.nih.gov/ij/plugins/kalman.html).

    This plugin implements a recursive prediction/correction algorithm which is based on the Kalman Filter
    (Piovoso and Laplante, 2003) commonly used for robotic vision and navigation to remove high gain noise
    from time lapse image streams. These filters remove camera/detector noise while recovering faint image detail.

    This method extends the original Kalman filter by introducing an additional filter gain G in the algorithm
    (Khmou and Safi, 2013). This gain is used at each iteration to compute a modified Kalman gain K2 = K + 1 - G,
    which is then used to compute the new posterior estimate:

    X_estim = K2 * X_measured + (1 - K2) * X_pred

    The formula then reduces to:

    X_estim = G * X_pred + (1 − G) * X_measured + K * (X_pred − X_measured)

    Additionally, the posterior variance estimate is calculated assuming a neutral transition process, with
    a unit diagonal transition matrix and zero noise. Under these assumptions, the posterior variance estimate becomes:

    EX_estim = EX_pred * (1 - K)

    The main advantages of this extension are:
    (1) wrong guesses of the initial variance will not prevent noise estimation but merely delay the fitting process.
    (2) values for the filter gain G renders the output less sensitive to momentary fluctuations and therefore adapted
    for the processing of recordings containing calcium transients.
    
    References:
    - Piovoso, M., and Laplante, P.A. (2003). Kalman filter recipes for real-time image processing. Real-Time Imaging 9, 433–439.
    - Khmou, Y., and Safi, S. (2013). Estimating 3D Signals with Kalman Filter. ArXiv:1307.4801 [Cs, Math].
    '''

    def __init__(self, G=0.5, V=0.05, npad: np.uint8=10, *args, **kwargs):
        '''
        Initialization

        :param G: filter gain (0-1).
        :param V: variance (i.e. noise) estimate (0-1).
        :param npad: baseline padding length (number of frames) to absorb initial errors.
        '''
        self.G = G
        self.V = V
        self.npad = npad
        super().__init__(*args, **kwargs)

    def __str__(self) -> str:
        return f'{self.__class__.__name__}(gain={self.G}, var={self.V}, npad={self.npad})'

    @property
    def code(self):
        return f'kd_G{self.G}_V{self.V}_npad{self.npad}'

    @property
    def G(self):
        return self._G

    @G.setter
    def G(self, value):
        if not 0 < value <= 1:
            raise ValueError('Gain must be between 0 and 1.')
        self._G = value

    @property
    def V(self):
        return self._V

    @V.setter
    def V(self, value):
        if not 0 < value <= 1:
            raise ValueError('Variance must be between 0 and 1.')
        self._V = value

    @property
    def npad(self):
        return self._npad

    @npad.setter
    def npad(self, value):
        if not isinstance(value, int) or value < 0:
            raise ValueError('npad must be a positive integer')
        self._npad = value

    def correct(self, x_prior: np.array, ex_prior: np.array, x_obs: np.array) -> tuple((np.array, np.array)):
        ''' 
        Correct estimate and associated variance of a state variable
        according to observation.

        :param x_prior: prior estimate of the variable
        :param ex_prior: prior variance estimate of the variable
        :param x_obs: observed variable
        :return: 2-tuple of a posteriori (i.e. corrected) variable estimate and variance.
        '''
        K = ex_prior / (ex_prior + self.V)  # Kalman gain = ratio of estimate variance / total variance
        x_post = self.G * x_prior + (1.0 - self.G) * x_obs + K * (x_obs - x_prior)  # posterior estimate
        ex_post = ex_prior * (1.0 - K)  # posterior variance estimate
        return x_post, ex_post

    def run(self, stack: np.array, full_output: bool=False):
        '''
        Performs Kalman denoising of a 3D image stack
    
        :param stack: 3D stack array (time, width, height)
        :param full_output: boolean stating whether to return also the history of estimated variance
        :return: filtered stack (with optionally the variance estimate history)
        '''
        assert len(stack) != 0, 'Stack is empty.'
        assert len(stack) != 1, 'Stack must contain more than one element.'

        # Cast to float64 for increased precision
        input_type = stack.dtype
        stack = stack.astype(np.float64)
        
        # Get stack array dimensions
        nframes, width, height = stack.shape

        # Optional: add initial padding to allow for initial variance fitting
        if self.npad > 0:
            stack = np.concatenate((
                self.get_baseline(stack[:min(nframes, 100)], self.npad),
                stack))

        logger.info(f'filtering {nframes}-frames stack with {self}')
             
        # Initialization
        filtered_stack = np.zeros_like(stack)  # Initialize output array
        ex_history = [self.V]  # Keep track of variance estimates
        x = stack[0]  # Use first frame as the state variable prediction seed
        ex = np.ones((width, height)) * self.V  # Use the variance estimate as the state error seed
        filtered_stack[0] = x

        # Recursive algorithm for each subsequent frame
        for i, x_obs in enumerate(tqdm(stack[1:])):
			# Correct estimates of frame state variable and its variance,
            # and use them as prior estimates for the next frame 
            x, ex = self.correct(x, ex, x_obs)
            # Store corrected frame estimate into output stack
            filtered_stack[i + 1] = x
            ex_history.append(np.mean(ex))

        # Remove padding output if present
        filtered_stack = filtered_stack[self.npad:]

        # Cast from floating point to input type (usually 16-bit integer
        filtered_stack = filtered_stack.astype(input_type) 

        # Return
        if full_output:
            return filtered_stack, np.array(ex_history)
        return filtered_stack

    def plot_variance_history(self, ex_history):
        fig, ax = plt.subplots()
        for sk in ['right', 'top']:
            ax.spines[sk].set_visible(False)
        ax.set_title('Variance estimate history')
        ax.set_xlabel('# iterations')
        ax.set_ylabel('mean variance estimate (%)')
        inds = np.arange(ex_history.size)
        ax.plot(inds, ex_history * 1e2, c='C0')
        ax.axhline(self.V * 1e2, c='k', ls='--', label='initial variance')
        if self.npad > 0:
            ax.axvspan(0, self.npad, fc='dimgray', ec='none', alpha=0.5, label='padding region')
        ax.legend(frameon=False)
        return fig
