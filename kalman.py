import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import get_stack_baseline
from logger import logger

class KalmanDenoiser:
    '''
    Main interface to Kalman filtering denoiser. 
    Based on ImageJ plugin written by Christopher Philip Mauer (https://imagej.nih.gov/ij/plugins/kalman.html).

    This plugin implements a recursive prediction/correction algorithm which is based on the Kalman Filter
    (commonly used for robotic vision and navigation) to remove high gain noise from time lapse image streams.
    These filters remove camera/detector noise while recovering faint image detail.
    '''

    def __init__(self, gain=0.8, variance=0.05, npad=10):
        '''
        Initialization

        :param gain: filter gain level (0-1). High filter gain renders the output less sensitive 
            to momentary fluctuations.
        :param variance: estimate of noise variance level (0-1). Wrong guesses for the initial variance
            will not prevent noise estimation, but merely delay the fitting process.
        :param npad: number of frames from which to construct initial baseline padding
        '''
        self.gain = gain
        self.variance = variance
        self.npad = npad

    def __str__(self) -> str:
        return f'{self.__class__.__name__}(gain={self.gain}, var={self.variance}, npad={self.npad})'

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, value):
        if not 0 < value <= 1:
            raise ValueError("Gain must be between 0 and 1.")
        self._gain = value

    @property
    def variance(self):
        return self._variance

    @variance.setter
    def variance(self, value):
        if not 0 < value <= 1:
            raise ValueError("Variance must be between 0 and 1.")
        self._variance = value

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
        k = ex_prior / (ex_prior + self.variance)
        x_post = self.gain * x_prior + (1.0 - self.gain) * x_obs + k * (x_obs - x_prior)
        ex_post = ex_prior * (1.0 - k)
        return x_post, ex_post
    

    # TODO: Speed up with numba???
    def filter(self, stack: np.array, full_output: bool=False):
        '''
        Performs Kalman denoising of a 3D image stack
    
        :param stack: 3D stack array (time, width, height)
        :param full_output: boolean stating whether to return also the history of estimated variance
        :return: filtered stack (with optionally the variance estimate history)
        '''
        assert len(stack) != 0, 'Stack is empty.'
        assert len(stack) != 1, 'Stack must contain more than one element.'
        # Default cast to float64 for increased precision
        stack = stack.astype(np.float64)
        
        # Get stack array dimensions
        nframes, width, height = stack.shape

        logger.info(f'filtering {nframes}-frames stack with {self}')

        # Optional: add initial padding to allow for initial variance fitting
        if self.npad > 0:
            stack = np.concatenate((get_stack_baseline(stack, self.npad), stack))
             
        # Initialization
        filtered_stack = np.zeros_like(stack)  # Initialize output array
        ex_history = [self.variance]  # Keep track of variance estimates
        x = stack[0]  # Use first frame as the state variable prediction seed
        ex = np.ones((width, height)) * self.variance  # Use the variance estimate as the state error seed
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
        ax.axhline(self.variance * 1e2, c='k', ls='--', label='initial variance')
        if self.npad > 0:
            ax.axvspan(0, self.npad, fc='dimgray', ec='none', alpha=0.5, label='padding region')
        ax.legend(frameon=False)
        return fig


if __name__ == '__main__':
    
    stack = np.random.rand(3, 4, 2)
    print(stack)
    kd = KalmanDenoiser(variance=0.05, gain=0.08)
    kd.filter(stack)