# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-28 16:29:23
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-10-28 19:33:56

''' Collection of stack interpolators utilities. '''

import abc
import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d

from logger import logger
from utils import StackProcessor, NoProcessor


class NoInterpolator(NoProcessor):
    ''' No interpolation instance to substitute in the code in case no interpolation is wanted '''
    pass


class StackInterpolator(StackProcessor):
    '''Main interface to Spline stack interpolator. '''

    order_to_kind_map = {
        0: 'zero',
        1: 'slinear',
        2: 'quadratic',
        3: 'cubic'
    }

    def __init__(self, order: np.uint8=3, npast:np.uint8=5):
        '''
        Initialization

        :param order: interpolation order (positive integer).
        :param npast: number of preceding samples to consider to build spline interpolatrs for each index to be replaced.
        '''
        self.order = order
        self.npast = npast
        super().__init__()

    def __str__(self) -> str:
        return f'{self.__class__.__name__}(order={self.order}, npast={self.npast})'

    @property
    def code(self):
        return f'si_order{self.order}_npast{self.npast}'
    
    @property
    def rootcode(self):
        return 'interpolated'

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        if not isinstance(value, int):
            raise ValueError('order must be a positive integer')
        try:
            self.kind = self.order_to_kind_map[value]
        except KeyError:
            raise ValueError(f'unknown order: {value} (choices are {list(self.order_to_kind_map.keys())})')
        self._order = value

    @property
    def npast(self):
        return self._npast

    @npast.setter
    def npast(self, value):
        if not isinstance(value, int) or value < 2:
            raise ValueError('npast must be an integer higher than 2')
        self._npast = value

    def run(self, stack: np.array, indexes: np.array) -> np.ndarray:
        '''
        Interpolate stack Performs Kalman denoising of a 3D image stack
    
        :param stack: 3D stack array (time, width, height)
        :param indexes: indexes to replace with interpolated estimates
        :return: interpolated stack
        '''
        # Cast to float64 for increased precision
        input_type = stack.dtype
        stack = stack.astype(np.float64)
        
        # Get stack array dimensions
        nframes, width, height = stack.shape

        logger.info(f'interpolating {len(indexes)} frames in {nframes}-frames stack with {self}')

        # Define time vector for preceding samples
        t = np.arange(self.npast + 1)

        # for each frame index to replace
        for i in tqdm(indexes):
            # Get past few frames
            logger.debug(f'predicting sample {i} using samples {i - self.npast} to {i - 1}')
            pre_frames = stack[i - self.npast:i]
            # Create interpolator function
            f = interp1d(
                t[:-1], pre_frames, kind=self.kind, axis=0, fill_value='extrapolate', assume_sorted=True)
            # Interpolate at frame index
            new_frame = f(t[-1])
            # Clip new frame to have only positive values
            new_frame = np.maximum(new_frame, 0.)
            # Replace frame in output stack
            stack[i] = new_frame

        # Cast from floating point to input type (usually 16-bit integer)
        stack = stack.astype(input_type)
        
        # Return
        return stack 
