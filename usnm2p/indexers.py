# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2024-08-15 17:32:10
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-07-01 16:34:35
import numpy as np

class FrameIndexer:
    ''' Frame indexer class for window slicing '''

    def __init__(self, iref, npre, npost, npertrial=None):
        ''' 
        Constructor

        :param iref: reference index
        :param npre: number of pre-reference samples
        :param npost: number of post-reference samples
        :param npertrial (optional): number of indexes per trial (i.e., "repetition unit"), if any
        '''
        self.iref = iref
        self.npre = npre
        self.npost = npost
        self.npertrial = npertrial

    @staticmethod    
    def from_time(tref, tpre, tpost, dt, **kwargs):
        '''
        Construct a FrameIndexer from time bounds

        :param tref: reference time (s)
        :param tpre: duration of pre-reference window (s)
        :param tpost: duration of post-reference window (s)
        :param dt: time step (s)
        :return: FrameIndexer instance
        '''
        iref, npre, npost = [int(np.round(t / dt)) for t in [tref, tpre, tpost]]
        return FrameIndexer(iref, npre + 1, npost + 1, **kwargs)
    
    def __repr__(self):
        return f'{self.__class__.__name__}(iref={self.iref}, npre={self.npre}, npost={self.npost})'
    
    def check_dtype(self, key, value):
        if not isinstance(value, (int, np.int64)):
            raise TypeError(f'{key} must be an integer (got {type(value)})')
    
    def check_window_size(self, key, value):
        if value < 1:
            raise ValueError(f'{key} must be >= 1')
    
    @property
    def iref(self):
        return self._iref
    
    @iref.setter
    def iref(self, value):
        key = 'reference index'
        self.check_dtype(key, value)
        if value < 0:
            raise ValueError(f'{key} must be >= 0')
        self._iref = value
    
    @property
    def npre(self):
        return self._npre
    
    @npre.setter
    def npre(self, value):
        key = 'pre-reference window size'
        self.check_dtype(key, value)
        self.check_window_size(key, value)
        self._npre = value
    
    @property
    def npost(self):
        return self._npost
    
    @npost.setter
    def npost(self, value):
        key = 'post-reference window size'
        self.check_dtype(key, value)
        self.check_window_size(key, value)
        self._npost = value
    
    @property
    def npertrial(self):
        return self._npertrial
    
    @npertrial.setter
    def npertrial(self, value):
        key = 'indexes per trial'
        if value is not None:
            self.check_dtype(key, value)
            self.check_window_size(key, value)
        self._npertrial = value
    
    def get_window_slice(self, kind):
        '''
        Get a window slice centered around a reference index

        :param kind: window kind ('pre' or 'post')
        :return: window slice
        '''
        # Check that window kind is valid
        if kind not in ('pre', 'post'):
            raise ValueError(f'unknown window kind "{kind}"') 
        
        # Preceding window: n elements, finishing on (including) index
        if kind == 'pre':
            return slice(self.iref - self.npre + 1, self.iref + 1)
        
        # Postceding window: n elements, starting at index + 1
        else:
            return slice(self.iref + 1, self.iref + self.npost + 1)
    
    def get_window_idxs(self, kind):
        '''
        Get indices for a specific analysis window 

        :param kind: window kind ('pre' or 'post')
        :return: window indexes as numpy array
        '''
        w = self.get_window_slice(kind)
        return np.arange(w.start, w.stop)
    
    def get_window_bounds(self, kind):
        '''
        Get bounds for a specific analysis window 

        :param kind: window kind ('pre' or 'post')
        :return: window bounding indexes as numpy array
        '''
        w = self.get_window_slice(kind)
        return np.array([w.start, w.stop - 1])
    
    def get_window_delta(self, kind):
        '''
        Get index delta for a specific analysis window

        :param kind: window kind ('pre' or 'post')
        :return: index delta
        '''
        bounds = self.get_window_bounds(kind)
        return bounds[1] - bounds[0]

    def get_time_vector(self, idx, dt):
        '''
        Transform indexes vector into time vector
        
        :param idx: indexes to transform
        :param dt: time step (s)
        :return: stimulus-aligned time vector as numpy array
        '''
        if self.npertrial is None or self.npertrial <= 0:
            raise ValueError('npertrial must be set to integer > 0 to get time vector')
        if dt <= 0:
            raise ValueError('dt must be > 0')
        if not isinstance(idx, np.ndarray):
            raise TypeError('idx must be a numpy array')
        if idx.ndim != 1:
            raise ValueError('idx must be a 1D numpy array')
        if not np.issubdtype(idx.dtype, np.integer):
            raise TypeError('idx must be an array of integers')
        return (idx - self.iref) * dt
    
    def resample(self, factor):
        ''' 
        Resample the indexer by a specific factor

        :param factor: resampling factor (must be an integer)
        :return: new FrameIndexer instance with resampled parameters
        '''
        return FrameIndexer(
            iref=self.iref * factor,
            npre=self.npre * factor,
            npost=self.npost * factor,
            npertrial=None if self.npertrial is None else self.npertrial * factor
        )
