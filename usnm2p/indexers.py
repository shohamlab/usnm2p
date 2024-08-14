import numpy as np

class FrameIndexer:
    ''' Frame indexer class for window slicing '''

    def __init__(self, iref, npre, npost):
        ''' 
        Constructor

        :param iref: reference index
        :param npre: number of pre-reference samples
        :param npost: number of post-reference samples
        '''
        self.iref = iref
        self.npre = npre
        self.npost = npost

    @staticmethod    
    def from_time(tref, tpre, tpost, dt):
        '''
        Construct a FrameIndexer from time bounds

        :param tref: reference time (s)
        :param tpre: duration of pre-reference window (s)
        :param tpost: duration of post-reference window (s)
        :param dt: time step (s)
        :return: FrameIndexer instance
        '''
        iref, npre, npost = [int(np.round(t / dt)) for t in [tref, tpre, tpost]]
        return FrameIndexer(iref, npre + 1, npost + 1)
    
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
