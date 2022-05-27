# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-28 16:29:23
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-05-27 13:31:30

''' Collection of stack substitutors utilities. '''

import numpy as np

from logger import logger
from utils import StackProcessor

class StackSubstitutor(StackProcessor):
    '''Main interface to stack substitutor. '''

    def __init__(self, submap, repeat_every=None):
        '''
        Initialization

        :param submap: list of (source_idx, dest_idx) subtitutions to be performed.
        :param repeat_every (optional): whether to repeat substitutions periodically
            (i.e. every "repeat_every" indexes)
        '''
        self.submap = submap
        self.repeat_every = repeat_every
        super().__init__()
    
    def submap_str(self):
        l  =[]
        for isource, idest in self.submap:
            if isource < idest:
                l.append(f'{isource}->{idest}')
            elif isource > idest:
                l.append(f'{idest}<-{isource}')
        return ', '.join(l)        

    def __str__(self) -> str:
        s = f'{self.__class__.__name__}(submap=[{self.submap_str()}]'
        if self.repeat_every is not None:
            s = f'{s}, repeat_every={self.repeat_every}'
        return f'{s})'

    @property
    def code(self):
        s = f'submap{self.submap_str()}'
        s = s.replace(', ', '_')
        if self.repeat_every is not None:
            s = f'{s}_every{self.repeat_every}'
        return s
     
    @property
    def rootcode(self):
        return 'substituted'

    @property
    def submap(self):
        return self._submap

    @submap.setter
    def submap(self, value):
        if not isinstance(value, list) or not all(isinstance(x, tuple) for x in value):
            raise ValueError('submap must be a list of tuples')
        for idxs in value:
            if len(idxs) != 2 or not all(isinstance(idx, int) for idx in idxs):
                raise ValueError(f'each substitution must a tuple of 2 integer indices') 
            if not all(idx >= 0 for idx in idxs):
                raise ValueError('all indexes must be >= 0')
            if idxs[0] == idxs[1]:
                raise ValueError('indexes must be different')
        isources, idests = [np.atleast_1d(np.asarray(x)) for x in zip(*value)]
        if len(set(isources).intersection(idests)) > 0:
            raise ValueError('intersecting source and destination indexes') 
        self._submap = value

    @property
    def repeat_every(self):
        return self._repeat_every

    @repeat_every.setter
    def repeat_every(self, value):
        if not isinstance(value, int) or value < 2:
            raise ValueError('repeat_every must be an integer higher than 1')
        imax = max(max(idxs) for idxs in self.submap)
        if value < imax:
            raise ValueError(f'cannot repeat every {value}: smaller than largest substitution index ({imax})')
        self._repeat_every = value

    def run(self, stack: np.array) -> np.ndarray:
        '''
        Apply subsitution on a 3D stack
    
        :param stack: 3D stack array (frames, width, height)
        :param axis: axis on which to perform substitutions
        :return: substituted stack
        '''
        # Get number of frames
        nframes, *_ = stack.shape
        # Extrat source and destination indexes
        isources, idests = [np.atleast_1d(np.asarray(x)) for x in zip(*self.submap)]
        idiffs = idests - isources

        # If specified, generate augmented list of sources and destination indexes
        if self.repeat_every is not None:
            nreps = nframes // self.repeat_every
            ireps = np.arange(nreps) * self.repeat_every
            isources = np.array([isources + irep for irep in ireps])
            idests = isources + idiffs
            isources, idests = isources.ravel(), idests.ravel()

        # Substitute frames
        logger.info(f'substituting frames {idests} by frames {isources} in {nframes}-frames stack')
        stack[idests] = stack[isources]

        # Return
        return stack 
