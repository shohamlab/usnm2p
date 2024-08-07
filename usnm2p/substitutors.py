# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-28 16:29:23
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2024-08-07 16:00:27

''' Collection of stack substitutors utilities. '''

import numpy as np

from .logger import logger
from .fileops import StackProcessor
from .constants import DataRoot


class StackSubstitutor(StackProcessor):
    '''Main interface to stack substitutor. '''

    def __init__(self, submap, *args, **kwargs):
        '''
        Initialization

        :param submap: list of (source_idx, dest_idx, repeat_every) items where:
            - source_idx is the source idx used for replacement
            - dest_idx is the destination index (i.e. replacement location)
            - repeat_every indicates whether to repeat substitutions periodically
            (i.e. every "repeat_every" indexes). If None, a single substitution is
            performed.
        '''
        self.submap = submap
        super().__init__(*args, **kwargs)
    
    def submap_str(self):
        ''' Rpresentative string for the substitution map '''
        l = []
        for isource, idest, repeat_every in self.submap:
            if isource < idest:
                s = f'{isource}->{idest}'
            elif isource > idest:
                s = f'{idest}<-{isource}'
            if repeat_every is not None:
                s = f'{s}every{repeat_every}'
            l.append(s)
        return ', '.join(l)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(submap=[{self.submap_str()}])'

    @property
    def code(self):
        s = f'submap_{self.submap_str()}'
        return s.replace(', ', '_')
     
    @property
    def rootcode(self):
        return DataRoot.SUBSTITUTED

    @property
    def submap(self):
        return self._submap

    @submap.setter
    def submap(self, value):
        if not isinstance(value, list) or not all(isinstance(x, tuple) for x in value):
            raise ValueError('submap must be a list of tuples')
        for items in value:
            if len(items) != 3:
                raise ValueError(f'each substitution must a tuple of 3 elements')
            if not all(isinstance(idx, int) for idx in items[:2]):
                raise ValueError(f'substitution indexes must be integers')
            if not all(idx >= 0 for idx in items[:2]):
                raise ValueError('substitution indexes must be >= 0')
            if items[0] == items[1]:
                raise ValueError('substitution indexes must be different')
            if items[2] is not None:
                if not isinstance(items[2], int):
                    raise ValueError('repeat item must be either None or an integer')
                if items[2] < 2:
                    raise ValueError('repeat item must be >= 2')
                imax = max(items[:2])
                if items[2] < imax:
                    raise ValueError(f'cannot repeat every {items[2]}: smaller than substitution index ({imax})')
        isources, idests, _ = [np.atleast_1d(np.asarray(x)) for x in zip(*value)]
        if len(set(isources).intersection(idests)) > 0:
            raise ValueError('intersecting source and destination indexes') 
        self._submap = value
    
    def get_expanded_subindexes(self, isource, idest, repeat, n):
        '''
        Generate expanded list of substitution indexes for a given stack size
        
        :param isource: source index
        :param idest: destination index
        :param repeat: optional repetition index
        :param n: stack size
        :return: expanded source and destination index vectors
        '''
        # If no repeat element provided, just assign it to stack size
        if repeat is None:
            repeat = n
        
        # Generate repetition indexes vector for input stack size
        ireps = np.arange(n // repeat) * repeat

        # Add original source and destination indexes as offsets to this vector,
        # in order to generate expanded source and destination index vectors 
        isources = isource + ireps
        idests = idest + ireps

        # Return expanded vectors
        return isources, idests
    
    def get_expanded_submap(self, n):
        '''
        Get expanded substitution map for a given stack size

        :param n: stack size
        :return: expanded subsitution map, with repeats incorporated
        '''
        all_isources, all_idests = np.array([], dtype=int), np.array([], dtype=int)
        for isource, idest, repeat in self.submap:
            isources, idests = self.get_expanded_subindexes(isource, idest, repeat, n)
            all_isources = np.hstack((all_isources, isources))
            all_idests = np.hstack((all_idests, idests))
        return all_isources, all_idests

    def run(self, stack: np.array) -> np.ndarray:
        '''
        Apply subsitution on a 3D stack
    
        :param stack: 3D stack array (frames, width, height)
        :param axis: axis on which to perform substitutions
        :return: substituted stack
        '''
        # Get number of frames
        nframes, *_ = stack.shape

        # Extract expanded source and destination indexes for that stack size
        isources, idests = self.get_expanded_submap(nframes)

        # Substitute frames
        logger.info(f'substituting frames {idests} by frames {isources} in {nframes}-frames stack')
        stack[idests] = stack[isources]

        # Return
        return stack