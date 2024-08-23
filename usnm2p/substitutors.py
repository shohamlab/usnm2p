# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-28 16:29:23
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2024-08-07 16:00:27

''' Collection of stack substitutors utilities. '''

import numpy as np
import ast
import operator as op

from .logger import logger
from .fileops import StackProcessor, process_and_save
from .constants import DataRoot
from .indexers import FrameIndexer


# Define supported operators for arithmetic expressions
operators = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.BitXor: op.xor,
    ast.USub: op.neg,
}

def _eval(node):
    '''
    Evaluate an arithmetic expression node.

    :param node: the expression node.
    :return: the evaluated result.
    '''
    if isinstance(node, ast.Num):  # <number>
        return node.n
    elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
        return operators[type(node.op)](_eval(node.left), _eval(node.right))
    elif isinstance(node, ast.UnaryOp):  # - <operand>
        return operators[type(node.op)](_eval(node.operand))
    else:
        raise TypeError(node)


def safe_eval(expr):
    '''
    Safely evaluate an arithmetic expression.

    :param expr: the expression as a string.
    :return: the evaluated result.
    '''
    try:
        return _eval(ast.parse(expr, mode='eval').body)
    except Exception as e:
        raise ValueError(f"Invalid expression: {expr}") from e


class StackSubstitutor(StackProcessor):
    '''Main interface to stack substitutor. '''

    def __init__(self, submap, fidx=None, **kwargs):
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
        self.fidx = fidx
        super().__init__(**kwargs)
    
    def submap_str(self):
        ''' Rpresentative string for the substitution map '''
        l = []
        # For each source desitnation pair
        for isource, idest, repeat_every in self.submap:
            # If any index is a string, replace 'stim' by placeholder and evaluate
            if any(isinstance(x, str) for x in [isource, idest]):
                proxy_source, proxy_dest = [safe_eval(s.replace('stim', '0')) for s in [isource, idest]]
                is_preceding_source = proxy_source < proxy_dest
            else:
                is_preceding_source = isource < idest
            if is_preceding_source:
                s = f'{isource}->{idest}'
            else:
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
        # Check that the substitution map is a list of tuples
        if not isinstance(value, list) or not all(isinstance(x, tuple) for x in value):
            raise ValueError('submap must be a list of tuples')

        # Loop through tuples
        for items in value:

            # Check that each tuple has 3 elements
            if len(items) != 3:
                raise ValueError(f'each substitution must a tuple of 3 elements')
            
            # Loop through index elements
            for idx in items[:2]:
                # Check that indexes are either integers or strings containing 'stim'
                if isinstance(idx, int):
                    if idx < 0:
                        raise ValueError('substitution indexes must be >= 0')
                elif isinstance(idx, str):
                    if 'stim' not in idx:
                        raise ValueError(f'string-based substitution index {idx} must contain "stim"')
                else:
                    raise ValueError(f'unknown substitution index type: {type(idx)}')
            
            # Check that substitution indexes are different
            if items[0] == items[1]:
                raise ValueError('substitution indexes must be different')
            
            # Inspect repeat element
            if items[2] is not None:
                # If integer, check that it is >= 2 and <= max of the first two indexes
                if isinstance(items[2], int):
                    if items[2] < 2:
                        raise ValueError('repeat item must be >= 2')
                    imax = max(items[:2])
                    if items[2] < imax:
                        raise ValueError(f'cannot repeat every {items[2]}: smaller than substitution index ({imax})')
                # If string, check that it is 'trial'
                elif isinstance(items[2], str):
                    if items[2] != 'trial':
                        raise ValueError('string-based repeat item must be "trial"')
                # Otherwise, raise error
                else:
                    raise ValueError(f'unknown repeat item type: {type(items[2])}')
        
        # Extract source and destination indexes
        isources, idests, _ = [np.atleast_1d(np.asarray(x)) for x in zip(*value)]

        # Check that there are no intersecting source and destination indexes
        if len(set(isources).intersection(idests)) > 0:
            raise ValueError('intersecting source and destination indexes') 
        
        # Assign substitution map
        self._submap = value
    
    @property
    def fidx(self):
        return self._fidx
    
    @fidx.setter
    def fidx(self, value):
        if value is not None and not isinstance(value, FrameIndexer):
            raise TypeError(f'fidx must be a FrameIndexer object (got {type(value)})')
        self._fidx = value
    
    def code_to_integer(self, s, fidx):
        ''' Translate string index code to integer index using frame indexer object '''
        # Replace "stim" occurences by their corresponding integer values
        if 'stim' in s:
            istim = fidx.iref
            s = s.replace('stim', str(istim))
        
        # Replace "trial" occurences by their corresponding integer values
        if 'trial' in s:
            if fidx.npertrial is None:
                raise ValueError('cannot convert "trial" code: frame indexer object does not have "npertrial" attribute')
            npertrial = fidx.npertrial
            s = s.replace('trial', str(npertrial))
        
        # Evaluate string expression and return corresponding integer
        return safe_eval(s)
    
    def parse_submap_element(self, x):
        ''' 
        Parse element of substitution map
        
        :param x: element of substitution map
        :return: parsed element
        '''
        if isinstance(x, str):
            if self.fidx is None:
                raise ValueError('substitution map contains strings -> frame indexer object is required')
            return self.code_to_integer(x, self.fidx)
        return x
    
    def parse_submap(self):
        '''
        Parse substitution map and convert all string codes to appropriate
        integer values using assigned frame indexer object. 
        '''
        return [tuple(self.parse_submap_element(x) for x in item) for item in self.submap]
    
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
        # Initialize empty source and destination vectors
        all_isources, all_idests = np.array([], dtype=int), np.array([], dtype=int)

        # Loop through parsed substitution map
        for isource, idest, repeat in self.parse_submap():
            # Generate expanded source and destination indexes for that stack size 
            isources, idests = self.get_expanded_subindexes(isource, idest, repeat, n)

            # Concatenate to global vectors
            all_isources = np.hstack((all_isources, isources))
            all_idests = np.hstack((all_idests, idests))
        
        # Return source and destination vectors
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
    

def substitute_tifs(input_fpaths, input_key, submap, fidx=None, nchannels=1, **kwargs):
    '''
    High-level stack frame substitution function

    :param input_fpaths: list of full paths to input TIF stacks
    :param input_key: input key for output path replacement
    :param submap: substituion map to apply
    :param fidx (optional): frame indexer object
    :param nchannels (optional): number of channels in the input stacks
    :return: list of substituted TIF stacks
    '''
    # Define substitutor object
    ss = StackSubstitutor(submap, fidx=fidx, nchannels=nchannels)

    # Substitute frames in all input TIFs 
    substituted_stack_fpaths = process_and_save(
        ss, input_fpaths, input_key, overwrite=False)

    # Return list of output filepaths     
    return substituted_stack_fpaths