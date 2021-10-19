# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-14 19:29:19
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-10-17 17:24:00

import re

''' Collection of parsing utilities. '''

# General tif file pattern
P_TIFFILE = re.compile('.*tif')

UNDEFINED_UNICODE = '\u00D8'

# Naming patterns for input folders and files 
P_LINE = '([A-z][A-z0-9]*)'
P_TRIAL_LENGTH = '([0-9]+)frames'
P_FREQ = '([0-9]+[.]?[0-9]*)Hz'
P_DUR = '([0-9]+[.]?[0-9]*)ms'
P_MPA = '([0-9]+[.]?[0-9]*)MPA'
P_DC = '([0-9]+)DC'
P_RUN = '([0-9]+)'
P_CYCLE = 'Cycle([0-9]+)'
P_CHANNEL = 'Ch([0-9])'
P_FRAME = '([0-9]+)'
P_RAWFOLDER = re.compile(f'^{P_LINE}_{P_TRIAL_LENGTH}_{P_FREQ}_{P_DUR}_{P_FREQ}_{P_MPA}_{P_DC}-{P_RUN}$', re.IGNORECASE)
P_RAWFILE = re.compile(f'{P_RAWFOLDER.pattern[:-1]}_{P_CYCLE}_{P_CHANNEL}_{P_FRAME}.ome.tif$', re.IGNORECASE)
P_STACKFILE = re.compile(f'{P_RAWFOLDER.pattern[:-1]}.tif$', re.IGNORECASE)


def parse_experiment_parameters(name):
    '''
    Parse experiment parameters from a file/folder name.
    
    :param name: file / folder name from which parameters must be extracted
    :return: dictionary of extracted parameters
    '''
    # Try first with folder pattern
    israwfile = False
    mo = P_RAWFOLDER.match(name)
    # If no match detected, try with file pattern
    if mo is None:
        israwfile = True
        mo = P_RAWFILE.match(name)
    # If still no match, try with stack file pattern
    if mo is None:
        israwfile = False
        mo = P_STACKFILE.match(name)
    # If still no match, throw error
    if mo is None:
        raise ValueError(f'"{name}" does not match the experiment naming pattern')
    # Extract and parse folder-level parameters
    params = {
        'line': mo.group(1),  # line name
        'trial_length': int(mo.group(2)),
        '???': float(mo.group(3)),
        'duration': float(mo.group(4)) * 1e-3,  # s
        'fps': float(mo.group(5)),  
        'P': mo.group(6),  # MPa
        'DC': float(mo.group(7)),  # %
        'run': int(mo.group(8))
    }
    # Fix for pressure (replacing first zero by decimal dot)
    if '.' not in params['P']:
        params['P'] = float(f'.{params["P"][1:]}')
    # If file, add file-level parameters
    if israwfile:
        params.update({
            'cycle': int(mo.group(9)),
            'channel': int(mo.group(10)),
            'frame': int(mo.group(11))
        })
    # Return parameters dictionary
    return params


def reg_search_list(query, targets):
    '''
    Search a list or targets for match if regexp query, and return first match.
    
    :param query: regexp format query
    :param targets: list of strings to be tested
    :return: first match (if any), otherwise None
    '''
    p = re.compile(query)
    for target in targets:
        positive = p.search(target)
        if positive:
            positive = positive.string
            break
    return positive


def to_unicode(query):
    ''' Translate regexp query into unicode (effectively a fix for undefined queries) '''
    if query == 'undefined':
        return UNDEFINED_UNICODE
    else:
        return query