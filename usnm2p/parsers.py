# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-14 19:29:19
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2024-08-09 15:14:12

''' Collection of parsing utilities. '''

# External modules
import re
import os
import numpy as np
import pandas as pd

# Internal modules
from .constants import *

# General tif file pattern
P_TIFFILE = re.compile('.*tif')
P_DATEMOUSEREGLAYER = re.compile(
    f'{Pattern.DATE}_({Pattern.MOUSE})_({Pattern.REGION})_?({Pattern.LAYER})?')
P_RAWFOLDER = re.compile(f'^{Pattern.LINE}_{Pattern.TRIAL_LENGTH}_{Pattern.FREQ}_{Pattern.DUR}_{Pattern.FREQ}_{Pattern.MPA}_{Pattern.DC}{Pattern.OPTIONAL_SUFFIX}-{Pattern.RUN}$', re.IGNORECASE)
P_RAWFILE = re.compile(f'{P_RAWFOLDER.pattern[:-1]}_{Pattern.CYCLE}_{Pattern.CHANNEL}_{Pattern.FRAME}.ome.tif$', re.IGNORECASE)
P_STACKFILE = re.compile(f'{P_RAWFOLDER.pattern[:-1]}.tif$', re.IGNORECASE)
P_RUNFILE = re.compile(
    f'^{Pattern.LINE}_{Pattern.TRIAL_LENGTH}_{Pattern.FREQ}_{Pattern.DUR}_{Pattern.FREQ}_{Pattern.MPA}_{Pattern.DC}-{Pattern.NAMED_RUN}.tif$', re.IGNORECASE)
P_TRIALFILE = re.compile(f'{P_RUNFILE.pattern[:-5]}_[0-9]*_?{Pattern.TRIAL}.tif$', re.IGNORECASE)
P_RUNFILE_SUB = r'\1_\2frames_\3Hz_\4ms_\5Hz_\6MPa_\7DC-run\8.tif'
P_TRIALFILE_SUB = r'\1_{nframes}frames_\3Hz_\4ms_{sr:.2f}Hz_\6MPa_\7DC-run\8_\9.tif'


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
    # If still not match, try with run file pattern:
    if mo is None:
        israwfile = False
        mo = P_RUNFILE.match(name)
    # If still no match, throw error
    if mo is None:
        raise ValueError(f'"{name}" does not match the experiment naming pattern')
    # Extract and parse folder-level parameters
    params = {
        Label.LINE: mo.group(1),  # line name
        Label.NPERTRIAL: int(mo.group(2)),
        Label.PRF: float(mo.group(3)),
        Label.DUR: float(mo.group(4)) * 1e-3,  # s
        Label.FPS: float(mo.group(5)),  
        Label.P: mo.group(6),  # MPa
        Label.DC: float(mo.group(7)),  # % 
        Label.RUNID: int(mo.group(9))
    }
    # Fix for folders with suffix
    if len(mo.group(8)) > 0:
        params[Label.SUFFIX] = mo.group(8)
    # Fix for pressure (replacing first zero by decimal dot)
    if '.' not in params[Label.P]:
        params[Label.P] = f'.{params[Label.P][1:]}'
    params[Label.P] = float(params[Label.P])
    # If file, add file-level parameters
    if israwfile:
        params.update({
            Label.CYCLE: int(mo.group(9)),
            Label.CH: int(mo.group(10)),
            Label.FRAME: int(mo.group(11))
        })
    # Return parameters dictionary
    return params


def get_info_table(folders, index_key=Label.RUN, ntrials_per_run=None, discard_unknown=True):
    '''
    Parse a list of input folders and aggregate extracted parameters into an info table.
    
    :param folders: list of absolute paths to data folders
    :param index_key (optional): name to give to the dataframe index.
    :param ntrials_per_run (optional): number of trials per run (added as an extra column if not none).
    :param discard_unknown (optional): whether to discard unknown (???) keys from table
    :return: pandas dataframe with parsed parameters for each folder.
    '''
    basenames = [os.path.basename(x) for x in folders]
    pdicts = [parse_experiment_parameters(x) for x in basenames]
    info_table = pd.DataFrame(pdicts)
    # for k in info_table:
    #     if isinstance(info_table[k][0], str):
    #         info_table[k] = pd.Categorical(info_table[k])
    info_table['code'] = [os.path.splitext(x)[0] for x in basenames]
    if index_key is not None:
        info_table.index.name = index_key
    if ntrials_per_run is not None:
        info_table[Label.NTRIALS] = ntrials_per_run
    if discard_unknown:
        if Label.UNKNOWN in info_table:
            del info_table[Label.UNKNOWN]
    return info_table


def parse_date_mouse_region(s):
    '''
    Parse a date, mouse and region from a concatenated string
    
    :param s: concatenated string
    :return: 3-tuple with (date, mouse, region)
    '''
    mo = P_DATEMOUSEREGLAYER.match(s)
    if mo:
        year, month, day, mouse, region, layer = mo.groups()
        date = f'{year}{month}{day}'
        if layer is None:
            layer = DEFAULT_LAYER
        return date, mouse, region, layer
    else:
        raise ValueError(
            f'{s} does not match date-mouse-reg-layer pattern ({P_DATEMOUSEREGLAYER.pattern})')
        

def group_by_run(fpaths):
    '''
    Group a large file list into consecutive trial files for each run
    
    :param fpaths: list of full paths to input files
    :return: dictionary of filepaths list per run index
    '''
    # Create output dictionary
    fbyrun = {}
    # For each file path
    for fpath in fpaths:
        # Split directory and filename
        fdir, fname = os.path.split(fpath)
        # Extract run and trial index from file name
        mo = P_TRIALFILE.match(fname)
        *_, irun, itrial = mo.groups()
        irun, itrial = int(irun), int(itrial)
        # Create run list if not already there
        if irun not in fbyrun:
            # Get run filename
            run_fname = P_TRIALFILE.sub(P_RUNFILE_SUB, fname)
            fbyrun[irun] = [run_fname, []]
        # Add filepath to appropriate run list
        fbyrun[irun][1].append(fpath)
    # Return dictionary
    return fbyrun


def parse_2D_offset(desc):
    '''
    Parse 2D offset (in mm) from string descriptor
    
    :param desc: string descriptor
    :return: 2D array with XY location values (in mm) 
    '''
    # Initialize null offset
    offset = np.array([0., 0.])
    # If composed offset descriptor, split across coordinates and construct offset iteratively
    sep = 'mm_'
    if sep in desc:
        subdescs = desc.split(sep)
        subdescs = [f'{s}{sep[:-1]}' for s in subdescs[:-1]] + [subdescs[-1]]
        for subdesc in subdescs:
            offset += parse_2D_offset(subdesc)
        return offset
    # If descriptor is not "center"
    if desc != 'center':
        # Parse descriptor and extract direction and magnitude
        mo = re.match(Pattern.OFFSET, desc)
        if mo is None:
            raise ValueError(f'unrecognized offset descriptor: {desc}')
        direction, magnitude = mo.group(1), float(mo.group(2))
        # Update XY offset depending on direction
        if direction == 'backward':
            offset[1] -= magnitude
        elif direction == 'right':
            offset[0] += magnitude
        elif direction == 'left':
            offset[0] -= magnitude    
    return offset


def parse_quantile(s):
    ''' Parse quantile expression '''
    mo = re.match(Pattern.QUANTILE, s)
    if mo is None:
        raise ValueError(f'expression "{s}" is not a valid quantile pattern')
    return float(mo.group(1))


def resolve_mouseline(s):
    ''' Resolve mouse line '''
    if 'line3' in s:
        return 'line3'
    elif 'pv' in s:
        return 'pv'
    elif 'sst' in s:
        return 'sst'
    else:
        raise ValueError(f'invalid mouse line: {s}')


def extract_FOV_area(map_ops):
    ''' Extract the field of view (FOV) area (in mm2)'''
    w = map_ops['micronsPerPixel'] * map_ops['Lx']  # um
    h = map_ops['micronsPerPixel'] * map_ops['Lx']  # um
    return w * h * UM2_TO_MM2  # mm2


def add_dataset_arguments(parser, analysis=True, line=True, date=True, mouse=True, region=True, layer=True):
    '''
    Add dataset-related arguments to a command line parser

    :param parser: command line parser
    :param analysis: flag to add analysis argument
    :param line: flag to add mouse line argument
    :param date: flag to add experiment date argument
    :param mouse: flag to add mouse number argument
    :param region: flag to add brain region argument
    :param layer: flag to add cortical layer argument
    '''
    if analysis:
        parser.add_argument('-a', '--analysis', default=DEFAULT_ANALYSIS, help='analysis type')
    if line:
        parser.add_argument('-l', '--mouseline', help='mouse line')
    if date:
        parser.add_argument('-d', '--expdate', help='experiment date')
    if mouse:
        parser.add_argument('-m', '--mouseid', help='mouse number')
    if region:
        parser.add_argument('-r', '--region', help='brain region')
    if layer:
       parser.add_argument('--layer', help='cortical layer')


def find_suffixes(folders):
    '''
    Extract suffixes from a list of folder names.
    
    :param folders: list of folder names
    :return: list of suffixes
    '''
    pref = os.path.commonprefix(folders)
    iprefend = pref.rindex('_')
    return [tf[iprefend + 1:] for tf in folders]