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
from .logger import logger

# General tif file pattern
P_TIFFILE = re.compile('.*tif')
P_DATEMOUSEREGLAYER = re.compile(
    f'{Pattern.DATE}_({Pattern.MOUSE})_({Pattern.REGION})_?({Pattern.LAYER})?')
P_RAWFOLDER = re.compile(f'^{Pattern.LINE}_{Pattern.TRIAL_LENGTH}_{Pattern.FREQ}_{Pattern.DUR}_{Pattern.FREQ}_{Pattern.MPA}_{Pattern.DC}{Pattern.OPTIONAL_SUFFIX}-{Pattern.RUN}$', re.IGNORECASE)
P_RAWFILE_BRUKER = re.compile(f'{P_RAWFOLDER.pattern[:-1]}_{Pattern.CYCLE}_{Pattern.CHANNEL}_{Pattern.FRAME}.ome.tif$', re.IGNORECASE)
P_STACKFILE = re.compile(f'{P_RAWFOLDER.pattern[:-1]}.tif$', re.IGNORECASE)
P_RUNFILE = re.compile(
    f'^{Pattern.LINE}_{Pattern.TRIAL_LENGTH}_{Pattern.FREQ}_{Pattern.DUR}_{Pattern.FREQ}_{Pattern.MPA}_{Pattern.DC}{Pattern.OPTIONAL_SUFFIX}{Pattern.NAMED_RUN}.tif$', re.IGNORECASE)
P_TRIALFILE = re.compile(f'{P_RUNFILE.pattern[:-5]}_[0-9]*_?{Pattern.TRIAL}.tif$', re.IGNORECASE)
P_RUNFILE_SUB = r'\1_{nframes}frames_\3Hz_\4ms_{fps}Hz_\6MPa_\7DC_run\8\9.tif'
P_TRIALFILE_SUB = r'\1_{nframes}frames_\3Hz_\4ms_{fps}Hz_\6MPa_\7DC_run\8_\9.tif'


def parse_experiment_parameters(name):
    '''
    Parse experiment parameters from a file/folder name.
    
    :param name: file / folder name from which parameters must be extracted
    :return: dictionary of extracted parameters
    '''
    # Dictionary of patterns to try and match
    patterns = {
        'raw folder': P_RAWFOLDER, 
        'raw bruker file': P_RAWFILE_BRUKER, 
        'stack file': P_STACKFILE, 
        'run file': P_RUNFILE, 
        'trial file': P_TRIALFILE,
    }

    # Attempt to match name with each pattern until a match is found
    for k, p in patterns.items():
        mo = p.match(name)
        if mo is not None:
            break
    
    # If no match detected, throw error
    if mo is None:
        raise ValueError(f'"{name}" does not match any of the experiment naming patterns')

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

    # If file pattern detected, add file-level parameters
    if k == 'trial file':
        params[Label.TRIAL] = int(mo.group(10))
    elif k == 'raw bruker file':
        params.update({
            Label.CYCLE: int(mo.group(10)),
            Label.CH: int(mo.group(11)),
            Label.FRAME: int(mo.group(12))
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
        

def group_by_run(flist, on_mismatch='raise', key_type='index'):
    '''
    Group a large file list into consecutive trial files for each run
    
    :param flist: list of file names
    :param on_mismatch: action to take if a file does not match the trial file naming pattern (raise, warn, or ignore)
    :param key_type: type of keys to include in the output dictionary ("index" for run index or "fname" run file name)
    :return: dictionary of sorted file names list per run index
    '''
    # If full paths provided, extract basenames
    is_fpath = any('/' in item for item in flist)
    if is_fpath:
        pardir = os.path.commonpath(flist)
        flist = [os.path.basename(item) for item in flist]

    # Create output dictionary
    fbyrun = {}

    # For each file path
    for fname in flist:
        # Extract run and trial index from file name
        mo = P_TRIALFILE.match(fname)
        if mo is None:
            err = f'"{fname}" does not match the trial file pattern'
            if on_mismatch == 'raise':
                raise ValueError(err)
            elif on_mismatch == 'warn':
                logger.warning(f'{err} -> ignoring')
                continue
            else:
                continue

        # If fname key requested, get run file name
        if key_type == 'fname':
            nframes = int(mo.group(2))
            fps = float(mo.group(5))
            outkey = P_TRIALFILE.sub(
                P_RUNFILE_SUB.format(
                    nframes=nframes,
                    fps=int(fps)
                ), fname)

        # If index key requested, get run index
        else:
            *_, irun, itrial = mo.groups()
            irun, itrial = int(irun), int(itrial)
            outkey = irun
        
        # Create run list if not already there
        if outkey not in fbyrun:
            fbyrun[outkey] = []
        
        # Add filepath to appropriate run list
        fbyrun[outkey].append(fname)
    
    # Sort trial files within each run
    fbyrun = {k: sorted(v) for k, v in fbyrun.items()}

    # If full paths provided, add them back to the output dictionary
    if is_fpath:
        fbyrun = {k: [os.path.join(pardir, fname) for fname in fl] for k, fl in fbyrun.items()}
    
    # Return
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


def find_suffixes(l):
    '''
    Extract suffixes from a list of names.
    
    :param l: list of names
    :return: list of suffixes
    '''
    # If full paths, extract basenames
    if any('/' in item for item in l):
        l = [os.path.basename(item) for item in l]
    # If extensions, remove them
    l = [os.path.splitext(item)[0] for item in l]
    # Find common prefix
    pref = os.path.commonprefix(l)
    # Find index of last underscore before suffixes
    iprefend = pref.rindex('_')
    # Return suffixes
    return [item[iprefend + 1:] for item in l]