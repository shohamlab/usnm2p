# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-14 19:29:19
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2024-08-07 12:21:25

''' Collection of parsing utilities. '''

from xml.dom import minidom
from tqdm import tqdm
import re
import os
import numpy as np
import pandas as pd
from argparse import ArgumentTypeError
from constants import *
from logger import logger
from utils import itemize

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


def parse_Bruker_element(elem):
    '''
    Parse an element from the Bruker XML tree.
    
    :param elem: XML DOM element
    :return: key, value pair representing the element
    '''
    # Fetch element type 
    elem_type = elem.nodeName
    # Determine element key type from element type
    key_key = {
        'PVStateValue': 'key',
        'IndexedValue': 'index',
        'SubindexedValues': 'index',
        'SubindexedValue': 'subindex'
    }[elem_type]
    # Fetch element key
    key = elem.attributes[key_key].value
    # Try to fetch element value
    try:
        val = elem.attributes['value'].value
        # If present, try to convert it to a float
        try:
            val = float(val)
        except:
            pass
        # Otherwise, convert to a boolean if applicable
        if val == 'True':
            val = True
        elif val == 'False':
            val = False
        # Fetch element description (if any) and combine it with its value
        try:
            desc = elem.attributes['description'].value
            val = (val, desc)
        except:
            pass
    # If element has no "value", then it must have children -> loop through them
    except KeyError:
        # Define values dictionary
        val = {}

        # Parse each child element and populate values dictionary
        for child in elem.childNodes:
            if isinstance(child, minidom.Element):
                k, v = parse_Bruker_element(child)
                val[k] = v

        # Post-processing
        if all(k.isnumeric() for k in val.keys()) and [int(k) for k in val.keys()] == list(range(len(val))):
            # If all keys are integers and form a range -> transform dict to list
            val = list(val.values())
            # If resulting list has only 1 element -> reduce it to its element
            if len(val) == 1:
                val = val[0]
            # If resulting list is made only of tuples -> recast as a dictionary
            elif all(isinstance(x, tuple) for x in val):
                val = {x[1]: x[0] for x in val}

        # If dictionary only has 1 key-value pair -> transform to tuple
        if isinstance(val, dict) and len(val) == 1:
            val = list(val.items())[0]

    # Return (key, val) pair
    return key, val


def get_Bruker_XML(folder):
    '''
    Get the path to a Bruker XML settings file associated to a specific data folder
    
    :param fodler: path to the folder containing the raw TIF files
    :return: full path to the corresponding Bruker XML file 
    '''
    xml_fname = f'{os.path.basename(folder)}.xml'
    return os.path.join(folder, xml_fname)


def parse_Bruker_XML(fpath, simplify=True):
    '''
    Extract data aquisition settings from Bruker XML file.
    
    :param fpath: full path to the XML file
    :return: data aquisition settings dictionary
    '''
    # Parse XML file
    xmltree = minidom.parse(fpath)
    # Extract relevant nodes from DOM
    PVStateShard = xmltree.getElementsByTagName('PVStateShard')[0]
    PVStateValues = PVStateShard.getElementsByTagName('PVStateValue')
    # Parse all these nodes into a info dictionary 
    settings = dict([parse_Bruker_element(item) for item in PVStateValues])
    if simplify:
        settings = simplify_Bruker_settings(settings)
    return settings


def simplify_Bruker_settings(settings):
    ''' Simplify dictionary of Bruker settings '''
    # Check spatial resolution uniformity across X and Y axes and simplify corresponding field
    mpp = settings['micronsPerPixel']
    del mpp['ZAxis']
    assert mpp['XAxis'] == mpp['YAxis'], f'differing spatial resolution across axes: {mpp}'
    settings['micronsPerPixel'] = mpp['XAxis']
    
    # Cast DAQ gain and pre-amp filter to float (if present)
    if 'daq' in settings:
        settings['DAQ gain'] = float(settings.pop('daq')[0][:-1])
    preampfilter = settings.pop('preampFilter')[1]
    filtval, filtunit = preampfilter.split(' ')
    filtfactor = np.power(10, SI_POWERS[filtunit.replace('Hz', '')])
    filtval = float(filtval) * filtfactor
    settings['preampFilter (MHz)'] = filtval * 1e-6

    # Curate Pockels power
    settings['Pockels power'] = settings.pop('laserPower')[0]

    # Simplify complex settings
    curated_settings = {}
    for k, v in settings.items():
        # Dictionary: expand
        if isinstance(v, dict):
            for kk, vv in v.items():
                curated_settings[f'{k} {kk}'] = vv
        # Tuple: only keep first item
        elif isinstance(v, tuple):
            curated_settings[f'{k}'] = v[0]
        # Normal fields: propagate
        else:
            curated_settings[k] = v

    # Return
    return curated_settings
    

def parse_acquisition_settings(folders):
    '''
    Extract data acquisition settings from raw data folders.
    
    :param folders: full list of data folders containing the raw TIF files.
    :return: 2-tuples with:
        - dictionary containing data aquisition settings that are common across all data folders 
        - list of folders for which acquisition settings vary significantly from reference
    '''
    logger.info(f'extracting acquisition settings across {len(folders)} folders...')

    # Identify common prefix across folders
    pref = os.path.commonprefix(folders)

    # Extract index of prefix end
    iprefend = pref.rindex('_')

    # Assemble dictionary of folder suffix: folder path pairs
    fdict = {folder[iprefend + 1:]: folder for folder in folders}

    # Parse aquisition settings of each data folder into common dataframe
    daq_settings = pd.DataFrame()
    for folder in tqdm(folders):
        fkey = folder[iprefend + 1:]
        daq_settings[fkey] = pd.Series(
            parse_Bruker_XML(get_Bruker_XML(folder)))
    daq_settings = daq_settings.transpose()

    # Convert all possible settings to float
    for k in daq_settings:
        try:
            daq_settings[k] = pd.to_numeric(daq_settings[k])
        except ValueError:
            pass

    # Identify reference (i.e., most common) value across runs, for each setting
    logger.info(f'identifying reference value for {daq_settings.shape[1]} settings')
    ref_daq_settings = (
        daq_settings
        .mode(axis=0)
        .iloc[0, :]
        .rename('settings')
    )

    # Initialize empty lists of outlier runs and "truly differing" settings
    outliers = []
    diff_settings = []

    logger.info('checking for settings consistency across folders...')
    
    # Identify mismatches across runs for each setting
    ismatch = daq_settings.eq(ref_daq_settings, axis=1).all(axis=0)
    nonmatching_settings = ismatch[~ismatch].index.values

    # For each differing setting
    for k in nonmatching_settings:
        # Extract values across runs, and reference value
        vals = daq_settings[k]
        ref_val = ref_daq_settings[k]

        # If numeric setting and not XYZ position
        if vals.dtype == 'float64' and 'positionCurrent' not in k:
            # Compute absolute relative deviations from reference value
            rel_devs = ((vals - ref_val) / ref_val).abs()
            logger.debug(f'absolute relative deviations from {k} reference:\n{rel_devs}')

            # Identify runs with significant relative deviations
            maxreldev = MAX_LASER_POWER_REL_DEV if k == 'twophotonLaserPower' else MAX_DAQ_REL_DEV  # for input laser power: allow 5% dev
            current_outliers = rel_devs[rel_devs > maxreldev].index.values.tolist()
        
        # Otherwise
        else:
            # Identify runs that differ from ref value 
            current_outliers = vals[vals != ref_val].index.values.tolist()

        # If outliers were detected, store differing setting
        if len(current_outliers) > 0:
            diff_settings.append(k)

        # If not XYZ position, add outliers runs to global list. The reason for
        # this exception is that DAQ position changes are not necessarily related
        # to changes in physical location (e.g. the position vector can be "zeroed" 
        # halfway through the expeirments without any actual translation). Hence, 
        # no automatic checks are performed for position vectors, but movies and
        # registration metrics should be inspected to check for potential drifts.
        # A warning message will still be issued though.
        if 'positionCurrent' not in k:
            outliers += current_outliers

    # If truly differing settings were found, log warning message
    if len(diff_settings) > 0:
        logger.warning(
            f'found {len(diff_settings)} acquisition setting(s) varying across runs:\n{daq_settings[diff_settings]}')
    
    # Extract folders for each run in outliers list
    outliers = [fdict[k] for k in list(set(outliers))]

    # Return common settings and list of outliers folders
    return ref_daq_settings, outliers


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
