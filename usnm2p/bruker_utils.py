# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2024-08-09 11:43:50
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2024-08-09 15:16:59

''' Utilities for Bruker data pre-processing '''

# External packages
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from IPython.utils import io
from xml.dom import minidom

# Internal modules
from usnm2p.constants import *
from usnm2p.logger import logger
from usnm2p.fileops import get_subfolder_names, get_data_folders, get_dataset_params, restrict_datasets, save_acquisition_settings
from usnm2p.parsers import resolve_mouseline, find_suffixes
from usnm2p.stackers import stack_tifs


def parse_bruker_element(elem):
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
                k, v = parse_bruker_element(child)
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


def get_bruker_XML(folder):
    '''
    Get the path to a Bruker XML settings file associated to a specific data folder
    
    :param fodler: path to the folder containing the raw TIF files
    :return: full path to the corresponding Bruker XML file 
    '''
    xml_fname = f'{os.path.basename(folder)}.xml'
    return os.path.join(folder, xml_fname)


def parse_bruker_XML(fpath, simplify=True):
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
    settings = dict([parse_bruker_element(item) for item in PVStateValues])
    if simplify:
        settings = simplify_bruker_settings(settings)
    return settings


def simplify_bruker_settings(settings):
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
    

def parse_bruker_acquisition_settings(folders):
    '''
    Extract data acquisition settings from Bruker raw data folders.
    
    :param folders: full list of data folders containing the raw TIF files.
    :return: pandas Series containing data aquisition settings that are common across all data folders. 
        An additional "outliers" field references the folders that have acquisitions settings that 
        vary significantly from the reference.
    '''
    logger.info(f'extracting acquisition settings across {len(folders)} folders...')

    # Identify unique suffix per folder
    fsuffixes = find_suffixes(folders)

    # Parse aquisition settings of each data folder into common dataframe
    daq_settings = pd.DataFrame()
    for fkey, folder in zip(fsuffixes, tqdm(folders)):
        daq_settings[fkey] = pd.Series(
            parse_bruker_XML(get_bruker_XML(folder)))
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
        .mode(axis=0)  # most common value across runs
        .iloc[0, :]  # first row
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
    
    # Add potential outliers to output series
    if len(outliers) > 0:
        logger.warning(
            f'found {len(outliers)} outlier run(s) with significantly different acquisition settings')
    ref_daq_settings['outliers'] = outliers

    # Return acquisition settings
    return ref_daq_settings


def preprocess_bruker_dataset(dataroot, analysis, mouseline, expdate, mouseid, region, layer):
    '''
    Pre-process input dataset from Bruker system, i.e.:
        - identify "acquisition" subfolders (i.e., those containing sequences of single-frame TIFs)
        - generate and save multi-frame TIF stacks for all subfolders
        - extract acquisition settings from Bruker XML file for each subfolder
        - save acquisition settings as JSON file in the output stacks directory
    
    :param dataroot: root directory for Bruker input data
    :param analysis: analysis type
    :param mouseline: mouse line
    :param expdate: experiment date
    :param mouseid: mouse ID
    :param region: imaged region
    :param layer: imaged cortical layer
    '''
    # Construct dataset ID
    dataset_id = f'{expdate}_{mouseid}_{region}'
    if layer != DEFAULT_LAYER:
        dataset_id = f'{dataset_id}_{layer}'

    # Extract input data directory
    datadir = os.path.join(dataroot, analysis, mouseline, dataset_id)
    if not os.path.exists:
        raise ValueError(f'input data directory "{datadir}" does not exist')
    logger.info(f'processing Bruker input from "{datadir}"')

    # Get raw list of subolders containing tifs, sorted by run ID
    tif_folders = get_data_folders(
        datadir, 
        exclude_patterns=FOLDER_EXCLUDE_PATTERNS, 
        include_patterns=[resolve_mouseline(mouseline)], 
        sortby=Label.RUNID
    )

    # Turn off TIF reading warning
    inkey, outkey = DataRoot.RAW_BRUKER, DataRoot.PREPROCESSED
    fpaths, nframes = [], []
    with io.capture_output() as captured:  
        # Generate stacks for all TIF folders in the input data directory
        for tf in tif_folders:
            fpath, shape = stack_tifs(
                tf, inkey, outkey, overwrite=False, verbose=False, full_output=True)
            fpaths.append(fpath)
            nframes.append(shape[0])

    # Gather number of frames per acquisition
    nframes = pd.Series(index=find_suffixes(tif_folders), data=nframes, name='nframes')

    # Identify most common number of frames per acquisition
    nframes_ref = nframes.mode().iloc[0]

    # Identify acquisitions that differ from reference number of frames
    nframes_outliers = nframes[nframes != nframes_ref].index.values.tolist()

    # Extract acquisition settings from Bruker XML files
    daq_settings = parse_bruker_acquisition_settings(tif_folders)

    # Add number of frames per acquisition to acquisition settings
    daq_settings['nFramesPerAcq'] = nframes_ref

    # Add potential nframes outliers to pre-existing acquisition settings outliers
    if len(nframes_outliers) > 0:
        daq_settings['outliers'] = list(set(daq_settings['outliers'] + nframes_outliers))
    
    # Save acquisition settings as JSON file in the output stacks directory
    save_acquisition_settings(os.path.dirname(fpaths[0]), daq_settings)


def preprocess_bruker_datasets(dataroot, analysis=None, **kwargs):
    '''
    Pre-process a set of input datasets from Bruker system

    :param dataroot: root directory for Bruker input data
    :param analysis (optional): analysis type
    '''
    # If analysis type not specified, extract analysis subfolders from input data directory
    # and call function recursively for each analysis type
    if analysis is None:
        for analysis in get_subfolder_names(dataroot):
            logger.info(f'processing Bruker inputs for "{analysis}" analysis')
            preprocess_bruker_datasets(dataroot, analysis=analysis, **kwargs)
        return

    # List datasets in input data directory, filtered by analysis type
    datasets = get_dataset_params(root=dataroot, analysis=analysis)
    logger.info(f'found {len(datasets)} datasets for "{analysis}" analysis')

    # Filter datasets to match related input parameters
    datasets = restrict_datasets(datasets, **kwargs)
    logger.info(f'found {len(datasets)} dataset(s) matching input parameters')
    
    # Process each dataset
    for dataset in datasets:
        preprocess_bruker_dataset(
            dataroot,
            analysis, 
            dataset['mouseline'], 
            dataset['expdate'], 
            dataset['mouseid'], 
            dataset['region'], 
            dataset['layer']
        )