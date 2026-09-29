# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2024-08-09 11:43:50
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-07-11 13:47:12

''' Utilities for Bruker data pre-processing '''

# External packages
import os
import glob
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from IPython.utils import io
from xml.dom import minidom
from datetime import datetime

# Internal modules
from .constants import *
from .logger import logger
from .fileops import get_subfolder_names, get_data_folders, get_dataset_params, restrict_datasets, save_acquisition_settings
from .parsers import resolve_mouseline, find_suffixes
from .stackers import stack_tifs
from .utils import itemize


# Generic naming pattern to extract acquisition number from folder/file name in Bruker 2026 experiments
BRUKER2026_NAMING_PATTERN = re.compile(f'([A-z0-9]+)_([A-z0-9]+)_([A-z0-9]+)-([0-9]+)*')  


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


def parse_bruker_frame(frame):
    ''' Parse Bruker Frame XML node '''
    d = {}
    for k, v in frame.attributes.items():
        if k == 'index':
            v = int(v)
        elif 'Time' in k:
            v = float(v)
        d[k] = v
    return d


def parse_bruker_sequence(seq):
    ''' Parse Bruker Sequence XML node '''

    # Parse sequence attributes
    attrs = {}
    for item in seq.attributes.values():
        k, v = item.nodeName, item.value
        if v in ('True', 'False'):
            v = bool(v)
        elif k == 'time':
            v = datetime.strptime(v[:15], '%H:%M:%S.%f').time()
        else:
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
        if k != 'type':
            attrs[k] = v

    # Parse children (especially frames)
    frames, voltage_output = [], None
    for child in seq.childNodes:
        if isinstance(child, minidom.Element):
            if child.nodeName == 'Frame':
                frames.append(parse_bruker_frame(child))
            elif child.nodeName == 'VoltageOutput':
                voltage_output = {}
                for k, v in child.attributes.items():
                    try:
                        v = float(v)
                    except ValueError:
                        pass
                    voltage_output[k] = v
            else:
                items = child.attributes.items()
                if len(items) > 0:
                    print(child.nodeName, items)
    frames = pd.DataFrame(frames).set_index('index')

    return attrs, frames, voltage_output



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
    :param
    :return: data aquisition settings dictionary
    '''
    # Parse XML file
    xmltree = minidom.parse(fpath)

    # Parse acquisition settings from relevant DOM nodes an info dictionary 
    PVStateShard = xmltree.getElementsByTagName('PVStateShard')[0]
    PVStateValues = PVStateShard.getElementsByTagName('PVStateValue')
    acq_settings = dict([parse_bruker_element(item) for item in PVStateValues])
    if simplify:
        acq_settings = simplify_bruker_settings(acq_settings)

    # Parse acquisition sequences data into sequence-indexed dataframes
    seqs = xmltree.getElementsByTagName('Sequence')
    seq_key = 'sequence'
    seq_info, seq_frames, seq_voutputs = list(zip(*[parse_bruker_sequence(seq) for seq in seqs]))
    seq_info = pd.DataFrame(seq_info)
    seq_info.index.name = seq_key
    seq_frames = pd.concat(seq_frames, axis=0, keys=np.arange(len(seq_frames)), names=[seq_key])
    seq_voutputs = pd.DataFrame({i: v for i, v in enumerate(seq_voutputs) if v is not None}).T
    seq_voutputs.index.name = seq_key

    # Return
    return acq_settings, seq_info, seq_frames, seq_voutputs


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
            parse_bruker_XML(get_bruker_XML(folder))[0])
    daq_settings = daq_settings.transpose()

    # Convert all possible settings to float
    for k in daq_settings:
        try:
            daq_settings[k] = pd.to_numeric(daq_settings[k])
        except (ValueError, TypeError) as e:
            logger.warning(f'failed to convert acquisition setting "{k}" to numeric type: {e}')

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
            maxreldev = MAX_LASER_POWER_REL_DEV if k == 'twophotonLaserPower' else MAX_DAQ_REL_DEV  # for input laser power: allow 15% dev
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
        # halfway through the experiments without any actual translation). Hence, 
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
    if not os.path.exists(datadir):
        raise ValueError(f'input data directory "{datadir}" does not exist')
    logger.info(f'processing Bruker input from "{datadir}"')

    # Get raw list of subolders containing tifs, sorted by run ID
    tif_folders = get_data_folders(
        datadir, 
        exclude_patterns=FOLDER_EXCLUDE_PATTERNS, 
        include_patterns=[resolve_mouseline(mouseline)], 
        sortby=Label.RUNID
    )

    if len(tif_folders) == 0:
        logger.warning(f'no tif folders found in "{datadir}" -> skipping pre-processing')
        return

    # Turn off TIF reading warning
    inkey, outkey = DataRoot.RAW_BRUKER, DataRoot.STACKED
    fpaths, ntrials, nptertrial = [], [], []
    with io.capture_output() as captured:  
        # Generate stacks for all TIF folders in the input data directory
        for tf in tif_folders:
            try:
                fpath, nt, npt = stack_tifs(
                    tf, inkey, outkey, overwrite=False, verbose=False, full_output=True)
                fpaths.append(fpath)
                ntrials.append(nt)
                nptertrial.append(npt)
            except ValueError as e:
                logger.error(f'failed to stack tifs in "{tf}": {e}')
                fpaths.append(None)
                ntrials.append(None)
                nptertrial.append(None)
                
    # Gather number of trials and number of frames per trial
    nframes_stats = pd.DataFrame(
        index=find_suffixes(tif_folders), 
        data={'ntrials': ntrials, 'nframes': nptertrial}
    )

    # Identify most common stats
    nframes_refstats = nframes_stats.mode().iloc[0]

    # Identify acquisitions that differ from reference stats
    is_match = nframes_stats.eq(nframes_refstats).all(axis=1)
    nframes_outliers = nframes_stats[~is_match].index.values.tolist()

    # Extract acquisition settings from Bruker XML files
    daq_settings = parse_bruker_acquisition_settings(tif_folders)

    # Add number of trials and number of frames per trials to acquisition settings
    daq_settings['nTrials'] = nframes_refstats['ntrials']
    daq_settings['nFramesPerTrial'] = nframes_refstats['nframes']

    # Add potential nframes outliers to pre-existing acquisition settings outliers
    if len(nframes_outliers) > 0:
        logger.warning(f'found {len(nframes_outliers)} outlier run(s) with frame stats significantly different from reference ({nframes_refstats}):\n{nframes_stats.loc[nframes_outliers]}')
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
    msg = f'found {len(datasets)} dataset(s) matching input parameters'
    if len(datasets) > 0:
        logger.info(msg + f':\n{pd.DataFrame(datasets).to_string(index=False)}')
    else:
        logger.warning(msg + ' -> skipping pre-processing')
    
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


def rename_files(folder_path, old_pattern, new_pattern, recursive=True, level=0, test=False, tab='  '):
    '''
    Rename files in a folder by replacing a specific pattern in their names with a new pattern.

    :param folder_path: path to the folder containing the files to rename
    :param old_pattern: pattern to be replaced in the file names
    :param new_pattern: new pattern to replace the old pattern in the file names
    :param recursive: whether to rename files in subfolders recursively (default: True)
    :param level: current recursion level (used for indentation in logging)
    :param test: if True, only print the old and new file names without renaming
    :param tab: string used for indentation in logging (default: two spaces)
    '''
    # Log
    folder_name = os.path.basename(folder_path)
    logger.info(f'replacing pattern "{old_pattern}" with "{new_pattern}" in folder "{folder_name}" contents')

    # Extract folder items
    items = sorted(os.listdir(folder_path))
    # If at the top level and not in test mode, wrap items in a tqdm progress bar
    if level == 0 and not test:
        items = tqdm(items)
    # Loop through each item in the folder
    for item in items:
        # Construct the full path to the item
        item_path = os.path.join(folder_path, item)
        
        # If the item is a directory and recursive renaming is enabled, 
        # call the function recursively on the subfolder
        if os.path.isdir(item_path) and recursive:
            print(f'{tab * level}found subfolder: {item_path}')
            rename_files(item_path, old_pattern, new_pattern, recursive=recursive, level=level+1, test=test)

        # Replace the old pattern with the new pattern in the item name
        new_file = item.replace(old_pattern, new_pattern)

        # If test mode, print the change
        if test:
            print(f'{tab * level}  old file: {item}')
            print(f'{tab * level}  new file: {new_file}')
        
        # Otherwise, rename the file
        else:
            os.rename(item_path, os.path.join(folder_path, new_file))

    # Log completion message
    logger.info('done.')


def rename_bruker_2026_tifs(datadir, mouseline, npertrial, fs, include_key='USNM', test=False, **kwargs):
    ''' 
    Rename Bruker 2026 TIF files to a more standard format, using stimulation log file as reference.
    
    :param datadir: path to directory containing Bruker 2026 experiment TIF files and stimulation log file.
    :param mouseline: mouse line
    :param npertrial: number of frames per trial
    :param fs: frame rate (Hz)
    :param include_key: string to identify files and folders pertaining to USNM acquisitions (default: 'USNM')
    :param test: if True, only print the changes without actually renaming the files
    :param kwargs: additional keyword arguments for filtering datasets (e.g., expdate, mouseid, region, layer)
    '''
    # Get raw list of subfolders containing tifs, with standard exclusion patterns
    # and specific inclusion pattern for the 2026 Bruker USNM acquisitions
    tif_folders = get_data_folders(
        datadir, 
        exclude_patterns=FOLDER_EXCLUDE_PATTERNS, 
        include_patterns=[include_key],
    )

    # Sort the list of folders to ensure consistent processing order by acquisition number
    tif_folders = sorted(tif_folders)
    logmsg = f'found {len(tif_folders)} tif folders in "{datadir}"'
    if len(tif_folders) > 0:
        logger.info(logmsg + f':\n{itemize(tif_folders)}')
    else:
        logger.warning(logmsg + ' -> skipping renaming')
        return

    # Extract log CSV file in the data directory
    csv_files = glob.glob(os.path.join(datadir, '*.csv'))
    if len(csv_files) != 1:
        raise ValueError(f'expected exactly one CSV file in "{datadir}", found {len(csv_files)}')
    log_fpath = csv_files[0]
    logger.info(f'found log file: {log_fpath}')

    # Load log file into a pandas DataFrame
    log_df = pd.read_csv(log_fpath)

    # Construct file codes based on mouse line and number of frames per trial
    prefix = f'{resolve_mouseline(mouseline)}_{npertrial}frames'
    log_df['filecode'] = ''
    for icond, cond in log_df.iterrows():
        Pcode = f'{cond[Label.P]:.2f}'.replace('.', '')
        fcode = f'{prefix}_{cond[Label.PRF]}Hz_{cond[Label.DUR_MS]}ms_{fs:.2f}Hz_{Pcode}MPa_{cond[Label.DC]}DC'
        log_df.at[icond, 'filecode'] = fcode

    logger.info(f'log data:\n{log_df.to_string()}')

    # Loop through each tif folder and corresponding log entry to rename files and folders
    for (folder_path, (iacq, new_pattern)) in zip(tif_folders, log_df['filecode'].items()):
        logger.info(f'acquisition run {iacq + 1} - tif folder: {folder_path}')
        folder_name = os.path.basename(folder_path)

        # Parse acquisition number from the folder name, and make sure it matches the log file acquisition number
        pre1, pre2, pre3, iacq_parsed = BRUKER2026_NAMING_PATTERN.match(folder_name).groups()
        if int(iacq_parsed) != iacq + 1:
            logger.warning(f'acquisition number mismatch: folder "{folder_name}" has acquisition number {iacq_parsed}, but log file has acquisition number {iacq + 1}')

        # Extract old pattern from folder name (everything except the last 4 characters, 
        # which are assumed to be the acquisition number)
        old_pattern = folder_name[:-4]

        # Call the rename_files function to rename files in the folder, 
        # replacing the old pattern with the new pattern
        rename_files(folder_path, old_pattern, new_pattern, test=test, **kwargs)

        # Rename the folder itself to reflect the new naming pattern
        pardir = os.path.join(folder_path, os.pardir)
        new_folder_name = folder_name.replace(old_pattern, new_pattern)
        if test:
            logger.info(f'  old folder: {folder_name}')
            logger.info(f'  new folder: {new_folder_name}')
        else:
            os.rename(folder_path, os.path.join(pardir, new_folder_name))

    logger.info(f'done renaming Bruker 2026 TIF files in "{datadir}"')



