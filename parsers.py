# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-14 19:29:19
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-05-28 17:29:52

''' Collection of parsing utilities. '''

from xml.dom import minidom
import re
import os
import pandas as pd
from constants import *

# General tif file pattern
P_TIFFILE = re.compile('.*tif')
P_DATEMOUSEREGLAYER = re.compile(
    f'{Pattern.DATE}_({Pattern.MOUSE})_({Pattern.REGION})_?({Pattern.LAYER})?')
P_RAWFOLDER = re.compile(f'^{Pattern.LINE}_{Pattern.TRIAL_LENGTH}_{Pattern.FREQ}_{Pattern.DUR}_{Pattern.FREQ}_{Pattern.MPA}_{Pattern.DC}-{Pattern.RUN}$', re.IGNORECASE)
P_RAWFILE = re.compile(f'{P_RAWFOLDER.pattern[:-1]}_{Pattern.CYCLE}_{Pattern.CHANNEL}_{Pattern.FRAME}.ome.tif$', re.IGNORECASE)
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
        Label.LINE: mo.group(1),  # line name
        Label.NPERTRIAL: int(mo.group(2)),
        Label.UNKNOWN: float(mo.group(3)),
        Label.DUR: float(mo.group(4)) * 1e-3,  # s
        Label.FPS: float(mo.group(5)),  
        Label.P: mo.group(6),  # MPa
        Label.DC: float(mo.group(7)),  # %
        Label.RUNID: int(mo.group(8))
    }
    # Fix for pressure (replacing first zero by decimal dot)
    if '.' not in params[Label.P]:
        params[Label.P] = float(f'.{params[Label.P][1:]}')
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
    info_table['code'] = [os.path.splitext(x)[0] for x in basenames]
    if index_key is not None:
        info_table.index.name = index_key
    if ntrials_per_run is not None:
        info_table[Label.NTRIALS] = ntrials_per_run
    if discard_unknown:
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
    return settings
    

def parse_acquisition_settings(folders):
    '''
    Extract data acquisition settings from raw data folders.
    
    :param folders: full list of data folders containing the raw TIF files.
    :return: dictionary containing data aquisition settings that are common across all data folders 
    '''
    # Parse aquisition settings of each data folder 
    daq_settings_list = [parse_Bruker_XML(get_Bruker_XML(folder)) for folder in folders]
    # Extract reference settings
    ref_settings, *other_settings_list = daq_settings_list
    # Check that aquisition settings keys are identical across data folders
    assert all(x.keys() == ref_settings.keys() for x in daq_settings_list), 'inconsistent settings list'
    # Gather keys of settings fields that vary across folders
    diffkeys = []
    for settings in other_settings_list:
        if settings != ref_settings:
            for k in settings.keys():
                if settings[k] != ref_settings[k]:
                    if k not in diffkeys:
                        diffkeys.append(k)
    # Remove those fields from reference settings dictionary
    for k in diffkeys:
        del ref_settings[k]
    # Return common aquisition settings dictionary
    return ref_settings


def parse_date_mouse_region(s):
    '''
    Parse a date, mouse and region from a concatenated string
    
    :param s: concatenated string
    :return: 3-tuple with (date, mouse, region)
    '''
    mo = P_DATEMOUSEREGLAYER.match(s)
    if mo:
        print(mo.groups())
        year, month, day, mouse, region, layer = mo.groups()
        date = f'{year}{month}{day}'
        if layer is None:
            layer = DEFAULT_LAYER
        return date, mouse, region, layer
    else:
        raise ValueError(
            f'{s} does not match date-mouse-reg-layer pattern ({P_DATEMOUSEREGLAYER.pattern})')