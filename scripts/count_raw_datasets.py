# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2023-01-31 10:20:34
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2024-08-07 17:18:14

import os
import pandas as pd

from usnm2p.config import dataroot

''' Get a detailed count of number of datasets per analysis type and mouse line '''

def get_subdirs(dir):
    ''' Function that returns the list of sub-directories names for a directory '''
    return [x for x in os.listdir(dir) if os.path.isdir(os.path.join(dir, x))]


if __name__ == '__main__':

    # Initialize count dictionary
    counts = {}

    # For each data directory
    for datadir in get_subdirs(dataroot):
        # Skip non-raw directories
        if not datadir.startswith('raw'):
            continue
        raw_dirpath = os.path.join(dataroot, datadir)

        # For each analysis type
        for analysis_type in get_subdirs(raw_dirpath):
            analysis_dirpath = os.path.join(raw_dirpath, analysis_type)

            # For each mouse line
            for line in get_subdirs(analysis_dirpath):
                # Count number of datasets
                line_dirpath = os.path.join(analysis_dirpath, line)
                ndatasets = len(get_subdirs(line_dirpath))
            
                # Add to count dictionary with appropriate code 
                counts[f'{analysis_type}_{line}'] = ndatasets

    # Transform to pandas Series 
    counts = pd.Series(counts, name='counts')

    # Print output
    print(f'detailed datasets count:\n{counts}')
    print(f'TOTAL: {counts.sum()}')