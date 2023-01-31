# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2023-01-31 10:20:34
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-01-31 10:36:23

import os
import pandas as pd
from config import dataroot

''' Get a detailed count of number of datasets per analysis type and mouse line '''

def get_subdirs(dir):
    ''' Function that returns the list of sub-directories names for a directory '''
    return [x for x in os.listdir(dir) if os.path.isdir(os.path.join(dir, x))]

# Initialize count dictionary
counts = {}
# For each analysis type
for analysis_type in get_subdirs(dataroot):
    # For each mouse line
    for line in get_subdirs(os.path.join(dataroot, analysis_type)):
        # Count number of datasets
        ndatasets = len(get_subdirs(os.path.join(dataroot, analysis_type, line)))
        # Add to count dictionary with appropriate code 
        counts[f'{analysis_type}_{line}'] = ndatasets

# Transform to pandas Series 
counts = pd.Series(counts, name='counts')

# Print output
print(f'detailed datasets count:\n{counts}')