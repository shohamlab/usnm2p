# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-10-06 14:59:30
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-10-06 16:52:25

import os
import shutil
from config import dataroot
from logger import logger


def get_subfolders(dir):
    ''' Get absolute path to all subfolders in directory '''
    return [f.path for f in os.scandir(dir) if f.is_dir()]

# Filtered data root
filtroot = dataroot.replace('raw', 'filtered')
segroot = filtroot.replace('filtered', 'segmented')

# Loop through subfolders until s2p output data folders
for ftype_fpath in get_subfolders(filtroot):
    for stype_fpath in get_subfolders(ftype_fpath):
        for atype_fpath in get_subfolders(stype_fpath):
            for line_fpath in get_subfolders(atype_fpath):
                for dataset_fpath in get_subfolders(line_fpath):
                    for s2p_fpath in get_subfolders(dataset_fpath):
                        # Path to source s2p data folder
                        s2pdata_fpath = os.path.join(s2p_fpath, 'plane0')
                        # Assemble subparts of path
                        subparts = [os.path.split(x)[1] for x in [
                            ftype_fpath, stype_fpath, atype_fpath, line_fpath, dataset_fpath
                        ]]
                        # Constitute path to destination folder
                        s2pfolder = os.path.split(s2p_fpath)[1]
                        dest_parts = [segroot, s2pfolder] + subparts
                        dest_fpath = os.path.join(*dest_parts)
                        # Make sure parent destination folder exists (create it if needed) 
                        dest_par = os.path.dirname(dest_fpath)
                        if not os.path.exists(dest_par):
                            os.makedirs(dest_par)
                        # If destination folder does not already exist
                        if not os.path.exists(dest_fpath):
                            # Copy s2p data folder to destination folder
                            logger.info(
                                f'moving contents of\n"{s2pdata_fpath}"to\n"{dest_fpath}"')
                            shutil.copytree(s2pdata_fpath, dest_fpath)
                            # Remove origin folder
                            shutil.rmtree(s2p_fpath)
                        else:
                            logger.warning(f'"{dest_fpath}" already exists -> skipping')
