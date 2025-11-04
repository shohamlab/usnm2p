# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2025-11-04 16:02:55
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-11-04 16:03:18
from usnm2p.bergamo_utils import stack_singleframe_tifs_across_trials

input_dir = '/gpfs/home/lemait01/scratch/data/usnm/raw_bergamo/main/cre_vip/20200309_mouse214_region1/tif'
stack_singleframe_tifs_across_trials(input_dir)