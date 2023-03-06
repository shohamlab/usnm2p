# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2023-03-03 16:33:20
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-03-06 18:24:14

''' Legacy list of excluded datasets (along with justification criteria) per mouse line '''

main_excludes = {
    'line3': [
        # Poor segmentation results
        '20191107_mouse1_region2',
        '20191023_mouse10_region1',
        '20191228_mouse13_region1',
        '20191230_mouse13_region1',
        '20191113_mouse9_region1',  # potential dead cells
        '20191113_mouse9_region2',  # potential dead cells
        '20191023_mouse3_region1',  # only 3 conditions

        # No (or very weak) response observed across conditions
        '20190829_mouse10_region1',
        '20191022_mouse1_region1',
        '20191107_mouse1_region1',

        # Abnormal response drop from 50% to 60% DC
        '20190807_mouse5_region1',
        '20191108_mouse6_region1',
        '20191110_mouse6_region1_layer5',        
        '20200313_mouse14_region1',

        # Abnormal response drop from 0.4 to 0.6 MPa
        '20191109_mouse7_region1',
        '20191108_mouse6_region1',

        # Large relative variation in fluorescence baseline (>25%) throughout experiment
        '20191109_mouse7_region3_layer5',
    ],
    'sarah_line3': [
        
    ],
    'sst': [
        # No (or very weak) response observed across conditions
        '20190516_mouse8_region1',
        '20190516_mouse8_region2',
        '20190808_mouse6_region1',  # huge motion artefacts in initial run 

        # Large relative variation in fluorescence baseline (>25%) throughout experiment
        '20190706_mouse7_region1',  # also huge response peak at 0.1MPa

        # Abnormally strong response profiles
        '20190511_mouse7_region1',
        '20190513_mouse2_region1',
    ],
    'pv': [
        # '20190606_mouse2_region1',  # incomplete (no high DC data)
        # '20190606_mouse2_region2',  # incomplete (no high DC data), negative dips for positive responders
        # '20190821_mouse3_region1',  # noisy DC dependencies
        # # '20190702_mouse5_region1',  # super negative dip for P = 0.4 MPa
        # # '20190821_mouse6_region1',  # strong activations for low pressures, very noisy dFF profiles
        # # '20190630_mouse3_region1',  # super strong outlier at P = 0.4 MPa
        # # '20190821_mouse7_region1',  # super oscillatory profile
    ]
}