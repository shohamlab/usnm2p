# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-14 19:29:19
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-10-14 19:30:28

from constants import *

''' Collection of parsing utilities. '''

EXPCODE = re.compile(
    f'^{P_LINE}_{P_TRIAL_LENGTH}_{P_FREQ}_{P_DUR}_{P_FREQ}_{P_MPA}_{P_DC}-{P_RUN}$',
    re.IGNORECASE)


def parse_experiment_parameters(name):
    '''
    Parse experiment parameters from a file name.
    
    :param name: file / folder name from which parameters must be extracted
    :return: dictionary of extracted parameters
    '''
    mo = EXPCODE.match(name)
    if mo is None:
        raise ValueError(f'"{name}" does not match the experiment naming pattern')
    return {
        'line': mo.group(1),
        'trial_length': int(mo.group(2)),
        '???': float(mo.group(3)),
        'duration': float(mo.group(4)),
        'fps': float(mo.group(5)),
        'P': float(mo.group(6)),
        'DC': float(mo.group(7)),
        'runID': int(mo.group(8))
    }
