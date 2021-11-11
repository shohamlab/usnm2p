# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-13 11:13:26
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-11-11 14:22:19

''' Collection of constants used throughout the code base. '''

import numpy as np
import seaborn as sns
from string import ascii_lowercase

# Miscellaneous
UNKNOWN = '???'  # unknown key
IND_LETTERS = list(ascii_lowercase[8:])  # generic index letters
TAB10 = sns.color_palette('tab10')  # default color palette

# Stimulation
DC_REF = 50.  # reference duty cycle value (in %) used to perform pressure amplitude sweeps
P_REF = .8  # reference pressure amplitude (in MPa) used to perform DC sweeps

# Acquisition
REF_NFRAMES = 1600  # reference number of frames in any given experimental run (used to check integrity of input stacks)
NFRAMES_PER_TRIAL = 100  # default number of frames per trial
STIM_FRAME_INDEX = 10  # index of the frame coinciding with the US stimulus in each trial

# Suite2p
REWRITTEN_S2P_KEYS = {'fast_disk', 'save_path0', 'save_folder', 'bidi_corrected', 'block_size'}  # set of suite2p options ckeys tha are rewritten upon suite2p processing
TAU_GCAMP6S_DECAY = 1.25  # exponential decay time constant for Calcium transients with GCaMP6s (s) 

# Post-processing
IS_VALID_KEY = 'is_valid' # key used to acces validity status of a given cell 
DFF_OUTLIER = 0.3  # upper bound threshold for dF/F0 (cells with absolute traces above this threshold get discarded). 0.8 works well for line3
I_RESPONSE = slice(STIM_FRAME_INDEX, STIM_FRAME_INDEX + 8)  # indexes used for response computation per trial. Considers the following 7 frames (i.e. ca. 2 seconds) following the stimulus onset.
ZSCORE_THR = 1.64  # threshold absolute z-score value
SUCCESS_RATE_THR = .3  # threshold success rate for a positive response
NPOS_CONDS_THR = 5  # threshold number of positive conditions for an ROI to be classified as positive responder

# SI units prefixes
SI_POWERS = {
    'y': -24,  # yocto
    'z': -21,  # zepto
    'a': -18,  # atto
    'f': -15,  # femto
    'p': -12,  # pico
    'n': -9,   # nano
    'u': -6,   # micro
    'm': -3,   # mili
    '': 0,     # None
    'k': 3,    # kilo
    'M': 6,    # mega
    'G': 9,    # giga
    'T': 12,   # tera
    'P': 15,   # peta
    'E': 18,   # exa
    'Z': 21,   # zetta
    'Y': 24,   # yotta
}

# Naming patterns for input folders and files 
P_LINE = '([A-z][A-z0-9]*)'
P_TRIAL_LENGTH = '([0-9]+)frames'
P_FREQ = '([0-9]+[.]?[0-9]*)Hz'
P_DUR = '([0-9]+[.]?[0-9]*)ms'
P_MPA = '([0-9]+[.]?[0-9]*)MPA'
P_DC = '([0-9]+)DC'
P_RUN = '([0-9]+)'
P_CYCLE = 'Cycle([0-9]+)'
P_CHANNEL = 'Ch([0-9])'
P_FRAME = '([0-9]+)'

# Labels for input stimulation/acquisition parameters
ROI_LABEL = 'ROI'
RUN_LABEL = 'run'
TRIAL_LABEL = 'trial'
P_LABEL = 'P (MPa)'
DC_LABEL = 'DC (%)'
DUR_LABEL = 'duration (s)'
FPS_LABEL = 'fps'
RUNID_LABEL = 'run ID'
LINE_LABEL = 'line'
NPERTRIAL_LABEL = 'trial_length'
NTRIALS_LABEL = 'ntrials'
CYCLE_LABEL = 'cycle'
FRAME_LABEL = 'frame'
CH_LABEL = 'channel'
TIME_LABEL = 'time (s)'
RESP_LABEL = 'response type'
F_ROI_LABEL = 'F_ROI (a.u.)'
F_NEU_LABEL = 'F_neu (a.u.)'
F_LABEL = 'F (a.u.)'
F0_LABEL = 'F0 (a.u.)'
DFF_LABEL = 'dF/F0'
STACK_AVG_INT_LABEL = 'Iavg (a.u.)'

# Plotting
NPOS_CONDS_LABEL = '# positive conditions'
LABEL_BY_TYPE = {-1: 'negative', 0: 'neutral', 1: 'positive'}  # mapping of response labels to specific integer codes
RGB_BY_TYPE = {-1: TAB10[1], 0: TAB10[7], 1: TAB10[2]}  # mapping of RGB colors to specific integer codes
CI = 95  # default confidence interval for bootstrapping