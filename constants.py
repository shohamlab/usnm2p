# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-13 11:13:26
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-11-17 16:55:27

''' Collection of constants used throughout the code base. '''

import numpy as np
import seaborn as sns
from string import ascii_lowercase

###################################### MISCELLANEOUS ######################################

IND_LETTERS = list(ascii_lowercase[8:])  # generic index letters

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


###################################### DATA ACQUISITION ######################################

REF_NFRAMES = 1600  # reference number of frames in any given experimental run (used to check integrity of input stacks)
NFRAMES_PER_TRIAL = 100  # default number of frames per trial
STIM_FRAME_INDEX = 10  # index of the frame coinciding with the US stimulus in each trial
DC_REF = 50.  # reference duty cycle value (in %) used to perform pressure amplitude sweeps
P_REF = .8  # reference pressure amplitude (in MPa) used to perform DC sweeps


###################################### SUITE2P ######################################

REWRITTEN_S2P_KEYS = {  # suite2p options keys that are rewritten upon suite2p processing
    'fast_disk',
    'save_path0',
    'save_folder',
    'bidi_corrected',
    'block_size'}  
TAU_GCAMP6S_DECAY = 1.25  # GCaMP6s exponential decay time constant (s) 


###################################### POST-PROCESSING ######################################

# neuropil subtraction factor (0-1):
# - small values (around 0) tend to produce smoother dF/F0 traces with less amplitude,
# - large values (around 1) tend to produce higher amplitude dF/F0 traces with more fluctuations
# - default is 0.7 (from literature), for PV 0.5 works better, for SST 0.6. But lately it seems that is ok with 0.7 for all regardless of the line 
ALPHA = .7

# Baseline computation
BASELINE_WLEN = 30.  # window length (in s) to compute the fluorescence baseline
BASELINE_QUANTILE = .05  # quantile used for the computation of the fluorescence baseline
BASELINE_RSD_THR = .5  # threshold for relative standard deviation of the fluorescence baseline across runs

# dFF noise level computation
DFF_NOISE_WLEN = 60.  # window length (in s) to compute the dFF noise level
DFF_NOISE_QUANTILE = .5  # quantile used for the computation of the dFF noise level

I_RESPONSE = slice(STIM_FRAME_INDEX, STIM_FRAME_INDEX + 10)  # indexes used for response computation per trial.
N_NEIGHBORS_PEAK = 1  # number of neighboring elements to consider to compute "averaged" peak value  
ZSCORE_THR = 1.64  # threshold absolute z-score value
SUCCESS_RATE_THR = .3  # threshold success rate for a positive response
NPOS_CONDS_THR = 5  # threshold number of positive conditions for an ROI to be classified as positive responder


###################################### PARSING ######################################

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


###################################### LABELS ######################################

# Acquisition
UNKNOWN = '???'  # unknown key
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

# Data indexes
ROI_LABEL = 'ROI'
RUN_LABEL = 'run'
TRIAL_LABEL = 'trial'

# Fluorescence signals
TIME_LABEL = 'time (s)'
F_ROI_LABEL = 'F_ROI (a.u.)'
F_NEU_LABEL = 'F_neu (a.u.)'
F_LABEL = 'F (a.u.)'
F0_LABEL = 'F0 (a.u.)'
DFF_LABEL = 'dF/F0'

# Stats
DFF_NOISE_LABEL = 'dFF noise'
ZSCORE_LABEL = 'z-score'
PEAK_ZSCORE_LABEL = 'peak z-score'
IS_RESP_LABEL = 'trial response?'
SUCCESS_RATE_LABEL = 'success rate'
CORRECTED_ZSCORE_LABEL = 'corrected z-score'
CORRECTED_PEAK_ZSCORE_LABEL = 'corrected peak z-score'
IS_POSITIVE_RUN_LABEL = 'positive run?'
NPOS_RUNS_LABEL = '# positive runs'
ROI_RESP_TYPE_LABEL = 'response type'


###################################### PLOTTING ######################################

LABEL_BY_TYPE = {-1: 'negative', 0: 'neutral', 1: 'positive'} 
TAB10 = sns.color_palette('tab10')  # default color palette
RGB_BY_TYPE = {'negative': TAB10[1], 'neutral': TAB10[7], 'positive': TAB10[2]}  # mapping of RGB colors to response types
CI = 95  # default confidence interval for bootstrapping