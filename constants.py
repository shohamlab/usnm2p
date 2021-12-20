# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-13 11:13:26
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-12-20 11:40:47

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
DC_REF = 50.  # reference duty cycle value (in %) used to perform pressure amplitude sweeps
P_REF = .8  # reference pressure amplitude (in MPa) used to perform DC sweeps


###################################### SUITE2P ######################################

S2P_UINT16_NORM_FACTOR = 2  # normalization factor applied by suite2p to input uint16 TIF files
REWRITTEN_S2P_KEYS = {  # suite2p options keys that are rewritten upon suite2p processing
    'fast_disk',
    'save_path0',
    'save_folder',
    'bidi_corrected',
    'block_size'}  
TAU_GCAMP6S_DECAY = 1.25  # GCaMP6s exponential decay time constant (s) 


###################################### POST-PROCESSING ######################################

NPIX_RATIO_THR = None  # threshold (# pixels ROI) / (# pixels soma) ratio (cells above that ratio get discarded) 

# neuropil subtraction factor (0-1):
# - small values (around 0) tend to produce smoother dF/F0 traces with less amplitude,
# - large values (around 1) tend to produce higher amplitude dF/F0 traces with more fluctuations
# - default is 0.7 (from literature), for PV 0.5 works better, for SST 0.6. But lately it seems that is ok with 0.7 for all regardless of the line 
ALPHA = 0.7

# Baseline computation
BASELINE_WLEN = 15.  # window length (in s) to compute the fluorescence baseline
BASELINE_QUANTILE = .05  # quantile used for the computation of the fluorescence baseline
BASELINE_RSD_THR = .5  # threshold for relative standard deviation of the fluorescence baseline across runs

# Trials discarding
ITRIALS_DISCARD = [0]  # indexes of trials to be automatically discarded for each ROI & run 

# Frame indexes
class FrameIndex:
    STIM = 10  # index of the frame coinciding with the US stimulus in each trial
    PRESTIM = slice(STIM - 3, STIM - 1)  # indexes used for analysis of pres-stimulus activity per trial.
    RESPONSE = slice(STIM, STIM + 10)  # indexes used for post-stimulus response computation per trial.

# Response & cell type classification
PTHR = 0.01  # significance threshold probability considered for activity detection in fluorescence signals (assuming directional effect)
N_NEIGHBORS_PEAK = 1  # number of neighboring elements to consider to compute "averaged" peak value  
SUCCESS_RATE_THR = .3  # threshold success rate for a positive response
NPOS_CONDS_THR = 5  # threshold number of positive conditions for an ROI to be classified as positive responder


###################################### PARSING ######################################

class Pattern:

    LINE = '([A-z][A-z0-9]*)'
    TRIAL_LENGTH = '([0-9]+)frames'
    FREQ = '([0-9]+[.]?[0-9]*)Hz'
    DUR = '([0-9]+[.]?[0-9]*)ms'
    MPA = '([0-9]+[.]?[0-9]*)MPA'
    DC = '([0-9]+)DC'
    RUN = '([0-9]+)'
    CYCLE = 'Cycle([0-9]+)'
    CHANNEL = 'Ch([0-9])'
    FRAME = '([0-9]+)'


###################################### LABELS ######################################

class Label:

    # Acquisition
    UNKNOWN = '???'  # unknown key
    P = 'P (MPa)'
    DC = 'DC (%)'
    DUR = 'duration (s)'
    FPS = 'fps'
    RUNID = 'run ID'
    LINE = 'line'
    NPERTRIAL = 'trial_length'
    NTRIALS = 'ntrials'
    CYCLE = 'cycle'
    FRAME = 'frame'
    CH = 'channel'

    # Data indexes
    ROI = 'ROI'
    RUN = 'run'
    TRIAL = 'trial'

    ISTART = 'istart'

    # Fluorescence signals
    TIME = 'time (s)'
    F_ROI = 'F_ROI (a.u.)'
    MAX_F_ROI = f'max {F_ROI}'
    F_NEU = 'F_neu (a.u.)'
    F = 'F (a.u.)'
    ALPHA = 'alpha'
    BETA = 'beta'
    F0 = 'F0 (a.u.)'

    # Relative fluorescence signals
    DFF = '\u0394F/F0'
    DFF_NOISE_LEVEL = f'{DFF} noise level'
    DFF_NOISE_AMP = f'{DFF} noise amplitude'

    # z-score signals
    ZSCORE = f'Z({DFF})'
    MAX_ZSCORE_PRESTIM = f'max pre-stim {ZSCORE}'
    REL_ZSCORE = f'{ZSCORE} - {ZSCORE}_stim'
    PEAK_REL_ZSCORE_POSTSTIM = f'peak post-stim [{REL_ZSCORE}]'
    REL_ZSCORE_RESPONLY = f'{REL_ZSCORE} (responses only)'
    PEAK_REL_ZSCORE_POSTSTIM_RESPONLY = f'{PEAK_REL_ZSCORE_POSTSTIM} (responses only)'

    # Trial activity & related measures
    PRESTIM_ACTIVITY = 'pre-stim activity?'
    PRESTIM_RATE = 'pre-stim rate'
    IS_RESP = 'trial response?'
    SUCCESS_RATE = 'success rate'
    SUCCESS_RATE_PRESTIM = f'{SUCCESS_RATE} (pre-stim activity)'
    SUCCESS_RATE_NOPRESTIM = f'{SUCCESS_RATE} (no pre-stim activity)'

    # ROI classification 
    IS_POSITIVE_RUN = 'positive run?'
    NPOS_RUNS = '# positive runs'
    ROI_RESP_TYPE = 'response type'

    # Labels that must be renamed upon averaging 
    RENAME_ON_AVERAGING = {
        IS_RESP: SUCCESS_RATE,
        PRESTIM_ACTIVITY: PRESTIM_RATE
    }


###################################### PLOTTING ######################################

LABEL_BY_TYPE = {0: 'non-responder', 1: 'responder'}
COLORS = sns.color_palette('Set2')  # default color palette
RGB_BY_TYPE = {'non-responder': COLORS[7], 'responder': COLORS[0]}  # mapping of RGB colors to response types
CI = 95  # default confidence interval for bootstrapping