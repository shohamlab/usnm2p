# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-13 11:13:26
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-06-02 09:22:16

''' Collection of constants used throughout the code base. '''

from string import ascii_lowercase
import pandas as pd

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

DEFAULT_LINE = 'line3'
DEFAULT_LAYER = 'layer2-3'
REF_NFRAMES = 1600  # reference number of frames in any given experimental run (used to check integrity of input stacks)
NFRAMES_PER_TRIAL = 100  # default number of frames per trial
DC_REF = 50.  # reference duty cycle value (in %) used to perform pressure amplitude sweeps
P_REF = .8  # reference pressure amplitude (in MPa) used to perform DC sweeps

####################################### PRE-PROCESSING ######################################

KALMAN_GAIN = 0.5  # gain of Kalman filter (0-1)

################################## FUNCTIONAL SEGMENTATION ##################################

S2P_UINT16_NORM_FACTOR = 2  # normalization factor applied by suite2p to input uint16 TIF files
REWRITTEN_S2P_KEYS = {  # suite2p options keys that are rewritten upon suite2p processing
    'fast_disk',
    'save_path0',
    'save_folder',
    'bidi_corrected',
    # 'block_size'
}  
TAU_GCAMP6S_DECAY = 1.25  # GCaMP6s exponential decay time constant (s) 


###################################### POST-PROCESSING ######################################

NPIX_RATIO_THR = None  # threshold (# pixels ROI) / (# pixels soma) ratio (cells above that ratio get discarded) 

# neuropil subtraction factor (0-1):
# - small values (around 0) tend to produce smoother dF/F0 traces with less amplitude,
# - large values (around 1) tend to produce higher amplitude dF/F0 traces with more fluctuations
# - default is 0.7 (from literature), for PV 0.5 works better, for SST 0.6. But lately it seems that is ok with 0.7 for all regardless of the line 
ALPHA = 0.7

# Baseline computation
BASELINE_WLEN = 10.  # window length (in s) to compute the fluorescence baseline
BASELINE_QUANTILE = .08  # quantile used for the computation of the fluorescence baseline
BASELINE_SMOOTHING = True  # whether to smooth the baseline with an extra moving average
BASELINE_RSD_THR = .5  # threshold for relative standard deviation of the fluorescence baseline across runs

# Trials discarding
ITRIALS_DISCARD = [0]  # indexes of trials to be automatically discarded for each ROI & run 

# Artifacts
VDISP_THR = 2.  # threshold peak displacement velocity (um/s). Trials with velocities higher than this value get discarded 
PCT_PREACTIVE_THR = 50.  # threshold percentage of pre-active cells for each trial. Trials with higher percentages get discarded  
NSTD_DEV_THR = 10  # number of standard deviations from timeseries distribution median outside which a trial is considered an outlier

# Baseline activity
NSEEDS_PER_TRIAL = 50  # number of detection windows along each trial interval to detect activity 

# Frame indexes
class FrameIndex:
    STIM = 10  # index of the frame coinciding with the US stimulus in each trial
    PRESTIM = slice(STIM - 9, STIM + 1)  # indexes used for analysis of pres-stimulus activity per trial.
    RESPONSE = slice(STIM, STIM + 10)  # indexes used for post-stimulus response computation per trial.
    RESP_EXT = slice(STIM, STIM + 40)  # indexes excluded for DFF detrending
    BASELINE = slice(70, 101)


# Response & cell type classification
N_NEIGHBORS_PEAK = 1  # number of neighboring elements to consider to compute "averaged" peak value
PTHR_DETECTION = 0.05  # significance threshold probability for activity detection in fluorescence signals (assuming directional effect)
NPOSCONDS_THR = 5  # minimum number of "positive" conditions for a cell to be classified as "US-responsive"  

# Datasets selection
MIN_CELL_DENSITY = 1000.  # minimum cell density (cells/mm2)

###################################### PARSING ######################################

class Pattern:

    LINE = '([A-z][A-z0-9]*)'
    DATE = '(\d{4})(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])'
    MOUSE = 'mouse[1-9][0-9]*'
    REGION = 'region[1-9][0-9]*[a-zA-Z]?'
    LAYER = 'layer[1-5]-?[1-5]*'
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
    ISPPA = 'I_SPPA (W/cm2)'  # spatial peak, pulse average acoustic intensity
    ISPTA = 'I_SPTA (W/cm2)'  # spatial peak, temporal average acoustic intensity
    DUR = 'duration (s)'
    FPS = 'fps'
    RUNID = 'run ID'
    LINE = 'line'
    NPERTRIAL = 'trial_length'
    NTRIALS = 'ntrials'
    CYCLE = 'cycle'
    FRAME = 'frame'
    CH = 'channel'
    DATASET = 'dataset'

    # Data indexes
    ROI = 'ROI'
    RUN = 'run'
    TRIAL = 'trial'
    ISTART = 'istart'

    # Time-varying signals
    TIME = 'time (s)'
    F_ROI = 'F_ROI (a.u.)'
    MAX_F_ROI = f'max {F_ROI}'
    F_NEU = 'F_neu (a.u.)'
    ALPHA = 'alpha'
    BETA = 'beta'
    F = 'F (a.u.)'
    F0 = 'F0 (a.u.)'
    F_DETRENDED = 'F_detrended (a.u.)'
    F0_DETRENDED = 'F0 detrended (a.u.)'
    DFF = '\u0394F/F0'
    ZSCORE = f'Z({DFF})'
    REL_DFF = f'{DFF} - {DFF}_stim'
    REL_ZSCORE = f'{ZSCORE} - {ZSCORE}_stim'

    # Displacement & velocities
    X_PX = 'x (pixels)'
    Y_PX = 'y (pixels)'
    DISTANCE_PX = 'd (pixels)'
    DISTANCE_UM = 'd (um)'
    SPEED_UM_FRAME = 'v (um/frame)'
    SPEED_UM_S = 'v (um/s)'

    # Statistics
    PEAK_DISP_VEL = 'peak displacement velocity (um/s)'
    DISCARDED = 'discarded'
    MOTION = 'motion'
    OUTLIER = 'outlier'
    EVENT = 'event'

    # Trial activity & related measures
    PRESTIM_ACTIVITY = 'pre-stim cell activity?'
    PCT_PREACTIVE_CELLS = '% pre-stim active cells'
    PRESTIM_POP_ACTIVITY = 'pre-stim population activity?'
    PRESTIM_INHIBITION = 'pre-stim cell inhibition?'
    PRESTIM_OUTLIER = 'pre-stimulus outlier?'
    RESP_TYPE = 'response type'
    PRESTIM_RATE = 'pre-stim rate'
    IS_RESP = 'trial response?'
    PCT_RESP_CELLS = '% responding cells'
    SUCCESS_RATE = 'success rate'
    POS_COND = 'positive condition?'

    # ROI classification 
    NPOS_CONDS = f'# {POS_COND[:-1]}s'
    IS_RESP_ROI = 'responsive ROI?'
    ROI_RESP_TYPE = 'responder type'

    # Labels that must be renamed upon averaging 
    RENAME_ON_AVERAGING = {
        IS_RESP: SUCCESS_RATE,
        PRESTIM_ACTIVITY: PRESTIM_RATE
    }


# Response and responders type
RTYPE_MAP = {
    -1: 'negative', 0: 'weak', 1: 'positive'}
RTYPE = pd.api.types.CategoricalDtype(
    categories=list(RTYPE_MAP.values()), ordered=True)


# Stats fields used to compute trial validity  
TRIAL_VALIDITY_KEYS = [
    Label.DISCARDED,
    Label.MOTION,
    Label.OUTLIER,
    Label.PRESTIM_ACTIVITY,
    Label.PRESTIM_POP_ACTIVITY,
    Label.PRESTIM_INHIBITION,
]

###################################### PLOTTING ######################################

class Palette:
    ''' Color palettes used to visualize various categories & dependencies '''
    
    DEFAULT = 'rocket_r'  # default (continuous)
    RTYPE = 'tab10'  # response type (categorical)
    P = 'flare'  # pressure (continuous)
    DC = 'crest'  # duty cycle (continuous)


CI = 95  # default confidence interval for bootstrapping