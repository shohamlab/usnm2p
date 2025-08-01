# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-13 11:13:26
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-08-01 16:51:12

''' Collection of constants used throughout the code base. '''

from string import ascii_lowercase
import numpy as np
import pandas as pd
import seaborn as sns

# Conda environment name
ENV_NAME = 'usnm2p'

###################################### LABELS ######################################

class Label:

    # Acquisition
    UNKNOWN = '???'  # unknown key
    P = 'P (MPa)'
    DC = 'DC (%)'
    ISPPA = 'I_SPPA (W/cm2)'  # spatial peak, pulse average acoustic intensity
    ISPTA = 'I_SPTA (W/cm2)'  # spatial peak, temporal average acoustic intensity
    PSPTA = 'P_SPTA (MPa)'   # spatial peak, temporal average pressure
    PSPPRMS = 'P_SPPRMS (MPa)'  # spatial peak, pulse RMS pressure
    PSPTRMS = 'P_SPTRMS (MPa)'  # spatial peak, temporal RMS pressure 
    ISPPRMS = 'I_SPPRMS (W/cm2)'  # spatial peak, pulse RMS intensity
    ISPTRMS = 'I_SPTRMS (W/cm2)'  # spatial peak, temporal RMS intensity
    PRF = 'PRF (Hz)'
    DUR = 'duration (s)'
    FPS = 'fps'
    LPF = 'lpf'  # lines per frame
    PULSE = 'pulse'
    PULSERELTIME = 'time w.r.t pulse onset (ms)'
    RUNID = 'run ID'
    LINE = 'line'
    NPERTRIAL = 'trial_length'
    NTRIALS = 'ntrials'
    CYCLE = 'cycle'
    SUFFIX = 'suffix'
    FRAME = 'frame'
    COL = 'col'
    ROW = 'row'
    FRAMEROW = 'frame row'
    STIMFRAME = 'stimulus frame'
    CH = 'channel'
    DATASET = 'dataset'
    LAYER = 'layer'
    OFFSET = 'offset (mm)'
    TRIG = 'trigger'
    DELAY = 'delay'

    # Frequency analysis
    FREQ = 'frequency (Hz)'
    PSPECTRUM = 'Power spectrum'
    PSPECTRUM_DB = 'Power spectrum (dB)'

    # Data indexes
    ROI = 'ROI'
    RUN = 'run'
    TRIAL = 'trial'
    ISTART = 'istart'
    ITI = 'iti'  # inter-trial interval

    # Time-varying signals
    TIME = 'time (s)'
    HOURS = 'hours'
    TRIALPHASE = 'trial phase (rad)'
    PHASE = 'phase (rad)'
    ANGLE = 'angle (rad)'  # i.e. unwrapped phase
    ITPC = 'ITPC'  # inter-trial phase coherence
    ENV = 'envelope'
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
    PI = '\u03C0'
    ZSCORE = f'Z({DFF})'
    REL_DFF = f'{DFF} - {DFF}_stim'
    REL_ZSCORE = f'{ZSCORE} - {ZSCORE}_stim'
    EVENT_RATE = 'event rate (Hz)'

    # Displacement & velocities
    X_PX = 'x (pixels)'
    Y_PX = 'y (pixels)'
    DISTANCE_PX = 'd (pixels)'
    DISTANCE_UM = 'd (um)'
    SPEED_UM_FRAME = 'v (um/frame)'
    SPEED_UM_S = 'v (um/s)'

    # Statistics
    PEAK_DISP_VEL = 'peak displacement velocity (um/s)'
    AVG_DISP_VEL = 'average displacement velocity (um/s)'
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
    RESP_FRAC = 'responder fraction'
    POS_COND = 'positive condition?'

    # ROI classification 
    CH2_ROI_PROB = 'CH2 ROI prob.'
    CH2_ROI = 'CH2 ROI?'
    ROI_COUNT = '# ROIs'
    ROI_DENSITY = 'cells / mm2'
    NPOS_CONDS = f'# {POS_COND[:-1]}s'
    IS_RESP_ROI = 'responsive ROI?'
    ROI_RESP_TYPE = 'responder type'

    # Stats columns that should be merged to timeseries before plotting 
    # response traces
    MERGE_UPON_PLT = [
        FPS, 
        DUR,
        P, 
        DC, 
        ISPTA
    ]

    # Labels that must be renamed upon aggregation 
    RENAME_UPON_AGG = {
        PRESTIM_ACTIVITY: PRESTIM_RATE
    }

    # Labels that must be "resolved" upon aggregation
    RESOLVE_UPON_AGG = [
        P,
        DC,
        ISPPA,
        ISPTA,
        PSPTA,
        PSPPRMS,
        PSPTRMS,
        ISPPRMS,
        ISPTRMS,
        PRF,
        DUR,
        FPS
    ]

    # Exclusion criteria
    POORSEG = 'poor segmentation'	
    DEADCELLS = 'potential dead cells'
    NORESP = 'no/weak response'
    STRONGRESP = 'abnormal response amplitudes'
    DROP5060DC = 'abnormal drop 50-60% DC'
    DROP0406MPA = 'abnormal drop 0.4-0.6 MPa'
    UNSTABLEFO = 'large variation in fluorescence baseline'


# Names used for intermediate data directories along the analysis pipeline
class DataRoot:
    
    FIG = 'figs'
    RAW_BRUKER = 'raw_bruker'
    RAW_BERGAMO = 'raw_bergamo'
    STACKED = 'stacked'
    ROWAVG = 'rowavg'
    PULSEDIP = 'pulsedip'
    RESAMPLED = 'resampled'
    SPLIT = 'split'
    MODEL = 'model'
    SUBSTITUTED = 'substituted'
    CORRECTED = 'corrected'
    FILTERED = 'filtered'
    SEGMENTED = 'segmented'
    CONDITIONED = 'conditioned'
    PROCESSED = 'processed'
    LINEAGG = 'lineagg'


# Response and responders type
RTYPE_MAP = {
    -1: 'negative',
    0: 'weak',
    1: 'positive'
}
RTYPE = pd.api.types.CategoricalDtype(
    categories=list(RTYPE_MAP.values()), ordered=True)


# Stats fields used to compute trial validity  
TRIAL_VALIDITY_KEYS = [
    Label.DISCARDED,
    Label.MOTION,
    # Label.OUTLIER,
    Label.PRESTIM_ACTIVITY,
    Label.PRESTIM_POP_ACTIVITY,
    Label.PRESTIM_INHIBITION,
] 

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

# Conversion constants
PA_TO_MPA = 1e-6
M2_TO_CM2 = 1e4
UM2_TO_MM2 = 1e-6
S_TO_MS = 1e3
HZ_TO_KHZ = 1e-3

###################################### DATA ACQUISITION ######################################

DEFAULT_ANALYSIS = 'main'
DEFAULT_LINE = 'line3'
DEFAULT_LAYER = 'layer2-3'
BERGAMO_SR = 30.00  # data sampling rate on Bergamo resonant scanning system (Hz)
BERGAMO_RESAMPLED_SR = 5.0  # data sampling rate on Bergamo system after resampling (Hz)
BRUKER_SR = 3.56    # data sampling rate on Bruker galvo scanning system (Hz)
FUNC_CHANNEL = 1  # index of functional channel in multi-channel Bergamo recordings
REF_NFRAMES = 1600  # reference number of frames in any given experimental run (used to check integrity of input stacks)
NFRAMES_PER_TRIAL = 100  # default number of frames per trial
DC_REF = 50.  # reference duty cycle value (in %) used to perform pressure amplitude sweeps
P_REF = .8  # reference pressure amplitude (in MPa) used to perform DC sweeps
OFFSET_DIRECTIONS = ('backward', 'left', 'right') # Potential lateral offset directions
MAX_LASER_POWER_REL_DEV = .15  # maximum allowed relative deviation from reference value for laser power (larger than other values because it typically warms up during experiment)
MAX_DAQ_REL_DEV = .01  # maximum allowed relative deviation from reference value for acquisition settings

####################################### PRE-PROCESSING ######################################

FOLDER_EXCLUDE_PATTERNS = ['MIP', 'References', 'incomplete', 'duplicated', 'excluded']  # folders exclusion patterns for raw TIF data

IREF_FRAMES_BERGAMO = slice(-200, None)  # index range of reference frames for detrending on corrupted Bergamo acquisitions 
NEXPS_DECAY_DETREND = 2  # number of exponentials for initial decay detrending on corrupted Bergamo acquisitions
NSAMPLES_DECAY_DETREND = 200  # number of samples for initial decay detrending on corrupted Bergamo acquisitions
DECAY_FIT_MAX_REL_RMSE = 1.2 # max relative RMSE allowed during stack decay detrending process

# Substitution maps
STIMSUB = ('stim-1', 'stim', 'trial')  # stimulus frames, every trial
FIRST_SUB = (1, 0, None)  # first run frame
SUBMAP_DICT = {  # line-specific substitution maps
    'cre': [STIMSUB],
    'default': [FIRST_SUB, STIMSUB],
}
def get_submap(line):
    ''' Get line-specific substitution map '''
    if isinstance(line, str) and line.startswith('cre'):
        return SUBMAP_DICT['cre']
    else:
        return SUBMAP_DICT['default']

# Global stack correction methods
GLOBAL_CORRECTION = {
    'line3': None,
    'sst': None,
    'pv': 'linreg_robust',
    'sarah_line3': None,
    'cre_sst': None, #'linreg_robust', 'linreg_nointercept_refch2'
    'cre_ndnf': None,
}
KALMAN_GAIN = 0.5  # gain of Kalman filter (0-1)

################################## FUNCTIONAL SEGMENTATION ##################################

REWRITTEN_S2P_KEYS = {  # suite2p options keys that are rewritten upon suite2p processing
    'fast_disk',
    'save_path0',
    'save_folder',
    'bidi_corrected',
    # 'block_size'
}  
GCAMP_DECAY_TAU = {  # GCaMP sensors exponential decay time constants (s)
    '6s': 1.25,  # GCaMP6s
    '7f': 0.7,  # GCaMP7f
}

def get_gcamp_key(line):
    ''' Get GCaMP key for a given line '''
    if isinstance(line, str) and line.startswith('cre'):
        return '7f'
    else:
        return '6s'

###################################### SIGNAL CONDITIONING ######################################

NPIX_RATIO_THR = None  # threshold (# pixels ROI) / (# pixels soma) ratio (cells above that ratio get discarded) 

# scaling coefficient for neuropil subtraction:
# - large values (around 1) tend to produce smoother dF/F0 traces with less amplitude,
# - small values (around 0) tend to produce higher amplitude dF/F0 traces with more fluctuations
# - default is 0.7 (from literature)
# - From Diego: for PV 0.5 works better, for SST 0.6. But lately it seems that is ok with 0.7 for all regardless of the line
NEUROPIL_SCALING_COEFF = 0.7

# Baseline computation
BASELINE_QUANTILE = None  #.08  # quantile used for the computation of the fluorescence baseline (if None, and adaptive quantile is used)
BASELINE_WQUANTILE = 30.  # quantile filter window size (s) to compute fluorescence baseline
BASELINE_WSMOOTHING = None  # gaussian filter window size (s) to smooth out fluorescence baseline

###################################### STATISTICS ######################################

# Trials discarding
ITRIALS_DISCARD = [0]  # indexes of trials to be automatically discarded for each ROI & run

# Activity events detection & quantification
MIN_EVENTS_DISTANCE = 2.  # minimum temporal interval between activity peaks (s)
EVENTS_BIN_INTERVAL = 1.4  # binning interval for events density quantification (s)

# Artifacts
VDISP_PEAK_THR = 10.  # threshold peak displacement velocity (um/s). Trials with peak velocities higher than this value get discarded 
VDISP_AVG_THR = 2.  # threshold average displacement velocity (um/s). Trials with average velocities higher than this value get discarded 
PCT_PREACTIVE_THR = 50.  # threshold percentage of pre-active cells for each trial. Trials with higher percentages get discarded  
NSTD_DEV_THR = 5  # number of standard deviations from timeseries distribution median outside which a trial is considered an outlier 
MIN_VALID_TRIALS = 5  # minimum of avaliable valid trials to average from for the ROI-condition to be valid 
ZTHR_REJECTION = 3000.  # threshold absolute z-score for signal rejection
PTHR_REJECTION = 0.01  # significance threshold probability for signal rejection

# Trial aggregation
TRIAL_AGGFUNC = np.median   # trial aggregation function

# Stimulation onset time (line specific)
STIM_ONSET_DICT = {
    'cre': 5.0,  # s
    'default': 2.81,  # s
}

DEFAULT_STIM_TRIG_DELAY = 9e-3  # Default stimulus trigger delay in seconds (inferred from data)
DATASET_STIM_TRIG_DELAY = {  # Dataset-specific variations in stimulus trigger delay (inferred from data) 
    '20191112_mouse12_region1': 0.,   # main line3 
    '20191112_mouse12_region2_layer5': 0.,  # main line3
    '20190516_mouse8_region2': 18e-3,   # main sst
}
def get_stim_trig_delay(dataset_ID):
    return DATASET_STIM_TRIG_DELAY.get(dataset_ID, DEFAULT_STIM_TRIG_DELAY)

MAX_TEMPLATE_MATCHING_SHIFT = 5e-3  # Maximum signal shift allowed during template matching (in s) 


def get_stim_onset_time(line):
    ''' Get stimulation onset time for a given line '''
    if isinstance(line, str) and line.startswith('cre'):
        return STIM_ONSET_DICT['cre']
    else:
        return STIM_ONSET_DICT['default']


# Analysis windows
TPRE = 1.4  # default size of the pre-stimulus window (s)
TPOST = 2.8  # default size of the post-stimulus window (s)

# Response classification
YKEY_CLASSIFICATION = Label.ZSCORE  # Reference variable for response classification
PTHR_DETECTION = 0.05  # significance threshold probability for activity detection in fluorescence signals
DIRECTIONAL_DETECTION = True  # whether to look for directional (i.e. positive only) effect for response detection

# Responder type classification
ISPTA_THR = 2.0  # ISPTA lower bound restricting the conditions on which to compute fraction of response occurence (W/cm2)
PROP_CONDS_THR = 0.50  # minimum proportion of conditions with given response type for a cell to be classified as that same respone type
OFFSET_MIN_PROP_POS = 0.33  # minimum proportion of positive responses in "best" condition to include datasets in offset analysis 

# Baseline fluorescence
MAX_F0_REL_DEV = .5  # max relative deviation of baseline fluorescence from its mean allowed during experiment

###################################### PARSING ######################################

class Pattern:

    LINE = '([A-z][A-z0-9_]*)'
    DATE = '(\d{4})(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])'
    MOUSE = '[A-z][a-z]*[1-9][0-9]*'
    REGION = 'region[1-9][0-9]*[a-zA-Z]?'
    LAYER = 'layer[1-5]-?[1-5]*'
    TRIAL_LENGTH = '([0-9]+)frames'
    FREQ = '([0-9]+[.]?[0-9]*)Hz'
    DUR = '([0-9]+[.]?[0-9]*)ms'
    MPA = '([0-9]+[.]?[0-9]*)MP[Aa]'
    DC = '([0-9]+)DC'
    OPTIONAL_SUFFIX = '_?(.*)'
    RUN = '([0-9]+)'
    NAMED_RUN = 'run([0-9]+)'
    TRIAL = '([0-9]+)'
    CYCLE = 'Cycle([0-9]+)'
    CHANNEL = 'Ch([0-9])'
    FRAME = '([0-9]+)'
    OFFSET = f'({"|".join(OFFSET_DIRECTIONS)})_([0-9]+[.]?[0-9]*)mm'
    QUANTILE = 'q(0.[0-9]*)'


###################################### PLOTTING ######################################

class Palette:
    ''' Color palettes used to visualize various categories & dependencies '''
    
    DEFAULT = 'rocket_r'  # default (continuous)
    TERNARY = {  # response type (categorical)
        -1: 'C1',
        0: 'silver',
        1: 'C2'
    }
    RTYPE = {  # response type (categorical)
        'negative': 'C1',
        'weak': 'silver',
        'positive': 'C2'
    }
    LINE = {  # mouse line (categorical)
        'line3': 'C0',
        'sarah_line3': 'b',
        'pv': 'C1',
        'sst': 'r',
        'cre_sst': 'C2',
        'cre_ndnf': 'C2',
    }
    P = 'flare'  # pressure (continuous)
    DC = 'crest'  # duty cycle (continuous)
    OFFSET = sns.cubehelix_palette(  # spatial offset distance (continous)
        start=.5, rot=-.5, reverse=True, as_cmap=True)
    GREEN = sns.dark_palette(  # For green-scale images
        'green', as_cmap=True)
    CH2_ROI = {  # CH2 ROI
        False: 'silver',
        True: 'tab:orange'
    }
    DEAFENING = {  # colors for pre- and post-deafening data
        'pre-deafening': 'C2', 
        'post-deafening': 'C5'
    }

# Sweep markers
sweep_markers = {
    Label.P: 'o',
    Label.DC: '^',
}

###################################### DATASETS ######################################

# Minimum cell density (cells/mm2) per cell line
MIN_CELL_DENSITY = {
    'line3': 1400.,
    'sarah_line3': None,
    'sst': None,
    'pv': None,
    'cre_sst': None,
    'cre_ndnf': None,
}


###################################### EXECUTION ######################################

NB_RETRY_ERRMSGS = [  # Notebook execution error messages for which to re-try execution
    'Resource temporarily unavailable',  # caused by concurrent access to TIF I/O manager (e.g. during suite2p processing)
    'Kernel died before replying to kernel_info',  # kernel address already in use (caused by ZMQ manager)
    "Kernel died",  # Caused by OOM error
    "Kernel didn't respond in 60 seconds", # ???
]


########################################## MODEL OPTIMIZATION ######################################

# Default optimization parameters
INVALID_COST = np.inf  # cost value for invalid optimization results
MAX_COST = 1e12  # maximum cost value for optimization results
MIN_ACTIVITY_LEVEL = 1e-3  # minimum activity level considered meaningful (to prevent emergence of "falsly varying" profiles upon normalization) 
DISPARITY_COST_FACTOR = 1e-3  # cost factor for activity profiles disparity
WDEV_COST_FACTOR = .9  # cost factor for connectivity matrix relative deviation from reference
SRELMAX = 5.0  # maximum relative stimulus sensitivity value
WMAX = 20.0  # maximum absolute value of coupling weights
RELWMAX = 0.3  # maximum relative deviation of coupling weights from their reference values  
OPT_METHOD = 'diffev'  # default optimization method
OPT_NRUNS = 5  # default number of optimization runs 
NORM_BEFORE_COMP = True  # whether to normalize activity profiles before optimization
CSV_DELIMITER = ','  # CSV delimiter
H5_KEY = 'opt_history'  # H5 key
LOG_FTYPE = 'h5' # 'csv' # default log file type