# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-13 11:13:26
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-10-22 16:41:07


''' Collection of constants used throughout the code base. '''

# Miscellaneous
UNKNOWN = '???'  # unknown key

# Acquisition
FPS = 3.56   # sampling rate (FPS)
REF_NFRAMES = 1600  # reference number of frames in any given experimental run
REF_LX = 256  # number of pixels in the x direction
REF_LY = 256  # number of pixels in the y direction
NTRIALS_PER_RUN = 16  # default number of trials per run
STIM_FRAME_INDEX = 10  # index of the frame coinciding with the US stimulus in each trial
STIM_DUR = 0.2  # reference stimulus duration (s)

# Stimulation
DC_REF = 50.  # reference duty cycle value (in %) used to perform pressure amplitude sweeps
P_REF = .8  # reference pressure amplitude (in MPa) used to perform DC sweeps

# Processing
TAU_GCAMP6_DECAY = 1.25  # exponential decay time constant for Calcium transients (s) 
NEUROPIL_FACTOR = 0.7  # default is 0.7, for PV 0.5 works better, for SST 0.6. But lately it seems that is ok with 0.7 for all regardless of the line
I_BASELINE = slice(STIM_FRAME_INDEX - 7, STIM_FRAME_INDEX)  # indexes used for baseline computation per trial. Considers the last 7 frames (i.e. ca. 2 seconds) preceding the stimulus onset.
I_RESPONSE = slice(STIM_FRAME_INDEX + 1, STIM_FRAME_INDEX + 8)  # indexes used for response computation per trial. Considers the following 7 frames (i.e. ca. 2 seconds) following the stimulus onset.
ZSCORE_THR_POSITIVE = 1.  # threshold absolute z-score value used to classify cell as "positively responding". 1.5 is default, 1 seems to work better for parvalbumin
ZSCORE_THR_NEGATIVE = -ZSCORE_THR_POSITIVE / 2  # threshold absolute z-score value used to classify cell as "negatively responding". Set to half the positive cutoff since negative responses are typically weaker than positive ones. 
LABEL_BY_TYPE = {-1: 'negative', 0: 'neutral', 1: 'positive'}  # mapping of response labels to specific integer codes
RGB_BY_TYPE = {-1: [1, 0, 0], 0: [.5, .5, .5], 1: [0, 1, 0]}  # mapping of RGB colors to specific integer codes 
DFF_OUTLIER = 0.3  # upper bound threshold for df/f (cells with traces above this threshold get discarded). 0.8 works well for line3
ROI_KEY = 'iROIs'  # key used to access list of ROIs from suite2p output dictionaries

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
P_LABEL = 'P (MPa)'
DC_LABEL = 'DC (%)'
DUR_LABEL = 'duration (s)'
FPS_LABEL = 'fps'
RUN_LABEL = 'run ID'
LINE_LABEL = 'line'
NPERTRIAL_LABEL = 'trial_length'
NTRIALS_LABEL = 'ntrials'
CYCLE_LABEL = 'cycle'
FRAME_LABEL = 'frame'
CH_LABEL = 'channel'
TIME_LABEL = 'time (s)'
RESP_LABEL = 'response type'


# Plotting
TBOUNDS = (-2., 8.)
CI = 95  # confidence interval for bootstrapping (float, 'std' or None)