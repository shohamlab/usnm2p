# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-13 11:13:26
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-10-18 22:27:11


''' Collection of constants used throughout the code base. '''

# Acquisition
FPS = 3.56   # sampling rate (FPS)
REF_NFRAMES = 1600  # reference number of frames in any given experimental run
REF_LX = 256  # number of pixels in the x direction
REF_LY = 256  # number of pixels in the y direction
NTRIALS_PER_RUN = 16  # default number of trials per run
STIM_FRAME_INDEX = 10  # index of the frame coinciding with the US stimulus in each trial

# Processing
TAU_GCAMP6_DECAY = 1.25  # exponential decay time constant for Calcium transients (s) 
NEUROPIL_FACTOR = 0.7  # default is 0.7, for PV 0.5 works better, for SST 0.6. But lately it seems that is ok with 0.7 for all regardless of the line
I_BASELINE = slice(STIM_FRAME_INDEX - 7, STIM_FRAME_INDEX)  # indexes used for baseline computation per trial. Considers the last 7 frames (i.e. ca. 2 seconds) preceding the stimulus onset.
I_RESPONSE = slice(STIM_FRAME_INDEX + 1, STIM_FRAME_INDEX + 8)  # indexes used for response computation per trial. Considers the following 7 frames (i.e. ca. 2 seconds) following the stimulus onset.
ZSCORE_THR_POSITIVE = 1.  # threshold absolute z-score value used to classify cell as "positively responding". 1.5 is default, 1 seems to work better for parvalbumin
ZSCORE_THR_NEGATIVE = -ZSCORE_THR_POSITIVE / 2  # threshold absolute z-score value used to classify cell as "negatively responding". Set to half the positive cutoff since negative responses are typically weaker than positive ones. 
RESPONSE_CODES = {-1: 'negative', 0: 'neutral', 1: 'positive'}  # mapping of response classes to specific integer codes  
DFF_OUTLIER = 0.3  # upper bound threshold for df/f (cells with traces above this threshold get discarded). 0.8 works well for line3


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

P_UNITS = {
    'fps': 'fps',
    'P': 'MPa',
    'duration': 's',
    'DC': '%',
}