# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-13 11:13:26
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-10-15 17:06:04


''' Collection of constants used throughout the code base. '''

# Miscellaneous
FPS = 3.56   # sampling rate (FPS)
REF_NFRAMES = 1600  # reference number of frames in any given experimental run
REF_LX = 256  # number of pixels in the x direction
REF_LY = 256  # number of pixels in the y direction
NTRIALS_PER_RUN = 16  # default number of trials per run
STIM_FRAME_INDEX = 10  # index of the frame coinciding with the US stimulus in each trial
NEUROPIL_FACTOR = 0.7  # default is 0.7, for PV 0.5 works better, for SST 0.6. But lately it seems that is ok with 0.7 for all regardless of the line
TAU_GCAMP6_DECAY = 1.25  # exponential decay time constant for Calcium transients (s) 
DFF_OUTLIER = 0.3  # upper bound threshold for df/f (cells with traces above this threshold get discarded). 0.8 works well for line3
NFRAMES_BASELINE = ...  # number of frames to average to compute the fluorescence baseline

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