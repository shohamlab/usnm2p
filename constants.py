import re

''' Collection of constants used throughout the code base. '''

# Miscellaneous
FPS = 3.56   # sampling rate (FPS)
REF_NFRAMES = 1600  # reference number of frames in any given experimental run 
NMAX_FRAMES_BASELINE_ESTIMATE = 100  # max number of frames from signal to use to compute baseline 
NTRIALS_PER_RUN = 16  # default number of trials per run
STIM_FRAME_INDEX = 10  # index of the frame coinciding with the US stimulus in each trial
NEUROPIL_FACTOR = 0.7  # default is 0.7, for PV 0.5 works better, for SST 0.6. But lately it seems that is ok with 0.7 for all regardless of the line
TAU_CA_DECAY = 1.25  # exponential decay time constant for Calcium transients (s) 
DFF_OUTLIER = 0.3  # upper bound threshold for df/f (cells with traces above this threshold get discarded). 0.8 works well for line3
N_MAV = 15  # number of frames in the moving average window to compute baseline
 
# Regexp patterns
TIF_PATTERN = re.compile('.*tif')
P_LINE = '([A-z0-9]+)'
P_TRIAL_LENGTH = '([0-9]+)frames'
P_FREQ = '([0-9]+[.]?[0-9]*)Hz'
P_DUR = '([0-9]+[.]?[0-9]*)ms'
P_MPA = '([0-9]+)MPA'
P_DC = '([0-9]+)DC'
P_RUN = '([0-9]+)'
EXPCODE = re.compile(
    f'^{P_LINE}_{P_TRIAL_LENGTH}_{P_FREQ}_{P_DUR}_{P_FREQ}_{P_MPA}_{P_DC}-{P_RUN}$',
    re.IGNORECASE)

UNDEFINED_UNICODE = '\u00D8'