# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-10-07 20:43:12
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-10-28 00:18:18

from constants import *
from fileops import get_data_root, get_output_equivalent
from substitutors import StackSubstitutor
from filters import KalmanDenoiser, NoFilter
from s2putils import *
from logger import logger


def get_dataset_group_id(mouseline, layer=None):
    ''' Construct dataset group ID '''
    dataset_group_id = mouseline
    if layer is not None:
        dataset_group_id = f'{dataset_group_id}_{layer}'
    return dataset_group_id


def get_prepro_id(kalman_gain=KALMAN_GAIN):
    '''
    Get code string corresponding to specific pre-processing settings
    
    :param klaman_gain: Kalmain gain (0 - 1)
    :return ID string
    '''
    submap = [(1, 0), (FrameIndex.STIM - 1, FrameIndex.STIM)]
    ss = StackSubstitutor(submap, repeat_every=NFRAMES_PER_TRIAL)
    kd = KalmanDenoiser(kalman_gain) if kalman_gain > 0 else NoFilter
    return os.path.join(kd.code, ss.code)


def get_s2p_id(tau=TAU_GCAMP6S_DECAY, fs=BRUKER_SR, do_registration=1,
                 reg_tif=True, nonrigid=True, denoise=False):
    ''' 
    Get code string corresponding to suite2p execution options
    
    :param tau: timescale of the sensor
    :param fs: sampling rate (per plane)
    :param do_registration: whether or not to perform image registration
    :param reg_tif: whether or not to write the registered binary to tiff files
    :param nonrigid: whether or not to perform non-rigid registration, which splits the
      field of view into blocks and computes registration offsets in each block separately.
    :param denoise: perform PCA denoising of the registered stack prior to ROI detection
    :return ID string
    '''
    s2p_ops = {
        'tau': tau,
        'fs': fs,
        'do_registration': do_registration,
        'reg_tif': reg_tif,
        'nonrigid': nonrigid,
        'denoise': denoise,
    }
    defops = default_ops()
    s2p_ops = get_normalized_options(s2p_ops, defops)            
    diff_from_default_ops = compare_options(s2p_ops, defops)['input'].to_dict()
    s2p_code = 'suite2p'
    if diff_from_default_ops is not None:
        s2p_code = f'{s2p_code}_{get_options_code(diff_from_default_ops)}'
    return s2p_code


def get_baseline_id(baseline_wlen, baseline_quantile, baseline_smoothing):
    ''' 
    Get baseline computation code string
    
    :param baseline_wlen: length of baseline detection rolling window 
    :param baseline_quantile: baseline computation quantile
    :param baseline_smoothing: whether baseline should be smoothed
    :return ID string
    '''
    # Construct post-processing ID
    if baseline_wlen is None:
        baseline_smoothing = False
        baseline_id = ''
    else:
        baseline_id = f'w{baseline_wlen:.1f}s'
    baseline_id = f'{baseline_id}_q{baseline_quantile:.2f}'
    if baseline_smoothing:
        baseline_id = f'{baseline_id}_smooth'
    return baseline_id


def get_postpro_id(baseline_wlen, baseline_quantile, baseline_smoothing, ykey_classification):
    ''' 
    Get post-processing code string
    
    :param *args: parameters used for baseline computation 
    :param ykey_classification: name of variable used for response classification
    :return ID string
    '''
    return f'{get_baseline_id(baseline_wlen, baseline_quantile, baseline_smoothing)}_{ykey_classification}'.replace('/', '')


def get_batch_settings(analysis_type, mouseline, layer, kalman_gain,
                       baseline_wlen, baseline_quantile, baseline_smoothing, 
                       ykey_classification):
    logger.info('assembling batch analysis settings...')
    # Construct dataset group ID
    dataset_group_id = get_dataset_group_id(mouseline, layer=layer)
    # Construct post-processing ID
    prepro_id = get_prepro_id(kalman_gain=kalman_gain)
    baseline_id = get_baseline_id(baseline_wlen, baseline_quantile, baseline_smoothing)
    postpro_id = f'{baseline_id}_{ykey_classification}'.replace('/', '')
    # Get figures PDF suffix
    figs_suffix = f'{analysis_type}_{dataset_group_id}_k{kalman_gain}_{postpro_id}'
    # Get trial-averaged input data directory
    dataroot = get_data_root()
    trialavg_root = get_output_equivalent(dataroot, 'raw', 'trial-averaged')
    trialavg_dir = os.path.join(
        trialavg_root, baseline_id, get_s2p_id(), prepro_id, analysis_type, mouseline)
    # Get figures directory
    figsdir = get_output_equivalent(dataroot, 'raw', 'figs')
    return dataset_group_id, trialavg_dir, figsdir, figs_suffix


def extract_from_batch_data(data):
    ''' Extract specific fields from batch data '''
    logger.info('extracting timeseries and stats from data...')
    return data['timeseries'], data['stats'], data['ROI_masks'], data['map_ops']

