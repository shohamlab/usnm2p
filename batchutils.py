# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-10-07 20:43:12
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-01-31 10:00:17

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


def get_baseline_id(baseline_quantile, baseline_wquantile, baseline_wsmoothing):
    ''' 
    Get baseline computation code string
    
    :param baseline_quantile: baseline evaluation quantile
    :param baseline_wquantile: quantile filter window size (s) 
    :param baseline_wsmoothing: gaussian filter window size (s)
    :return ID string
    '''
    if baseline_wquantile is None:
        baseline_wsmoothing = None
        baseline_id = ''
    else:
        baseline_id = f'wq{baseline_wquantile:.1f}s'
    if baseline_wsmoothing is not None:
        baseline_id = f'{baseline_id}_ws{baseline_wsmoothing:.2f}s'
    baseline_quantile_str = 'adaptive' if baseline_quantile is None else f'{baseline_quantile:.2f}'
    return f'q{baseline_quantile_str}_{baseline_id}'


def get_postpro_id(neuropil_scaling_coeff, *args):
    ''' 
    Get post-processing code string
    
    :param *args: parameters used for baseline computation 
    :return ID string
    '''
    return f'alpha{neuropil_scaling_coeff}_{get_baseline_id(*args)}'


def get_stats_id(ykey_classification):
    ''' 
    
    Get stats code string 
    
    :param ykey_classification: name of variable used for response classification
    '''
    return f'class{ykey_classification.replace("/", "")}'


def get_batch_settings(analysis_type, mouseline, layer, kalman_gain, neuropil_scaling_coeff,
                       baseline_quantile, baseline_wquantile, baseline_wsmoothing, 
                       trial_aggfunc, ykey_classification, directional):
    logger.info('assembling batch analysis settings...')
    # Construct dataset group ID
    dataset_group_id = get_dataset_group_id(mouseline, layer=layer)
    # Construct post-processing ID
    prepro_id = get_prepro_id(kalman_gain=kalman_gain)
    baseline_id = get_baseline_id(baseline_quantile, baseline_wquantile, baseline_wsmoothing)
    postpro_id = f'alpha{neuropil_scaling_coeff}_{baseline_id}'
    ykey_classification_str = {Label.DFF: 'dff', Label.ZSCORE: 'zscore'}[ykey_classification]
    stats_id = f'agg{trial_aggfunc.__name__}_class{ykey_classification_str}'
    if directional:
        stats_id = f'{stats_id}_directional'
    # Get figures PDF suffix
    figs_suffix = f'{analysis_type}_{dataset_group_id}_k{kalman_gain}_{postpro_id}_{stats_id}'
    # Get trial-averaged input data directory
    dataroot = get_data_root()
    trialavg_root = get_output_equivalent(dataroot, 'raw', 'trial-averaged')
    trialavg_dir = os.path.join(
        trialavg_root, stats_id, postpro_id, get_s2p_id(), prepro_id, analysis_type, mouseline)
    # Get figures directory
    figsdir = get_output_equivalent(dataroot, 'raw', 'figs')
    return dataset_group_id, trialavg_dir, figsdir, figs_suffix


def extract_from_batch_data(data):
    ''' Extract specific fields from batch data '''
    logger.info('extracting timeseries and stats from data...')
    return data['timeseries'], data['trialagg_stats'], data['ROI_masks'], data['map_ops']

