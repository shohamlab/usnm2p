# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-10-07 20:43:12
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-10-16 11:44:30

from constants import *
from fileops import get_data_root, get_output_equivalent
from substitutors import StackSubstitutor
from filters import KalmanDenoiser, NoFilter
from correctors import LinRegCorrector, NoCorrector
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
    # List of applied stack processors, in forward order
    processors = [
        StackSubstitutor([
            (1, 0, None),
            (FrameIndex.STIM - 1, FrameIndex.STIM, NFRAMES_PER_TRIAL),
        ]),
        LinRegCorrector(robust=False),
        KalmanDenoiser(kalman_gain) if kalman_gain > 0 else NoFilter,
    ]
    # Return code string
    return os.path.join(*[p.code for p in processors[::-1]])


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


def get_conditioning_id(neuropil_scaling_coeff, *args):
    ''' 
    Get signal conditioning code string
    
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
    '''
    Get batch analysis settings

    :param analysis_type: type of analysis
    :param mouseline: mouse line
    :param layer: cortical layer
    :param kalman_gain: Kalman gain
    :param neuropil_scaling_coeff: neuropil scaling coefficient
    :param baseline_quantile: baseline evaluation quantile
    :param baseline_wquantile: quantile filter window size (s)
    :param baseline_wsmoothing: gaussian filter window size (s)
    :param trial_aggfunc: trial aggregation function
    :param ykey_classification: name of variable used for response classification
    :param directional: whether or not to perform directional analysis
    '''
    logger.info('assembling batch analysis settings...')
    # Construct dataset group ID
    if mouseline is None:
        dataset_group_id = 'all'
    else:
        dataset_group_id = get_dataset_group_id(mouseline, layer=layer)

    # Construct processing IDs
    prepro_id = get_prepro_id(kalman_gain=kalman_gain)
    baseline_id = get_baseline_id(baseline_quantile, baseline_wquantile, baseline_wsmoothing)
    conditioning_id = f'alpha{neuropil_scaling_coeff}_{baseline_id}'
    ykey_classification_str = {Label.DFF: 'dff', Label.ZSCORE: 'zscore'}[ykey_classification]
    processing_id = f'agg{trial_aggfunc.__name__}_class{ykey_classification_str}'
    if directional:
        processing_id = f'{processing_id}_directional'
    # Get figures PDF suffix
    figs_suffix = f'{analysis_type}_{dataset_group_id}_k{kalman_gain}_{conditioning_id}_{processing_id}'
    # Get stats input data directory
    dataroot = get_data_root()
    if mouseline is not None:
        input_root = get_output_equivalent(dataroot, DataRoot.RAW, DataRoot.PROCESSED)
        input_dir = os.path.join(
            input_root, processing_id, conditioning_id, get_s2p_id(), prepro_id, analysis_type, mouseline)
    else:    
        input_root = get_output_equivalent(dataroot, DataRoot.RAW, DataRoot.LINESTATS)
        input_dir = os.path.join(
            input_root, processing_id, conditioning_id, get_s2p_id(), prepro_id, analysis_type)
    # Get figures directory
    figsdir = get_output_equivalent(dataroot, DataRoot.RAW, DataRoot.FIG)
    return dataset_group_id, input_dir, figsdir, figs_suffix


def extract_from_batch_data(data):
    ''' Extract specific fields from batch data '''
    logger.info('extracting timeseries and stats from data...')
    keys = [
        'trialagg_timeseries',
        'popagg_timeseries', 
        'trialagg_stats', 
        'stats',
        'ROI_masks',
        'map_ops'
    ]
    return tuple([data[k].copy() for k in keys])

