# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-10-07 20:43:12
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2024-08-09 10:40:33

from .constants import *
from .fileops import get_data_root
from .substitutors import StackSubstitutor
from .filters import KalmanDenoiser
from .correctors import LinRegCorrector
from .resamplers import StackResampler
from .s2putils import *
from .logger import logger


def get_dataset_group_id(mouseline, layer=None):
    ''' Construct dataset group ID '''
    dataset_group_id = mouseline
    if layer is not None:
        dataset_group_id = f'{dataset_group_id}_{layer}'
    return dataset_group_id


def get_prepro_id(submap=None, global_correction=None, kalman_gain=KALMAN_GAIN):
    '''
    Get code string corresponding to specific pre-processing settings
    
    :param submap (optional): stack substitution map
    :param global_correction (optional): global correction method
    :param kalman_gain (optional): Kalmain gain (0 - 1)
    :return ID string
    '''
    # If global correction is a dictionary, recursively call this function for each entry
    if isinstance(global_correction, dict):
        return {
            k: get_prepro_id(submap=submap, global_correction=v, kalman_gain=kalman_gain)
            for k, v in global_correction.items()
        }

    # Create dictionary of applied stack processors
    processors = {}

    # Add stack substitor, if any
    if submap is not None:
        processors['stack_substitutor'] = StackSubstitutor(submap)
    
    # Add global corrector, if any
    if global_correction is not None:
        processors['lrc'] = LinRegCorrector.from_string(global_correction)
    
    # Add Kalman denoiser, if any
    if kalman_gain is not None and kalman_gain > 0:
        processors['kd'] = KalmanDenoiser(kalman_gain)
    
    # Generate codes list
    codes = [p.code for p in processors.values()]

    # Reverse codes list order and return code string
    return os.path.join(*codes[::-1])


def get_s2p_id(fs, tau, do_registration=1, reg_tif=True, nonrigid=True, denoise=False):
    ''' 
    Get code string corresponding to suite2p execution options
    
    :param fs: sampling rate (per plane)
    :param tau: timescale of the sensor
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


def get_batch_settings(analysis_type, mouseline, layer, global_correction, kalman_gain, 
                       neuropil_scaling_coeff, baseline_quantile, baseline_wquantile, 
                       baseline_wsmoothing, trial_aggfunc, ykey_classification, directional):
    '''
    Get batch analysis settings

    :param analysis_type: type of analysis
    :param mouseline: mouse line
    :param layer: cortical layer
    :param global_correction: global correction method
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

    # Get mouse-specific substitution map
    submap = get_submap(mouseline)

    # Construct processing IDs
    prepro_id = get_prepro_id(submap=submap, global_correction=global_correction, kalman_gain=kalman_gain)
    if mouseline == 'cre_sst':
        channel_id = f'channel{FUNC_CHANNEL}'
        resampling_id = StackResampler(BERGAMO_SR, BERGAMO_RESAMPLED_SR, smooth=True).code
        prepro_id = os.path.join(prepro_id, channel_id, resampling_id)

    baseline_id = get_baseline_id(baseline_quantile, baseline_wquantile, baseline_wsmoothing)
    conditioning_id = f'alpha{neuropil_scaling_coeff}_{baseline_id}'
    ykey_classification_str = {Label.DFF: 'dff', Label.ZSCORE: 'zscore'}[ykey_classification]
    processing_id = f'agg{trial_aggfunc.__name__}_class{ykey_classification_str}'
    if directional:
        processing_id = f'{processing_id}_directional'
    
    # Get figures PDF suffix
    prepro_code = []
    if global_correction is not None:
        if isinstance(global_correction, dict):
            global_correction = 'line-specific'
        prepro_code.append(global_correction)
    if kalman_gain is not None:
        prepro_code.append(f'k{kalman_gain}')
    prepro_code = '_'.join(prepro_code)
    figs_suffix = f'{analysis_type}_{dataset_group_id}_{prepro_code}_{conditioning_id}_{processing_id}'

    # Infer GCaMP decay time from mouseline
    gcamp_key = '7f' if mouseline == 'cre_sst' else '6s'
    tau = GCAMP_DECAY_TAU[gcamp_key]
    fs = BERGAMO_RESAMPLED_SR if mouseline == 'cre_sst' else BRUKER_SR
    
    # Get stats input data directory
    if mouseline is not None:
        input_root = get_data_root(kind=DataRoot.PROCESSED)
        input_dir = os.path.join(
            input_root, processing_id, conditioning_id, get_s2p_id(fs, tau), prepro_id, analysis_type, mouseline)
    else:    
        input_root = get_data_root(kind=DataRoot.LINEAGG)
        if isinstance(prepro_id, dict):
            input_dir = {
                k: os.path.join(
                    input_root, processing_id, conditioning_id, get_s2p_id(fs, tau), v, analysis_type)
                for k, v in prepro_id.items()
            }
            intput_dir = {k: v for k, v in input_dir.items() if os.path.exists(v)}
        else:
            input_dir = os.path.join(
                input_root, processing_id, conditioning_id, get_s2p_id(fs, tau), prepro_id, analysis_type)
    
    # Get figures directory
    figsdir = get_data_root(kind=DataRoot.FIG)
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

