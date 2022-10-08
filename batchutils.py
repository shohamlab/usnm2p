# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-10-07 20:43:12
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-10-07 20:55:48

from constants import *
from fileops import get_data_root, get_output_equivalent
from substitutors import StackSubstitutor
from filters import KalmanDenoiser, NoFilter
from s2putils import *
from logger import logger


def get_batch_settings(analysis_type, mouseline, layer, kalman_gain,
                   baseline_wlen, baseline_quantile, baseline_smoothing, 
                   ykey_postpro):

    logger.info('assembling batch analysis settings...')

    # Get trial-averaged data root folder
    dataroot = get_data_root()
    trialavg_root = get_output_equivalent(dataroot, 'raw', 'trial-averaged')
    # Construct dataset group ID
    dataset_group_id = mouseline
    if layer is not None:
        dataset_group_id = f'{dataset_group_id}_{layer}'
    # Construct post-processing ID
    if baseline_wlen is None:
        baseline_smoothing = False
        baseline_id = ''
    else:
        baseline_id = f'w{baseline_wlen:.1f}s'
    baseline_id = f'{baseline_id}_q{baseline_quantile:.2f}'
    if baseline_smoothing:
        baseline_id = f'{baseline_id}_smooth'
    postpro_id = f'{baseline_id}_{ykey_postpro}'.replace('/', '')
    # Get figures PDF suffix
    figs_suffix = f'{dataset_group_id}_k{kalman_gain}_{postpro_id}'
    # Get trial-averaged input data directory
    submap = [(1, 0), (FrameIndex.STIM - 1, FrameIndex.STIM)]
    ss = StackSubstitutor(submap, repeat_every=NFRAMES_PER_TRIAL)
    kd = KalmanDenoiser(kalman_gain) if kalman_gain > 0 else NoFilter
    s2p_ops = {
        'tau': TAU_GCAMP6S_DECAY,  # timescale of the sensor
        'fs': BRUKER_SR,  # sampling rate (per plane)
        'do_registration': 1,  # whether or not to perform image registration,
        'reg_tif': True,  # whether or not to write the registered binary to tiff files
        'nonrigid': True,  # whether or not to perform non-rigid registration, which splits the field of view into blocks and computes registration offsets in each block separately.
        'denoise': False,  # perform PCA denoising of the registered stack prior to ROI detection
    }
    defops = default_ops()
    s2p_ops = get_normalized_options(s2p_ops, defops)            
    diff_from_default_ops = compare_options(s2p_ops, defops)['input'].to_dict()
    s2p_code = 'suite2p'
    if diff_from_default_ops is not None:
        s2p_code = f'{s2p_code}_{get_options_code(diff_from_default_ops)}'
    trialavg_dir = os.path.join(
        trialavg_root, baseline_id, s2p_code, kd.code, ss.code, analysis_type, mouseline)

    figsdir = get_output_equivalent(dataroot, 'raw', 'figs')

    return dataset_group_id, trialavg_dir, figsdir, figs_suffix


def extract_from_batch_data(data):
    ''' Extract specific fields from batch data '''
    logger.info('extracting timeseries and stats from data...')
    return data['timeseries'], data['stats'], data['ROI_masks'], data['map_ops']

