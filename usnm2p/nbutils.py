# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-01-06 11:17:50
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2025-07-18 14:40:10

''' Notebook running utilities '''

# External modules
import jupyter_slack
import os
import papermill as pm
from nbclient.exceptions import DeadKernelError
from argparse import ArgumentParser, ArgumentTypeError

# Internal modules
from .logger import logger
from .batches import Batch, create_queue
from .constants import *
from .utils import as_iterable
from .parsers import add_dataset_arguments


def none_or_float(value):
    ''' Custom function to parse None or a float '''
    if isinstance(value, str):
        if value.lower() == 'none':
            return None
        try:
            return float(value)
        except ValueError:
            raise ArgumentTypeError(f'Could not parse "{value}" into None/float')
    elif isinstance(value, (int, float)):
        return value
    else:
        raise ArgumentTypeError(f'Could not parse {type(value)}-typed input into None/float')


def none_or_str(value):
    ''' Custom function to parse None or a string '''
    if not isinstance(value, str):
        raise ArgumentTypeError(f'Could not parse {type(value)}-typed input into string/float')
    if value.lower() == 'none':
        return None
    return value


def get_notebook_parser(input_nb, analysis=True, line=False, date=False, mouse=False, region=False, layer=False, exec=True):
    ''' Create command line parser for notebook execution '''
    # Create command line parser
    parser = ArgumentParser()
    
    # Add input / output / mpi / check arguments
    parser.add_argument(
        '-i', '--input', default=input_nb,
        help='path to input notebook')
    parser.add_argument(
        '-o', '--outdir', default='../nboutputs', 
        help='relative path to output directory w.r.t. this script')
    parser.add_argument(
        '--mpi', default=False, action='store_true', help='enable multiprocessing')
    parser.add_argument(
        '--go', default=False, action='store_true', help='start without user check')
    
    # Add slack notification argument
    parser.add_argument(
        '--slack_notify', action='store_true', help='Notify on slack')
    parser.add_argument(
        '--no-slack_notify', dest='slack_notify', action='store_false')
    parser.set_defaults(slack_notify=True)

    # Add specfied dataset arguments
    add_dataset_arguments(parser, analysis=analysis, line=line, date=date, mouse=mouse, region=region, layer=layer)

    # Add arguments about other execution parameters
    if exec:
        parser.add_argument(
            '--inspect', default=False, action='store_true',
            help='Inspect data from random run along processing')
        parser.add_argument(
            '-c', '--global_correction', type=none_or_str, default='line', nargs='+',
            help='Global correction method')
        parser.add_argument(
            '-k', '--kalman_gain', type=none_or_float, default=KALMAN_GAIN, nargs='+',
            help='Kalman filter gain (s)')
        parser.add_argument(
            '--alpha', type=float, default=NEUROPIL_SCALING_COEFF, nargs='+',
            help='scaling coefficient for neuropil subtraction')    
        parser.add_argument(
            '-q', '--baseline_quantile', type=none_or_float, default=BASELINE_QUANTILE, nargs='+',
            help='Baseline evaluation quantile')
        parser.add_argument(
            '--wq', type=float, default=BASELINE_WQUANTILE, nargs='+',
            help='Baseline quantile filter window size (s)')
        parser.add_argument(
            '--ws', type=none_or_float, default=BASELINE_WSMOOTHING, nargs='+',
            help='Baseline gaussian filter window size (s)')
        parser.add_argument(
            '-y', '--ykey_classification', type=str, default='zscore', choices=['dff', 'zscore', 'evrate'], nargs='+',
            help='Classification variable')
        parser.add_argument(
            '--directional', action='store_true', help='Directional classification')
        parser.add_argument(
            '--non-directional', dest='directional', action='store_false')
        parser.set_defaults(directional=True)
    
    # Return parser
    return parser


def parse_notebook_exec_args(args):
    ''' 
    Parse notebook execution arguments
    
    :param args: command line arguments dictionary
    :return: 5-tuple with:
        - input notebook path
        - output directory path
        - MPI flag
        - ask confirmation flag
        - processing parameters queue
    '''
    # Extract execution arguments
    input_nbpath = args.pop('input')
    outdir = args.pop('outdir')
    mpi = args.pop('mpi')
    ask_confirm = not args.pop('go')
    
    # Extract notebook processing parameters
    proc_keys = [
        'slack_notify',
        'inspect',
        'global_correction',
        'kalman_gain',
        'alpha',
        'baseline_quantile',
        'wq',
        'ws',
        'ykey_classification',
        'directional'
    ]
    proc_args = {}
    for k in proc_keys:
        if k in args:
            proc_args[k] = args.pop(k)

    # Cast processing parameters as lists
    proc_args = {k: as_iterable(v) for k, v in proc_args.items()}

    # Rename some processing parameters for consistency with notebook definitions
    if 'alpha' in proc_args:
        proc_args['neuropil_scaling_coeff'] = proc_args.pop('alpha')
    if 'wq' in proc_args:
        proc_args['baseline_wquantile'] = proc_args.pop('wq')
    if 'ws' in proc_args:
        proc_args['baseline_wsmoothing'] = proc_args.pop('ws')
    if 'ykey_classification' in proc_args:
        proc_args['ykey_classification'] = [
            {
                'evrate': Label.EVENT_RATE, 
                'dff': Label.DFF,
                'zscore': Label.ZSCORE
            }[y]
            for y in proc_args['ykey_classification']]
    
    # Create queue from processing parameters
    proc_queue = create_queue(proc_args)

    # Return outputs
    return input_nbpath, outdir, mpi, ask_confirm, proc_queue


def execute_notebook(pdict, input_nbpath, outdir):
    '''
    Wrapper around papermill's notebook execution function
    
    :param pdict: parameters dictionary (will be injected into the notebook)
    :param input_nbpath: path (relative or absolute) to the input notebook
    :param outdir: path (relative or absolute) to the output directory in which to save the output notebook
    :return: absolute path to the output notebook
    '''
    # Convert input notebook path to absolute path
    input_nbpath = os.path.abspath(input_nbpath)

    # Convert output directory path to absolute path
    outdir = os.path.abspath(outdir)

    # Extract notebook name and extension
    input_nbname, nbext = os.path.splitext(os.path.basename(input_nbpath))

    # Determine output notebook name from input and parameters
    pstr = '_'.join([str(v) for k, v in pdict.items() if k != 'no_slack_notify'])
    pstr = pstr.replace('/', '_')
    output_nbname = f'{input_nbname}_{pstr}{nbext}'
    output_nbpath = os.path.join(outdir, output_nbname)

    # Set up infinite loop for notebook execution
    while True:
        # Execute notebook, and return output notebook filepath if successful
        try:
            logger.info(f'executing "{output_nbname}"...')
            pm.execute_notebook(input_nbpath, output_nbpath, parameters=pdict)
            logger.info(f'{output_nbpath} notebook successfully executed')
            return output_nbpath
        
        # Catch generated errors
        except Exception as err:
            # If execution error
            if isinstance(err, (pm.exceptions.PapermillExecutionError, DeadKernelError, RuntimeError)):
                # Log error and notify on slack
                s = f'"{output_nbname}" execution error: {err}'
                logger.error(s)
                if pdict['slack_notify']:
                    jupyter_slack.notify_self(s)

                # If error is due resource sharing limitation, notify and retry execution
                if any([x in str(err) for x in NB_RETRY_ERRMSGS]):
                    logger.info(f're-trying execution of "{output_nbname}"...')            
                # Otherwise, abandon execution and return None
                else:
                    return None
            
            # If other error, log and notify on slack, and return None
            else:
                s = f'"{output_nbname}" unknown error: {err}'
                logger.error(s)
                if pdict['slack_notify']:
                    jupyter_slack.notify_self(s)
                return None


def execute_notebooks(pdicts, input_nbpath, outdir, **kwargs):
    '''
    Carry out multiple executions of a notebook with different input parameters combinations.

    :param pdicts: list of parameters dictionary used for each notebook execution
    :param input_nbpath: path to the input notebook
    :param outdir: path to the output directory in which to save the output notebooks
    :return: list of paths to the output notebooks
    '''
    # Establish batch queue with each parameter combination
    queue = [list(x) for x in zip(*[pdicts, [input_nbpath] * len(pdicts), [outdir] * len(pdicts)])]

    # For each queue entry, if "global correction" field is present and set to "line", 
    # change it to corresponding line-specific value
    for entry in queue:
        if 'global_correction' in entry[0] and entry[0]['global_correction'] == 'line':
            entry[0]['global_correction'] = GLOBAL_CORRECTION[entry[0]['mouseline']]

    # Create and run job batch
    batch = Batch(execute_notebook, queue)
    nbpaths = batch.run(loglevel=logger.getEffectiveLevel(), **kwargs)
    # Log completion message
    if nbpaths is None:
        is_completed = [False] * len(queue)
    else:
        is_completed = [x is not None for x in nbpaths]
    logger.info(f'{sum(is_completed)}/{len(is_completed)} jobs completed.')
    # Return output filepaths
    return nbpaths


class DirectorySwicther:
    ''' Context manager ensuring temporary code inside a specified directory '''

    def __init__(self, exec_dir):
        self.exec_dir = exec_dir
        self.call_dir = os.getcwd()
        self.move = self.call_dir != self.exec_dir
          
    def __enter__(self):
        if self.move:
            logger.info(f'moving to "{self.exec_dir}"')
            os.chdir(self.exec_dir)
        return self
      
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.move:
            logger.info(f'moving back to "{self.call_dir}"')
            os.chdir(self.call_dir)
 