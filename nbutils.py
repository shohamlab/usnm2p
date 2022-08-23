# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-01-06 11:17:50
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-08-23 10:27:43

''' Notebook running utilities '''

import jupyter_slack
import os
import papermill as pm
from nbclient.exceptions import DeadKernelError

from logger import logger
from batches import Batch


def execute_notebook(pdict, input_nbpath, outdir):
    '''
    Wrapper around papermill's notebook execution function
    
    :param pdict: parameters dictionary (will be injected into the notebook)
    :param input_nbpath: path to the input notebook
    :param outdir: path to the output directory in which to save the output notebook
    :return: path to the output notebook
    '''
    # Extract notebook name and extension
    nbname, nbext = os.path.splitext(input_nbpath)
    # Determine output notebook name from input and parameters
    pstr = '_'.join([str(v) for k, v in pdict.items() if k != 'no_slack_notify'])
    pstr = pstr.replace('/', '_')
    output_nbname = f'{nbname}_{pstr}{nbext}'
    output_nbpath = os.path.join(outdir, output_nbname)
    # Execute notebook
    logger.info(f'executing "{output_nbname}"...')
    try:
        pm.execute_notebook(input_nbpath, output_nbpath, parameters=pdict)
    except (pm.exceptions.PapermillExecutionError, DeadKernelError) as err:
        s = f'"{output_nbname}" execution error: {err}'
        logger.error(s)
        jupyter_slack.notify_self(s)
        return None
    logger.info(f'{output_nbpath} notebook successfully executed')
    return output_nbpath


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
    # Create and run job batch
    batch = Batch(execute_notebook, queue)
    nbpaths = batch.run(loglevel=logger.getEffectiveLevel(), **kwargs)
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
 