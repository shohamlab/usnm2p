# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-08-22 14:33:04
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-11-10 11:08:20

''' Batch processing utilities '''

import logging
from datetime import datetime
import multiprocessing as mp
import numpy as np
import pandas as pd

from logger import logger


class Consumer(mp.Process):
    ''' Generic consumer process, taking tasks from a queue and outputing results in
        another queue.
    '''

    def __init__(self, queue_in, queue_out):
        mp.Process.__init__(self)
        self.queue_in = queue_in
        self.queue_out = queue_out
        logger.info(f'Starting {self.name}')

    def run(self):
        while True:
            nextTask = self.queue_in.get()
            if nextTask is None:
                logger.debug(f'Exiting {self.name}')
                self.queue_in.task_done()
                break
            answer = nextTask()
            self.queue_in.task_done()
            self.queue_out.put(answer)
        return


class Worker:
    ''' Generic worker class calling a specific function with a given set of parameters. '''

    def __init__(self, wid, func, args, kwargs, loglevel):
        ''' Worker constructor.

            :param wid: worker ID
            :param func: function object
            :param args: list of method arguments
            :param kwargs: dictionary of optional method arguments
            :param loglevel: logging level
        '''
        self.id = wid
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.loglevel = loglevel

    def __call__(self):
        ''' Caller to the function with specific parameters. '''
        logger.setLevel(self.loglevel)
        return self.id, self.func(*self.args, **self.kwargs)


class Batch:
    ''' Generic interface to run batches of function calls. '''

    def __init__(self, func, queue):
        ''' Batch constructor.

            :param func: function object
            :param queue: list of list of function parameters
        '''
        self.func = func
        self.queue = queue
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.func.__name__}, queue = \n{self.queue_str(self.queue)}\n)'

    def __call__(self, *args, **kwargs):
        ''' Call the internal run method. '''
        return self.run(*args, **kwargs)

    def getNConsumers(self):
        ''' Determine number of consumers based on queue length and number of available CPUs. '''
        return min(mp.cpu_count(), len(self.queue))

    def start(self):
        ''' Create tasks and results queues, and start consumers. '''
        mp.freeze_support()
        self.tasks = mp.JoinableQueue()
        self.results = mp.Queue()
        self.consumers = [Consumer(self.tasks, self.results) for i in range(self.getNConsumers())]
        for c in self.consumers:
            c.start()

    @staticmethod
    def resolve(params):
        if isinstance(params, list):
            args = params
            kwargs = {}
        elif isinstance(params, tuple):
            args, kwargs = params
        return args, kwargs

    def assign(self, loglevel):
        ''' Assign tasks to workers. '''
        for i, params in enumerate(self.queue):
            args, kwargs = self.resolve(params)
            worker = Worker(i, self.func, args, kwargs, loglevel)
            self.tasks.put(worker, block=False)

    def join(self):
        ''' Put all tasks to None and join the queue. '''
        for i in range(len(self.consumers)):
            self.tasks.put(None, block=False)
        self.tasks.join()

    def get(self):
        ''' Extract and re-order results. '''
        outputs, idxs = [], []
        for i in range(len(self.queue)):
            wid, out = self.results.get()
            outputs.append(out)
            idxs.append(wid)
        return [x for _, x in sorted(zip(idxs, outputs))]

    def stop(self):
        ''' Close tasks and results queues. '''
        self.tasks.close()
        self.results.close()

    def run(self, mpi=False, loglevel=logging.INFO, ask_confirm=False):
        ''' Run batch with or without multiprocessing. '''
        if ask_confirm:
            if not self.ask_confirm(mpi):
                logger.info('aborting batch execution')
                return None
        s = 'en' if mpi else 'dis'
        logger.info(f'Starting {len(self.queue)}-job(s) batch (multiprocessing {s}abled)')
        tstamp_start = datetime.now()
        if mpi:
            self.start()
            self.assign(loglevel)
            self.join()
            outputs = self.get()
            self.stop()
        else:
            outputs = []
            for params in self.queue:
                args, kwargs = self.resolve(params)
                outputs.append(self.func(*args, **kwargs))
        tstamp_end = datetime.now()
        logger.info(f'Batch completed in {tstamp_end - tstamp_start} s')
        return outputs

    @staticmethod
    def queue_str(queue):
        jobsdf = []
        for ijob, jobargs in enumerate(queue):
            jobdict = {}
            iarg = 0
            for item in jobargs:
                if isinstance(item, dict):
                    jobdict.update(item)
                else:
                    jobdict[f'arg{iarg}'] = item
                    iarg += 1
            jobsdf.append(pd.DataFrame(jobdict, index=[ijob]))
        jobsdf = pd.concat(jobsdf, axis=0)
        return jobsdf
    
    def ask_confirm(self, mpi):
        ''' Ask user for confirmation before running batch '''
        mpistr = 'with' if mpi else 'without'
        answer = input(f'run {self} {mpistr} multiprocessing ? (y/n):')
        return answer.lower() == 'y'


def create_queue(params):
    ''' 
    Create a serialized 2D array of all parameter combinations for a series
    of individual parameter sweeps.
    
    :param dims: dictionary of (name: value(s)) for input parameters
    :return: list of (name: value) dictionaries for all parameter combinations
    '''
    # Get data types
    dtypes = {k: list(set([type(vv) for vv in v])) for k, v in params.items()}
    # Make sure data types are uniform for each parameter
    for k, v in dtypes.items():
        if len(v) > 1:
            raise ValueError(f'non-uniform data type for {k} parameter: {v}')
    # Reduce to 1 data type per parameter
    dtypes = {k: v[0] for k, v in dtypes.items()}
    # Construct meshgrid array from parameter values
    pgrid = np.meshgrid(*params.values(), indexing='ij')
    # Reshape to 2D array
    queue = np.stack(pgrid, -1).reshape(-1, len(params))
    # Re-assign keys to each row
    queue = [dict(zip(params.keys(), r)) for r in queue]
    # Re-assign data types
    parsed_queue = []
    for pdict in queue:
        parsed_pdict = {}
        for k, v in pdict.items():
            if v is None:
                parsed_pdict[k] = None
            elif dtypes[k] == bool:
                parsed_pdict[k] = {'True': True, 'False': False}[v]
            else:
                parsed_pdict[k] = dtypes[k](v)
        parsed_queue.append(parsed_pdict)
    # Return queue
    return parsed_queue