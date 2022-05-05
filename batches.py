# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-08-22 14:33:04
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-05-05 09:53:10

''' Batch processing utilities '''

from datetime import datetime
import logging
import multiprocessing as mp

from logger import logger
from utils import itemize


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
    def queue_str(queue, nmax=20):
        if len(queue) <= nmax:
            return itemize(queue)
        else:
            s_start = itemize(queue[:nmax // 2])
            s_end = itemize(queue[-nmax // 2:])
            nbetween = len(queue) - nmax
            return f'{s_start}\n... {nbetween} more entries ...{s_end}'
    
    def ask_confirm(self, mpi):
        ''' Ask user for confirmation before running batch '''
        mpistr = 'with' if mpi else 'without'
        answer = input(f'run {self} {mpistr} multiprocessing ? (y/n):')
        return answer.lower() == 'y'
