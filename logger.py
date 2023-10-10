# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-11 13:30:15
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-10-10 12:02:36

''' Collection of logging utilities. '''

import sys
import colorlog
import logging
import tqdm


# Define custom log formatter with date:time and level-dependent colors
my_log_formatter = colorlog.ColoredFormatter(
    '%(log_color)s %(asctime)s %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S:',
    reset=True,
    log_colors={
        'DEBUG': 'green',
        'INFO': 'white',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
    style='%'
)


def initLogHandler(formatter):
    ''' Initialize log handler. '''
    # Initialize log handler
    handler = colorlog.StreamHandler()
    # Set handler formatter
    handler.setFormatter(formatter)
    # Set handler target stream
    handler.stream = sys.stdout
    # Return handler
    return handler


def assignLogHandler(logger, handler):
    ''' Assign handler to logger. '''
    # Remove all previous handlers
    while logger.handlers:
        logger.handlers.pop()

    # Add hander to logger
    logger.addHandler(handler)

    # Check that only one handler is assigned to logger
    if len(logger.handlers) > 1:
        raise ValueError('multiple handlers assigned to logger')


def setLogger(name, formatter, level=logging.INFO):
    ''' 
    Set up logger with given name and formatter.
    
    :param name: logger name
    :param formatter: log formatter
    :return: logger
    '''
    # Fetch logger
    logger = colorlog.getLogger(name)

    # Initialize log handler
    handler = initLogHandler(formatter)
    # print(f'initialized {handler} handler')

    # Assign handler to logger
    assignLogHandler(logger, handler)
    # print(f'assigned {handler} to {logger}')

    # Set logger level
    logger.setLevel(level)

    # Return logger
    return logger


def getMyLogger():
    ''' wrapper function to define logger '''
    return setLogger('mylogger', my_log_formatter, level=logging.INFO)


logger = getMyLogger()