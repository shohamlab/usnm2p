import sys
import colorlog
import logging
import tqdm

# Package logger
my_log_formatter = colorlog.ColoredFormatter(
    '%(log_color)s %(asctime)s %(message)s',
    datefmt='%d/%m/%Y %H:%M:%S:',
    reset=True,
    log_colors={
        'DEBUG': 'green',
        'INFO': 'white',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
    style='%')


def setHandler(logger, handler):
    for h in logger.handlers:
        logger.removeHandler(h)
    logger.addHandler(handler)
    return logger


def setLogger(name, formatter):
    handler = colorlog.StreamHandler()
    handler.setFormatter(formatter)
    handler.stream = sys.stdout
    logger = colorlog.getLogger(name)
    logger.addHandler(handler)
    return logger


class TqdmHandler(logging.StreamHandler):

    def __init__(self, formatter):
        logging.StreamHandler.__init__(self)
        self.setFormatter(formatter)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)


logger = setLogger('mylogger', my_log_formatter)