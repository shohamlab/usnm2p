
from argparse import ArgumentParser
import numpy as np
from nbutils import execute_notebook
from logger import logger

# Parse projection function from command line
parser = ArgumentParser()
parser.add_argument(
    '-p', '--projfunc', type=str, default='median', 
    choices=['median', 'mean'], help='Project function')
args = parser.parse_args()

# Construct parameter dictionary
pdict = {'projfunc': args.projfunc}

# Execute notebook
input_nbpath = 'fov_analysis.ipynb'
outdir = 'outputs'
execute_notebook(pdict, input_nbpath, outdir)

# Log termination
logger.info('done.')