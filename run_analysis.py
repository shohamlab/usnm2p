# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-12-29 12:43:46
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-12-29 16:45:52

import os
import papermill as pm
from argparse import ArgumentParser

''' Utility script to run analysis notebook '''

# Input & output
input_nbpath = 'analysis.ipynb'  # path to input notebook 
outdir = 'outputs'  # relative path to output directory (w.r.t. this script)

# Default parameters
defaults = {
    'line': 'line3',  # mouse line
    'mouse': 'mouse12',  # mouse number
    'region': 'region1',  # brain region
    'date': '11122019'  # experiment date
}

if __name__ == '__main__':
    # Get absolute path to script directory
    script_fpath = os.path.realpath(__file__)
    script_dir = os.path.split(script_fpath)[0]

    # Moving to script directory if not there already
    call_dir = os.getcwd()
    if call_dir != script_dir:
        print(f'moving to "{script_dir}"')
        os.chdir(script_dir)

    # Extract input notebook and parameters from command line arguments
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', default=input_nbpath, help='input notebook path')
    for k, v in defaults.items():
        parser.add_argument(f'-{k[0]}', f'--{k}', default=v)
    args = vars(parser.parse_args())
    input_nbpath = args.pop('input')
    parameters = args

    # Determine output notebook name from input and parameters
    nbname, nbext = os.path.splitext(input_nbpath)
    output_nbpath = os.path.join(outdir, f'{nbname}_{"_".join(parameters.values())}{nbext}')

    print(f'executing "{input_nbpath}" as "{output_nbpath}" with parameters {parameters}...')
    pm.execute_notebook(input_nbpath, output_nbpath, parameters=parameters)
    print(f'notebook successfully executed')

    # Moving back to calling directory if it was different
    if call_dir != script_dir:
        print(f'moving back to "{call_dir}"')
        os.chdir(call_dir)

