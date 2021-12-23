# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-12-22 21:17:15
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-12-23 00:18:34

''' Notebook cell execution skip utilities '''


def skip(line, cell=None):
    '''Skips execution of the current line/cell if line evaluates to True.'''
    if eval(line):
        return
    get_ipython().ex(cell)


def load_ipython_extension(shell):
    '''Registers the skip magic when the extension loads.'''
    shell.register_magic_function(skip, 'line_cell')


def unload_ipython_extension(shell):
    '''Unregisters the skip magic when the extension unloads.'''
    del shell.magics_manager.magics['cell']['skip']