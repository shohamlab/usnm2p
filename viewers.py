# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-05 17:56:34
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-10-22 13:03:47

import numpy as np
import matplotlib.pyplot as plt
import numpy_image_widget as niw
from ipywidgets import IntSlider, VBox, HBox, HTML, interact
from suite2p.io import BinaryFile

from logger import logger

''' Notebook image viewing utilities, inspired from Robert Haase's
    stackview package (https://github.com/haesleinhuepf/stackview/).
'''

class StackViewer:
    ''' Class implementing single stack viewer. '''

    def __init__(
        self,
        iframe: int = 0,
        display_width: int = 240,
        display_height: int = 240,
        continuous_update: bool = True):
        ''' Initialization.

            :param iframe: index of frame to display upon rendering
            :param display_width: diplay width (in pixels)
            :param display_height: diplay height (in pixels):
            :param continuous_update: update the image while dragging the mouse (default: True)
        '''
        self.iframe = iframe
        self.display_width = display_width
        self.display_height = display_height
        self.continuous_update = continuous_update
    
    def init_view(self, image):
        ''' Initialize view .'''
        view = niw.NumpyImage(image)
        if self.display_width is not None:
            view.width_display = self.display_width
        if self.display_height is not None:
            view.height_display = self.display_height
        return view

    def init_slider(self, iframe, nframes):
        ''' Initialize slider control to change frame. '''
        return IntSlider(
            value=iframe,
            min=0,
            max=nframes - 1,
            continuous_update=self.continuous_update,
            description='Frame'
        )
    
    def init_title(self, text):
        ''' Get title text component '''
        # return Text(value=text)
        return HTML(value=f'<center>{text}</center>')

    def update(self, event):
        ''' Event handler: update view upon change in slider index. '''
        self.view.data = self.stack[self.slider.value]

    def render(self, stack: np.array, title=None) -> VBox:
        ''' Render stack.
            
            :param stack: 3D numpy array containing the image stack
            :param title: optional title to render above the image
            :return: ipywidget VBox object dynamically rendering the stack
        '''
        logger.info(f'stack size: {stack.shape}')
        logger.info('rendering stack view...')
        self.stack = stack
        nframes, width, height = self.stack.shape
        # Initialize view
        self.view = self.init_view(self.stack[self.iframe])
        # Set up slider control to change frame
        self.slider = self.init_slider(self.iframe, nframes)
        # Connect slider to image view
        self.slider.observe(self.update)
        # Return ipywidget
        view = self.view
        if title is not None:
            view = HBox([VBox([self.init_title(title), view])])
        return VBox([view, self.slider])


def viewstack(stack, *args, **kwargs):
    ''' Interface function for single stack viewing. '''
    title = kwargs.pop('title', None)
    viewer = StackViewer(*args, **kwargs)
    return viewer.render(stack, title=title)


class StacksViewer(StackViewer):
    ''' Class implementing multiple stacks viewer. '''

    def __init__(self, *args, **kwargs):
        ''' Initialization. '''
        super().__init__(*args, **kwargs)

    def update(self, event):
        ''' Event handler: update views upon change in slider index. '''
        for i in range(len(self.stacks)):
            self.views[i].data = self.stacks[i][self.slider.value]

    def render(self, stacks_dict, suptitle=None):
        ''' Render stacks.
            
            :param stacks_dict: dictionary of 3D stacks
            :param suptitle: optional super-title to render above the 2 images
            :return: ipywidget VBox object dynamically rendering the stack
        '''
        titles = list(stacks_dict.keys())
        dims = stacks_dict[titles[0]].shape
        assert all(x.shape == dims for x in stacks_dict.values()), ''' Stacks are of different sizes '''
        logger.info(f'stack size: {dims}')
        logger.info('rendering stacks view...')
        self.stacks = list(stacks_dict.values())
        nframes, width, height = dims
        # Initialize views
        self.views = []
        for stack in self.stacks:
            self.views.append(self.init_view(stack[self.iframe]))
        # Set up slider control to change frame
        self.slider = self.init_slider(self.iframe, nframes)
        # Connect slider to image view
        self.slider.observe(self.update)
        # Return ipywidget
        container = HBox([VBox([self.init_title(title), x]) for title, x in zip(titles, self.views)])
        if suptitle is not None:
            container = HBox([VBox([self.init_title(suptitle), container])])
        return VBox([container, self.slider])



def viewstacks(stacks_dict, *args, **kwargs):
    ''' Interface function for multiple stack viewing '''
    suptitle = kwargs.pop('suptitle', None)
    viewer = StacksViewer(*args, **kwargs)
    return viewer.render(stacks_dict, suptitle=suptitle)



def plot_registered_frame(iframe, ops):
    '''
    Plot a specific frame of a suite2p registered stack file.
    
    :param iframe: frame index
    :param ops: suite2p output options dictionary
    '''
    with BinaryFile(Ly=ops['Ly'], Lx=ops['Lx'], read_filename=ops['reg_file']) as f:
        plt.figure(figsize=(10, 10))
        plt.imshow(f[iframe][0])


def view_registered_stack(ops):
    '''
    Initiate an interactive view for the suite2p registered stack.
    
    :param ops: suite2p output options dictionary
    '''
    interact(lambda frame: plot_registered_frame(frame, ops), frame=(0, ops['nframes'] -1, 1))
