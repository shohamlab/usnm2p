# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-05 17:56:34
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-10-14 19:32:08

import numpy as np
import numpy_image_widget as niw
from ipywidgets import IntSlider, VBox, HBox, HTML

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


def view(stack, *args, **kwargs):
    ''' Interface function for single stack viewing. '''
    title = kwargs.pop('title', None)
    viewer = StackViewer(*args, **kwargs)
    return viewer.render(stack, title=title)


class DualStackViewer(StackViewer):
    ''' Class implementing dual stack viewer. '''

    def __init__(self, *args, view_diff=False, **kwargs):
        ''' Initialization. '''
        super().__init__(*args, **kwargs)
        self.view_diff = view_diff

    def get_diff(self, iframe):
        ''' Get differential view between the two stacks for a given frame index. '''
        return self.stack2[iframe] - self.stack1[iframe]

    def update(self, event):
        ''' Event handler: update views upon change in slider index. '''
        self.view1.data = self.stack1[self.slider.value]
        self.view2.data = self.stack2[self.slider.value]
        if self.view_diff:
            self.view3.data = self.get_diff(self.slider.value)

    def render(self, stack1: np.array, stack2: np.array, title1=None, title2=None, suptitle=None):
        ''' Render stacks.
            
            :param stack1: 3D numpy array containing the 1st image stack
            :param stack2: 3D numpy array containing the 2nd image stack
            :param title1: optional title to render above the 1st image
            :param title2: optional title to render above the 2nd image
            :param suptitle: optional super-title to render above the 2 images
            :return: ipywidget VBox object dynamically rendering the stack
        '''
        assert stack1.shape == stack2.shape, ''' Stacks are of different sizes '''
        logger.info(f'stack size: {stack1.shape}')
        assert (title1 is None and title2 is None) or (title1 is not None and title2 is not None), ''' Labels cannot be partially provided '''
        logger.info('rendering stacks view...')
        titles = [title1, title2]
        self.stack1, self.stack2 = stack1, stack2
        nframes, width, height = self.stack1.shape
        # Initialize views
        self.view1 = self.init_view(self.stack1[self.iframe])
        self.view2 = self.init_view(self.stack2[self.iframe])
        if self.view_diff:
            self.view3 = self.init_view(self.get_diff(self.iframe))
        # Set up slider control to change frame
        self.slider = self.init_slider(self.iframe, nframes)
        # Connect slider to image view
        self.slider.observe(self.update)
        # Return ipywidget
        views = [self.view1, self.view2]
        if self.view_diff:
            views.append(self.view3)
            titles.append('Diff')
        if title1 is not None:
            views = [VBox([self.init_title(title), x]) for title, x in zip(titles, views)]
        views = HBox(views)
        if suptitle is not None:
            views = HBox([VBox([self.init_title(suptitle), views])])
        return VBox([views, self.slider])


def dualview(stack1, stack2, *args, **kwargs):
    ''' Interface function for dual stack viewing '''
    title1 = kwargs.pop('title1', None)
    title2 = kwargs.pop('title2', None)
    suptitle = kwargs.pop('suptitle', None)
    viewer = DualStackViewer(*args, **kwargs)
    return viewer.render(stack1, stack2, title1=title1, title2=title2, suptitle=suptitle)
