# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-05 17:56:34
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-10-25 13:25:00

''' Notebook image viewing utilities. '''

import abc
import numpy as np
import matplotlib.pyplot as plt
import numpy_image_widget as niw
from ipywidgets import IntSlider, VBox, HBox, HTML, interact
from suite2p.io import BinaryFile
from tifffile import TiffFile

from logger import logger


class StackViewer(metaclass=abc.ABCMeta):
    '''
    Simple stack viewer, inspired from Robert Haase's stackview package
    (https://github.com/haesleinhuepf/stackview/).
    '''
    
    def __init__(self, fpaths, headers, title=None, iframe=0, continuous_update=True,
                 display_width=240, display_height=240, cmap='viridis'):
        '''
        Initialization.

        :param fpaths: list of path(s) to the file(s) containing the image stack(s)
        :param headers: list of header(s) associated to each stack file
        :param title: optional title to render above the image(s)
        :param iframe: index of frame to display upon rendering
        :param continuous_update: update the image while dragging the mouse (default: True)
        :param display_width: diplay width (in pixels)
        :param display_height: diplay height (in pixels)
        :param cmap (optional): colormap used to display grayscale image. If none, a gray colormap is used by default. 
        '''          
        logger.info('initializing stack viewer')
        self.stackobjs = [self.get_fileobj(fp) for fp in fpaths]
        self.headers = headers
        self.title = title
        nframes = [self.get_nframes(x) for x in self.stackobjs]
        shapes = [self.get_frame_shape(x) for x in self.stackobjs]
        if not len(list(set(nframes))) == 1:
            raise ValueError(f'Inconsistent number of frames: {nframes}')
        if not len(list(set(shapes))) == 1:
            raise ValueError(f'Inconsistent frame shapes: {shapes}')
        self.nframes = nframes[0]
        self.shape = shapes[0]
        logger.info(f'stack size: {(self.nframes, *self.shape)}')

        # Initialize other attributes
        self.iframe = iframe
        self.display_width = display_width
        self.display_height = display_height
        self.continuous_update = continuous_update

        # Initialize colormap and normalizers
        logger.info('computing stack intensity range...')
        lims = [self.get_intensity_range(fobj) for fobj in self.stackobjs]
        self.norms = [plt.Normalize(vmin=x[0], vmax=x[1]) for x in lims]
        if cmap is not None:
            cmap = plt.get_cmap(cmap)
        self.cmap = cmap

    @abc.abstractmethod
    def get_fileobj(self, fpath):
        ''' Get the binary file object corresponding to a file path '''
        raise NotImplementedError

    @abc.abstractmethod
    def get_nframes(self, fobj):
        ''' Get the number of frames in a stack '''
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_frame_shape(self, fobj, i=0):
        ''' Get the shape of a frame in a stack '''
        raise NotImplementedError

    @abc.abstractclassmethod
    def get_frame(self, fobj, i):
        '''Get a particular frame in the stack '''
        raise NotImplementedError

    # def __del__(self):
    #     ''' Making sure to close all open binary file objects upon deletion. '''
    #     for sobj in self.stackobjs:
    #         sobj.close()

    def get_intensity_range(self, fobj):
        ''' Get the range of pixel intensity across the whole stack. '''
        return (
            min([self.get_frame(fobj, i).min() for i in range(self.nframes)]),
            max([self.get_frame(fobj, i).max() for i in range(self.nframes)])
        )

    def get_slider(self, iframe, nframes):
        ''' Get slider control to change frame. '''
        return IntSlider(
            value=iframe,
            min=0,
            max=nframes - 1,
            continuous_update=self.continuous_update,
            description='Frame'
        )
    
    def get_header(self, text):
        ''' Get header text component '''
        return HTML(value=f'<center>{text}</center>')

    def transform(self, arr, norm):
        ''' Transform a grayscale intensity image to a colored image using a specific colormap '''
        arr = norm(arr)  # normalize frame acccording to entire stack intensity range
        if self.cmap is not None:  # transform into colormap if specified
            arr = self.cmap(arr)[:, :, :-1]
        return arr

    def init_view(self, fobj, norm):
        ''' Initialize view(s).'''
        arr = self.get_frame(fobj, self.iframe)
        view = niw.NumpyImage(self.transform(arr, norm))
        if self.display_width is not None:
            view.width_display = self.display_width
        if self.display_height is not None:
            view.height_display = self.display_height
        return view

    def update(self, event):
        ''' Event handler: update view(s) upon change in slider index. '''
        for i in range(len(self.stackobjs)):
            self.views[i].data = self.transform(
                self.get_frame(self.stackobjs[i], self.slider.value),
                self.norms[i])
            
    def render(self):
        '''
        Render stacks.
        
        :return: ipywidget VBox object dynamically rendering the stack
        '''       
        logger.info('rendering stack view...')
        # Initialize views
        self.views = []
        for sobj, norm in zip(self.stackobjs, self.norms):
            self.views.append(self.init_view(sobj, norm))
        # Set up slider control to change frame
        self.slider = self.get_slider(self.iframe, self.nframes)
        # Connect slider to image view
        self.slider.observe(self.update)
        # Return ipywidget
        container = HBox([VBox([self.get_header(h), v]) for h, v in zip(self.headers, self.views)])
        if self.title is not None:
            container = HBox([VBox([self.get_header(self.title), container])])
        return VBox([container, self.slider])


class TifStackViewer(StackViewer):
    
    def get_fileobj(self, fpath):
        ''' Get the binary file object corresponding to a file path '''
        return TiffFile(fpath)

    def get_nframes(self, fobj):
        ''' Get the number of frames in a stack '''
        return len(fobj.pages)
    
    def get_frame_shape(self, fobj, i=0):
        ''' Get the shape of a frame in a stack '''
        return fobj.pages[i].shape

    def get_frame(self, fobj, i):
        '''Get a particular frame in the stack '''
        return fobj.pages[i].asarray()


class Suite2pRegisteredStackViewer(StackViewer):
    
    def get_fileobj(self, ops):
        ''' Get the binary file object corresponding to a suite2p options dictionary '''
        return BinaryFile(Ly=ops['Ly'], Lx=ops['Lx'], read_filename=ops['reg_file'])

    def get_nframes(self, fobj):
        ''' Get the number of frames in a stack '''
        return fobj.n_frames
    
    def get_frame_shape(self, fobj, i=0):
        ''' Get the shape of a frame in a stack '''
        return fobj[i][0].shape

    def get_frame(self, fobj, i):
        '''Get a particular frame in the stack '''
        return fobj[i][0]


def view_stack(fpaths, *args, **kwargs):
    ''' Interface function for stack(s) viewing. '''
    vclass = TifStackViewer
    if isinstance(fpaths, dict):
        if 'reg_file' in fpaths:
            headers, fpaths = [kwargs.pop('title', None)], [fpaths]
            title = None
            vclass = Suite2pRegisteredStackViewer
        else:
            headers, fpaths = zip(*fpaths.items())
            title = kwargs.pop('title', None)
    else:
        headers, fpaths = [kwargs.pop('title', None)], [fpaths]
        title = None
    viewer = vclass(fpaths, headers, *args, title=title, **kwargs)
    return viewer.render()


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

