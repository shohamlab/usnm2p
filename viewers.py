# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-05 17:56:34
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-12-02 14:52:59

''' Notebook image viewing utilities. '''

import abc
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy_image_widget as niw
from ipywidgets import IntSlider, VBox, HBox, HTML, Button, Output
from IPython.display import display

from suite2p.io import BinaryFile
from tifffile import TiffFile
import imageio as iio
from constants import S2P_UINT16_NORM_FACTOR

from logger import logger
from utils import float_to_uint8


class StackViewer:
    '''
    Simple stack viewer, inspired from Robert Haase's stackview package
    (https://github.com/haesleinhuepf/stackview/).
    '''

    npix_label = 10 # number of pixels used for upper-right labeling
    
    def __init__(self, fpaths, headers, title=None, continuous_update=True,
                 display_width=240, display_height=240, ):
        '''
        Initialization.

        :param fpaths: list of path(s) to the file(s) containing the image stack(s)
        :param headers: list of header(s) associated to each stack file
        :param title: optional title to render above the image(s)
        :param continuous_update: update the image while dragging the mouse (default: True)
        :param display_width: diplay width (in pixels)
        :param display_height: diplay height (in pixels)
        '''
        self.fpaths = fpaths
        logger.info('initializing stack viewer')
        self.fobjs = [self.get_fileobj(fp) for fp in self.fpaths]
        self.headers = headers
        self.title = title
        nframes = [self.get_nframes(x) for x in self.fobjs]
        shapes = [self.get_frame_shape(x) for x in self.fobjs]
        if not len(list(set(nframes))) == 1:
            raise ValueError(f'Inconsistent number of frames: {nframes}')
        if not len(list(set(shapes))) == 1:
            raise ValueError(f'Inconsistent frame shapes: {shapes}')
        self.nframes = nframes[0]
        self.shape = shapes[0]
        logger.info(f'stack size: {(self.nframes, *self.shape)}')

        # Initialize other attributes
        self.display_width = display_width
        self.display_height = display_height
        self.continuous_update = continuous_update

    def get_fileobj(self, fpath):
        ''' Get the binary file object corresponding to a file path '''
        if isinstance(fpath, dict):
            return BinaryFile(Ly=fpath['Ly'], Lx=fpath['Lx'], read_filename=fpath['reg_file'])
        else:
            return TiffFile(fpath)

    def get_nframes(self, fobj):
        ''' Get the number of frames in a stack '''
        if isinstance(fobj, BinaryFile):
            return fobj.n_frames
        else:
            return len(fobj.pages)

    def get_frame_shape(self, fobj, i=0):
        ''' Get the shape of a frame in a stack '''
        if isinstance(fobj, BinaryFile):
            return fobj[i][0].shape
        else:
            return fobj.pages[i].shape

    def get_frame(self, fobj, i):
        '''Get a particular frame in the stack '''
        if isinstance(fobj, BinaryFile):
            return fobj[i][0] * S2P_UINT16_NORM_FACTOR
        else:
            return fobj.pages[i].asarray()

    # def __del__(self):
    #     ''' Making sure to close all open binary file objects upon deletion. '''
    #     for sobj in self.fobjs:
    #         sobj.close()

    def get_frame_range(self, bounds):
        ''' Get a range of frames indexes given specified index bounds '''
        if bounds is None:
            bounds = [0, self.nframes - 1]
        else:
            logger.info(f'frame frange: {bounds}')
        return range(bounds[0], bounds[1] + 1)
    
    def get_intensity_range(self, i, frange):
        ''' Get the range of pixel intensity across the whole stack. '''
        # Get evolution of min and max frame intensity
        Imin, Imax = self.get_frame_metric_evolution(
            self.fobjs[i], frange, func=lambda x: (x.min(), x.max())).T
        Imin, Imax = np.min(Imin), np.max(Imax)
        logger.info(f'intensity range: {Imin} - {Imax}')
        return (Imin, Imax)

    def get_slider(self, frange):
        ''' Get slider control to change frame. '''
        return IntSlider(
            value=frange.start,
            min=frange.start,
            max=frange.stop - 1,
            continuous_update=self.continuous_update,
            description='Frame'
        )
    
    def get_header(self, text):
        ''' Get header text component '''
        return HTML(value=f'<center>{text}</center>')

    def reload_binary_file(self, fobj):
        ''' Close and re-open file-object '''
        fpath, Lx, Ly = fobj.read_filename, fobj.Lx, fobj.Ly
        findex = self.fobjs.index(fobj)
        fobj.close()
        self.fobjs[findex] = BinaryFile(Ly=Ly, Lx=Lx, read_filename=fpath)

    def iter_frames(self, fobj, frange, func=None):
        '''
        Iterate through frames and apply func

        :param fobj: file object
        :param frange: frame index range
        :param func: function applied on the frame at each iteration within the range
        '''
        if func is None:
            func = lambda x: None
        out = []
        if isinstance(fobj, BinaryFile):
            refindex = None
            with tqdm(total=frange.stop - frange.start, position=0, leave=True) as pbar:
                for (index, *_), frame in fobj.iter_frames():
                    if refindex is None:
                        refindex = index
                    real_index = index - refindex
                    if real_index >= frange.start and real_index < frange.stop:
                        # Multiply stack by factor 2 to compensate for suite2p input normalization
                        out.append(func(real_index, frame[0] * S2P_UINT16_NORM_FACTOR))
                        pbar.update()
            # FIX: reload binary file object to reset internal index and make sure
            # next iter_frames works correctly
            self.reload_binary_file(fobj)
        else:
            for k in tqdm(list(frange)):
                frame = self.get_frame(fobj, k)
                out.append(func(k, frame))
        return out

    def get_frame_metric_evolution(self, fobj, frange, func=None):
        ''' Compute the time course of a given frame metrics across a specific range of frames. '''
        if func is None:
            func = lambda x: x.mean()
        xlist = self.iter_frames(fobj, frange, func=lambda _, x: func(x))
        return np.array(xlist)

    def transform(self, arr):
        ''' Transform a grayscale intensity image to a colored image using a specific colormap '''
        return self.cmap(arr)[:, :, :-1]
    
    def label(self, arr):
        ''' Label a frame by setting pixels on the upper-right corner to red. '''
        arr[:self.npix_label, -self.npix_label:, :] = [arr.max(), 0, 0]
        return arr

    def process_frame(self, arr, norm, i):
        '''
        Process frame.
        
        :param arr: frame array
        :param norm: normalizer object
        :return: processed frame (i.e. normalized, transformed and labeled) 
        '''
        if norm is not None:
            arr = norm(arr)
        arr = self.transform(arr)
        if self.is_labeled[i]:
            arr = self.label(arr)
        return arr

    def init_view(self):
        ''' Initialize view with random data.'''
        view = niw.NumpyImage(self.transform(np.random.rand(*self.shape)))
        if self.display_width is not None:
            view.width_display = self.display_width
        if self.display_height is not None:
            view.height_display = self.display_height
        return view
    
    def update(self, event):
        ''' Event handler: update view(s) upon change in slider index. '''
        for i in range(len(self.fpaths)):
            arr = self.get_frame(self.fobjs[i], self.slider.value)
            self.views[i].data = self.process_frame(arr, self.norms[i], self.slider.value)

    def init_render(self, norm=True, cmap='viridis', bounds=None, ilabels=None):
        '''
        Initialize stacks rendering.
        
        :param norm (default = True): whether to normalize the stack data to [0-1] range upon rendering
        :param cmap (optional): colormap used to display grayscale image. If none, a gray colormap is used by default.
        :param bounds (optional): boundary frame indexes. If none, the entire stack is rendered.
        :param ilabels (optional): array of frame indexes to label.
        '''
        # Get frame range
        self.frange = self.get_frame_range(bounds)
        # Initialize label arrray
        is_labeled = np.zeros(self.nframes)
        if ilabels is not None:
            is_labeled[ilabels] = 1.
        self.is_labeled = is_labeled.astype(bool)
        # Initialize colormap
        if cmap is None:
            cmap = 'gray'
        self.cmap = plt.get_cmap(cmap)
        # Initialize normalizers
        if norm:
            logger.info(f'computing stack intensity range across {self.frange.start} - {self.frange.stop -1} frame range...')
            lims = [self.get_intensity_range(i, self.frange) for i in range(len(self.fobjs))]
            self.norms = [plt.Normalize(vmin=x[0], vmax=x[1]) for x in lims]
        else:
            self.norms = [None] * len(self.fobjs)
            
    def render(self, *args, **kwargs):
        '''
        Render stacks
        
        :param args: positional arguments passed on to init_render method
        :param kwargs: keyword arguments passed on to init_render method
        :return: VBox ipywidget to render the stacks interactively
        '''
        self.init_render(*args, **kwargs)
        logger.info('rendering stack view...')
        # Initialize views - DO NOT USE LIST COMPREHENSION HERE TO PRESERVE CLOSURE!!!
        self.views = [] 
        for i in range(len(self.fpaths)):
            self.views.append(self.init_view())
        # Set up slider control to change frame
        self.slider = self.get_slider(self.frange)
        # Connect slider to image view and update view with first frame in range
        self.slider.observe(self.update)
        self.update(None)
        # Return ipywidget
        container = HBox([VBox([self.get_header(h), v]) for h, v in zip(self.headers, self.views)])
        if self.title is not None:
            container = HBox([VBox([self.get_header(self.title), container])])
        return VBox([container, self.slider])

    def save_as_gif(self, outdir, fps):
        ''' Save stack(s) as GIF(s) '''
        # For each stack object
        for i, (header, fpath, fobj, norm) in enumerate(zip(self.headers, self.fpaths, self.fobjs, self.norms)):
            # Fetch output file path from header (if any) or from input file path
            if header is None:
                fcode = os.path.splitext(os.path.basename(fpath))[0]
            else:
                fcode = header
            new_fpath = os.path.join(os.path.abspath(outdir), f'{fcode}.gif')
            # Export stack to gif, frame by frame
            logger.info(f'exporting "{fcode}.gif"...')
            with iio.get_writer(new_fpath, mode='I', duration=1 / fps) as writer:
                def add_frame(iframe, x):
                    writer.append_data(float_to_uint8(self.process_frame(x, self.norms[i], iframe)))
                self.iter_frames(fobj, self.frange, add_frame)


def is_s2p_dict(d):
    ''' Determine if dictionary is a suite2p output option dictionary '''
    return isinstance(d, dict) and 'reg_file' in d


def get_stack_viewer(fpaths, *args, **kwargs):
    ''' Interface function to instanciate stack(s) viewers. '''
    # If fpaths if a dictionary but not a suite2p options dictionary
    if isinstance(fpaths, dict) and not is_s2p_dict(fpaths):
        # Extract headers and filepaths from dictionary items
        headers, fpaths = zip(*fpaths.items())
        # Get title from input arguments
        title = kwargs.pop('title', None)

    # If fpaths is a single file instance or a suite2p output options dictionary
    else:
        # Extract header from title and populate single filepaths list
        headers, fpaths = [kwargs.pop('title', None)], [fpaths]
        # Set no title (because single view does not need it)
        title = None
    # Return stack viewer initialized with appropriate fpaths, headers and title
    return StackViewer(fpaths, headers, *args, title=title, **kwargs)


def view_stack(*args, **kwargs):
    ''' Interface function to view stacks '''
    norm = kwargs.pop('norm', True)
    cmap = kwargs.pop('cmap', 'viridis')
    bounds = kwargs.pop('bounds', None)
    ilabels = kwargs.pop('ilabels', None)
    return get_stack_viewer(*args, **kwargs).render(
        norm=norm, cmap=cmap, bounds=bounds, ilabels=ilabels)



class InteractivePlotViewer:
    ''' Class to plot figure across range of conditions '''

    def __init__(self, pltfunc, n):
        self.pltfunc = pltfunc
        self.n = n
        self.prev_button = Button(description='<', tooltip='Prev')
        self.next_button = Button(description='>', tooltip='Next')
        self.prev_button.on_click(self.plot_prev)
        self.next_button.on_click(self.plot_next)
        self.out = Output()
        self.view = VBox([self.out, HBox([self.prev_button, self.next_button])])
        self.i = 0
        self.pltwrap()
        display(self.view)

    def plot_next(self, *args):
        if self.i < self.n - 1:
            self.i += 1
            self.pltwrap()

    def plot_prev(self, *args):
        if self.i > 0:
            self.i -= 1
            self.pltwrap()

    def pltwrap(self):
        self.out.clear_output()
        with self.out:
            lvl = logger.getEffectiveLevel() 
            logger.setLevel(logging.ERROR)
            self.pltfunc(self.i)
            logger.setLevel(lvl)
            plt.show()

    
def view_interactive_plot(*args, **kwargs):
    return InteractivePlotViewer(*args, **kwargs)