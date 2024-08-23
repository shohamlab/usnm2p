# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-05 17:56:34
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2024-08-07 16:02:25

''' Notebook image viewing utilities. '''

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy_image_widget as niw
from ipywidgets import Label, IntSlider, Play, VBox, HBox, HTML, Button, Output, jslink
from IPython.display import display
from suite2p.io import BinaryFile
from tifffile import TiffFile
import imageio as iio
import matplotlib.colors as mcolors

from .logger import logger
from .utils import float_to_uint8, is_iterable, idx_format, as_iterable
from .imlabel import add_label_to_image

# RGB colormaps for the stack viewer
RGB_COLORS = {
    'green': (0, 1, 0),
    'red': (1, 0, 0),
    'blue': (0, 0, 1),
}
VIEWER_CMAPS = {
    k: mcolors.LinearSegmentedColormap.from_list(k, [(0, 0, 0), c])
    for k, c in RGB_COLORS.items()
}


class StackViewer:
    '''
    Simple stack viewer, inspired from Robert Haase's stackview package
    (https://github.com/haesleinhuepf/stackview/).
    '''

    npix_label = 10 # number of pixels used for upper-right labeling
    
    def __init__(self, fpaths, headers, title=None, continuous_update=True, display_size=350, 
                 nchannels=1, verbose=True):
        '''
        Initialization.

        :param fpaths: list of path(s) to the file(s) containing the image stack(s)
        :param headers: list of header(s) associated to each stack file
        :param title: optional title to render above the image(s)
        :param continuous_update: update the image while dragging the mouse (default: True)
        :param display_size: size of the render (in pixels per axis)
        '''
        self.logfunc = logger.info if verbose else logger.debug
        self.fpaths = fpaths
        self.logfunc('initializing stack viewer')
        self.fobjs = [self.get_fileobj(fp) for fp in self.fpaths]
        self.headers = headers
        self.title = title
        self.nchannels = nchannels
        nframes = [self.get_nframes(x) for x in self.fobjs]
        shapes = [self.get_frame_shape(x) for x in self.fobjs]
        if not len(list(set(nframes))) == 1:
            raise ValueError(f'Inconsistent number of frames: {nframes}')
        if not len(list(set(shapes))) == 1:
            raise ValueError(f'Inconsistent frame shapes: {shapes}')
        self.nframes = nframes[0]
        self.shape = shapes[0]
        self.logfunc(f'stack size: {(self.nframes, *self.shape)}')

        # Initialize other attributes
        self.display_width = self.display_height = display_size
        self.continuous_update = continuous_update
    
    def __repr__(self):
        dims = {
            'stack': len(self.fpaths), 
            'channel': self.nchannels,
            'frame': self.nframes, 
        }
        l = [f'{v} {k}{"s" if v > 1 else ""}' for k, v in dims.items()]
        l.append(f'{self.shape[1]}-by-{self.shape[0]} pixels')
        return f'{self.__class__.__name__}({", ".join(l)})'

    def get_fileobj(self, fpath):
        ''' Get the binary file object corresponding to a file path '''
        if isinstance(fpath, dict):
            return BinaryFile(
                Ly=fpath['Ly'], 
                Lx=fpath['Lx'], 
                read_filename=fpath['reg_file']
            )
        elif isinstance(fpath, np.ndarray):
            return fpath
        else:
            return TiffFile(fpath)

    def get_nframes(self, fobj):
        ''' Get the number of frames in a stack '''
        if isinstance(fobj, BinaryFile):
            return fobj.n_frames
        elif isinstance(fobj, np.ndarray):
            return fobj.shape[0]
        else:
            return len(fobj.pages) // self.nchannels
    
    def pageidx(self, i, ichannel=0):
        ''' Get "page" (i.e., serialized) index from frame index and channel index '''
        return i * self.nchannels + ichannel

    def get_frame_shape(self, fobj, i=0, **kwargs):
        ''' Get the shape of a frame in a stack '''
        if isinstance(fobj, BinaryFile):
            return fobj[i][0].shape
        elif isinstance(fobj, np.ndarray):
            return fobj[i].shape[-2:]
        else:
            return fobj.pages[self.pageidx(i, **kwargs)].shape

    def get_frame(self, fobj, i, ichannel=0):
        ''' 
        Get a particular frame in the stack
        
        :param fobj: file object
        :param i: frame index
        :param ichannel (optional): channel index (default: 0)
        :return: frame array
        '''
        if isinstance(fobj, BinaryFile):
            return fobj[i][0]
        elif isinstance(fobj, np.ndarray):
            return fobj[i, ichannel]
        else:
            return fobj.pages[self.pageidx(i, ichannel=ichannel)].asarray()
    
    # def __del__(self):
    #     ''' Making sure to close all open binary file objects upon deletion. '''
    #     for sobj in self.fobjs:
    #         sobj.close()

    def get_frame_index_range(self, fbounds):
        ''' Get a range of frames indexes given specified index bounds '''
        if fbounds is None:
            fbounds = [0, self.nframes - 1]
        else:
            self.logfunc(f'frame frange: {fbounds}')
        return range(fbounds[0], fbounds[1] + 1)
    
    def get_dynamic_range(self, i, frange, **kwargs):
        ''' 
        Get dynamic range of pixel values across a stack.
        
        :param i: stack index
        :param frange: frame index range
        :return: lower and upper bounds of pixel values across the stack
        '''
        self.logfunc(f'computing stack dynamic range across frames {frange.start} - {frange.stop -1}...')
        # Get vectors of min and max intensity values across frames
        Imin, Imax = self.get_frame_metric_evolution(
            self.fobjs[i], frange, func=lambda x: (x.min(), x.max()), **kwargs).T
        # Get min and max values across the stack
        Imin, Imax = np.min(Imin), np.max(Imax)
        # Log and return
        self.logfunc(f'stack dynamic range range: {Imin} - {Imax}')
        return (Imin, Imax)
    
    def slider_format(self, val):
        ''' Format slider readout value '''
        if self.fps is not None:
            return f'{val / self.fps:.2f} s'
        else:
            return f'{val}'

    def set_frame_slider(self, frange):
        ''' Set slider control to change frame. '''
        # Define default slider parameters
        slider_params = dict(
            value=frange.start,
            min=frange.start,
            max=frange.stop - 1,
            step=1,
        )

        # Create play widget
        playback_interval = 100. if self.fps is None else 1000 / self.fps
        playback_interval /= self.playback_speed
        self.play = Play(
            interval=playback_interval, 
            **slider_params
        )
        
        # Create slider object
        self.frame_slider = IntSlider(
            continuous_update=self.continuous_update,
            readout=False,
            **slider_params
        )
        
        # Link frame slider and play widgets
        jslink((self.play, 'value'), (self.frame_slider, 'value'))

        # Create slider label
        self.frame_slider_label = Label(value=self.slider_format(self.frame_slider.value))
    
    def get_header(self, text):
        ''' Get header text component '''
        return HTML(value=f'<center>{text}</center>')

    def reload_binary_file(self, fobj):
        ''' Close and re-open file-object '''
        fpath, Lx, Ly = fobj.read_filename, fobj.Lx, fobj.Ly
        findex = self.fobjs.index(fobj)
        fobj.close()
        self.fobjs[findex] = BinaryFile(Ly=Ly, Lx=Lx, read_filename=fpath)

    def iter_frames(self, fobj, frange, func=None, **kwargs):
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
                        out.append(func(real_index, frame[0]))
                        pbar.update()
            # FIX: reload binary file object to reset internal index and make sure
            # next iter_frames works correctly
            self.reload_binary_file(fobj)
        else:
            for k in tqdm(list(frange)):
                frame = self.get_frame(fobj, k, **kwargs)
                out.append(func(k, frame))
        return out

    def get_frame_metric_evolution(self, fobj, frange, func=None, **kwargs):
        ''' Compute the time course of a given frame metrics across a specific range of frames. '''
        if func is None:
            func = lambda x: x.mean()
        self.logfunc(f'computing frame-average metrics across frames {frange.start} - {frange.stop - 1}...')
        xlist = self.iter_frames(fobj, frange, func=lambda _, x: func(x), **kwargs)
        return np.array(xlist)

    def transform(self, arr, cmap):
        ''' 
        Transform an grayscale image to a colored image using a specific colormap
        
        :param arr: 2D pixel-by-pixel array of grayscale values
        :param cmap: colormap object(s)
        :return: 3D pixel-by-pixel-by-color array
        '''
        # Transform array through colormap to get a (0, 1) float RGB image
        img = cmap(arr)[:, :, :-1]

        # Convert to uint8 and return
        return float_to_uint8(img)
    
    def label(self, img, label):
        ''' Add label to top-right corner of a frame. '''
        return add_label_to_image(img, label, color='red')

    def process_frame(self, arr, norm, cmap, label):
        '''
        Process frame.
        
        :param arr: frame array(s)
        :param norm: normalizer object
        :param cmap: colormap object(s)
        :param label: label to add to the frame
        :return: processed frame (i.e. normalized, transformed and labeled) 
        '''
        # If arr is a list, process each frame independently and sum outputs
        if isinstance(arr, list):
            outarr = np.zeros((*arr[0].shape, 3))
            for x, n, c in zip(arr, norm, cmap):
                outarr += self.process_frame(x, n, c, None)
            if label:
                outarr = self.label(outarr, label)
            return outarr
        
        # If norm is not None, normalize the frame
        if norm is not None:
            arr = norm(arr)

        # Transform the frame with colormap
        arr = self.transform(arr, cmap)

        # Add label to top-right corner of the frame, if any
        if label:
            arr = self.label(arr, label)

        # Return processed frame
        return arr

    def init_view(self):
        ''' Initialize view with random data.'''
        view = niw.NumpyImage(
            self.transform(np.random.rand(*self.shape), self.cmaps[0]))
        if self.display_width is not None:
            view.width_display = self.display_width
        if self.display_height is not None:
            view.height_display = self.display_height
        return view
    
    def update(self, event):
        ''' Event handler: update view(s) upon change in slider index. '''
        # Get current frame index from slider value
        iframe = self.frame_slider.value

        # Get potential label for current frame
        label = self.labels[iframe]

        # Update slider readout
        self.frame_slider_label.value = self.slider_format(iframe - self.frange.start)

        # Loop through files
        for ifile in range(len(self.fpaths)):
            arrs = []
            # Loop through channels
            for ichannel in range(self.nchannels):
                # Get frame array
                arr = self.get_frame(self.fobjs[ifile], iframe, ichannel=ichannel)
                
                # Append to list if merge is ON
                if self.merge_channels:
                    arrs.append(arr)
                
                # Otherwise, update view with processed frame
                else:
                    img = self.process_frame(
                        arr, self.norms[ifile][ichannel], self.cmaps[0], label)
                    if self.nrows > 1:
                        self.views[ifile][ichannel].data = img
                    else:
                        iview = ifile if self.nchannels == 1 else ichannel
                        self.views[iview].data = img
            
            # If merge is ON, update view with processed frame
            if self.merge_channels:
                img = self.process_frame(arrs, self.norms[ifile], self.cmaps, label)
                self.views[ifile].data = img
    
    def get_vbounds(self, x, rel_vbounds):
        ''' Get value bounds from relative value bounds. '''
        vrange = x[1] - x[0]
        lb = x[0] + rel_vbounds[0] * vrange
        ub = x[0] + rel_vbounds[1] * vrange
        return (lb, ub)

    def init_render(self, fps=None, norm='stack', rel_vbounds=None, cmap=None, fbounds=None, ilabels=None, playback_speed=1.):
        '''
        Initialize stacks rendering.
        
        :param fps (optional): frame rate (in Hz) used to convert frame index to time (in s)
        :param norm: method used to normalize frame intensity prior to rendering (default: 'stack'):
            - 'frame': render each frame indepdendently using its full dynamic range
            - 'stack': render each frame using the dynamic range of the entire stack 
            - 'all': render each frame using the dynamic range of all stacks
        :param rel_vbounds (optional): relative value bounds used to normalize the stack data.
        :param cmap (optional): colormap used to display grayscale image. If none, a gray colormap is used by default.
        :param fbounds (optional): boundary frame indexes. If none, the entire stack is rendered.
        :param ilabels (optional): dictionary of (label:frame indexes) pairs.
        :param playback_speed (optional): playback speed factor (default = 1.)
        '''
        # Initialize fps and playback speed
        self.fps = fps
        self.playback_speed = playback_speed

        # Get frame range
        self.frange = self.get_frame_index_range(fbounds)
        
        # Initialize label arrray (e.g., to dynamically label stim frames)
        self.labels = [''] * self.nframes
        if ilabels is not None:
            for label, iframes in ilabels.items():
                for iframe in iframes:
                    self.labels[iframe] = label
        self.labels = np.array(self.labels)
        
        # Initialize colormap(s)
        if cmap is None:
            if self.nchannels == 1 or not self.merge_channels:
                cmap = 'gray'  # 'viridis'
            else:
                cmap = list(VIEWER_CMAPS.values())[:self.nchannels]
        cmaps = as_iterable(cmap)
        self.cmaps = [plt.get_cmap(cmap) for cmap in cmaps]

        # Initialize normalizers
        if norm in ('stack', 'all'):
            # Get limits for each channel of each stack
            lims = [
                [self.get_dynamic_range(i, self.frange, ichannel=ich) for ich in range(self.nchannels)] 
                 for i in range(len(self.fobjs))
            ]

            # Adapt limits to relative value bounds if provided
            if rel_vbounds is not None:
                lims = [
                    [self.get_vbounds(xx, rel_vbounds) for xx in x]
                    for x in lims
                ]
            
            # If norm is 'all', use the same limits for all stacks, per channel
            if norm == 'all':
                lims_per_channel = []
                for ich in range(self.nchannels):
                    vmin = np.min([x[ich][0] for x in lims])
                    vmax = np.max([x[ich][1] for x in lims])
                    lims_per_channel.append((vmin, vmax))
                lims = [lims_per_channel for i in range(len(self.fobjs))]

            self.norms = [
                [plt.Normalize(vmin=vmin, vmax=vmax) for vmin, vmax in x]
                for x in lims
            ]
            
        else:
            self.norms = [[None] * self.nchannels for i in range(len(self.fobjs))]
            
    def render(self, *args, merge_channels=True, **kwargs):
        '''
        Render stacks
        
        :param args: positional arguments passed on to init_render method
        :param merge_channels: boolean stating whether or not to merge all channels in a single view
        :param kwargs: keyword arguments passed on to init_render method
        :return: VBox ipywidget to render the stacks interactively
        '''
        self.merge_channels = merge_channels
        self.init_render(*args, **kwargs)
        self.logfunc('rendering stack view...')

        # Initialize views
        # DO NOT USE LIST COMPREHENSION HERE TO PRESERVE CLOSURE!!!
        if self.merge_channels:
            self.nrows = 1
        else:
            self.nrows = self.nchannels if len(self.fpaths) > 1 else 1
        self.views = []
        for i in range(len(self.fpaths)):
            if self.nrows > 1:
                    fviews = []
                    for j in range(self.nrows):
                        fviews.append(self.init_view())
                    self.views.append(fviews)
            else:
                for j in range(1 if self.merge_channels else self.nchannels):
                    self.views.append(self.init_view())
        
        # Set up slider control to change frame
        self.set_frame_slider(self.frange)
        
        # Connect slider to image view and update view with first frame in range
        self.frame_slider.observe(self.update)
        self.update(None)

        # If 1 row only, pack all views in single HBox, and 
        # add channel suffix to headers only if multiple channels
        if self.nrows == 1:
            headers = [h for h in self.headers]
            if self.nchannels > 1 and not self.merge_channels:
                headers = [f'{headers[0]} (channel {ich + 1})' for ich in range(self.nchannels)]
            headers = [self.get_header(h) for h in headers]
            rows = [HBox([VBox([h, v]) for h, v in zip(headers, self.views)])]
        
        # If multiple rows (i.e. multiple channels), pack each channel row in an HBox
        else:           
            rows = []
            for ich in range(self.nchannels):
                headers = [h for h in self.headers]
                if self.nchannels > 1:
                    headers = [f'{h} (channel {ich + 1})' for h in headers]
                headers = [self.get_header(h) for h in headers]
                rows.append(HBox([
                    VBox([h, v[ich]]) for h, v in zip(headers, self.views)]))
        
        # Pack all rows (and potential title) in a VBox
        if self.title is not None:
            container = VBox([self.get_header(self.title), *rows])
        else:
            container = VBox(rows)
        
        # # Pack container in an 
        # container = HBox([container])

        # Pack all controls in a HBox
        controls = HBox([self.play, self.frame_slider, self.frame_slider_label])

        # Return VBox with container and controls
        return VBox([container, controls])

    def save_as_gif(self, outdir, fps):
        ''' Save stack(s) as GIF(s) '''
        # For each stack object
        for header, fpath, fobj, norm in zip(self.headers, self.fpaths, self.fobjs, self.norms):
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
                    writer.append_data(float_to_uint8(self.process_frame(x, norm, iframe)))
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
        kwargs['display_size'] = 300
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
    if args[0] is None:
        return
    norm = kwargs.pop('norm', 'stack')
    rel_vbounds = kwargs.pop('rel_vbounds', None)
    fbounds = kwargs.pop('fbounds', None)
    fps = kwargs.pop('fps', None)
    ilabels = kwargs.pop('ilabels', None)
    playback_speed = kwargs.pop('playback_speed', 1.)
    merge_channels = kwargs.pop('merge_channels', True)
    viewer = get_stack_viewer(*args, **kwargs)
    cmap = kwargs.pop('cmap', None)
    return viewer.render(
        fps=fps, 
        norm=norm, 
        rel_vbounds=rel_vbounds, 
        cmap=cmap,
        fbounds=fbounds, 
        ilabels=ilabels, 
        playback_speed=playback_speed,
        merge_channels=merge_channels
    )


def save_stack_to_gif(folder, *args, **kwargs):
    ''' High level function to save stacks to gifs. '''
    fps = kwargs.pop('fps', 10)
    norm = kwargs.pop('norm', True)
    cmap = kwargs.pop('cmap', 'viridis')
    bounds = kwargs.pop('bounds', None)
    ilabels = kwargs.pop('ilabels', None)
    viewer = get_stack_viewer(*args, **kwargs)
    viewer.init_render(norm=norm, cmap=cmap, bounds=bounds, ilabels=ilabels)
    viewer.save_as_gif(folder, fps)



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


def extract_registered_frames(ops, irun, ntrials_per_run, nframes_per_trial, itrial=None, iframes=None, 
                              aggtrials=False, aggfunc=np.median, verbose=True):
    '''
    Extract a sequence of frames from registered movie for a given run and trial

    :param ops: suite2p output options dictionary
    :param irun: run index
    :param itrial: trial(s) index. If not provided, frames will be extracted from all available trials.
    :param iframes: list of frame indexes to extract. If none provided, all frames per trial are extracted.
    :param ntrials_per_run: number of trials per run.
    :param aggtrials: boolean stating whether or not to aggregatre frames across selected trials
    :return: frames stack array
    '''
    # Derive number of frames per run
    nframes_per_run = ntrials_per_run * nframes_per_trial

    # Cast frames list to array
    if iframes is None:
        iframes = np.arange(nframes_per_trial)
    else:
        iframes = np.atleast_1d(np.asarray(iframes))

    # If trial index not provided, extract frames from all trials
    if itrial is None:
        itrial = np.arange(ntrials_per_run)

    # If multiple trials provided, transpose trial vector
    if is_iterable(itrial):
        itrial = np.atleast_2d(itrial).transpose()

    # Compute extended frame indexes for given run and trial
    iframes_ext = irun * nframes_per_run + itrial * nframes_per_trial + iframes
    iframes_ext = np.ravel(iframes_ext)
    itrial = np.ravel(itrial)

    # Initialize stack viewer and extract frames
    viewer = get_stack_viewer(ops, verbose=verbose)
    if verbose:
        logger.info(
            f'extracting frames {idx_format(iframes)} from run {irun}, trial(s) {idx_format(itrial)} (indexes = {idx_format(iframes_ext)})')
    frames = []
    for i in iframes_ext:
        frames.append(viewer.get_frame(viewer.fobjs[0], int(i)))
        viewer.reload_binary_file(viewer.fobjs[0])
    frames = np.array(frames)

    # If specified, aggregate frames across trials
    if aggtrials:
        if verbose:
            logger.info(
                f'aggregating frames across trials {itrial} with {aggfunc.__name__} function...')
        frames = np.reshape(frames, (len(itrial), len(iframes), *frames.shape[1:]))
        frames = aggfunc(frames, axis=0)

    # Return stack 
    return frames
