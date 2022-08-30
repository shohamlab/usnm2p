
import numpy as np
import glob
import sys
import os
from tifffile import imread, TiffFile
import matplotlib.pyplot as plt
from packaging import version
from dialog import open_folder_dialog
# from lab_instruments.logger import logger

# Check that python version enables correct ScanImage data parsing
VCRITICAL = version.parse('3.8.10')
v = version.parse(sys.version.split(' ')[0])
if v < VCRITICAL:
    print(f'error: python version should be greater than {VCRITICAL}')


def loadtif(fpath):
    ''' Load TIF and potentially reshape it according to number of channels '''
    with TiffFile(fpath) as f:
        meta = f.scanimage_metadata
        y = f.asarray()
    nchannels = len(meta['FrameData']['SI.hChannels.channelSave'])
    if y.ndim < 4 and nchannels > 1:
        print(f'splitting {nchannels} from {y.shape} shaped array')
        y = np.reshape(y, (y.shape[0] // nchannels, nchannels, *y.shape[1:]))
    return y


def plot_frameavg_profiles(tif_fpaths, save=False):
    '''
    Plot frame-average profiles of a given acquisition
    
    :param tif_fpaths: list of full paths to tif_files
    :param save: whether to save the output figure to disk
    :return: frame average profiles figure
    '''
    # Find common prefix
    out_fpath = os.path.commonprefix(tif_fpaths)
    acqname = os.path.basename(out_fpath)
    # For each file
    frameavgs = []
    for i, f in enumerate(tif_fpaths):
        # Load TIF stack
        print(f'loading data from {os.path.basename(f)}')
        # stack = imread(f)
        stack = loadtif(f)
        print(f'loaded {stack.shape} stack')

        # Compute frame average for each channel
        frameavg = stack.mean(axis=(-2, -1))
        if frameavg.ndim == 1:
            frameavg = np.array([frameavg]).T
        print(f'computed {frameavg.shape[0]} frame average vector on {frameavg.shape[1]} channels')

        # Append to list
        frameavgs.append(frameavg)

    # Transform frame averages list to array, and swap axes
    frameavgs = np.stack(frameavgs)
    frameavgs = np.swapaxes(frameavgs, 1, 2)
    frameavgs = np.swapaxes(frameavgs, 0, 1)
    nchannels, nacqs, nsamples = frameavgs.shape
    isamples = np.arange(nsamples)

    # Plot
    print('plotting frame-average profiles...')
    fig, axes = plt.subplots(nchannels, figsize=(10, 3 * nchannels))
    if nchannels == 1:
        axes = [axes]
    axes[0].set_title(acqname)
    for i, ax in enumerate(axes):
        ax.set_xlabel('frames')
        ax.set_ylabel(f'channel {i + 1}')
    for ax, ychannel in zip(axes, frameavgs):
        for i, yacq in enumerate(ychannel):
            ax.plot(isamples, yacq, label=f'acq {i + 1}', alpha=0.5, lw=1)
        yavg = ychannel.mean(axis=0)
        ystd = ychannel.std(axis=0)
        ax.plot(isamples, yavg, label='avg', c='k')
        ax.fill_between(isamples, yavg - ystd, yavg + ystd, fc='k', alpha=0.3)
    for ax in axes:
        ax.legend(loc='center right')    
    if save:
        fig.savefig(f'{out_fpath}.png')
    
    return fig


# Select root data directory
datadir = open_folder_dialog()
if datadir is None:
    print('no input data directory chosen')
    quit()

# Look for a batch of files in that directory
print(f'looking for TIF files in {datadir}...')
tif_files = glob.glob(os.path.join(datadir, '*.tif'))
# Find common root
root = os.path.commonprefix(tif_files)
# Make sure all file have the same size
sizes = [os.path.getsize(f) for f in tif_files]
assert all(x == sizes[0] for x in sizes), 'differing stack sizes'
print(f'found {len(tif_files)} files under the common root {root}*.tif')

# Plot stacks frame-average profiles
fig = plot_frameavg_profiles(tif_files, save=True)
plt.show()