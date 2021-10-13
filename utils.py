import os
import glob
import numpy as np
from suite2p import run_s2p
import operator

from constants import *
from logger import logger
from tifffile import imread, imsave

''' Collection of generic utilities. '''


def isIterable(x):
    for t in [list, tuple, np.ndarray]:
        if isinstance(x, t):
            return True
    return False


# SI units prefixes
SI_powers = {
    'y': -24,  # yocto
    'z': -21,  # zepto
    'a': -18,  # atto
    'f': -15,  # femto
    'p': -12,  # pico
    'n': -9,   # nano
    'u': -6,   # micro
    'm': -3,   # mili
    '': 0,     # None
    'k': 3,    # kilo
    'M': 6,    # mega
    'G': 9,    # giga
    'T': 12,   # tera
    'P': 15,   # peta
    'E': 18,   # exa
    'Z': 21,   # zetta
    'Y': 24,   # yotta
}
si_prefixes = {k: np.power(10., v) for k, v in SI_powers.items()}
sorted_si_prefixes = sorted(si_prefixes.items(), key=operator.itemgetter(1))


def getSIpair(x, scale='lin', unit_dim=1):
    ''' Get the correct SI factor and prefix for a floating point number. '''
    if isIterable(x):
        # If iterable, get a representative number of the distribution
        x = np.asarray(x)
        x = x.prod()**(1.0 / x.size) if scale == 'log' else np.mean(x)
    if x == 0:
        return 1e0, ''
    else:
        vals = np.array([tmp[1] for tmp in sorted_si_prefixes])
        if unit_dim != 1:
            vals = np.power(vals, unit_dim)
        ix = np.searchsorted(vals, np.abs(x)) - 1
        if np.abs(x) == vals[ix + 1]:
            ix += 1
        return vals[ix], sorted_si_prefixes[ix][0]


def si_format(x, precision=0, space=' ', **kwargs):
    ''' Format a float according to the SI unit system, with the appropriate prefix letter. '''
    if isinstance(x, float) or isinstance(x, int) or isinstance(x, np.float) or\
       isinstance(x, np.int32) or isinstance(x, np.int64):
        factor, prefix = getSIpair(x, **kwargs)
        return f'{x / factor:.{precision}f}{space}{prefix}'
    elif isinstance(x, list) or isinstance(x, tuple):
        return [si_format(item, precision, space) for item in x]
    elif isinstance(x, np.ndarray) and x.ndim == 1:
        return [si_format(float(item), precision, space) for item in x]
    else:
        raise ValueError(f'cannot si_format {type(x)} objects')


def parse_overwrite(overwrite):
    '''
    Parse a user input overwrite parameter.
    
    :param overwrite: one of (True, False, '?') defining what to do in case of overwrite dilemma.
    :return: parsed overwrite decision (True/False)
    '''
    # Check that input is valid
    valids = (True, False, '?')
    if overwrite not in valids:
        raise ValueError(f'"overwrite must be one of {valids}')
    # If not a question, must be True or False -> return as is
    if overwrite != '?':
        return overwrite
    # Otherwise ask user to input (y/n) response
    overwrite = input('overwrite (y/n)?:')
    if overwrite not in ['y', 'n']:
        raise ValueError('"overwrite" argument must be one of ("y", "n")')
        # Parse response into True/False and return boolean
    return {'y': True, 'n': False}[overwrite]


def check_for_existence(fpath, overwrite):
    '''
    Check file for existence
    
    :param fpath: full path to candidate file
    :param overwrite: one of (True, False, '?') defining what to do if file already exists.
    :return: boolean stating whether to move forward or not.
    '''
    if os.path.exists(fpath):
        logger.warning(f'"{fpath}" already exists')
        return parse_overwrite(overwrite)
    else:
        return True


def loadtif(fpath):
    ''' Load stack/image from .tif file '''
    stack = imread(fpath)
    if stack.ndim > 2:
        logger.info(f'loaded {stack.shape} {stack.dtype} stack from "{fpath}"')
    return stack


def savetif(fpath, stack, overwrite=True):
    ''' Save stack/image as .tif file '''
    move_forward = check_for_existence(fpath, overwrite)
    if not move_forward:
        return
    if stack.ndim > 2:
        logger.info(f'saving {stack.shape} {stack.dtype} stack as "{fpath}"...')
    imsave(fpath, stack)


def get_stack_baseline(stack: np.array, n: int, noisefactor: float=1.) -> np.array:
    '''
    Construct baseline stack from image stack. 
    
    :param stack: 3D image stack
    :param n: number of frames to put in the baseline
    :param noisefactor: uniform multiplying factor applied to the noise matrix to construct the baseline
    :return: generated baseline stack
    '''
    # Extract frame dimensions from stack
    nframes, *framedims = stack.shape
    logger.info(f'constructing {n}-frames baseline from {nframes}-frames stack...')
    # Estimate median and standard deviation of each pixel across the images of the stack
    pixel_med, pixel_std = np.median(stack, axis=0), np.std(stack, axis=0)
    # Tile both matrices along the required number of baseline frames
    pixel_med, pixel_std = [np.tile(x, (n, 1, 1)) for x in [pixel_med, pixel_std]]
    # Construct noise matrix spanning [-noisefactor, noisefactor] and matching the required baseline stack dimensions
    noise = noisefactor * (2 * np.random.rand(n, *framedims) - 1)
    # Construct baseline stack by summing up median and std-scale noise matrices
    baseline = pixel_med + noise * pixel_std
    # Bound to [0, MAX] interval and return
    return np.clip(baseline, 0, np.amax(stack))


def get_output_equivalent(inpath, basein, baseout):
    '''
    Get the "output equivalent" of a given file or directory, i.e. its corresponding path in
    an identified output branch of the file tree structure, while creating the intermediate
    output subdirectories if needed.

    :param inpath: absolute path to the input file or directory
    :param basein: name of the base folder containing the input data (must contain inpath)
    :param baseout: name of the base folder containing the output data (must not necessarily exist)
    :return: absolute path to the equivalent output file or directory
    '''
    if not os.path.exists(inpath):
        raise ValueError(f'"{inpath}" does not exist')
    pardir, dirname = os.path.split(inpath)
    logger.debug(f'input path: "{inpath}"')
    subdirs = []
    if os.path.isdir(inpath):
        subdirs.append(dirname)
        fname = None
    else:
        fname = dirname
    logger.debug(f'moving up the filetree to find "{basein}"')
    while dirname != basein:
        if len(pardir) < 2:
            raise ValueError(f'"{basein}"" is not a parent of "{inpath}"')
        pardir, dirname = os.path.split(pardir)
        subdirs.append(dirname)
    logger.debug(f'found "{basein}" in "{pardir}"')
    subdirs = subdirs[::-1]
    subdirs[0] = baseout
    logger.debug(f'moving down the file tree in "{baseout}"')
    outpath = pardir
    for subdir in subdirs:
        outpath = os.path.join(outpath, subdir)
        if not os.path.isdir(outpath):
            logger.info(f'creating "{outpath}" directory')
            os.mkdir(outpath)
    if fname is not None:
        outpath = os.path.join(outpath, fname)
    return outpath


def run_suite2p(*args, overwrite=True, **kwargs):
    '''
    Wrapper around run_s2p function that first checks for existence of suite2p output files
    and runs only if files are absent or if user allowed overwrite.
    '''
    suite2p_keys = ['iscell', 'stat', 'F', 'Fneu', 'spks']
    # For each input directory
    for inputdir in kwargs['db']['data_path']:
        # Check for existence of suite2p subdirectory
        suite2pdir = os.path.join(inputdir, 'suite2p', 'plane0')
        print(suite2pdir)
        if os.path.isdir(suite2pdir):
            # Check for existence of any suite2p output file
            if any(os.path.isfile(os.path.join(suite2pdir, f'{k}.npy')) for k in suite2p_keys):
                # Warn user if any exists, and act according to defined overwrite behavior
                logger.warning(f'suite2p output files already exist in "{suite2pdir}"')
                if not parse_overwrite(overwrite):
                    return
    # If execution was not canceled, run the standard function
    return run_s2p(*args, **kwargs)


def get_suite2p_data(dirpath, cells_only=True, withops=False):
    '''
    Locate suite2p output files given a specific output directory, and load them into a dictionary.
    
    :param dirpath: full path to the output directory (must contain a suite2p subdirectory).
    :param cells_only: boolean stating whether to filter out non-cell entities from dataset.
    :param withops: whether include a dictionary of options and intermediate outputs.
    :return: suite2p output dictionary.
    '''
    if not os.path.isdir(dirpath):
        raise ValueError(f'"{dirpath}" directory does not exist')
    for subdir in ['suite2p', 'plane0']:
        dirpath = os.path.join(dirpath, subdir)
        if not os.path.isdir(dirpath):
            raise ValueError(f'"{dirpath}" directory does not exist')
    keys = ['iscell', 'stat', 'F', 'Fneu', 'spks']
    data = {k : np.load(os.path.join(dirpath, f'{k}.npy'), allow_pickle=True) for k in keys} 
    iscell = data.pop('iscell')[:, 0]  # the full "iscell" has a second column with the probability of being a cell
    # If specified, restrict dataset to cells only
    if cells_only:
        cell_idx = np.array(iscell.nonzero()).reshape(-1)
        data = {k : v[cell_idx] for k, v in data.items()}
    if withops:
        data['ops'] = np.load(os.path.join(dirpath, f'ops.npy'), allow_pickle=True).item()
    return data


def moving_average(x, n=5):
    ''' Apply a monving average on a signal. '''
    if n % 2 == 0:
        raise ValueError('you must specify an odd MAV window length')
    return np.convolve(np.pad(x, n // 2, mode='symmetric'), np.ones(n), 'valid') / n


def get_F_baseline(x):
    '''
    Compute the baseline of a signal. This function assumes that time
    is the last dimension of the signal array.
    '''
    if x.ndim == 1:
        return moving_average(x, n=N_MAV)
    else:
        return np.apply_along_axis(lambda a: moving_average(a, n=N_MAV), -1, x, )


def separate_trials(x, ntrials):
    '''
    Split suite2p data array into separate trials.
    
    :param x: 2D (ncells, nframes) suite2p data array
    :return: 3D (ncells, ntrials, npertrial) data array
    '''
    ncells, nframes = x.shape
    npertrial = nframes // ntrials
    return np.array([np.reshape(xx, (ntrials, npertrial)) for xx in x])


def get_df_over_f(F, ibaseline):
    '''
    Calculate relative fluorescence signal (dF/F0) from absolute fluorescence signal.

    Baseline is calculated on a per-trial basis as the average of the pre-stimiulus interval.

    :param F: 3D (ncells, ntrials, npertrial) fluorescence signal array
    :param ibaseline: baseline evaluation indexes
    '''
    # Extract F dimensions
    ncells, ntrials, npertrial = F.shape
    # Extract baseline fluoresence signals and average across time for each cell and trial 
    F0 = F[:, :, ibaseline].mean(axis=-1)
    # Tile F0 along the time dimension
    F0 = np.moveaxis(np.tile(F0, (npertrial, 1, 1)), 0, -1)
    # Return relative change in fluorescence
    return (F - F0) / F0


def exponential_kernel(t, tau=1, pad=True):
    '''
    Generate exponential kernel.

    :param t: 1D array of time points
    :param tau: time constant for exponential decay of the indicator
    :param pad: whether to pad kernel at init with length = len(t)
    :return: a kernel or generative decaying exponenial function
    '''
    hemi_kernel = 1 / tau * np.exp(-t / tau)
    if pad:
        len_pad = len(hemi_kernel)
        hemi_kernel = np.pad(hemi_kernel, (len_pad, 0))
    return hemi_kernel


def locate_datafiles(line, layer, filter_key=None):
    ''' Construct a list of suite2p data files to be used as input for an analysis. '''
    
    # Determine data directory and potential exclusion patterns based on input parameters
    exclude = []
    if line == 'sarah_line3':
        base_dir = '/gpfs/scratch/asuacd01/sarah_usnm/line3/20210804/mouse28/region1'
        file_dir = f'{base_dir}/suite2p'
        # base_dir = '/gpfs/scratch/asuacd01/sarah_usnm/line3'
        # assert os.path.isdir(base_dir), f'"{base_dir}" directory does not exist'
        # for x in [date, mouse, region]:
        #     base_dir = f'{base_dir}/{x}'
        #     assert os.path.isdir(base_dir), f'"{base_dir}" directory does not exist'
        # file_dir = f'{base_dir}/suite2p'
    if line == 'yi_line3':        
        base_dir = '/gpfs/scratch/asuacd01/yi_usnm/line3'
        file_dir = f'{base_dir}/suite2p_results'
        exclude = ['mouse9', 'mouse10', 'mouse1_region2'] # list of excluded subjects, empty list if all included, for yi_line3
    elif line == 'sst':
        base_dir = '/gpfs/data/shohamlab/shared_data/yi_recordings/yi_new_holder_results/sst'
        file_dir = f'{base_dir}/suite2p_results_frame_norm'
        exclude = ['mouse6'] # list of excluded subjects, empty list if all included, for sst
    elif line == 'celia_line3':
        base_dir = '/gpfs/data/shohamlab/shared_data/celia/line3'
        file_dir = f'{base_dir}/suite2p_results' # new data
    elif line == 'pv':
        base_dir = '/gpfs/data/shohamlab/shared_data/yi_recordings/yi_new_holder_results/PV/'
        file_dir = f'{base_dir}/suite2p_results_framenorm'        
        exclude = ['mouse6'] # list of excluded subjects, empty list if all included, for PV

    # Locate and sort all files in the file directory
    group_files = sorted(glob.glob(os.path.join(file_dir,'*')))

    # Remove unwanted layers from the analysis
    if layer == 'layer5':
        group_files = [x for x in group_files if 'layer5' in x] 
    elif layer == 'layer2_3':
        group_files = [x for x in group_files if 'layer5' not in x]
    else:
        logger.warning('Performing analysis disregarding layer parameter')

    # Exclude relevant subjects from the analysis
    for subject in exclude:
        group_files = [x for x in group_files if subject not in x]
    
    # Restrict analysis to files containing the filter key, if any
    if filter_key is not None:
        group_files = [x for x in group_files if filter_key in x]

    # Get file base names
    file_basenames = [os.path.basename(x) for x in group_files]

    # # Add potential suffixes to line label
    # if layer:
    #     line = line + '_' + layer
    # if exclude:
    #     line = line + '_exclude'+ '-'.join(exclude)
    # if filter_key:
    #     line = line + '_individual'+ '-'.join(filter_key)

    return file_dir, file_basenames


def parse_experiment_parameters(name):
    '''
    Parse experiment parameters from a file name.
    
    :param name: file / folder name from which parameters must be extracted
    :return: dictionary of extracted parameters
    '''
    mo = EXPCODE.match(name)
    if mo is None:
        raise ValueError(f'"{name}" does not match the experiment naming pattern')
    return {
        'line': mo.group(1),
        'trial_length': int(mo.group(2)),
        '???': float(mo.group(3)),
        'duration': float(mo.group(4)),
        'fps': float(mo.group(5)),
        'P': float(mo.group(6)),
        'DC': float(mo.group(7)),
        'runID': int(mo.group(8))
    }


def reg_search_list(query, targets):
    '''
    Search a list or targets for match if regexp query, and return first match.
    
    :param query: regexp format query
    :param targets: list of strings to be tested
    :return: first match (if any), otherwise None
    '''
    p = re.compile(query)
    for target in targets:
        positive = p.search(target)
        if positive:
            positive = positive.string
            break
    return positive


def to_unicode(query):
    ''' Translate regexp query into unicode (effectively a fix for undefined queries) '''
    if query == 'undefined':
        return UNDEFINED_UNICODE
    else:
        return query
