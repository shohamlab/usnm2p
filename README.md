# USNM 2P imaging data analysis pipeline

This repository hosts the code base to process and analyze 2-photon calcium imaging data upon stimulation of different cortical regions by ultrasound.

## Installation

### Prerequisites

- Python 3.8+
- Conda environment manager

### Instructions

- Download and install the suite2p functional segmentation pipeline in a new conda environment, following the instructions at https://suite2p.readthedocs.io/en/latest/installation.html
- While in that environment, install the remaining package dependencies: `pip install -r requirements.txt`
- You're all set!

## Usage

### Acquisition protocols & raw data

Raw data consist of invididual 16-bit, 256 by 256 pixels TIF images acquired from the microscope during experiments. The current sampling rate of these images is ca. 3.5 Hz, corresponding to an inter-frame interval of ca. 280 ms.

Each acquisition protocol (or run) consists of 16 sonication trials with an inter sonication interval of approximately 30s. 

Each trial is divided in 2 cycles:
- The first cycle (always indicated by an odd number) consists of 10 frames of the pre-stimulus interval of the trial (ca. 2.8s).
- The second cycle (always indicated by an even number) consists of 90 frames of the peri and post-stimulus intervals of the trial (ca. 25.2s).
The stimulus is therefore delivered concurrently with the acquisition of the 10th frame of the first cycle (i.e. ca. at t = 2.8s).

The TIF images from a given run are all stored in a single folder, which is named according to a specific pattern depending on the specimen, stimulation and acquisitation parameters:

`foldername = <mouse_line>_<nframes_per_trial>frames_<???>Hz_<stim_duration>ms_<sampling_rate>Hz_<stim_amplitude>MPA_<stim_DC>DC-<run_ID>`

Inside this folder, individual files are named using this same pattern, together with unique cycle and frame identifers:

`filename = <foldername>_Cycle<cycle_number>_Ch2_<frame_number>.ome.tif`

This rich nomenclature is used as a way to store metadata associated with each experimental run.

### Processing steps

The raw data is typically processed in different successive steps, described below.
1. **Stacking**: raw TIF images are assembled into stacked TIF files containing all the frames of an entire run. Each resulting stacked TIF file should contain a 1600x256x256 uint16 array and is named after the directory containing the corresponding individual TIF files.

2. **Denoising**: the main aim of this step is to remove Speckle noise present in raw microscope aqcuisitions. To this end, we use a Kalman filter with specific parameters (TO COMPLETE).

3. **Functional segmentation**: the denoised TIF stacks are fed into the *suite2p* pipeline to extract cell-specific fluorescence timeseries. This consists of several substeps:
    - conversion from TIF to binary data
    - motion correction (parametrized, rigid / non-rigid registration using FFTs)
    - denoising using principal component analysis (optional)
    - non-negative matrix factorization to find relevant regions (neurons, axonal processes, neuropil artefacts…), optimizing and removing z-artefacts
    - naïve Bayesian classification (using model trained on cortical data) to identify cells based on parameters (shape of neuron, size, shape of activity, etc)
    - spike deconvolution (optional , and somewhat useless with a sampling rate of 3.5 Hz)
Upon completion, a */suite2p/plane0/* folder is created for each input stack that typically contains the following output files:
    - `data.bin`: the input stack converted to binary format ???
    - `F.npy`: 2D numpy array with fluorescence timeseries for each identified cell
    - `Fneu.npy`: 2D numpy array with fluorescence timeseries for each identified neuropil
    - `iscell.npy`: 2D numpy array with ???
    - `ops.npy`: options suite2p was run with
    - `spks.npy`: 2D numpy array with attempted deconvoluted spike info for each identified cell
    - `stat.npy`: 2D numpy array with quantified features for each identified cell

4. **Calcium transients analysis**: the suite2p input files are used as input to derive and analyze calcium transient traces. This analysis consists of the following substeps:
    - subtraction cell – neuropil
	- stim onset artefact removing
    - baseline normalization -> df/f
    - removing outliers (discards cells above dff_outlier threshold)
    - classify by response type
    - pandas data cleaning / re-organizing
    - plot across trials / cells (time series & summary plots) 

TO COMPLETE

## Authors & constributors

This code base has received contributions from many people, including
- Diego Asua: original author???
- Theo Lemaire (theo.lemaire@nyulangone.org): current contributor

TO COMPLETE

## References

- Pachitariu, M., Stringer, C., Dipoppa, M., Schröder, S., Rossi, L.F., Dalgleish, H., Carandini, M., and Harris, K.D. (2016). Suite2p: beyond 10,000 neurons with standard two-photon microscopy (Neuroscience).

TO COMPLETE