# USNM 2P imaging data analysis pipeline

This repository hosts the code base to process and analyze 2-photon calcium imaging data upon stimulation of different cortical regions by ultrasound.

## Installation

### Prerequisites

- Python 3.8+
- Conda environment manager

### Instructions

- Download and install the suite2p functional segmentation pipeline [1] in a new conda environment, following the instructions at https://suite2p.readthedocs.io/en/latest/installation.html
- Clone this repository and open a terminal at its root directory 
- While in the suite2p conda environment, install remaining package dependencies: `pip install -r requirements.txt`
- Create a user configuration file called `config.py`, and define in it a variable called `dataroot` indicating the root directory for the raw data to be analyzed. Save the file in the repository top folder.
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

`foldername = <mouse_line>_<nframes_per_trial>frames_<PRF>Hz_<stim_duration>ms_<sampling_rate>Hz_<stim_amplitude>MPA_<stim_DC>DC-<run_ID>`

Inside this folder, individual files are named using this same pattern, together with unique cycle and frame identifers:

`filename = <foldername>_Cycle<cycle_number>_Ch2_<frame_number>.ome.tif`

This rich nomenclature is used as a way to store metadata associated with each experimental run.

### Processing steps

The raw data is typically processed in different successive steps, described below.
1. **Stacking**: raw TIF images are assembled into stacked TIF files containing all the frames of an entire run. Each resulting stacked TIF file should contain a 1600x256x256 uint16 array and is named after the directory containing the corresponding individual TIF files.

2. **Denoising**: the main aim of this step is to remove Speckle noise present in raw microscope aqcuisitions. To this end, we use a modified implementation of the Kalman filter [2]. The main parameter influencing the outcome of this processing step is the *specified filter gain* (`G`). From collective experience, it seems that values around 0.5 work well when using GCaMP6s as a fluorescence reporter.

3. **Functional segmentation**: the denoised TIF stacks are fed into the *suite2p* pipeline to extract cell-specific fluorescence timeseries. This consists of several substeps:
	- conversion from TIF to binary data
	- motion correction & image registration (parametrized, rigid vs. non-rigid)
	- denoising using principal component analysis (optional) ???
	- regions of interest (ROIs) detection over contaminating signals originating from the surrounding neuropil (i.e. axons & dendrites located outside of the plane of interest but in the acquisition volume)
	- ROI labelling into cell (i.e. soma) and non-cell (e.g. axons, dendrites...) ROIs, using a naive Bayes classifier trained on cortical data to identify cells based on extracted features of ROI activity (skewness, variance, correlation to surrounding pixels) and anatomy (area, aspect ratio).
	- extraction of ROI's calcium fluorescence timecourse
	- spike deconvolution (optional , and somewhat useless with a sampling rate of 3.5 Hz)

**Important**: if multiple stacked TIF files are provided as input, suite2p will **stack them sequentially (in which order???) prior to processing**. Therefore, **all stacked TIF files in the input folder must correspond to the same brain region**.

Upon completion, a `/suite2p/plane0/` folder is created for each input stack that typically contains the following output files:
	- `F.npy`: array of fluorescence traces (ROIs by timepoints)
	- `Fneu.npy`: array of neuropil fluorescence traces (ROIs by timepoints)
	- `spks.npy`: array of deconvolved traces (ROIs by timepoints)
	- `stat.npy`: array of statistics computed for each cell (ROIs by 1)
	- `ops.npy`: options and intermediate outputs (identical to the output of the run_s2p function)
	- `iscell.npy`: specifies whether an ROI is a cell, first column is 0/1, and second column is probability that the ROI is a cell based on the default classifier
	- `data.bin` (optional): registered image stack in binary format format

4. **Calcium transients analysis**: the suite2p input files are used as input to derive and analyze calcium transient traces. This analysis consists of the following substeps:
	- subtraction of cell and associated neuropil fluorescence traces (with an constant neuropil factor `k = 0.7`).
	- *(stim onset artefact removing)*
	- baseline normalization on a per trial basis to obtain relative fluorescence traces (`dF/F0`)
	- removal of outlier cells which exhibit abnormally high peaks of relative fluorescence activity
	- classification of cells by response type (positively responding, negatively responding, and neutral)
	- data cleaning and re-organization
	- visualization of the results across trials / cells (time series & summary plots)

TO COMPLETE

## Authors & contributors

This code base has received contributions from many people, including
- Diego Asua: original author???
- Theo Lemaire (theo.lemaire@nyulangone.org): current contributor

TO COMPLETE

## References

- [1] Pachitariu, M., Stringer, C., Dipoppa, M., Schröder, S., Rossi, L.F., Dalgleish, H., Carandini, M., and Harris, K.D. (2016). Suite2p: beyond 10,000 neurons with standard two-photon microscopy (Neuroscience).
- [2] Khmou, Y., and Safi, S. (2013). Estimating 3D Signals with Kalman Filter. ArXiv:1307.4801 [Cs, Math].

TO COMPLETE
