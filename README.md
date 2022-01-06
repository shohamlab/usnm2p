# USNM 2P imaging data analysis pipeline

This repository hosts the code base to process and analyze 2-photon calcium imaging data acquired upon stimulation of different cortical regions by ultrasound.

## Installation & usage

### Prerequisites

- Python 3.8+
- Conda environment manager

These can be downloaded from https://www.anaconda.com/products/individual.

### Installation

- Download and install the suite2p functional segmentation pipeline in a new conda environment, following the instructions at https://suite2p.readthedocs.io/en/latest/installation.html.
- Clone this repository and open a terminal at its root directory.
- While in the suite2p conda environment, install remaining package dependencies: `pip install -r requirements.txt`
- Create a user configuration file called `config.py`, and define in it 3 variables:
	- `dataroot` indicating the root directory for the raw data to be analyzed
	- `figsdir` indicating the folder in which to save the final figures file.
	- `statsdir` indicating the folder in which to save the output stats file.
- Save the configuration file in the repository top folder. You're all set!

### Usage

To run the analysis pipeline of a single mouse region interactively:
- start Jupyter Lab: `jupyter lab`
- execute the `single_region_analysis.ipynb` notebook

To automatically run the same analysis over several datasets, you can use the `run_analyses.py` script directly from the terminal:
- to run the script: `python run_analyses.py`
- to access command line options: `python run_analyses.py -h`

TO COMPLETE.

## Description

### Mouse lines & datasets

The full dataset contains data from different mouse lines, each genetically derived to target specifc neuron type(s):

| Line              | Marker       | Neuron type                                   | Morphology | Cortical Layer(s) |
| ----------------- | ------------ | --------------------------------------------- | ---------- | ----------------- |
| Line3 - Layer 2/3 | ???          | Excitatory neuron                             | Pyramidal  | 2-3               |
| Line3 - Layer 5   | ???          | Excitatory neuron                             | Pyramidal  | 5                 |
| SST               | Somatostatin | GABAergic inhibitory interneuron              | ???        | 2-3               |
| PV                | Parvalbumin  | GABAergic fast-spiking inhibitory interneuron | ???        | 2-3               |

TO COMPLETE.

### Acquisition protocols & raw data

Raw data consist of inividual 16-bit deep, 256-by-256 pixels grayscale TIF images acquired at a sampling rate of 3.56 Hz and spatial resolution of 1.97 μm / pixel by the Bruker microscope during experiments.

Each acquisition protocol (or run) consists of 16 sonication trials with an inter sonication interval of approximately 30s.

Each trial is divided in 2 cycles:
- The first cycle (indicated by an odd number) consists of 10 frames of the pre-stimulus interval of the trial (ca. 2.8s).
- The second cycle (indicated by an even number) consists of 90 frames of the peri and post-stimulus intervals of the trial (ca. 25.2s).

The ultrasound stimulus is delivered concurrently with the acquisition of the 10th frame of the first cycle (i.e. ca. at t = 2.8s).

The TIF images from a given run are all stored in a single folder, which is named according to a specific pattern depending on the specimen, stimulation and acquisitation parameters:

`foldername = <mouse_line>_<nframes_per_trial>frames_<PRF>Hz_<stim_duration>ms_<sampling_rate>Hz_<stim_amplitude>MPA_<stim_DC>DC-<run_ID>`

Inside this folder, individual files are named using this same pattern, together with unique cycle and frame identifers:

`filename = <foldername>_Cycle<cycle_number>_Ch2_<frame_number>.ome.tif`

This informative nomenclature is used as a way to store metadata associated with each experimental run.

Additionally, a file named `<foldername>.xml`, containing information about the microscope acquisition parameters for the run, is also stored in the data folder.

### Processing steps

The raw data is typically processed in different successive steps, described below.

1. **Pre-processing**: this consists of multiple steps to convert raw, single-frame TIF files into chronologically organized, stacked (i.e. multi-frame) TIF files (1 for each run) that can be presented to our functional segmentation algorithm. This pre-processing consists of multiple steps:
	- **Stacking**: raw TIF images are assembled into stacked TIF files containing all the frames of an entire run. Each resulting stacked TIF file should contain a 1600x256x256 uint16 array and is named after the directory containing the corresponding individual TIF files.
	- **Substitution of stimulus frames**: frames acquired during sonication periods are typically "polluted" with a large amount of noise, making it virtually impossble to reliably detect regiosn of interest and extract their activity. Therefore, these frames are substituted by their preceding (unpolluted) frames.
	- **Denoising**: the aim of this step is to remove the Speckle noise present in raw microscope aqcuisition frames. To this end, we use a modified implementation of the Kalman filter [1]. The main parameter influencing the outcome of this processing step is the *specified filter gain* (`G`). From collective experience and cmomparative visual inspections, it seems that values around 0.5 work well when using GCaMP6s as a fluorescence reporter.

2. **Functional segmentation**: the denoised TIF stacks are fed into the *suite2p* [2] pipeline to extract cell-specific fluorescence timeseries. This consists of several sub-steps:
	- conversion from TIF to binary data "movie"
	- motion correction using (rigid & non-rigid) movie registration
	- movie projection at various spatial scales to compute a "correlation map"
	- iterative peak detection on correlation map, and extension around these peaks to determine regions of interest (ROIs)
	- extraction of associated "neuropil" areas around each ROI (contaning contaminating signals originating from axons & dendrites located outside of the plane of interest but in the acquisition volume)
	- ROI classification into cell (i.e. soma) and non-cell (e.g. axons, dendrites...) ROIs based on extracted features of ROI activity (skewness, variance, correlation to surrounding pixels) and anatomy (area, aspect ratio), using a naive Bayes classifier trained on cortical data
	- extraction of ROI's calcium fluorescence timecourse and optional spike deconvolution.

	**Important**: if multiple stacked TIF files are provided as input, suite2p will **stack them sequentially prior to processing**. Therefore, **all stacked TIF files in the input folder must correspond to the same brain region**.

	Upon completion, a `/suite2p/plane0/` folder is created that typically contains the following output files:
	- `F.npy`: array of fluorescence traces (ROIs by timepoints)
	- `Fneu.npy`: array of neuropil fluorescence traces (ROIs by timepoints)
	- `spks.npy`: array of deconvolved traces (ROIs by timepoints)
	- `stat.npy`: array of statistics computed for each cell (ROIs by 1)
	- `ops.npy`: options and intermediate outputs (identical to the output of the run_s2p function)
	- `iscell.npy`: specifies whether an ROI is a cell, first column is 0/1, and second column is probability that the ROI is a cell based on the default classifier
	- `data.bin` (optional): registered image stack in binary format format.

3. **Post-processing**: the main aim of this step is to convert the raw fluorescence timeseries of each ROI (extracted from the suite2p output files) into z-score timeseries of the corresponding relative fluorescence variation. This is carried out in successive sub-steps:
	- extraction of cell and associated neuropil fluorescence traces
	- subtraction of neuropil background with an appropriate coeefficient (currently 0.7) to obtain a corrected flueorescence timecourse (F) of each ROI
	- baseline (F0) computation and baseline correction of neuropil-corrected fluorescence traces
	- baseline normalization to obtain relative change fluorescence traces ΔF/F0
	- noise level and variation range estimation (using gaussian fits of ΔF/F0 distributions) and subsequent noise-normalization of relative change fluorescence traces into z-score traces.

	Upon completion, a `/suite2p/processed/` folder is created in which the following post-processing output files are saved:
	- `info_table.csv`: summary table of the parameters pertaining to each run
	- `zscores_run*.csv`: z-score traces of each ROI for a given run (along with their ROI, run, trial and frame index information)
	- `ROI_masks.csv`: table of the pixel masks of each selected ROI in the reference frame.

4. **Statistics**: using the extracted z-score timeseries as a basis, transient activity events are detected and used to derive statistics on ultrasound-evoked (& spontaneous) neural activity. This analysis consists of the following sub-steps:
	- quantification of lateral motion over time (using the registration offsets timeseries outputed by suite2p), detection of motion artifacts, and exclusion of associated trials
	- detection of transient activity events in defined pre-stimulus and post-stimulus windows, but also on a continous detection window moving along the trial intervals
	- characterization of baseline, pre-stimulus and stimulus-evoked neural activity (using the peak z-score value as a proxy) for each ROI and each trial, and exclusion of trials with pre-stimulus activity
	- quantification of success rate & response strength for every ROI & run
	- classification of cells by response type
	- derivation of stimulus-evoked transient activity traces for each cell type and stimulus condition
	- characteriztion of parameter-dependency of success rate & response strength for each cell type.

## Authors & contributors

The data was acquired by Yi Yuan (???) and Sarah Sarah Haiken (Research Associate).

The analysis code base has received contributions from many people, including
- Celia Gellman (Project Student)
- Ben Stetler (Research Associate)
- Diego Asua (Research Associate)
- Theo Lemaire (Postdoctoral Researcher, theo.lemaire@nyulangone.org): current contributor

## References

- [1] Khmou, Y., and Safi, S. (2013). Estimating 3D Signals with Kalman Filter. ArXiv:1307.4801 [Cs, Math].
- [2] Pachitariu, M., Stringer, C., Dipoppa, M., Schröder, S., Rossi, L.F., Dalgleish, H., Carandini, M., and Harris, K.D. (2016). Suite2p: beyond 10,000 neurons with standard two-photon microscopy (Neuroscience).