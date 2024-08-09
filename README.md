# USNM 2P imaging data analysis pipeline

This repository hosts the code base to process and analyze 2-photon calcium imaging data acquired upon stimulation of different cortical regions by ultrasound.

## Acquisition protocols & raw data

### Bruker 2P imaging system

Raw data from the Bruker imaging system consist of individual 16-bit deep, 256-by-256 pixels grayscale TIF images of fluorescence data acquired at a sampling rate of 3.56 Hz and spatial resolution of 1.97 μm / pixel.

Each acquisition protocol (or run) consists of 16 sonication trials with an inter sonication interval of approximately 30s.

Each trial is divided in 2 cycles:
- The first cycle (indicated by an odd number) consists of 10 frames of the pre-stimulus interval of the trial (ca. 2.8s).
- The second cycle (indicated by an even number) consists of 90 frames of the peri and post-stimulus intervals of the trial (ca. 25.2s).

The ultrasound stimulus is delivered concurrently with the acquisition of the 10th frame of the first cycle (i.e. ca. at t = 2.8s).

For each acquisition protocol, the data is stored in a folder named after the specimen, stimulation and acquisition parameters:

`foldername = <mouse_line>_<nframes_per_trial>frames_<PRF>Hz_<stim_duration>ms_<sampling_rate>Hz_<stim_amplitude>MPA_<stim_DC>DC-<run_ID>`

Inside this folder, data is stored as single-frame TIF files named after the same pattern, together with unique cycle and frame identifers:

`filename = <foldername>_Cycle<cycle_number>_Ch2_<frame_number>.ome.tif`

This informative nomenclature is used as a way to store metadata associated with each experimental run.

Additionally, a file named `<foldername>.xml`, containing information about the microscope acquisition parameters for the run, is also stored in the data folder.

### Bergamo 2P imaging system

TO COMPLETE

## Analysis code base

### Prerequisites

- Python 3.8+
- Conda environment manager

These can be downloaded from https://www.anaconda.com/products/individual.

### Installation

- Create a new anaconda environment called `usnm2p`: `conda create -n usnm2p python=3.8`
- Activate this anaconda environment: `conda activate usnm2p`
- Clone this repository: `git clone https://github.com/shohamlab/usnm2p.git`
- Install the downloaded code base as an editable Python package: `pip install -e ./usnm2p`. This will install all required dependencies.
- Install the pre-built package for PyTables directly from conda: `conda install pytables`
- Move to the `usnm2p` subfolder inside the repository: `cd usnm2p/usnm2p` 
- Create a user configuration file called `config.py`, and define in it:
    - a variable called `dataroot` indicating the path to the root directory continaing all the data folders to be analyzed.
    - a dictionary called `default_dataset` indicating parameters of the default dataset to analyze, with the following keys:
        - `mouseline`: mouse line
        - `expdate`: experiment date
        - `mouseid`: mouse number
        - `region`: brain region
- Save the configuration file in the repository top folder. You're all set!

### Usage

Always start by activating the `usnm2p` anaconda environment: `conda activate usnm2p`

#### Notebooks

The `notebooks` subfolder contains several analysis notebooks:
- `dataset_analysis.ipynb`: analysis pipeline for a single dataset (date, mouse, region): from fluorescence movies to ROI-specific statistics.
- `mouseline_analysis.ipynb`: analysis pipeline for processed data from all datasets of a given mouse line
- `main_analysis.ipynb`: to run the main parameter-dependency analysis across mouse lines
- `spatial_offset_analysis.ipynb`: to run the analysis of datasets pertaining to the effect of spatial offsets
- `buzzer_analysis.ipynb`: to run the analysis of datasets pertaining to the effects of ultrasound vs. noise

#### Command-line scripts

The `scripts` subfolder contains several command-line scripts to run preprocess/analyze data:
Several command-line scripts can also be run from the terminal to execute analysis notebooks over specific subsets of datasets:
- `preprocess_bruker_data.py`: preprocess raw datasets from Bruker 2P imaging system
- `run_dataset.py`: analyze one/several single dataset(s) from the command line (runs the `dataset_analysis.ipynb` notebook in the background).
- `run_mouseline.py`: analyze one/several mouseline(s) from the command line (runs the `mouseline_analysis.ipynb` notebook in the background).

To access command line options, type in: `python <script_name> -h`

## Authors & contributors

- Data acquired by various members of the [Shoham Lab](https://nie-lab.org/) at New York University.
- Code written and maintained by [Theo Lemaire](mailto:theo.lemaire1@gmail.com)
