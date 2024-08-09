# USNM 2P imaging data analysis pipeline

This repository hosts the code base to process and analyze 2-photon calcium imaging data acquired upon stimulation of different cortical regions by ultrasound.

## Installation & usage

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
    - a variable called `dataroot` indicating the path to the root directory for the raw data to be analyzed.
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

Code written and maintained by [Theo Lemaire](mailto:theo.lemaire1@gmail.com)
