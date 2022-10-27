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
- Move inside the repository: `cd usnm2p` 
- Install the package dependencies: `pip install -r requirements.txt`
- Install the pre-built package for PyTables directly from conda: `conda install pytables`
- Create a user configuration file called `config.py`, and define in a variable called `dataroot` indicating the path to the root directory for the raw data to be analyzed.
- Save the configuration file in the repository top folder. You're all set!

### Usage

Always start by activating the `usnm2p` anaconda environment: `conda activate usnm2p`

#### Notebooks

Several jupyter notebooks can be executed interactively:
- `single_dataset_analysis.ipynb`: to run the analysis pipeline of a single dataset (date, mouse, region)
- `mouseline_analysis.ipynb`: to run the analysis of all datasets pertaining to a given mouse line (main analysis)
- `spatial_offset_analysis.ipynb`: to run the analysis of datasets pertaining to the effect of spatial offsets
- `buzzer_analysis.ipynb`: to run the analysis of datasets pertaining to the effects of ultrasound vs. noise

#### Command-line scripts

Several command-line scripts can also be run from the terminal to execute analysis notebooks over specific subsets of datasets:
- `run_datasets.py`: to excute the analysis of several single datasets
- `run_mouselines.py`: to excute the analysis of several mouselines

To access command line options, type in: `python <script_name> -h`

## Authors & contributors

The data was acquired by Yi Yuan, Amy LeMessurier and Sarah Rachel Haiken. The analysis code base has received contributions from many people, listed below.

*Past contributors:*
- Celia Gellman (Project Student)
- Junhyook Lee (Project Student)
- Ben Stetler (Research Associate)
- Diego Asua (Research Associate)

*Current contributor:*
- Theo Lemaire (Postdoctoral Researcher): theo.lemaire@nyulangone.org
