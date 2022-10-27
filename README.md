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

**TO COMPLETE**

#### Command-line scripts

To automatically run the same analysis over several datasets, you can use the `run_analyses.py` script directly from the terminal:
- to run the script: `python run_analyses.py`
- to access command line options: `python run_analyses.py -h`

## Authors & contributors

The data was acquired by Yi Yuan, Amy LeMessurier and Sarah Rachel Haiken. The analysis code base has received contributions from many people, listed below.

*Past contributors:*
- Celia Gellman (Project Student)
- Junhyook Lee (Project Student)
- Ben Stetler (Research Associate)
- Diego Asua (Research Associate)

*Current contributor:*
- Theo Lemaire (Postdoctoral Researcher): theo.lemaire@nyulangone.org

## References

- [1] Khmou, Y., and Safi, S. (2013). Estimating 3D Signals with Kalman Filter. ArXiv:1307.4801 [Cs, Math].
- [2] Pachitariu, M., Stringer, C., Dipoppa, M., Schröder, S., Rossi, L.F., Dalgleish, H., Carandini, M., and Harris, K.D. (2016). Suite2p: beyond 10,000 neurons with standard two-photon microscopy (Neuroscience).