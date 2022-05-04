# How to set up BigPurple for analyses with Suite2p

## Log in

- Log in to bigpurple (`xbigpurple`)

## Set up a coding environment

- Go to your personal lab storage space: `cd /gpfs/data/shohamlab`
- Create a directory with your name (where you will store code and configuration files): `mkdir theo`
- Go to your personal directory: `cd theo`
- Create directory that will contain your code repositories: `mkdir code`
- Create a directory that will contain your bash scripts: `mkdir bash`
- Create directories that will contain your configuration files: `mkdir conda config ipython local misc`
- Go to home folder: `cd ~`
- Add a symbolic link to your personal directory: `ln -s /gpfs/data/shohamlab/theo ~/theo`
- Add a symbolic link to your scratch directory (where you will store computation data): `ln -s /gpfs/scratch/lemait01 ~/scratch`
- Remove placeholders added by default upon creation of your user space: `rm .conda .config .ipython .local`
- Replace these placeholders with symbolic links towards directories in your personal space:
    - `ln -s /gpfs/data/shohamlab/theo/conda ~/.conda`
    - `ln -s /gpfs/data/shohamlab/theo/config ~/.config`
    - `ln -s /gpfs/data/shohamlab/theo/ipython ~/.ipython`
    - `ln -s /gpfs/data/shohamlab/theo/local ~/.local` 
- Load git: `module load git`
- Load miniconda: `module load miniconda3/cpu/4.9.2`
- Got to your personal repositories folder: `cd ~/theo/code` 
- Download the suite2p repository from GitHub using Git: `git clone https://github.com/MouseLand/suite2p`
- Move to the suite2p directory: `cd suite2p`
- Run `conda env create -f environment.yml`
- Activate this new environment: `conda activate suite2p`
- Install suite2p into this environment: `pip install suite2p`
- Download the USNM2P analysis directory: `git clone https://github.com/shohamlab/usnm2p.git`
- Move to that directory: `cd usnm2p`
- Install the dependencies: `pip install -r requirements.txt`

# Run an analysis notebook interactively on BigPurple

TO COMPLETE

# Run batch analyses on BigPurple

- Switch node and load resources to run parallelized job: `mpijob`
- Load git: `module load git`
- Load miniconda: `module load miniconda3/cpu/4.9.2`
- Activate anaconda environment: `conda activate suite2p`
- Go to repository's folder: `cd theo/code/usnm2p`
- Pull latest changes: `git pull`
- Start screen session: `screen -S mysession`
- Detect datsets in file system and run parallel analyses: `python run_analyses.py --mpi --locate` (**the script may be slow to start because of cellpose import attempts**)
- Confirm datasets and multiprocessing: `y`
- Exit screen session: `Ctrl-a + d`
- Let analyses run...
- Re-attach to screen session `screen -r mysession`
- Observe completion output