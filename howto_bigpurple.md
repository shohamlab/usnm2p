# How to set up BigPurple for analyses with Suite2p

# Initial setup

- Log in to bigpurple: `xbigpurple`
- Open your `.bashrc` file: `nano .bashrc`
- Copy paste the following block into the file (adpting environment variables to your needs):

```
# User-specific environment variables
export MYID="lemait01"  # Kerberos ID
export MYNAME="theo"   # Name of your personal directory in your lab space
export MYLABNAME="shohamlab"  # Name of your lab space in /gpfs/data/
export MYLABDRIVE="shohas01labspace"  # Name of your lab's research drive

# Slack webhook
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/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"

# Alias commands
alias datamover="srun -p data_mover -n 2 -t=8:00:00 --mem-per-cpu=1G --pty bash"  # open special node for file transfer with rsync
alias mountlab="mount /mnt/$MYID/$MYLABDRIVE/"  # mount the labs's research drive on your BigPurple user session
alias job="srun -c2 --partition=$MYLABNAME --mem=8G --pty /bin/bash"  # run a normal bash job
alias xjob="srun --x11 -c2 --partition=$MYLABNAME --mem=16G --pty /bin/bash"  # run an interactive bash job
alias mpijob="srun -c10 --partition=$MYLABNAME --mem=64G --pty /bin/bash"  # run a massively parallelized bash job
alias myjobs="squeue -u $MYID"  # list all jobs assosicated to your user ID
```

- Save and close to file (`Ctrl-o` followed by `Ctrl-x`)
- Activate these changes: `source .bashrc`
- Go to your personal lab storage space: `cd /gpfs/data/$MYLABNAME`
- Create a personal directory (where you will store code, configuration files, etc): `mkdir $MYNAME`
- Go to your personal directory: `cd $MYNAME`
- Create directory that will contain your code repositories: `mkdir code`
- Create a directory that will contain your bash scripts: `mkdir bash`
- Create directories that will contain your configuration files: `mkdir .conda .config .ipython .local`
- Go to home folder: `cd ~`
- Add a symbolic link to your personal directory: `ln -s /gpfs/data/$MYLABNAME/$MYNAME ~/$MYNAME`
- Add a symbolic link to your scratch directory (where you will store computation data): `ln -s /gpfs/scratch/$MYID ~/scratch`
- Remove placeholders added by default upon creation of your user space: `rm .conda .config .ipython .local`
- Replace these placeholders with symbolic links towards directories in your personal space:
    - `ln -s /gpfs/data/$MYLABNAME/$MYNAME/.conda ~/.conda`
    - `ln -s /gpfs/data/$MYLABNAME/$MYNAME/.config ~/.config`
    - `ln -s /gpfs/data/$MYLABNAME/$MYNAME/.ipython ~/.ipython`
    - `ln -s /gpfs/data/$MYLABNAME/$MYNAME/.local ~/.local`

## Set tup the USNM2P analysis coding environment

- Log in to bigpurple: `xbigpurple`
- Load git: `module load git`
- Load miniconda: `module load miniconda3/cpu/4.9.2`
- Got to your personal repositories folder: `cd ~/$MYNAME/code` 
- Download the suite2p repository from GitHub using Git: `git clone https://github.com/MouseLand/suite2p`
- Move to the suite2p directory: `cd suite2p`
- Run `conda env create -f environment.yml`
- Activate this new environment: `conda activate suite2p`
- Install suite2p into this environment: `pip install suite2p`
- Download the USNM2P analysis directory: `git clone https://github.com/shohamlab/usnm2p.git`
- Move to that directory: `cd usnm2p`
- Install the dependencies: `pip install -r requirements.txt`

# Run an analysis notebook interactively on BigPurple

- Log in to bigpurple: `xbigpurple`

TO COMPLETE

# Run batch analyses on BigPurple

- Log in to bigpurple: `xbigpurple`
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