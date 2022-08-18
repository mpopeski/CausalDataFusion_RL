# Causal Data Fusion for Model-Based Reinforcement Learning

The code for the experimental section of my master's thesis 'Causal Data Fusion for Model-Based Reinforcement Learning' done at ETH Zurich.

## Environment

The Python version and packages used are in the environment.yml file. 

Can create conda environment with the following command:

`conda env create -f environment.yml`

Navigate to the folder containing this project and activate the environmnet

`conda activate thesis`

## Running the Experiments

In order to get the results presented in the thesis run the following commands.

For experiment 1:

`python exp1.py ./configs/exp1Config.yaml`

For experiment 2:

`python exp2.py ./configs/exp2Config.yaml`

These experiments take a while to run. When they finnish running a Results folder will appear with rewards history in csv files for different algorithms and methods, and plot images that summarize the results. 

Faster less intensive variants can be run in 10-15min using the following commands:

`python exp1.py ./configs/exp1Config_fast.yaml`

`python exp2.py ./configs/exp2Config_fast.yaml`
