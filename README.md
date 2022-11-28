﻿
# Hierarchical GAN for large dimensional financial market data Implementation

This repository is an implementation of the [Hierarchical (Sig-Wasserstein) GAN] algorithm for large dimensional Time Series Generation.

Link to the paper: https://doi.org/10.3905/jfds.2022.1.109 

This codebase is based on work from COST Action 19130, supported by COST (European Cooperation in Science and Technology; www.cost.eu). We also thank an anonymous reviewer for useful comments.

## Installation
 There are two possible installation setups:

-  **YML Installation** (Recommended). Run the following command to automatically setup the **hiersigcwganenv** environment.
	```
	conda env create -f environment.yml
	```
-  **Manual Installation**. Setup a new conda environment with python==3.8.3 and then run  
	```
	pip install -r requirements.txt
	```

## Training on new datasets

In order to train on a new dataset the following changes to the code are needed: 

- Create a folder **src/data**, add your data file with the name of your dataset and add a data pipeline to **data.py** with the name `get_{}_dataset` with {} being the name of your dataset. 

- In **hyperparameters_hierarchicalgan.py** add an entry with the name of your dataset and the desired parameters to the following dictionaries:
  
  1. Clustering_Hierarchical_GAN_CONFIGS
  2. Base_SIGCWGAN_CONFIGS
  3. CrossDim_SIGCWGAN_CONFIGS
  
All training results will be saved to **generated_data/{}/seed=i** with {} being the name of your dataset and as many seeds as specified.

## Generating Scenarios

Once a model has been trained the parser in *evaluate_hierarchical_gan.py* can be modified to load the trained model and generate scenarios via the function `generate_series_hierarchical_gan`.

## Replicating results from the paper

The data used in the first experiment are the closing prices from $03/05/2000$ to $07/05/2021$ (dd/mm/yyyy) of the continously rolled futures from Tables 2 and 4 in the paper. Due to copyright sharing issues we are not able to share the datasets but we do include the trained models in  `src/generated_data`.

![Alt Text](https://media.giphy.com/media/hMvcLdbpVqRRNCwPsj/giphy.gif)

Enjoy!
