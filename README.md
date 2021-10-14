# Hierarchical GAN for large dimensional financial market data Implementation

This repository is an implementation of the [Hierarchical (Sig-Wasserstein) GAN] algorithm for large dimensional Time Series Generation.

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

- Add your data file to **src/data** and a data pipeline to **data.py** with the name get_{}_dataset with {} being the name of your dataset. 

- In **hyperparameters_hierarchicalgan.py** add an entry with the name of your dataset and the desired parameters to the following dictionaries:
  
  1. Clustering_Hierarchical_GAN_CONFIGS
  2. Base_SIGCWGAN_CONFIGS
  3. CrossDim_SIGCWGAN_CONFIGS
  
All training results will be saved to *generated_data/{}/seed=i* with {} being the name of your dataset and as many seeds as specified.

## Generating Scenarios

Once a model has been trained the parser in *evaluate_hierarchical_gan.py* can be modified to load the trained model and generate scenarios via the function generate_series_hierarchical_gan.
"# Hierarchical-SigCWGAN" 
