# Scalable Causal Structure Learning via Amortized Conditional Independence Testing
Implementation of Scalable Causal Structure Learning. 

The datasets currently used in the dataset are not public and we are awaiting confirmation from data owners to share them publically. This repo currently shares the scripts to implement the procedure for a generic dataset alongside instructions for how to run these scripts. An anonymized public repo is currently being maintained that mirrors these files. Upon confirmation that the datasets can be shared publically, we will update this repo with the datasets and additonal instructions for replicating figures. 

https://anonymous.4open.science/r/scsl-B00F/

## Dependencies
To load the libraries that this code is dependent on use the scsl.yml file via the following command:
```
conda env update -n scsl --file scsl.yaml
```
In addition, we use the pytetrad project to benchmark these methodologies, which can be downloaded [here](https://github.com/cmu-phil/py-tetrad). We assume these files are downloaded into the root directory into a subfolder entitled "pytetrad". 


## Instructions
We assume the dataset is structured into two csv files without headers containing only binary $0/1$ data. In what follows, we label these **dataset_X.csv** and **dataset_y.csv**. As described in the paper, we assume the $X$ nodes are temporally separated from the $Y$ nodes, such that edges between these two sets of nodes can only oriented away from $X$ towards $Y$. 

1. Choose a specific edge $X_{j} \rightarrow Y_{k}$ to test. Denote the column in dataset_X corresponding to $X_{j}$ as **X_target** and the column in dataset_Y corresponding to $Y_{K}$ as **Y_target**. To learn all edges, these steps can be executed in parallel for each one. 
2. Create predictive models to model $\mathbb{E} \left[ X_{j} | S, X_{-j} \right]$ and $\mathbb{E} \left[ Y_{k} | S, X_{-j} \right]$.
   * If logistic models are desired for prediction, use
    ```
    python create_models_x.py --label dataset --X_target X_target
    python create_models_y.py --label dataset --Y_target Y_target
    ```
  * If neural networks are desired for prediction, use
    ```
    python model_mlp_x.py --label dataset --X_target X_target
    python model_mlp_y.py --label dataset --Y_target Y_target
    ```
3. To find the optimized $p$-value using neural networks for prediction, run
```
python optimize_pvals.py --label  dataset --X_target X_target --Y_target Y_target --NUM_EPOCHS 1000 --stop_pval 0.8 --model_type logit 
```
or if logistic models are to be used for prediction
```
python optimize_pvals.py --label  dataset --X_target X_target --Y_target Y_target --NUM_EPOCHS 1000 --stop_pval 0.8 --model_type mlp 
```
Here, stop_pval refers to the $p$-value to use for an early stopping rule and NUM_EPOCHS refers to the number of training steps to train the model over. These can be changed arbitrarily as the user desires.

## Instructions for Replicating Figures 
There are three steps required to recreate figures: recreate the semi-synthetic datasets, run the proceure using the instructions above, and then compling the results. Detailed below are instructions for each step:
### Creating Semi-Synthetic Datasets
The following raw data files are contained in the data folder. The source is [Genomic characterization of metastatic patterns from prospective clinical sequencing of 25,000 patients](https://zenodo.org/records/5801902).
* missing_gene_alteration_matrix.csv: For each patient, indicates wheter a mutation is present at each location.
* metastasis_bysamples.csv: For each patient, indicate whether a secondary metastsis is present at each body location. 
* primary_sites.csv: Primary site location for each patient.

The Jupyter Notebook *Create Datasets.ipynb" merges these datasets and creates the semi-synthetic datasets based off of the above. All semi-syntehtic datasets are pre-saved in the data folder as well to aid in replicability. 

### Running SCSL in server environment
After creating the semi-synthetic datasets, the next steps are to create the predictive models and then optimize the $p$-values using the instructions above. To do this efficiently, we execute the code in a server environment. 

To build the predictive models, we can run the following scripts in a bash server environment:
```
sbatch batch_model_x
sbatch batch_model_y
sbatch batch_model_x_actual
sbatch batch_model_y_actual
```
This will require a significant amount of compute resource and the user may wish to modify the text files *params_actual.txt*, *params_x,txt*, and *params_y.txt* so only a specific dataset of interest has predictive models built for it.

After building predictive models, we can then compute the $p$-values in batch as:
```
sbatch batch_gcm_gso
sbatch batch_gcm_gso_primary_subset
```
### Creating Benchmarks 
In addition to running SCSL, we benchmark this to comparison methods. As before, we assume that the pytetrad environment has been installed. 

* run_benchmarks.py runs through different methodologies used in pytetrad for a choice of dataset.
* run_pcalgo.py is a custom implementation of the PC algorithm that is designed to work with the GCM conditional independence tests used in this project
* run_pcalgo_cont.py is a custom implementation of the above but also is designed to be used for continuous data only.

To run these in batch, we can execute the following scripts:
```
sbatch batch_gcm_gso
sbatch batch_gcm_gso_primary_subset
```

### Compile Results
To generate the experimental results and figures in the papers, the Jupyter Notebook 'Compile Results.ipynb' recreates the figures in the paper from the experimental results. 




