# Represention Matters

This repository contains the code and information needed to replicate and expand on the analyis in the paper **"Representation Matters: Assessing the Importance of Subgroup Allocations in Training Data" by Esther Rolf, Theodora Worledge, Benjamin Recht, and Michael I. Jordan**, _to appear in ICML 2021_. You can access the ArXiv version of the paper [here](https://arxiv.org/abs/2103.03399).


## Getting started:
To get the conda environment loaded, run `conda env create -f torch_env.yml` from the root folder in this repo. Then you can run `conda activate torch_env`, and all the packages you need should be loaded for you. 


## Downloading and processing datasets:
Follow these instructions to download each of the datasets you are interested in:

Modified CIFAR-10:
1. From 'multi-acc/cifar4/scripts' run `python process_cifar4_data.py`.

Goodreads [History/Scify]:
1. Download 'goodreads_reviews_history_biography.json.gz' and 'goodreads_reviews_fantasy_paranormal.json.gz' from https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home , and place them in 'multi-acc/data/goodreads'.

2. From 'multi-acc/goodreads/scripts' run `process_goodreads_data.py`. This will place the CSV to use for analysis in 'multi-acc/data/goodreads/goodreads_history_fantasy_5_fold_splits.csv'. 

ISIC [up to 2019]
1. follow installation instructions from https://github.com/GalAvineri/ISIC-Archive-Downloader

2. from 'ISIC-Archive-Downloader' run `$ python download_archive.py --images-dir='<relative_path_to_multi-acc>/multi-acc/data/isic/Images' --descs-dir='<relative_path_to_multi-acc>/multi-acc/multi-acc/data/isic/Descriptions'`
when prompted, confirm download

3. From '/multi-acc/isic/scripts' run `process_isic_data.py`

note: We only used data collected prior to 2019. If you want all of the most recent data, this requires chaning an argument in `compile_isic_data.py`.

Adult:
1. Download 'adult.data' and 'adult.test' from http://archive.ics.uci.edu/ml/machine-learning-databases/adult/

2. Move both files into '/multi-acc/data/adult'

3. From '/multi-acc/adult/scripts' run `process_adult_data.py`

MOOC [HarvardX-MITX]:
1. Download 'HXPC13_DI_v3_11-13-2019.tab' as a CSV from https://dataverse.harvard.edu/file.xhtml?persistentId=doi:10.7910/DVN/26147/91P6G7&version=11.2

2. Rename the file 'harvard_mit.csv'

3. Move the file into '/multi-acc/data/mooc'

4. From '/multi-acc/mooc/scripts' run `process_mooc_data.py`

## Examples  

### Running ERM subsetting experiment on CIFAR4:  
To process the data: `python process_cifar4_data.py`  
To run 5-fold cross-validation for hyperparameter tuning: `python cv_hyperparameter_search.py 'animal' --dataset_name 'cifar' --obj 'ERM' --results_tag 'example'`  
You can read in and visualize the cross-validation results with `multi-acc/cifar4/notebooks/read_subsetting_cv_results_cifar.ipynb`.

To run 10 seeds on the test set for the subsetting experiment: `python subsetting_exp.py 'animal' --dataset_name 'cifar4' --run_type 'subsetting'  --obj 'ERM' --results_tag 'example'`  

You can read in and visualize the results with `multi-acc/cifar4/notebooks/read_uplot_results.ipynb`.

The procedure above closely mirrors the process for running the CIFAR4 IS and GDRO subsetting experiments, as well as the ISIC ERM, IS, and GDRO subsetting experiments (see below for how to modify for these settings).  

### Running ERM subsetting experiment on Adult using random forest classifier:   
To process the data: `python process_adult_data.py`  

To run 5-fold cross-validation for hyperparameter tuning: `python cv_hyperparameter_search_nonimage.py 'adult' --pred_fxn_name 'rf_classifier' --results_tag 'example'`.
You can read in and visualize the cross-validation results with `multi-acc/adult/notebooks/read_crossval_results.ipynb`.

To run 10 seeds on the test set for the subsetting experiment: `python subsetting_exp_nonimage.py 'adult' --run_type 'subsetting' --pred_fxn_name 'rf_classifier' --results_tag 'example'`.

You can read in and visualize the results with `multi-acc/adult/notebooks/read_uplot_results_adult.ipynb`.

The procedure above closely mirrors the process for running the Adult IW subsetting experiment, as well as the Goodreads and Mooc ERM and IW subsetting experiments.  

### Running scaling model fits on Goodreads using ERM and logistic regression:   
To process the data: `python process_goodreads_data.py`  
To run 5-fold cross-validation for hyperparameter tuning: `python cv_hyperparameter_search_nonimage.py 'goodreads' --pred_fxn_name 'logistic_regression' --results_tag 'example'`  

To obtain the subsetting results needed to fit the scaling law model (if you've already run the first of these, there's no need to re-run):  
`python subsetting_exp_nonimage.py 'goodreads' --run_type 'subsetting' --pred_fxn_name 'logistic_regression' --results_tag 'example'`  
`python subsetting_exp_nonimage.py 'goodreads' --run_type 'additional' --pred_fxn_name 'logistic_regression' --results_tag 'example'`  
`python subsetting_exp_nonimage.py 'goodreads' --run_type 'additional_equal_group_sizes' --pred_fxn_name 'logistic_regression' --results_tag 'example'`  

To fit the scaling law model and visualize the outputs, use the notebook: `multi-acc/goodreads/notebooks/pilot_sample_read_in.ipynb`.


### Running the "pilot sample" experiment with Goodreads using ERM:
Process the data and run 5-fold cross-validation as above.
To run the pilot sample experiment, run:
`python pilot_sample_experiment_nonimage.py goodreads --pred_fxn_name 'logistic_regression'`. To run the gridded baseline, run `python pilot_baseline_goodreads.py <n_new>` for each n_new value you want to run the baseline for (in the paper we show [5000,10000,20000,40000]).
To read in and plot the results, use the notebook: `multi-acc/goodreads/notebooks/pilot_sample_read_in.ipynb`.


### Running the "leave one study out" experiment on ISIC using ERM:  
To process the data: `python process_isic_data.py`  
To run the leave-one-study-out experiment, run:
`python subsetting_exp.py 'study_name_aggregated_id '--run_type 'leave_one_group_out' --dataset_name 'isic_sonic'  --results_tag 'example'`.
To run the leave-one-substudy-out experiment, replace `study_name_aggregated_id` with `study_name_id`.
To read in and plot the results, use the notesbooks: `multi-acc/isic/notebooks/read_loo_results_aggregated_studies.ipynb` and `multi-acc/isic/notebooks/read_loo_results.ipynb`.

## Running Scripts:
This gives the functionality for each of the experimental scripts, to aid in modifying the example runs shown above. All of the following scripts can be found in `code/scripts`.

**python cv_hyperparameter_search.py < subset group key > --dataset_name < dataset name > --obj < objective > --results_tag < experiment name >**  
< subset group key > is 'animal' when < dataset name > is 'cifar'  
< subset group key > is 'age_over_50_id' when < dataset name > is 'isic'  
< objective > is one of ['ERM', 'IS', 'GDRO']  
< experiment name > is a string descriptor of the experiment  

**python subsetting_exp.py < subset group key > --dataset_name < dataset name > --run_type < run type >  --obj < objective > --results_tag < experiment name >**  
< subset group key > is 'animal' when < dataset name > is 'cifar'  
< subset group key > is 'age_over_50_id' when < dataset name > is 'isic'  
< subset group key > is 'age_over_50_id' when < dataset name > is 'isic_sonic'  
< run type > is one of ['additional_equal_group_sizes', 'additional', 'subsetting', 'leave_one_group_out']   
< objective > is one of ['ERM', 'IS', 'GDRO']  
< experiment name > is a string descriptor of the experiment  

**python cv_hyperparameter_search_nonimage.py < dataset name > --reweight < reweight > --pred_fxn_name < classifier > --results_tag < experiment name >**  
< dataset name > is one of ['adult', 'goodreads', 'mooc']  
< reweight > should be excluded for ERM (defaults to False), < reweight > is True for IW  
< classifier > is one of ['logistic_regression', 'rf_classifier', 'rf_regressor', 'ridge']  
< experiment name > is a string descriptor of the experiment  


**python subsetting_exp_nonimage.py < dataset name > --run_type < run type > --reweight < reweight > --pred_fxn_name < classifier > --results_tag < experiment name >**  
< dataset name > is one of ['adult', 'goodreads', 'mooc']  
< run type > is one of ['additional_equal_group_sizes', 'additional', 'subsetting']   
< reweight > should be excluded for ERM (defaults to False), < reweight > is True for IW  
< classifier > is one of ['logistic_regression', 'rf_classifier', 'rf_regressor', 'ridge']  
< experiment name > is a string descriptor of the experiment  

**python pilot_sample_nonimage.py < dataset name >  --pred_fxn_name < classifier >**  
< dataset name > for now must be ['goodreads']  
< classifier > is one of ['logistic_regression', 'ridge']  
