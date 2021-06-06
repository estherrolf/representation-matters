import torch
#
torch.multiprocessing.set_sharing_strategy('file_system')

import pandas as pd
import numpy as np
import sys
import os
import pickle
import argparse

# local imports
import train_model
import experiment_routines as experiments
from subsetting_exp import subset_experiment
from dataset_chunking_fxns import add_stratified_kfold_splits, get_single_cv_kfold_split

# can add different 
from dataset import get_data_loaders 

data_dir = '../../data'
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('group_key', type=str, 
                        help='which key to subset on')
    parser.add_argument('--num_seeds', type=int, default=5, 
                        help='number of seeds to run')
    parser.add_argument('--seed_beginning', type=int, default=0, 
                        help='The split index to start with is determined by seed_beginning; \
                        seed_beginning must be within [0, 5 - num_seeds]')
    parser.add_argument('--dataset_name', type=str, default='isic', 
                        help='string identifier of the dataset')
    parser.add_argument('--obj', type=str, default='ERM', 
                        help='string identifier of objective [ERM, IS, GDRO]')
    parser.add_argument('--results_tag', type=str, default='debug', 
                        help='string identifier for the results folder that will be \
                        concatenated with the dataset name and objective')
    num_cv_splits = 5
    args = parser.parse_args()
    group_key = args.group_key
    num_seeds, seed_beginning = args.num_seeds, args.seed_beginning
    assert seed_beginning + num_seeds <= num_cv_splits, \
            "The split index to start with is determined by seed_beginning; seed_beginning must be within [0, 5 - num_seeds]"
    dataset_name = args.dataset_name
    objs = [args.obj]
    results_tag = args.results_tag
    
    # do one random seed per split since there's num_splits splits
    run_type = 'subsetting'
    num_seeds_each_split = 1
    
    fracs_group_a = [0.02, 0.05, 0.2, 0.5, 0.8, 0.95, 0.98]
    
    learning_rates = [0.01, 0.001, 0.0001]     
    weight_decays = [0.01, 0.001, 0.0001]
    momentums = [0.9]

    number_of_epochs = [20] 
    
    # GDRO is a special case since it has more params
    objs_excluding_gdro = objs.copy()
    
    do_gdro=False
    if 'GDRO' in objs:
        do_gdro=True
        objs_excluding_gdro.remove('GDRO')
    
        if (len(objs) == 1):
            learning_rates = [0.001]
            weight_decays = [.001, 0.0001]
    
        gdro_step_sizes = [1e-1, 1e-2, 1e-3]
        gdro_group_adjs = [1.0, 4.0, 8.0]
    
        gdro_params_base = {'num_groups': 2, 
                            'group_key': group_key}
    
    
    # get parameters for each dataset
    if dataset_name.lower() == 'isic':
        label_colname_ = 'benign_malignant_01'
        original_data_csv_ = 'isic/df_no_sonic_age_over_50_id.csv'
        image_fp_  = 'isic/ImagesSmaller'
        all_group_colnames_ = ['study_name_id',
                                'age_approx_by_decade_id', 
                                'age_over_45_id', 
                                'age_over_50_id',
                                'age_over_60_id', 
                                'anatom_site_general_id', 
                                'sex_id']
        results_descriptor_ ='isic_subsetting_hpSel_'+results_tag+'_'
        group_keys_to_stratify_cv = [group_key]
        
    if dataset_name.lower() == 'isic_sonic':
        label_colname_ = 'benign_malignant_01'
        original_data_csv_ = 'isic/df_with_sonic_study_name_id.csv'
        image_fp_  = 'isic/ImagesSmallerWithSonic'
        all_group_colnames_ = ['study_name_id',
                                'age_approx_by_decade_id', 
                                'age_over_45_id', 
                                'age_over_50_id',
                                'age_over_60_id', 
                                'anatom_site_general_id', 
                                'sex_id']
        results_descriptor_ ='isic_sonic_hpSel_'+results_tag+'_'
        group_keys_to_stratify_cv = [group_key]
        
        # This one is a special case - we want to do just train_fracs = [1.0]
        fracs_group_a = [1.0]
        run_type = 'max_all_groups'
        
    elif dataset_name.lower() == 'cifar':
        
        group_keys_to_stratify_cv = ['bird','horse','airplane','automobile']
        
        if (group_key == "air"):
            label_colname_ = "animal"
        else:
            label_colname_ = "air"
        original_data_csv_ = 'cifar4/df_cifar4_labels.csv'
        image_fp_ = 'cifar4/images'
        all_group_colnames_ = ["air", "animal"]
        results_descriptor_ ='cifar4_subsetting_hpSel_'+results_tag+'_'
        
    else:
        print('TODO: need to input the data and image files for {0}'.format(dataset_name))
            
    # instantiate the folds to be read, if that csv doesn't already exists
    split_data_csv = original_data_csv_.replace('.csv',
                                                '_{0}_fold_splits.csv'.format(num_cv_splits))
    split_data_csv_fp = os.path.join(data_dir, split_data_csv)
    if not os.path.exists(split_data_csv):
        # add the file with the splits
        print('5 fold split csv does not exist yet; making it now')
        add_stratified_kfold_splits(os.path.join(data_dir,
                                                 original_data_csv_),
                                                 group_keys_to_stratify_cv,
                                                 num_splits= num_cv_splits,
                                                 overwrite=False)
        
    # if the directory for the intermediate csvs doesn't exist, make it now
    tmp_dir = 'tmp'
    if not os.path.exists(os.path.join(data_dir, tmp_dir)):
        print(os.path.join(data_dir, tmp_dir), ' did not exist; making it now')
        os.mkdir(os.path.join(data_dir, tmp_dir))
        
    tmp_dir_for_splits = os.path.join(data_dir, 
                                      tmp_dir, 
                                      split_data_csv.split('/')[0])
    
    if not os.path.exists(tmp_dir_for_splits):
        print(tmp_dir_for_splits, ' did not exist; making it now')
        os.mkdir(tmp_dir_for_splits)
    print()
    
    # save in the tmp dir
    fn_template = split_data_csv.replace('.csv','_split_{0}.csv')
    # relative to data
    tmp_fn_template = os.path.join(tmp_dir, fn_template)
    
    print('searching over the following hyperparameters:')
    print('learning_rates', learning_rates)
    print('weight_decays', weight_decays)
    print('momentums', momentums)
    print('number_of_epochs', number_of_epochs)
    
    if 'GDRO' in objs:
        print('gdro_step_sizes', gdro_step_sizes)
        print('gdro_group_adjs', gdro_group_adjs)
    print()
    
    print('Will be evaluating the following (fold, seed) pairs:')
    for k in range(seed_beginning, seed_beginning + num_seeds):
        print(k, k)
    print()
    
    for k in range(seed_beginning, seed_beginning + num_seeds):
        # instantiate and name the split for this fold.
        this_fold_split_csv = tmp_fn_template.format(k)
        this_fold_split_save_fn = os.path.join(data_dir,
                                               this_fold_split_csv)
                                   
        # instantiate the file
        get_single_cv_kfold_split(split_data_csv_fp,
                                  k,
                                  save_fn = this_fold_split_save_fn)
        for num_epochs in number_of_epochs:
            for lr in learning_rates:
                for wd in weight_decays:
                    for momentum in momentums:
                
                        sgd_params = {'lr': lr, 
                                      'weight_decay': wd, 
                                      'momentum': momentum}
                    
                        for obj in objs:
                            print()
                            print('will be saving results to', results_descriptor_+obj)
                            print()
                        
                        # run the subset experiment on this fold - IS / ERM
                        subset_experiment(num_seeds_each_split, 
                                          k, 
                                          group_key,
                                          label_colname = label_colname_,
                                          original_data_csv = this_fold_split_csv,
                                          image_fp = image_fp_,
                                          all_group_colnames = all_group_colnames_,
                                          fracs_group_a = fracs_group_a,
                                          eval_key = 'val',
                                          dataset_name = dataset_name,
                                          run_type = run_type,
                                          results_descriptor = results_descriptor_,
                                          objs = objs_excluding_gdro,
                                          num_epochs = num_epochs,
                                          sgd_params = sgd_params,
                                          model_name = 'resnet18',
                                          data_dir=data_dir
                                          )
                        
                        # run the subset experiment on this fold - GDRO
                        if do_gdro:
                            for gdro_step_size in gdro_step_sizes:
                                for gdro_group_adj in gdro_group_adjs:
                                    # update gdro params for this hp config
                                    gdro_params = gdro_params_base.copy()
                                    gdro_params['gdro_step_size'] = gdro_step_size
                                    gdro_params['group_adjustment'] = gdro_group_adj
                                    
                                    subset_experiment(num_seeds_each_split, 
                                          k, 
                                          group_key,
                                          label_colname = label_colname_,
                                          original_data_csv = this_fold_split_csv,
                                          image_fp = image_fp_,
                                          all_group_colnames = all_group_colnames_,
                                          fracs_group_a = fracs_group_a,
                                          eval_key = 'val',
                                          dataset_name = dataset_name,
                                          results_descriptor = results_descriptor_,
                                          objs = ['GDRO'],
                                          num_epochs=num_epochs,
                                          sgd_params = sgd_params,
                                          gdro_params = gdro_params,
                                          model_name = 'resnet18',
                                          data_dir=data_dir
                                          )
                        
