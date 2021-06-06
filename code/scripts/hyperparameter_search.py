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

# can add different 
from dataset import get_data_loaders 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('group_key', type=str, 
                        help='which key to subset on')
    parser.add_argument('--num_seeds', type=int, default=5, 
                        help='number of seeds to run')
    parser.add_argument('--seed_beginning', type=int, default=0, 
                        help='seed to start with')
    parser.add_argument('--dataset_name', type=str, default='isic', 
                        help='seed to start with')
    
    args = parser.parse_args()
    group_key = args.group_key
    num_seeds, seed_beginning = args.num_seeds, args.seed_beginning
    dataset_name = args.dataset_name
    
    fracs_group_a = [0.02, 0.05, 0.2, 0.5, 0.8, 0.95,  0.98]
    
    learning_rates = [0.01, 0.001, 0.0001]
    weight_decays = [.01, .001, 0.0001]
#    momentums = [0, 0.9]
    momentums = [0.9]
#    number_of_epochs = [10,20]
    number_of_epochs = [20]
    
    
    if dataset_name == 'isic':
        label_colname_ = 'benign_malignant_01'
        original_data_csv_ = 'isic/fresh/int_data/df_no_sonic.csv'
        image_fp_ = 'isic/fresh/Data/Images'
        all_group_colnames_ = ['study_name_id',
                                'age_approx_by_decade_id', 
                                'age_over_45_id', 
                                'age_over_50_id',
                                'age_over_60_id', 
                                'anatom_site_general_id', 
                                'sex_id']
        results_descriptor_ ='isic_hpSel'
        
    elif dataset_name == 'cifar':
        
        if (group_key == "air"):
            label_colname_ = "animal"
        else:
            label_colname_ = "air"
        original_data_csv_ = 'cifar4/df_cifar4_labels.csv'
        image_fp_ = 'cifar4/images'
        all_group_colnames_ = ["air", "animal"]
        results_descriptor_ ='cifar4_subsetting_hpSel_debug'
        
        
    else:
        print('TODO: need to input the data and image files')
    
    for lr in learning_rates:
            for wd in weight_decays:
                for momentum in momentums:
                        for num_epochs in number_of_epochs:
                            sgd_params = {'lr': lr, 'weight_decay': wd, 'momentum': momentum}
                            
                            subset_experiment(num_seeds, 
                                              seed_beginning, 
                                              group_key,
                                              label_colname = label_colname_,
                                              original_data_csv = original_data_csv_,
                                              image_fp = image_fp_,
                                              all_group_colnames = all_group_colnames_,
                                              fracs_group_a = fracs_group_a,
                                              eval_key='val',
                                              results_descriptor = results_descriptor_,
                                              num_epochs=num_epochs,
                                              sgd_params = sgd_params,
                                              )
