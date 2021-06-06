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

# can add different 
from dataset import get_data_loaders 
    
def augment_experiment(num_seeds, 
                      seed_beginning, 
                      group_key,
                      label_colname,
                      original_data_csv,
                      image_fp,
                      all_group_colnames,
                      augment_sizes_by_group,
                      results_descriptor='debug',
                      skip_ERM = False):
    
    """Run the subsetting experiment. Assumes subsetting is performed on a binary variable.

    Arguments:
    num_seeds (int) -- how many random trials to run
    seed_beginning (int) -- beginning seed #
    group_key (str) -- column identifier of the column on which to subset, should be a column name of
                        original_data_csv
    label_colname (str) -- column idetifier of the column to predict, should be a column name of 
                           original_data_csv
    original_data_csv (str) -- file path (relative to multi-acc/data) of the full data csv
    image_fp (str) -- file path (relative to multi-acc/data) of the images listed in the full data csv
    all_group_colnames (list of strs) -- all columns that we might want to subset analysis on later
    augment_fracs_by_group ([2 x _ ] array of floats) -- fractions of the dataset to assign augmentation for
                                                         each group
    results_descriptor (str) - string to store and later identify results. Results from the experiment
                               will be stored in multi-acc/results/<results_descriptor>
    """
    
    sgd_params = {'lr': 0.001, 'weight_decay': 0.001, 'momentum': 0.9}

    # subset_groups is all of the different study_ids
    
    original_data = pd.read_csv(os.path.join('../../data',original_data_csv))
    print(original_data.keys())
    
    # get subset groups
    subset_groups = [[x] for x in list(original_data[group_key].unique())]
    subset_groups.sort()
    print('augmenting groups are ', subset_groups)
        
    # select training data
    training_data = original_data[original_data['fold'] == 'train']
    num_training_pts = len(training_data)
     
    # double zero index b/c we know there's just one key in the group
    num_group_a = (training_data[group_key] == subset_groups[0][0]).sum()
    num_group_b = num_training_pts - num_group_a
    
    train_str = 'there are {0} group {2} training pts and {1} other training pts'

    
    for t in range(seed_beginning, seed_beginning + num_seeds):
        print('seed ',t)

        # data dict will store all the results
        data_dict = {}
        data_dict['augment_groups'] = subset_groups

        sgd_str = "lr_{0}_weight_decay_{1}_momentum_{2}"
        sgd_param_specifier = sgd_str.format(sgd_params['lr'],
                                             sgd_params['weight_decay'],
                                             sgd_params['momentum'])
        
        
        results_general_path = '../../results/augment_results/'
        this_results_path = os.path.join(results_general_path, results_descriptor)
        
        for fp in [results_general_path,this_results_path]:
            if not os.path.exists(fp):
                os.makedirs(fp)
          
     
            # 1. run the experiment and save in fn_out for ERM
            fn_out_str_basic = os.path.join(this_results_path,
                                         'augment_{1}_seed_{0}_{2}_basic.pkl')

            fn_out_basic = fn_out_str_basic.format(t,
                                                   group_key,
                                                   sgd_param_specifier)

            experiments.augment_and_train(original_data_csv,
                                         image_fp,
                                         label_colname,
                                         all_group_colnames,
                                         group_key,
                                         subset_groups, 
                                         augment_sizes_by_group, 
                                         experiments.train_and_eval_basic, 
                                         sgd_params,
                                         fn_out_basic,
                                         train_fxn_params = {'num_epochs': 20},
                                         seed=t)
        

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
    
    
   # gonna need to fix this
    num_total = 4000
    
    num_group_a = [0, 1000, 2000, 3000, 4000]
    num_group_b = [num_total - x for x in num_group_a]
    augment_sizes_by_group = np.vstack((num_group_a, num_group_b)).T
    
    if dataset_name == 'isic':
        
        augment_experiment(num_seeds, 
                          seed_beginning, 
                          group_key,
                          label_colname = 'benign_malignant_01',
                          original_data_csv = 'isic/fresh/int_data/df_no_sonic.csv',
                          image_fp = 'isic/fresh/Data/Images',
                          all_group_colnames = ['study_name_id',
                                                'age_approx_by_decade_id', 
                                                'age_over_45_id', 
                                                'age_over_50_id',
                                                'age_over_60_id', 
                                                'anatom_site_general_id', 
                                                'sex_id'],
                          augment_sizes_by_group = augment_sizes_by_group,
                          results_descriptor='isic'
                          )
        
    elif dataset_name == 'cifar':
        print('todo: implement what the call to main should be')
    else:
        print('TODO: need to input the data and image files')